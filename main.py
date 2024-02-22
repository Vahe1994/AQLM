import os
import time
from argparse import Namespace
from itertools import chain
from typing import Any, Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
from tqdm import trange
from tqdm.auto import trange
from transformers import PreTrainedModel

from aq_engine import AQEngine
from src.aq import QuantizedLinear
from src.datautils import get_loaders
from src.finetune import finetune_groupwise
from src.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
)
from src.utils import using_tf32

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False


def quantize_model(model, args):
    """main entry point to functions for model quantization"""
    tick = time.time()

    print("Loading data ...")
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )
    results = quantize_aq(model, dataloader, args)
    print(f"quantization time: {time.time() - tick:.1f}")
    return results


@torch.no_grad()
def get_inps(
    model: PreTrainedModel, data_iterable: Iterable, args: Namespace, nsamples: Optional[int] = None
) -> Sequence[torch.Tensor]:
    """
    mocks model launch to collect inputs to the first model layer
    :returns: a list of torch tensors with activations for each device in args.devices.
    Each tensor has shape [nsample_per_device, seq_len, hid_size]
    """
    print("catching layer inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples or len(data_iterable)
    device = args.devices[0] if not args.offload_activations else torch.device("cpu")
    assert nsamples is not None

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(device)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_device = emb.weight.device
    if emb_device.type != "cuda":
        emb = emb.to(device)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(device)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(device)
    device = emb.weight.device  # now default device is the one where the embeddings are.
    layer_device = next(layers[0].parameters()).device
    layers[0] = layers[0].to(device)

    dtype = next(iter(model.parameters())).dtype
    nsamples_per_device = (nsamples - 1) // len(args.devices) + 1
    inps = [
        torch.zeros(
            (min(nsamples_per_device, nsamples - i * nsamples_per_device), model.seqlen, model.config.hidden_size),
            dtype=dtype,
            device=args.devices[i] if not args.offload_activations else "cpu",
            pin_memory=args.offload_activations,
        )
        for i in range(len(args.devices))
    ]
    forward_arg_names = ["attention_mask", "position_ids"]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "alibi": None}

    class CatcherExit(Exception):
        pass

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"] // nsamples_per_device][cache["i"] % nsamples_per_device] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise CatcherExit()

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch_inps in data_iterable:
        try:
            if isinstance(batch_inps, (list, tuple)):
                batch_inps, *_ = batch_inps
            batch_inps = batch_inps.to(device)
            # call model.forward to trigger the Catcher
            model(batch_inps, attention_mask=torch.ones_like(batch_inps))
        except CatcherExit:
            pass  # exit after catcher finished without running the rest of the model layers
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_device)
    model.get_input_embeddings().to(emb_device)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_device)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_device)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    assert cache["i"] == nsamples
    return inps, forward_args


@torch.no_grad()
def quantize_aq(model: PreTrainedModel, dataloader: Iterable, args: Namespace):
    assert not torch.backends.cuda.matmul.allow_tf32
    print("\nStarting AQ quantization ...")
    inps, forward_args = get_inps(model, dataloader, args)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    num_codebooks = args.num_codebooks
    use_cache = model.config.use_cache
    model.config.use_cache = False

    quantizers = {}
    overall_bits = 0
    model_number_of_params = 0
    layers = get_layers(model)

    for layer_index in range(len(layers)):
        print(f"\n---------------- Layer {layer_index} of {len(layers)} ----------------")
        stats_payload = {}
        start_time = time.time()

        # quantized layer will return there
        layer_device_original = next(layers[layer_index].parameters()).device
        # backup layer dtype
        layer_dtype_original = next(layers[layer_index].parameters()).dtype
        print(f"{layer_device_original=}")
        layer = layers[layer_index].to(args.devices[0])
        for k, v in forward_args.items():
            forward_args[k] = v.to(args.devices[0]) if isinstance(v, torch.Tensor) else v

        if args.true_sequential:
            sequential = get_sequential_groups(model)
        else:
            sequential = [list(find_sublayers(layer).keys())]

        for names in sequential:
            if len(args.devices) == 1:
                assert len(inps) == len(outs) == 1  # number of per-device inputs/outputs
                aq_handlers = init_aq_engines(
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps[0],
                    outs[0],
                    **forward_args,
                )
            else:
                aq_handlers = init_aq_engines_parallel(
                    args.devices,
                    layer,
                    [
                        name
                        for name in names
                        if ((".gate" not in name.lower()) or ("mixtral" not in model.config.model_type.lower()))
                    ],
                    inps,
                    outs,
                    **forward_args,
                )
            for sublayer_name in aq_handlers.keys():
                print(f"Quantizing module {sublayer_name} of layer {layer_index}")
                if "mixtral" in model.config.model_type.lower() and args.mix_compression:
                    assert "mixtral" in model.config.model_type.lower()
                    if "self_attn" in sublayer_name.lower():
                        args.num_codebooks = 2 * num_codebooks
                    else:
                        args.num_codebooks = num_codebooks
                    print(sublayer_name.lower(), " mixtral num codebooks", args.num_codebooks)
                quantized_weight = aq_handlers[sublayer_name].quantize(args=args, verbose=True)

                with torch.no_grad():
                    assert aq_handlers[sublayer_name].layer.weight in set(
                        layer.parameters()
                    )  # test that this is not a replica

                    new_linear = QuantizedLinear(quantized_weight, aq_handlers[sublayer_name].layer.bias)
                    if args.use_checkpointing:
                        new_linear.use_checkpoint = True
                        print("ENABLED CHECKPOINTING FOR", sublayer_name)
                    found_original = False
                    for submodule in layer.modules():
                        for child_name, child_module in submodule.named_children():
                            if child_module is aq_handlers[sublayer_name].layer:
                                setattr(submodule, child_name, new_linear)
                                found_original = True  # note: do not break to handle tied layers

                    assert found_original, f"could not find {sublayer_name}"

                weight_avg_bits = quantized_weight.estimate_nbits_per_parameter()
                overall_bits += int(weight_avg_bits * torch.numel(aq_handlers[sublayer_name].layer.weight.data))
                model_number_of_params += torch.numel(aq_handlers[sublayer_name].layer.weight.data)
                print("curent_avg_bits", overall_bits / model_number_of_params)
                quantizers["model.layers.%d.%s" % (layer_index, sublayer_name)] = ()  # to be updated

            print("PREPARING TO FINETUNE")
            print(layer)
            layer = layer.to(dtype=torch.float32)
            with using_tf32(enabled=True):
                layer = finetune_groupwise(layer=layer, inps=inps, outs=outs, args=args, **forward_args)
            layer = layer.to(dtype=layer_dtype_original)
            print("FINISHED FINETUNING")
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            print(f"Saving layer {layer_index}... to {layer_save_path}")
            torch.save(layer, layer_save_path)

        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            out_losses = update_outs(layer, inps[0], outs[0], compute_mse=not args.skip_out_loss, **forward_args)
        else:
            out_losses = update_outs_parallel(
                args.devices, layer, inps, outs, compute_mse=not args.skip_out_loss, **forward_args
            )

        layers[layer_index] = layer.to(layer_device_original)
        del layer
        del aq_handlers
        torch.cuda.empty_cache()

        inps, outs = outs, inps

        # Logging
        stats_payload["layer_time"] = time.time() - start_time
        stats_payload["out_loss"] = torch.mean(torch.Tensor(out_losses)).item()
        stats_payload["Step"] = layer_index
        if args.wandb:
            wandb.log({"out_loss": stats_payload["out_loss"]}, step=layer_index)
            wandb.log({"layer_time": stats_payload["layer_time"]}, step=layer_index)
        print(stats_payload)

    print("=====================\nFinal stats:")
    if args.save:
        torch.save(vars(args), args.save + "/args.pt")
        already_saved_weights = set()
        for layer in get_layers(model):
            for param in layer.parameters():
                already_saved_weights.add(param)
        not_quantized_weights = {
            name: param for name, param in model.named_parameters() if param not in already_saved_weights
        }
        torch.save(not_quantized_weights, args.save + "/not_quantized_weights.pt")

    if args.wandb:
        wandb.log({"max_cuda_mem_quantize": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
        wandb.log({"Avg_bits": overall_bits / model_number_of_params})
    model.config.use_cache = use_cache
    print(f"quantize: {torch.cuda.max_memory_allocated()=:,}")
    return quantizers


@torch.no_grad()
def perplexity_eval(model, testenc, args):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")

    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False

    inps, forward_args = get_inps(model, testenc, args, nsamples=nsamples)
    outs = [torch.zeros_like(inp_tensor, pin_memory=inp_tensor.is_pinned()) for inp_tensor in inps]
    device = args.devices[0]
    for k, v in forward_args.items():
        forward_args[k] = v.to(device) if isinstance(v, torch.Tensor) else v

    layers = get_layers(model)
    for i in trange(len(layers), desc="processing eval data by layer"):
        layer = layers[i].to(device)
        if len(args.devices) == 1:
            assert len(inps) == len(outs) == 1
            update_outs(layer, inps[0], outs[0], compute_mse=False, **forward_args)
        else:
            update_outs_parallel(args.devices, layer, inps, outs, compute_mse=False, **forward_args)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    get_model_head(model).to(device)
    testenc = testenc.to(device)
    nsamples_per_device = len(inps[0])
    assert len(set(map(len, inps[:-1]))) <= 1 and len(inps[-1]) <= len(inps[0])

    nlls = []
    for i in range(nsamples):
        inp = inps[i // nsamples_per_device][i % nsamples_per_device].to(args.devices[0], non_blocking=True)
        lm_logits = get_lm_logits(inp.to(device), model)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\n{args.dataset_name} perplexity = {ppl.item():.4f}\n")

    get_model_head(model).to(torch.device("cpu"))

    if args.wandb:
        wandb.log({args.dataset_name: ppl.item()})

    model.config.use_cache = use_cache


@torch.no_grad()
def init_aq_engines(
    layer: nn.Module,
    names: Sequence[str],
    inps_tensor: torch.Tensor,
    outs_tensor: torch.Tensor,
    **forward_args: Dict[str, Any],
) -> Dict[str, AQEngine]:
    """
    Create a dictionary of AQUtil instances for each quantized layer;
    Run forward pass on each sample in inps_tensor; write output activations to outs_tensor (in-plance)
    Accumulate XTX to each one of aq_handlers
    :param layer: transformer layer with one or more linear layer to be quantized
    :param names: a list/tuple of string names for linear sub-layers inside :layer: that shall be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a dictionary where keys are full layer names and values are AQUtil instances ready to run .quantize
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    all_sublayers = find_sublayers(layer)
    subset = {name: all_sublayers[name] for name in names}
    assert len(subset) > 0
    aq_handlers = {}
    for sublayer_name in subset:
        aq_handlers[sublayer_name] = AQEngine(subset[sublayer_name])

    # wrap all quantized sub-layers with a wrapper that accumulates inputs on forward
    # note: the code below uses wrappers instead of hooks because hooks cause bugs in multi-gpu code
    wrapped_layer_to_hander = {aq_handler.layer: aq_handler for aq_handler in aq_handlers.values()}
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if child in wrapped_layer_to_hander:
                setattr(module, child_name, _LayerWrapperThatAccumulatesXTX(child, wrapped_layer_to_hander[child]))

    # compute output activations and accumulate XTX
    for j in trange(len(inps_tensor), desc="calc outs before quantization", leave=False):
        outs_tensor[j].copy_(
            layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0].view_as(outs_tensor[j]), non_blocking=True
        )

    # remove wrappers
    for module in list(layer.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, _LayerWrapperThatAccumulatesXTX):
                setattr(module, child_name, child.wrapped_layer)
    return aq_handlers


class _LayerWrapperThatAccumulatesXTX(nn.Module):
    def __init__(self, layer: nn.Module, aq_handler: AQEngine):
        super().__init__()
        self.wrapped_layer, self.aq_handler = layer, aq_handler

    def forward(self, input, *args, **kwargs):
        self.aq_handler.add_batch(input)
        return self.wrapped_layer(input, *args, **kwargs)


@torch.no_grad()
def init_aq_engines_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    names: Sequence[str],
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    **forward_args,
):
    """Parallel version of init_aq_engines; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    layer_replicas[0] = layer  # this ensures that aq_handlers returned by 0-th replica operate on the main layer
    funcs_by_device = [init_aq_engines for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], names, inps[i], outs[i]))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    aq_handles_by_device: Sequence[Dict[str, AQEngine]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    aq_handlers = aq_handles_by_device[0]
    for key, aq_handler in aq_handlers.items():
        replica_handlers = [device_aq_handlers[key] for device_aq_handlers in aq_handles_by_device]
        replica_nsamples = [replica_handler.nsamples for replica_handler in replica_handlers]
        total_nsamples = sum(replica_nsamples)
        aq_handler.XTX = sum(
            (replica_handlers[i].XTX * (replica_nsamples[i] / total_nsamples)).to(devices[0], non_blocking=True)
            for i in range(len(devices))
        )
        aq_handler.nsamples = total_nsamples
    return aq_handlers


@torch.no_grad()
def update_outs(
    layer: nn.Module, inps_tensor: torch.Tensor, outs_tensor: torch.Tensor, compute_mse: bool, **forward_args
) -> Sequence[float]:
    """
    Update outs_tensor with new activations and optionally compute sample-wise mse loss with previous activations
    :param layer: transformer layer with one or more linear layer to be quantized
    :param inps_tensor: a tensor of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs_tensor: a tensor to write output activations into, [nsamples_per_device, seq_len, hidden_size]
    :note: outs_tensor must contain previous activations with which to compute MSE loss
    :param compute_mse: if True, return a list of sample-wise mse losses; if False, return an empty sequence
    :param forward_args: additional keyword arguments, e.g. attention mask
    :returns: a list of mean squared errors for each sequence
    """
    device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    out_losses = []
    for j in trange(len(inps_tensor), desc="calc outs after quantization", leave=False):
        outs_batch = layer(inps_tensor[j].to(device).unsqueeze(0), **forward_args)[0]
        if compute_mse:
            outs_batch_loss = (
                (outs_batch - outs_tensor[j].to(device))
                .float()
                .square()
                .view(outs_batch.shape[0], -1)
                .mean(dim=1)
                .sqrt()
            )
            outs_batch_loss /= outs_batch.view(outs_batch.shape[0], -1).float().std(dim=1)
            out_losses.append(outs_batch_loss.item())
        outs_tensor[j].copy_(outs_batch.reshape_as(outs_tensor[j]), non_blocking=True)
    return out_losses


@torch.no_grad()
def update_outs_parallel(
    devices: Sequence[torch.device],
    layer: nn.Module,
    inps: Sequence[torch.Tensor],
    outs: Sequence[torch.Tensor],
    compute_mse: bool,
    **forward_args,
) -> Sequence[float]:
    """Parallel version of update_outs_and_compute_losses; works on lists of input/output tensors"""
    layer_replicas = torch.nn.parallel.replicate(layer, devices=devices, detach=True)
    funcs_by_device = [update_outs for _ in devices]
    inputs_by_device = []
    kwargs_by_device = []
    for i in range(len(devices)):
        inputs_by_device.append((layer_replicas[i], inps[i], outs[i], compute_mse))
        kwargs_by_device.append(
            {
                k: (v.to(devices[i], non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for k, v in forward_args.items()
            }
        )
    out_losses_by_device: Sequence[Sequence[float]] = torch.nn.parallel.parallel_apply(
        funcs_by_device, inputs_by_device, kwargs_by_device, devices=devices
    )
    return list(chain(*out_losses_by_device))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=None,
        help="Number of calibration data samples.If None take all calibration data.",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--use_checkpointing",
        action="store_true",
        help="Whether to use checkpoining in finetuning",
    )
    parser.add_argument(
        "--mix_compression",
        action="store_true",
        help="Compress .self_attn in 4 bits, .block_sparse_moe.experts to 2.3 for mixtral.",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=None, help="Path to save quantized statistics.")
    parser.add_argument("--devices", metavar="N", type=str, nargs="+", default=None, help="List of devices")
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--num_codebooks",
        type=int,
        default=1,
        help="#Number of codebooks per layer",
    )
    parser.add_argument(
        "--nbits_per_codebook",
        type=int,
        default=16,
        help="each codebook will contain 2 ** nbits_per_codebook vectors",
    )
    parser.add_argument(
        "--out_group_size",
        type=int,
        default=1,
        help="How many output units are quantized together",
    )
    parser.add_argument(
        "--in_group_size",
        type=int,
        default=8,
        help="How many input features are quantized together",
    )
    parser.add_argument(
        "--scale_nbits",
        type=int,
        default=0,
        help="Number of bits dedicated to the learnable group-wise scale. 0 means do not use group-wise scales "
        "(still has row-wise scales), 1-15 means using per-group scales quantized to this many bits, "
        "16+ means use per-group scales but do not quantize them",
    )
    parser.add_argument(
        "--codebook_value_nbits",
        type=int,
        default=16,
        help="If below 16, quantize the values in each codebook with the specified number of bits",
    )
    parser.add_argument(
        "--codebook_value_num_groups",
        type=int,
        default=1,
        help="Split codebook vectors into this many groups for quantizations. Only used when quantized codebooks.",
    )

    parser.add_argument(
        "--init_max_iter",
        type=int,
        default=100,
        help="Number of K-Means iterations used to initialize codebooks and codes",
    )
    parser.add_argument(
        "--use_faiss",
        action="store_true",
        help="Whether to use faiss.Kmeans when initializing codebooks and codes",
    )
    parser.add_argument(
        "--init_max_points_per_centroid",
        type=int,
        default=None,
        help="During K-means initialzation, sample (this_many * 2 ^ nbits_per_codebook) points for training K-means",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for Adam optimizer",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Keep top-(this_many) best candidates for each codebook when finding optimal codes",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of beam search rounds before the optimization is forcibly stopped.",
    )
    parser.add_argument(
        "--relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=100,
        help="Run (this many) Adam updates before every beam search round",
    )
    parser.add_argument(
        "--finetune_max_epochs",
        type=int,
        default=1000,
        help="Run this many passes over training data when doing finetuning; No finetuning if set to 0.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=1e-5,
        help="Finetuning learning rate",
    )
    parser.add_argument(
        "--finetune_relative_mse_tolerance",
        type=float,
        default=None,
        help="Stop finetuning when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)",
    )
    parser.add_argument(
        "--finetune_batch_size",
        type=int,
        default=1,
        help="(finetuning only) train on batches of this many sequences, globally across all GPUs",
    )
    parser.add_argument(
        "--finetune_adam_beta1",
        type=float,
        default=0.9,
        help="Finetuning adam_beta1",
    )
    parser.add_argument(
        "--finetune_adam_beta2",
        type=float,
        default=0.95,
        help="Finetuning adam_beta2",
    )
    parser.add_argument("--finetune_keep_best", action="store_true")
    parser.add_argument(
        "--local_batch_size",
        type=int,
        default=None,
        help="(finetuning only) Per-device and per-forward-pass batch size used to accumulate global --batch_size",
    )
    parser.add_argument(
        "--print_frequency",
        type=int,
        default=10,
        help="Print Adam progress after each print_frequency updates",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help="Skip model quantization and immediately evaluate the loaded model",
    )

    torch.set_num_threads(min(16, torch.get_num_threads()))
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.devices is None:
        args.devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    else:
        args.devices = [torch.device(device_str) for device_str in args.devices]
    assert all(isinstance(device, torch.device) for device in args.devices)

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "AQ")
            + f"_num_codebooks_{args.num_codebooks}"
            + f"_out_group_size_{args.out_group_size}"
            + f"_in_group_size_{args.in_group_size}"
            + f"_nbits_per_codebook_{args.nbits_per_codebook}"
            + f"_codebook_value_nbits_{args.codebook_value_nbits}"
            + f"_codebook_value_num_groups_{args.codebook_value_num_groups}"
            + f"_scale_nbits_{args.scale_nbits}"
            + f"_steps_per_epoch_{args.steps_per_epoch}"
            + f"_init_max_iter{args.init_max_iter}"
            + f"_{len(args.devices)}gpus"
        )
        args.group_size = args.in_group_size * args.out_group_size

        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )

    print("\n============ Load model... ============")
    model = get_model(args.model_path, args.load, args.dtype, args.model_seqlen).train(False)

    if not args.load and not args.no_quant:
        print("\n============ Quantizing model... ============")
        quantize_model(model, args)

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.model_path,
            seqlen=model.seqlen,
            eval_mode=True,
        )
        args.dataset_name = dataset
        perplexity_eval(model, testloader, args)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
