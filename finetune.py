import os
import shutil
import argparse
from copy import deepcopy
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from accelerate.hooks import remove_hook_from_submodules

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.modelutils import get_model, get_layers, save_not_quantized_weights
from src.datautils import get_loaders
from src.utils import maybe_get_0th_element, _extract_into_tensor

from main import perplexity_eval


@torch.inference_mode()
def cache_logits(model, dataloader, device):
    cached_logits = []
    for i in trange(len(dataloader),  total=len(dataloader), desc='Caching teacher logits', leave=False):
        with torch.autocast(device_type='cuda', enabled=args.amp):
            batch = maybe_get_0th_element(dataloader[i]).to(device)
        cached_logits.append(model(batch).logits.cpu())
    return cached_logits

def kl_div(student_logits, teacher_logits, temp):
    C = student_logits.shape[-1] # num classes
    return temp ** 2 * F.kl_div(
        input=F.log_softmax(student_logits.view(-1, C) / temp, dim=-1),
        target=F.log_softmax(teacher_logits.view(-1, C) / temp, dim=-1),
        log_target=True,
        reduction="batchmean",
    )

@torch.no_grad()
def evalulate(model, loader, logits, batch_size):
    model.eval()
    loss_numerator, loss_denominator = 0, 0
    # convert tensor to list
    for i in range(0, len(loader), batch_size):
        batch_ids = range(i, i + batch_size)
        inputs = _extract_into_tensor(loader, batch_ids, device=device)
        targets = _extract_into_tensor(logits, batch_ids, device=device)
        outputs = model(inputs).logits
        loss = kl_div(outputs, targets, args.temperature)
        loss_numerator += loss.item()
        loss_denominator += 1
    return loss_numerator / loss_denominator


def finetune(
    model, 
    train_loader, 
    train_logits, 
    args, 
    device, 
    val_loader=None, 
    val_logits=None
):
    # cast model to float
    model.float()
    diff_params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    print(f"Fine-tuning {sum(param.numel() for _, param in diff_params.items())} parameters")
    opt = torch.optim.Adam(diff_params.values(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    num_accumulation_steps = args.batch_size // args.microbatch_size
    num_samples = len(train_loader)
    epoch_samples = num_samples - num_samples % args.microbatch_size
    microbatches_per_epoch = epoch_samples // args.microbatch_size

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    run_validation = val_loader is not None and val_logits is not None
    # validate before training
    if run_validation:
        valid_loss_epoch = evalulate(model, val_loader, val_logits, args.microbatch_size)
        print(f"Evaluation before training.")
        print(f"valid loss={valid_loss_epoch:.2e}\t")
        best_loss = valid_loss_epoch
        best_params = deepcopy(diff_params)
        worse_count = 0

    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_numerator, loss_denominator = 0, 0
        steps_accumulated = 0
        # prepare batch indices
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(microbatches_per_epoch)

        for batch_indices in tqdm(batch_indices_epoch, desc=f"Train epoch {epoch}", leave=False):
            # convert tensor to list
            batch_indices = batch_indices.tolist()
            inputs = _extract_into_tensor(train_loader, batch_indices, device=device)
            targets = _extract_into_tensor(train_logits, batch_indices, device=device)

            with torch.autocast(device_type='cuda', enabled=args.amp):
                outputs = model(inputs).logits
                loss = kl_div(outputs, targets, args.temperature)

            if not torch.isfinite(loss).item():
                raise ValueError(f"Fine-tuning loss is {loss}")

            scaler.scale(loss / num_accumulation_steps).backward()

            if steps_accumulated == num_accumulation_steps:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                # reset accumulated step and loss
                steps_accumulated = 0

            steps_accumulated += 1
            loss_numerator += loss.item()
            loss_denominator += 1
        train_loss_epoch = loss_numerator / loss_denominator

        if run_validation:
            valid_loss_epoch = evalulate(model, val_loader, val_logits, args.microbatch_size)
        
        # log losses in the end of the epoch
        print('-' * 10)
        print(f"epoch={epoch}")
        print(f"train loss={train_loss_epoch:.2e}\t")
        if run_validation:
            print(f"valid loss={valid_loss_epoch:.2e}\t")

        if args.wandb:
            wandb.log({'train_loss': train_loss_epoch}, step=epoch)
            if run_validation:
                wandb.log({'valid_loss': valid_loss_epoch}, step=epoch)

        if run_validation:
            if valid_loss_epoch < best_loss:
                print(f"new best loss {valid_loss_epoch:.2e} on epoch {epoch}")
                best_loss = valid_loss_epoch
                best_params = deepcopy(diff_params)
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= args.early_stop:
                    break
    
    if run_validation:
        model.load_state_dict(best_params, strict=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=True)
    # Model params
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="path or name of the teacher model",
    )
    parser.add_argument(
        "--quant_model",
        type=str,
        required=True,
        help="path to quantized model",
    )
    # Data params
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name [c4, pajama] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=1024,
        help="number of samples",
    )
    parser.add_argument(
        "--model_seqlen",
        type=int,
        default=4096,
        help="Model seqlen and calibration data context length.",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=0,
        help="size of validation split",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    # Training params
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="finetuning learning rate",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.90,
        help="Adam beta1",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="Adam beta2",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="training batch size",
    )
    parser.add_argument(
        "--microbatch_size",
        type=int,
        default=None,
        help="training microbatch size",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Distillation temperature",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether to apply gradient checkpointing",
    )
    parser.add_argument(
        "--amp",
         action="store_true",
        help="Whether to use amp",
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=3,
        help="Terminate finetuning if loss doesn't improve after this number of epochs.",
    )
    # Logging params
    parser.add_argument(
        "--wandb", 
        action="store_true", 
        help="Whether to use wandb or store locally."
    )
    # Save params
    parser.add_argument(
        "--save", 
        type=str, 
        default=None, 
        help="Path to save quantized statistics."
    )
    # Misc params
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for calibration data and initialization. "
        "Note that the main training is not strictly deterministic.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model in",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default=None,
        choices=[None, "auto"],
        help="accelerate device map",
    )
    args = parser.parse_args()
    args.microbatch_size = args.microbatch_size or args.batch_size
    # get device
    assert torch.cuda.is_available()
    device = 'cuda'
    args.devices = [device] # needed for perplexity eval
    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # get data
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.base_model,
        seqlen=args.model_seqlen,
    )
    if args.val_size > 0:
        all_ids = torch.randperm(len(dataloader))
        train_ids, val_ids = all_ids[args.val_size:], all_ids[:args.val_size]
        train_dataloader = [dataloader[i] for i in train_ids]
        val_dataloader = [dataloader[i] for i in val_ids]
    else:
        train_dataloader = dataloader
        val_dataloader = None
    # create original model
    orig_model = get_model(args.base_model, None, args.device_map, args.dtype, args.model_seqlen)
    if not args.device_map:
        orig_model = orig_model.to(device)
    # cache logits
    orig_train_logits = cache_logits(orig_model, train_dataloader, device)
    if val_dataloader:
        orig_val_logits = cache_logits(orig_model, val_dataloader, device)
    else:
        orig_val_logits = None
    del orig_model
    torch.cuda.empty_cache()
    quant_model = get_model(args.base_model, args.quant_model, args.device_map, args.dtype, args.model_seqlen)
    if not args.device_map:
        quant_model = quant_model.to(device)

    # finetune
    finetune(
        quant_model, 
        train_loader=train_dataloader,
        train_logits=orig_train_logits,
        args=args,
        device=device,
        val_loader=val_dataloader,
        val_logits=orig_val_logits
    )

    # offload model to cpu
    quant_model = quant_model.cpu()
    if args.device_map:
        remove_hook_from_submodules(quant_model)

    # save model
    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for layer_index, layer in enumerate(get_layers(quant_model)):
            layer_save_path = os.path.join(args.save, f"{layer_index}.pth")
            torch.save(layer, layer_save_path)
        save_not_quantized_weights(quant_model, args.save)
        # copy args
        shutil.copy(os.path.join(args.quant_model, "args.pt"), os.path.join(args.save, "args.pt"))

    print("\n============ Evaluating perplexity... ============")
    torch.cuda.reset_peak_memory_stats()
    datasets = ["wikitext2", "ptb", "c4"]
    if args.new_eval:
        datasets = ["wikitext2", "ptb-new", "c4-new"]
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            seed=args.seed,
            model_path=args.base_model,
            seqlen=quant_model.seqlen,
            eval_mode=True,
        )
        args.dataset_name = dataset
        perplexity_eval(quant_model, testloader, args)

    print(f"eval: {torch.cuda.max_memory_allocated()=:,}")
    if args.wandb:
        wandb.log({"max_cuda_mem_eval": round(torch.cuda.max_memory_allocated() / 1e9, 2)})
