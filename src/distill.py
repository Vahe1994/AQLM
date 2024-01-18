from collections import defaultdict
from typing import List, Dict, Iterable, Sequence
from tqdm import trange

import torch
import torch.nn.functional as F

try:
    import wandb
    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.utils import maybe_get_0th_element


class DistillationWrapper:

    def __init__(self, model, layer_ids: Sequence[int]) -> None:
        self.model = model
        self.hooks = {}
        self.activations = {}
        for layer_id in layer_ids:
            def cache_output(l_id):
                def _hook(mod, inp, out):
                    self.activations[l_id] = maybe_get_0th_element(out)
                return _hook

            self.hooks[layer_id] = model.model.layers[layer_id].register_forward_hook(cache_output(layer_id))

    def __call__(self, *input_args, **input_kwargs) -> Dict[str, torch.Tensor]:
        self.model(*input_args, **input_kwargs)
        return self.activations

    def free(self):
        # remove hooks
        for _, hook in self.hooks.items():
            hook.remove()
        # clean dict with activations
        self.activations.clear()
        torch.cuda.empty_cache()


def _extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int], device=None):
    extracted_items = [maybe_get_0th_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, dim=0).to(device)


def _extract_into_tensor_dict(tensor_dict: List[Dict[str, torch.Tensor]], indices: Iterable[int], device=None):
    extracted_dicts = defaultdict(list)
    for i in indices:
        for k, v in tensor_dict[i].items():
            extracted_dicts[k].append(v)
    # concatenate tensors
    return {k: torch.cat(v, dim=0).to(device) for k, v in extracted_dicts.items()}


@torch.inference_mode()
def cache_teacher_activations(teacher_model, dataloader, args) -> List[Dict[str, torch.Tensor]]:
    cached_teacher_activations = []
    # get device
    device = maybe_get_0th_element(args.devices)
    # create wrapper
    teacher_wrapper = DistillationWrapper(teacher_model, args.distill_layers)

    for i in trange(len(dataloader),  total=len(dataloader), desc='Caching teacher activations', leave=False):
        batch = maybe_get_0th_element(dataloader[i]).to(device)
        # get activations
        teacher_activations = teacher_wrapper(batch)
        # add to dict
        cached_teacher_activations.append({k: v.cpu() for k, v in teacher_activations.items()})
        # clear
        teacher_wrapper.activations.clear()   

    teacher_wrapper.free()

    return cached_teacher_activations


def distill_activations(model, dataloader, teacher_activations, args):
    # set model into train mode
    model.train()
    # cast model to float
    model.float()
    # get device
    device = maybe_get_0th_element(args.devices)
    # initialize trainable parameters on main device; prepare to send them to replicas
    differentiable_parameters = [param for param in model.parameters() if param.requires_grad]
    print(f"Num trainable params: {(sum(param.numel() for param in differentiable_parameters) / 2 ** 30):.2f}B.")
    # init optimizer
    opt = torch.optim.Adam(
        differentiable_parameters,
        lr=args.distill_lr,
        betas=(args.distill_adam_beta1, args.distill_adam_beta2)
    )
    # create wrapper
    wrapper = DistillationWrapper(model, args.distill_layers)
    # prepare loss_fn
    if args.distill_loss == 'l2':
        loss_fn = F.mse_loss
    elif args.distill_loss == 'l2_norm':
        loss_fn = lambda x, y: (x - y).pow(2).mean() / y.pow(2).mean()
    elif args.distill_loss == 'cosine':
        def cosine_distance(x, y):
            x_n = x.div(x.norm(dim=-1, keepdim=True))
            y_n = y.div(y.norm(dim=-1, keepdim=True))
            return (x_n * y_n).sum(dim=-1).mean().mul(-0.5).add(0.5)
        loss_fn = cosine_distance
    else:
        raise ValueError(f"Unknown loss_fn {args.distill_loss}")

    num_samples = len(dataloader)
    epoch_samples = num_samples - num_samples % args.distill_batch_size # number of samples in epoch
    local_batches_per_epoch = epoch_samples // args.distill_local_batch_size
    num_accumulation_steps = args.distill_batch_size // args.distill_local_batch_size

    step = 0

    if args.distill_gradient_checkpointing:
        model.gradient_checkpointing_enable()

    for epoch in trange(
        args.distill_max_epochs,
        total=args.distill_max_epochs,
        desc="Distilling quantized model",
        leave=False
    ):
        # prepare batch indices
        batch_indices_epoch = torch.randperm(num_samples)[:epoch_samples].chunk(local_batches_per_epoch)
        # reset number of accumulated steps and losses
        steps_accumulated = 0
        loss_accumulated = 0
        epoch_loss = 0

        for batch_indices in batch_indices_epoch:
            # convert tensor to list
            batch_indices = batch_indices.tolist()
            # prepare inputs
            inputs = _extract_into_tensor(dataloader, batch_indices, device)
            # prepare targets
            targets =_extract_into_tensor_dict(teacher_activations, batch_indices, device)
            # get student outputs
            with torch.autocast(device_type='cuda', enabled=args.distill_amp):
                outputs = wrapper(inputs)
                loss = 0
                for layer_id in outputs:
                   loss += loss_fn(outputs[layer_id], targets[layer_id].float())
                loss.div_(num_accumulation_steps).div_(len(outputs))

            loss.backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Distillation loss is {loss}")
            
            loss_accumulated += loss.item()
            epoch_loss += (num_accumulation_steps / local_batches_per_epoch) * loss.item()

            if steps_accumulated == num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                # log stats if requested
                if step % args.print_frequency == 0:
                    print(f"step={step}\tloss={loss_accumulated:.4f}\t")
                    if args.wandb:
                        assert has_wandb
                        wandb.log({'distillation/loss': loss_accumulated, 'distillation/step': step})
                # reset step and losses
                steps_accumulated = 0
                loss_accumulated = 0
                step += 1
        print(f"epoch={epoch}\tloss={epoch_loss:.4f}\t")
    # reset student wrapper
    wrapper.free()
    # set student back to eval
    model.eval()
