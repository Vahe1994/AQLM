from typing import List, Iterable
from tqdm import trange

import torch
import torch.nn.functional as F

from .utils import maybe_get_0th_element


def kl_div_loss(student_logits, teacher_hiddens, temperature):
    "Kullback-Leibler divergence loss"
    num_tokens = student_logits.numel() / student_logits.size(-1)
    return (
        F.kl_div(
            input=F.log_softmax(student_logits / temperature, dim=-1),
            target=F.log_softmax(teacher_hiddens / temperature, dim=-1),
            log_target=True,
            reduction="sum",
        )
        * (temperature ** 2)
        / num_tokens
    )


def _extract_into_tensor(tensor_list: List[torch.Tensor], indices: Iterable[int]):
    extracted_items = [maybe_get_0th_element(tensor_list[i]) for i in indices]
    return torch.cat(extracted_items, axis=0)


@torch.inference_mode()
def cache_teacher_hiddens(teacher_model, dataloader, args) -> List[torch.Tensor]:
    cached_teacher_hiddens = []
    # get device
    device = maybe_get_0th_element(args.devices)

    for i in trange(len(dataloader),  total=len(dataloader), desc='Caching teacher outputs', leave=False):
        batch = maybe_get_0th_element(dataloader[i]).to(device)
        # get activations
        teacher_hiddens = teacher_model.model(batch).last_hidden_state
        cached_teacher_hiddens.append(teacher_hiddens.cpu())

    return cached_teacher_hiddens


def distill_logits(model, dataloader, teacher_hiddens, args):
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

    num_samples = len(dataloader)
    epoch_samples = num_samples - num_samples % args.distill_batch_size # number of samples in epoch
    local_batches_per_epoch = epoch_samples // args.distill_local_batch_size
    num_accumulation_steps = args.distill_batch_size // args.distill_local_batch_size

    step = 0
    previous_best_loss = float("inf")

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
        # reset number of accumulated steps
        steps_accumulated = 0
        # reset loss numerator and denominator
        loss_numerator = 0
        loss_denominator = 0 

        for batch_indices in batch_indices_epoch:
            # convert tensor to list
            batch_indices = batch_indices.tolist()
            # prepare inputs
            inputs = _extract_into_tensor(dataloader, batch_indices).to(device)
            # prepare targets
            pre_targets = _extract_into_tensor(teacher_hiddens, batch_indices).to(device)
            with torch.no_grad():
                teacher_logits = model.lm_head(pre_targets.float())
            # get student outputs
            with torch.autocast(device_type='cuda', enabled=args.distill_amp):
                student_logits = model(inputs).logits
                loss = kl_div_loss(student_logits, teacher_logits, args.distill_temperature)
            loss.div(num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss).item():
                raise ValueError(f"Distillation loss is {loss}")
            
            loss_numerator += loss.item()
            loss_denominator += 1

            if steps_accumulated == num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0
                step += 1
            else:
                continue

            if step % args.print_frequency == 0:
                print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        if step != args.print_frequency == 0:
            print(f"epoch={epoch}\tstep={step}\tloss={loss_numerator / loss_denominator:.10f}\t")

        if args.distill_relative_mse_tolerance is not None:
            epoch_loss = loss_numerator / loss_denominator
            if epoch_loss / previous_best_loss > (1.0 - args.distill_relative_mse_tolerance):
                print(f"Early stopping on epoch {epoch}.")
                return
            previous_best_loss = min(epoch_loss, previous_best_loss)
    # set student back to eval
    model.eval()
