# PV-Tuning

Supplementary code for **PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression**.

Will gradually publish models this week, stay tuned!

# Installation

####
Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

#### Loading / caching datasets and tokenizer

The script will require downloading and caching locally the relevant tokenizer and the datasets. 
They will be saved in default Huggingface Datasets directory unless alternative location is provided by env variables.
See [relevant Datasets documentation section](https://huggingface.co/docs/datasets/main/en/cache#cache-directory)

#### Models

This repository currently supports `LLaMA`, `Mistral` and `Phi` model families.


#### WandB logging

One can optionally log the data to `Weights and Biases` service (wandb).
Run `pip install wandb` for W&B logging.
Specify `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables prior to running experiments. use `--wandb` argument to enable logging

### GPU and RAM requirements

This code was developed and tested using several A100 and H100 GPUs with 80GB of VRAM. Some experiments require a combined amount of 640GB VRAM (e.g. 8xH100).

#### Model downloading

The code requires the LLaMA model to be downloaded in Huggingface format and saved locally. The scripts below assume that `$TRANSFORMERS_CACHE` variable points to the Huggingface Transformers cache folder.
To download and cache the models, run this in the same environment:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"  # or whatever else you wish to download
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
```




#### Reproducing Preliminary Analysis

We offer separate instructions for our reproducing preliminary analysis from supplementary materials:

* Instructions of estimating L smoothness computation experiments can be found in [`./L_smoothness_of_LLM`](L_smoothness_of_LLM)
* Instructions on small scale experiments with quadratic objective can be found in [`./pv_small_scale`](pv_small_scale)


# Usage

This guide will walk you through reproducing the full PV-Tuning pipeline over VQ. The pipeline consists of:

1. Initial model calibration
2. Preparing calibration dataset for fine-tuning
3. Fine-tuning with PV-Tuning
4. Exporting the saved model
5. Evaluation

We will use `Llama 2` 7B with 1.58-bit quantization throughout this guide; the rest of configurations can be achieved
by changing nbits per codebook and group size (see below). 

## 1. Initial model calibration


This script quantizes the model and then tests its performance in terms of perplexity using WikiText2, and C4 datasets.
For this, we adapt the original code from [AQLM](https://github.com/Vahe1994/AQLM) and modify it to better support VQ.

To quantize the model, run 

```sh
export MODEL_PATH=meta-llama/Llama-2-7b-hf  # path or huggingface id of the base model
export DATASET_PATH=pajama
export MODEL_SEQLEN=4096    # model-specific maximal sequence length, 4096 for llama2, 8192 for mistral
export NBITS_PER_CODEBOOK=16
export GROUP_SIZE=16
# this corresponds to having a single 16-bit codebook for 16-dimensional vectors

export BLOCKWISE_FINETUNE_EPOCHS=25
# set to 0 to disable blockwise finetuning during calibration

export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export SAVE_PATH=/path/to/save/quantized/model/
export WANDB_PROJECT=MY_EXPS
export WANDB_NAME=YOUR_EXP_NAME

ython main.py \
    $MODEL_PATH \
    $DATASET_PATH \
    --nsamples=2048 \
    --val_size=256 \
    --model_seqlen=4096 \
    --num_codebooks=1 \
    --nbits_per_codebook=$NBITS_PER_CODEBOOK \
    --out_group_size=1 \
    --in_group_size=$GROUP_SIZE \
    --beam_size=1 \
    --relative_mse_tolerance=0.01 \
    --max_epochs=100 \
    --finetune_lr=1e-4 \
    --finetune_adam_beta1=0.90 \
    --finetune_adam_beta2=0.999 \
    --finetune_keep_best \
    --finetune_batch_size=64 \
    --local_batch_size=4 \
    --finetune_max_epochs=$BLOCKWISE_FINETUNE_EPOCHS \
    --finetune_early_stop=3 \
    --offload_activations \
    --save $SNAPSHOT_PATH \
    --wandb --resume

```

Main CLI arguments (this is the original help message from AQLM code):
- `CUDA_VISIBLE_DEVICES` - by default, the code will use all available GPUs. If you want to use specific GPUs (or one GPU), use this variable.
- `MODEL_PATH` - a path to either Hugging Face hub (e.g. meta-llama/Llama-2-7b-hf) or a local folder with transformers model and a tokenizer.
- `DATASET_PATH` - either a path to calibration data (see above) or a standard dataset `[c4, ptb, wikitext2]`
   - for llama-2 models, you can use `DATASET_PATH=./data/red_pajama_n=1024_4096_context_length.pth` for a slice of RedPajama (up to 1024 samples)
- `--nsamples` - the number of calibration data _sequences_ (train + validation). If this parameter is not set, take all calibration data avaialble.
- `--val_size` - the number of validation sequences for early stopping on block finetuning. By default equal to 0. Must be smaller than `--nsamples`.
- `--num_codebooks` - number of codebooks per layer
- `--nbits_per_codebook` - each codebook will contain 2 ** nbits_per_codebook vectors
- `--in_group_size` - how many weights are quantized together (aka "g" in the arXiv paper)
- `--finetune_batch_size` - (for fine-tuning only) the total number of sequences used for each optimization step
- `--local_batch_size` - when accumulating finetune_batch_size, process this many samples per GPU per forward pass (affects GPU RAM usage)
- `--relative_mse_tolerance`- (for initial calibration) - stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--finetune_max_epochs` - maximal number of passes through calibration data on block tuning.
- `--finetune_early_stop` -  maximal number of passes through calibration data without improvement on validation.
- `--offload_activations` -- during calibration, move activations from GPU memory to RAM. This reduces VRAM usage while slowing calibration by ~10% (depending on your hardware). 
- `--save` -- path to save/load quantized model. (see also: `--load`)
- `--wandb` - if this parameter is set, the code will log results to wandb
- `--attn_implementation` - specify attention (for transformers >= `4.38`). Sdpa attention sometimes causes issues and it is recommended to use `eager` implementation.

There are additional hyperparameters aviailable. Run `python main.py --help` for more details on command line arguments, including compression parameters.




## 2. Preparing fine-tuning dataset


This is a script is used to pre-tokenize a subset of RedPajama data for future fine-tuning.

```sh
TARGET_MODEL=meta-llama/Llama-2-7b-hf  # used for tokenization
SEQLEN=4096
DATASET=togethercomputer/RedPajama-Data-1T-Sample
OUTPUT_PATH=./redpajama_tokenized_llama2

CUDA_VISIBLE_DEVICES=0 HF_HOME=/mnt/LLM OMP_NUM_THREADS=16 torchrun --master-port 3456 --nproc-per-node=1 finetune_fsdp.py --base_model $TARGET_MODEL --quantized_model ./doesnt_matter --dtype bfloat16 --block_type LlamaDecoderLayer --dataset_name=$DATASET --split train --cache_dir=./cache_dir --trust_remote_code --model_seqlen=$SEQLEN --preprocessing_num_workers=64 --preprocessing_chunk_length 100000 --save_dataset_and_exit $OUTPUT_PATH

tar -cvf tokenized_data_llama2.tar $OUTPUT_PATH   # optionally pack for distribution
```

The tokenized dataset is specific the model family (or more specifically, its tokenizer). For instance, Llama-3 8B is compatible with Llama-3 70B, but not with Llama-2 because it uses a different tokenizer.
To tokenize the data for another model, you need to set 1) --base_model 2) model_seqlen and 3) the path to --save_dataset_and_exit .

You can also set --preprocessing_num_workers to something hardware-appropriate. Note that setting --download_num_workers > 1 may cause download errors, possibly due to rate limit. These and other parameters are explained in the script's --help.
The job requires 150-200 GiB of disk space to store the dataset sample and preprocessing cache. Both are stored in ./cache_dir and can be deleted afterwards.


## 3. Refining Quantized Model with PV-Tuning


The code below starts from the initially calibrated model and iteratively refines with PV-tuning.
We use the same hyperparameters for all model sizes and families, varying only the model and sequence length.

```sh
export MODEL_PATH=meta-llama/Llama-2-7b-hf  # path or huggingface id of the base model
export QUANTIZED_MODEL_PATH=<PATH_TO_QUANTIZED_MODEL>  # path to the model created by initial calibration
export TOKENIZED_DATASET_PATH=<PATH_TO_TOKENIZED_DATASET>  # yet again, red pajama adviced
export CACHE_DIR=./cache_dir
export SNAPSHOT_PATH=<PATH_FOR_THE_TUNED_MODEL_TO_BE_SAVED_TO>
export SEQLEN=4096

export WANDB_PROJECT=PV_TUNE_LLAMA_2
export WANDB_NAME=llama-2-7b-1x16gs16-pv

torchrun --nproc-per-node=$NUM_GPUS finetune_fsdp.py \
    --base_model $MODEL_PATH --quantized_model $QUANTIZED_MODEL_PATH  --monkeypatch_old_pickle \
    --model_seqlen=$SEQLEN --block_type LlamaDecoderLayer --limit_parallel_inits 4 \
    --load_dtype bfloat16 --amp_dtype bfloat16 --code_dtype uint16 \
    --straight_through_buffer_dtype float32 \
    --dataset_name=$TOKENIZED_DATASET_PATH --split none --seed 1337 \
    --preprocessing_chunk_length 100000 --cache_dir=$CACHE_DIR --trust_remote_code \
    --update_codes --update_codebooks_and_scales --update_non_quantized_parameters \
    --lamb --debias --lr 3e-4 --adam_beta1 0.9 --adam_beta2 0.95 \
    --code_lr 3e-3 --code_beta1 0.0 --code_beta2 0.95 --beam_size 1 --delta_decay 0 \
    --max_code_change_per_step 1e-2 --code_trust_ratio 1e-2 --code_selection_temperature 0 \
    --batch_size=256 --microbatch_size=8 --max_epochs 10 --gradient_checkpointing \
    --print_every_steps=1 --verbose_optimizer --wandb  --eval_every_steps=10 --keep_best_model \
    --save $SNAPSHOT_PATH --save_every_steps 100
```


### 4. Exporting the saved model

The code above saves the model in a sharded format due to using a custom FSDP extension.

To use standard inference kernels and evaluation tools, one must revert it to the original format.
Since our evaluation code was inherited from AQLM, you need a script to convert PV-Tuned checkpoints to the AQLM format. 

```sh

python convert_legacy_model_format.py\
    --base_model $ORIG_MODEL_PATH\
    --pv_fsdp_dir $MODEL_PATH\
    --code_dtype int32 --load_dtype auto --quantized_model=./doesnt_matter \
    --save CONVERTED_CHECKPOINT_PATH

```

You can also use this converted checkpoint to export model to `inference_lib`.



### 5. Evaluaton

The code above evaluates validation perplexity by default. For few-shot evaluation

```sh
MODEL_PATH=meta-llama/Llama-2-7b-hf
CONVERTED_CHECKPOINT_PATH=<path to converted checkpoint>
SOURCE_CODE_PATH=<path to the root of this repository>

lm_eval --model hf --model_args pretrained=$MODEL_PATH,aqlm_checkpoint_path=$CONVERTED_CHECKPOINT_PATH,aqlm_src_path=$SOURCE_CODE_PATH,parallelize=True,dtype=float16 \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 4

```

By default, the evaluation runs in float16 precision. For models native to bfloat16, we recommend chaning to dtype=bfloat16.

That's it! You now know all you need to use PV-tuning for your models or reproduce our experiments.
