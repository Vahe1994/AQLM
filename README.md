# AQLM

Official PyTorch implementation for [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118.pdf)

**UNDER CONSTRUCTION** -- we have just opened the codebase and are still changing it every few hours. If you need a more stable codebase, please return here after Jan 13 23:59 CET


## Installation

### Packages

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Loading / caching datasets and tokenizer

The script will require downloading and caching locally the relevant tokenizer and the datasets. 
They will be saved in default Huggingface Datasets directory unless alternative location is provided by env variables.
See [relevant Datasets documentation section](https://huggingface.co/docs/datasets/main/en/cache#cache-directory)
## Models

This repository is expected to work with models of `LLaMA ` families so far.

## Data

When quantizing models with AQLM, we recommend that you use a subset of the original data the model was trained on (or something similar).

For Llama-2 models, the closest available dataset is [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) .

We also provide subsets of Redpajama for 2048 and 4096 context lengths stored in `data` directory:
* `red_pajama_n=1024_2048_context_length.pth`
* `red_pajama_n=1024_4096_context_length.pth`
  
**Warning:** These subsets are already processed with the corresponding model tokenizer. If you want to quantize another model (e.g. mistral/mixtral), please re-tokenize the data.

__We shall add step-by-step instructions for this before Jan 13 23:59 AOE.__


### WandB logging

For the sake of convenience one can optionally log the data to `Weights and Biases` service (wandb).
Run `pip install wandb` for W&B logging.
Specify `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables prior to running experiments. use `--wandb` argument to enable logging
# Launching

### GPU and RAM requirements
This code was developed and tested using a several A100 GPU with 80GB GPU RAM. 
`--offload activations` option, reduce VRAM usage.
For `Language Model Evaluation Harness` evaluation one needs to have enough memory to load whole model
on one or several devices + activation tensors.

### Model downloading
The code requires the LLaMA model to be downloaded in Huggingface format and saved locally. The scripts below assume that `$TRANSFORMERS_CACHE` variable points to the Huggingface Transformers cache folder.
To download and cache the model, run this in any python code:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
```


### How to quantize a model with AQLM
This script compresses the model and then tests its performance in terms of perplexity using WikiText2, C4, and Penn Treebank datasets. 

The command to launch the script should look like this: 

```bash
export CUDA_VISIBLE_DEVICES=0   # or e.g. 0,1,2,3
export MODEL_PATH=<PATH_TO_MODEL_ON_HUB>
export DATASET_PATH=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export SAVE_PATH=/path/to/save/quantized/model/
export WANDB_PROJECT=MY_AQ_EXPS
export WANDB_NAME=COOL_EXP_NAME

python main.py $MODEL_PATH $DATASET_PATH --nsamples=1024 \
 --num_codebooks=1 --nbits_per_codebook=16 --in_group_size=8 \
 --relative_mse_tolerance=0.01 --go_relative_mse_tolerance=0.001 \
 --batch_size=32 --local_batch_size=2 --wandb --save $SAVE_PATH

```

Main CLI arguments:
- `CUDA_VISIBLE_DEVICES` - by default, the code will use all available GPUs. If you want to use specific GPUs (or one GPU), use this variable.
- `MODEL_PATH` - a path to either hugginface hub (e.g. meta-llama/Llama-2-7b-hf) or a local folder with transformers model and a tokenizer.
- `DATASET_PATH` - either a path to calibration data (see above) or a standard dataset `[c4, ptb, wikitext2]`
   - for llama-2 models, you can use `DATASET_PATH=./data/red_pajama_n=1024_4096_context_length.pth` for a slice of RedPajama (up to 1024 samples)
- `--nsamples` - the number of calibration data _sequences_. If this parameter is not set, take all calibration data avaialble.
- `--num_codebooks` - number of codebooks per layer
- `--nbits_per_codebook` - each codebook will contain 2 ** nbits_per_codebook vectors
- `--in_group_size` - how many weights are quantized together (aka "g" in the arXiv paper)
- `--local_batch_size` - (for fine-tuning only) how many sequences are processed on each GPU in a single forward pass
- `--batch_size` - (for fine-tuning only) the total number of sequences used for each optimization step
- `--relative_mse_tolerance`- (for initial calibration) - stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--go_relative_mse_tolerance`- (for fine-tuning only) - stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--offload activations` -- during calibration, move activations from GPU memory to RAM. This reduces VRAM usage while slowing calibration by ~10% (depending on your hardware). 
- `--save` -- path to save/load quantized model. (see also: `--load`)
- `--wandb` - if this parameter is set, the code will log results to wandb

There are additional hyperparameters aviailable. Run `python main.py --help` for more details on command line arguments, including compression parameters.

### Zero-shot benchmarks via LM Evaluation Harness

To perform zero-shot evaluation, we use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework with slight modifications. This repository contains a copy of LM Evaluation Harness repo from early 2023 in `lm-eval-harness` folder. 

Before running the code make sure that you have all the requirements and dependencies of `lm-eval-harness` installed. To install them run:
```
pip install -r lm-evaluation-harness/requirements.txt
```

The main script launching the evaluation procedure is `lmeval.py` .


```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # optional: select GPUs
export QUANTZED_MODEL=<PATH_TO_SAVED_QUANTIZED_MODEL_FROM_MAIN.py>
export MODEL_PATH=<INSERT_PATH_TO_ORIINAL_MODEL_ON_HUB>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export WANDB_PROJECT=MY_AQ_LM_EVAL
export WANDB_NAME=COOL_EVAL_NAME

python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --load $QUANTZED_MODEL \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1
```

## Contributing

If you want to contribute something substantial (more than a typo), please open an issue first.
We use black and isort for all pull requests. Before committing your code run `black . && isort .`

## Cite

If you found this work useful, please consider citing:

```
@misc{egiazarian2024extreme,
      title={Extreme Compression of Large Language Models via Additive Quantization}, 
      author={Vage Egiazarian and Andrei Panferov and Denis Kuznedelev and Elias Frantar and Artem Babenko and Dan Alistarh},
      year={2024},
      eprint={2401.06118},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
