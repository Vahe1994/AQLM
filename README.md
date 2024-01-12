**UNDER CONSTRUCTION** -- we have just opened the codebase and are still changing it every few hours. If you want a stable codebase, please return after Jan 13 23:59 CET
# AQLM
Official Pytorch repository for [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118.pdf)

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

For quantization with AQLM its is recommended to use the subset of the data model 
was trained on. I.e. for quantization of `LLaMA 2` models we recommend to use the subset
of [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) .The subset of Redpajama for  2048 and 4096 context length stored in `data` directory: 
* `red_pajama_n=1024_2048_context_length.pth`
* `red_pajama_n=1024_4096_context_length.pth`
  
**Note** These subsets are already processed with the corresponding model tokenizer. Use for different model will lead to
unexpected behavior.

### W&B logging

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

### Perplexity benchmarks:
This script compresses the model and then tests its performance in terms of perplexity using WikiText2, C4, and Penn Treebank datasets. 

The command to launch the script should look like this: 

```
export MODEL_PATH=<PATH_TO_MODEL_DIR>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>

python main.py $MODEL_PATH $DATASET \
    --num_codebooks=2 \
    --
    --relative_mse_tolerance=0.01 \
    --go_relative_mse_tolerance=0.001 \
    --nsamples=1024 \  
    --nbits_per_codebook=15 \
    --in_group_size=8 \
    --scale_nbits=0 \
    --local_batch_size=4 \
    --save="save_path"\
    --batch_size=32 \
    --wandb
```

Note the launch arguments:
- `<PATH_TO_MODEL_DIR>` - path to model folder, which contains `config.json `
- `one of [c4, ptb, wikitext2, pajama, refinedweb, none]` -- name of dataset to use for compression, or path to an alternative preprocessed and tokenized dataset.
- `--num_codebooks` - #Number of codebooks per layer
- `--batch_size` - Size of sequences fot fine-tuning the layer (GO), globally across all GPUs
- `--local_batch_size` - Per-device and per-forward-pass batch size used to accumulate global --batch_size
- `--nsamples` - Number of calibration data samples.If None take all calibration data.
- `--relative_mse_tolerance`- Stop training when (current_epoch_mse / previous_epoch_mse) > (1 - relative_mse_tolerance)
- `--in_group_size` - How many input features are quantized together
- `--nbits_per_codebook` - Codebook size. Each codebook will contain 2 ** nbits_per_codebook vectors
-  `--scale_nbits` - Number of bits dedicated to the learnable group-wise scale.0 will use row-wise scales
- `--offload activations` -- moves activations to RAM when not used. Reduces VRAM usage while slowing work by ~10%. 
run `python main.py --help` for more details on command line arguments, including compression parameters.
- `--save --load` -- path to save/load quantized model.
- `--wandb` - log to wandb

### LM Evaluation Harness benchmark.

To perform zero-shot evaluation, we use [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework with slight modifications. This repository contains a copy of LM Evaluation Harness repo from early 2023 in `lm-eval-harness` folder. 
#### Installation
Before running the code make sure that you have all the requirements and dependencies of `lm-eval-harness` installed. To install them run:
```
pip install -r lm-evaluation-harness/requirements.txt
```
#### Execution

The main script launching the evaluation procedure is `lmeval.py` .


```
export MODEL_PATH=<INSERT PATH_TO_MODEL_DIR>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>

python lmeval.py \
    --model hf-causal \
    --model_args pretrained=$MODEL_PATH,dtype=float16,use_accelerate=True \
    --load $QUANTZED_MODEL \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size 1
```

## Contributing
We use black and isort for all pull requests. Before committing your code run black . && isort . .

## Citation
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
