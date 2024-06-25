# AQLM

Official PyTorch implementation for [Extreme Compression of Large Language Models via Additive Quantization](https://arxiv.org/pdf/2401.06118.pdf)

**[2024.05]** AQLM was accepted to [ICML'2024](https://icml.cc/Conferences/2024)! If you're attending, meet us around [this poster](https://icml.cc/virtual/2024/poster/34964).

**[2024.06]** there's a more effective way to tune quantized models with [PV-tuning](https://arxiv.org/abs/2405.14852)). We're releasing PV-tuned AQLM models [**in this collection**](https://huggingface.co/collections/ISTA-DASLab/aqlmpv-66564dff5d84f00a893ba93f) and the code is in the [pv-tuning branch](https://github.com/Vahe1994/AQLM/tree/pv-tuning). We'll merge the pv-tuning code into main after several technical improvements.

## Inference

### Demo

Learn how to run the prequantized models using this Google Colab examples:

| Basic AQLM <br> generation | Streaming with <br> GPU/CPU | Inference with CUDA <br> graphs (3x speedup) | Fine-tuning <br> with PEFT | Serving with <br> `vLLM` |
|:-----------:|:-------:|:---------------:|:----------:|:--------:|
| <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/colab_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="AQLM In Colab"/></a>         | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/streaming_example.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="AQLM In Colab"/></a> | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_cuda_graph.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_2bit_training.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>  | <a target="_blank" href="https://colab.research.google.com/github/Vahe1994/AQLM/blob/main/notebooks/aqlm_vllm.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |


### Models

This repository is currently designed to work with models of `LLaMA`, `Mistral` and `Mixtral` families.
The models reported below use **full model fine-tuning** as described in appendix A, with cross-entropy objective with teacher logits.

We provide a number of prequantized AQLM models without PV-Tuning (scroll down for PV-Tuned models):

| Model      | AQLM scheme | WikiText-2 PPL | MMLU (5-shot) FP16→AQLM | Model size, Gb | Hub link                                                                 |
|------------|-------------|----------------|---------------|----------------|--------------------------------------------------------------------------|
| Llama-3-8b | 1x16        | -          | 0.65→0.56 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-2Bit-1x16) |
| Llama-3-8b-Instruct | 1x16        | -          | 0.66→0.59 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-Instruct-AQLM-2Bit-1x16) |
| Llama-3-70b | 1x16        | -          | 0.79→0.75 | 21.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-2Bit-1x16) |
| Llama-3-70b-Instruct | 1x16        | -          | 0.80→0.76 | 21.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-Instruct-AQLM-2Bit-1x16) |
| Command-R | 1x16      | -           | 0.68→0.57 | 12.7            | [Link](https://huggingface.co/ISTA-DASLab/c4ai-command-r-v01-AQLM-2Bit-1x16)|
| Command-R+ | 1x16      | -           | 0.74→0.68 | 31.9            | [Link](https://huggingface.co/ISTA-DASLab/c4ai-command-r-plus-AQLM-2Bit-1x16)|
| Mistral-7b| 1x16       | 5.40           | - | 2.5            | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-v0.1-AQLM-2Bit-1x16-hf)|
| Mistral-7B-Instruct-v0.2 | 2x8       | -           | 0.59→0.44 | 2.5            | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-Instruct-v0.2-AQLM-2Bit-2x8)|
| Mixtral-8x7b| 1x16       | 3.35           | -| 12.6            | [Link](https://huggingface.co/ISTA-DASLab/Mixtral-8x7b-AQLM-2Bit-1x16-hf)|
| Mixtral-8x7b-Instruct| 1x16       | -           | -| 12.6            | [Link](https://huggingface.co/ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf)|
| Llama-2-7b | 1x16        | 5.92          | 0.46→0.39 | 2.4            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf) |
| Llama-2-7b | 2x8         | 6.69          | - | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-2x8-hf)  |
| Llama-2-7b | 8x8         | 6.61          | - | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-2Bit-8x8-hf)  |
| Llama-2-13b| 1x16        | 5.22           | 0.55→0.49 | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-2Bit-1x16-hf)|
| Llama-2-13b| 2x8        |  5.63          | - | 3.8            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-2Bit-2x8-hf)|
| Llama-2-70b| 1x16        | 3.83           | 0.69→0.65 | 18.8           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-2Bit-1x16-hf)|
| Llama-2-70b| 2x8         | 4.21           | - | 18.2           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-2Bit-2x8-hf) |
| gemma-2b | 1x16      | -           | - | 1.7            | [Link](https://huggingface.co/ISTA-DASLab/gemma-2b-AQLM-2Bit-1x16-hf)|
| gemma-2b | 2x8      | -           | - | 1.6            | [Link](https://huggingface.co/ISTA-DASLab/gemma-2b-AQLM-2Bit-2x8-hf)|

You can also download AQLM models tuned via PV-tuning:

| Model      | AQLM scheme | WikiText-2 PPL | Model size, Gb | Hub link                                                                 |
|------------|-------------|----------------|----------------|--------------------------------------------------------------------------|
| Llama-2-7b | 1x16g8        | 5.68          | 2.4            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-1x16-hf) |
| Llama-2-7b | 2x8g8         | 5.90          | 2.2            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-7b-AQLM-PV-2Bit-2x8-hf)  |
| Llama-2-7b | 1x16g16     | 9.21          | 1.7            | [Link](https://huggingface.co/justheuristic/Llama-2-7b-AQLM-PV-1Bit-1x16-hf)  |
| Llama-2-13b| 1x16g8        | 5.05           | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-13b-AQLM-PV-2Bit-1x16-hf)|
| Llama-2-70b| 1x16g8        | 3.78           | 18.8           | [Link](https://huggingface.co/ISTA-DASLab/Llama-2-70b-AQLM-PV-2Bit-1x16-hf)|
| Meta-Llama-3-8B | 1x16g8        | 6.99          | 4.1            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-2Bit-1x16) |
| Meta-Llama-3-8B  | 1x16g16        | 9.43          | 3.9            | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-8B-AQLM-PV-1Bit-1x16) |
| Meta-Llama-3-70B | 1x16g8        | 4.57           | 21.9           | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-2Bit-1x16)|
| Meta-Llama-3-70B | 1x16g16        | 8.67           | 13           | [Link](https://huggingface.co/ISTA-DASLab/Meta-Llama-3-70B-AQLM-PV-1Bit-1x16)|
| Mistral-7B-v0.1 | 1x16g8  | 5.22 | 2.51 | [Link](https://huggingface.co/ISTA-DASLab/Mistral-7B-v0.1-AQLM-PV-2Bit-1x16-hf) |
| Phi-3-mini-4k-instruct | 1x16g8 | 6.63 | 1.4 | [Link](https://huggingface.co/ISTA-DASLab/Phi-3-mini-4k-instruct-AQLM-PV-2Bit-1x16-hf) |



Note that models with "g16" in their scheme require aqlm inference library v1.1.6 or newer: 
```bash
pip install aqlm[gpu,cpu]>=1.1.6
```

Above perplexity is evaluated on **4k** context length for Llama 2 models and **8k** for Mistral/Mixtral and Llama 3. 
Please also note that token-level perplexity can only be compared within the same model family, but should not be compared between models that use different vocabularies.
While Mistral has a lower perplexity than Llama 3 8B but this does not mean that Mistral is better: Llama's perplexity is computed on a much larger dictionary and has higher per-token perplexity because of that.

For more evaluation results and detailed explanations, please see our papers: [Egiazarian et al. (2024)](https://arxiv.org/abs/2401.06118) for pure AQLM and [Malinovskii et al. (2024)](https://arxiv.org/abs/2405.14852) for PV-Tuned models.

### Inference kernels

AQLM quantization setpus vary mainly on the number of codebooks used as well as the codebook sizes in bits. The most popular setups, as well as inference kernels they support are:
 
| Kernel | Number of codebooks | Codebook size, bits | Scheme Notation | Accuracy | Speedup     | Fast GPU inference | Fast CPU inference |
|---|---------------------|---------------------|----------|-------------|-------------|--------------------|--------------------|
| Triton | K                   | N                  | KxN     | -        | Up to ~0.7x | ✅                  | ❌                  |
| CUDA | 1                   | 16                  | 1x16     | Best        | Up to ~1.3x | ✅                  | ❌                  |
| CUDA | 2                   | 8                   | 2x8      | OK          | Up to ~3.0x | ✅                  | ❌                  |
| Numba | K                   | 8                   | Kx8      | Good        | Up to ~4.0x | ❌                  | ✅                  |

### Installation



To run the models, one would have to install an inference library:
```bash
pip install aqlm[gpu,cpu]
```
, specifying either `gpu`, `cpu` or both based on one's inference setting.


Then, one can use the familiar `.from_pretrained` method provided by the [transformers](https://github.com/huggingface/transformers) library:
```python
from transformers import AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Llama-2-7b-AQLM-2Bit-1x16-hf",
    trust_remote_code=True, torch_dtype="auto"
).cuda()
```
Notice that `torch_dtype` should be set to either `torch.float16` or `"auto"` on GPU and `torch.float32` on CPU. After that, the model can be used exactly the same as one would use and unquantized model. 



## Quantization

### Dependencies

Install packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Loading / caching datasets and tokenizer

The script will require downloading and caching locally the relevant tokenizer and the datasets. 
They will be saved in default Huggingface Datasets directory unless alternative location is provided by env variables.
See [relevant Datasets documentation section](https://huggingface.co/docs/datasets/main/en/cache#cache-directory)

### Data

When quantizing models with AQLM, we recommend that you use a subset of the original data the model was trained on.

For Llama-2 models, the closest available dataset is [RedPajama](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample) . To load subset of RedPajama provide "pajama" in --dataset argument.
This will process nsamples data and tokenize it using provided model tokenizer.

Additionally we provide tokenized Redpajama for LLama and Solar/Mistral models for 4096 context lengths stored in [Hunggingface](https://huggingface.co/datasets/Vahe1994/AQLM) .
To load it, use:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="Vahe1994/AQLM", filename="data/name.pth", repo_type="dataset")
```

To use downloaded data from HF, place it in data folder(optional) and set correct path to it in "--dataset" argument in main.py.

**Warning:** These subsets are already processed with the corresponding model tokenizer. If you want to quantize another model (e.g. mistral/mixtral), please re-tokenize the data with provided script in src/datautils.

### WandB logging

One can optionally log the data to `Weights and Biases` service (wandb).
Run `pip install wandb` for W&B logging.
Specify `$WANDB_ENTITY`, `$WANDB_PROJECT`, `$WANDB_NAME` environment variables prior to running experiments. use `--wandb` argument to enable logging

### GPU and RAM requirements
This code was developed and tested using a several A100 GPU with 80GB GPU RAM. 
You can use the `--offload activations` option to reduce VRAM usage.
For `Language Model Evaluation Harness` evaluation one needs to have enough memory to load whole model  + activation tensors 
on one or several devices.

### Quantization time

AQLM quantization takes considerably longer to calibrate than simpler quantization methods such as GPTQ. This only impacts quantization time, not inference time.

For instance, quantizing a 7B model with default configuration takes about 1 day on a single A100 gpu. Similarly, quantizing a 70B model on a single GPU would take 10-14 days. If you have multiple GPUs with fast interconnect, you can run AQLM multi-gpu to speed up comparison - simply set CUDA_VISIBLE_DEVICES for multiple GPUs. Quantizing 7B model on two gpus reduces quantization time to ~14.5 hours. Similarly, quantizing a 70B model on 8 x A100 GPUs takes 3 days 18 hours.

If you need to speed up quantization without adding more GPUs, you may also increase `--relative_mse_tolerance` or set `--init_max_points_per_centroid` or limit `--finetune_max_epochs`. 
However, that usually comes at a cost of reduced model accuracy.

### Model downloading
The code requires the LLaMA model to be downloaded in Huggingface format and saved locally. The scripts below assume that `$TRANSFORMERS_CACHE` variable points to the Huggingface Transformers cache folder.
To download and cache the models, run this in the same environment:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = "meta-llama/Llama-2-7b-hf"  # or whatever else you wish to download
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

python main.py $MODEL_PATH $DATASET_PATH \
 --nsamples=1024 \
 --val_size=128 \
 --num_codebooks=1 \
 --nbits_per_codebook=16 \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=1 \
 --offload_activations \
 --wandb \
 --resume \
 --save $SAVE_PATH
```

Main CLI arguments:
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

### Finetuning

**Note** this code will only fine-tune continuous parameters. To fine-tune both continuous and discrete parameters, please switch to [pv-tuning](https://github.com/Vahe1994/AQLM/tree/pv-tuning) branch and follow instructions in its readme.

The accuracy of the quantized model can be further improved via block finetuning. First, the logits 
of the float16/bfloat16 are cached in RAM. Then the differentiable parameters of the quantized model
are optimized to minimize KL-divergence with teacher logits. Typically, we use the same calibration data that was used for model quantization.

The command to launch the script should look like this: 

```bash
python finetune.py \
  --base_model $MODEL_PATH \
  --quant_model $INPUT_PATH \
  --dataset $DATASET_PATH \
  --nsamples=<TOTAL_SIZE> \
  --val_size=<VAL_SIZE> \
  --lr=1e-5 \
  --adam_beta1=0.90 \
  --adam_beta2=0.999 \
  --epochs=5 \
  --early_stop=3 \
  --batch_size=8 \
  --microbatch_size=4 \
  --save $DATA_PATH \
  --gradient_checkpointing
```

Main CLI arguments:
- `--base_model` - path or name of the original floating-point model
- `--quant_model` - path to quantized model weights.
- `--dataset` - path or name of the calibration dataset
- `--nsamples` - the number of calibration data _sequences_ (train + validation). If this parameter is not set, take all calibration data avaialble.
- `--val_size` - the number of validation sequences for early stopping on end-to-end finetuning. By default equal to 0. Must be smaller than `--nsamples`.
- `--gradient_checkpointing` - whether to use gradient checkpointing. Reduces peak memory usage at the cost of longer runtime.
- `--finetune_dtype` - which dtype should be used on finetuning. By default `float32`. 
- `--amp` - whether to use amp on finetuning. Requires `--finetune_dtype=float32`.

For larger models one would need multi-GPU training. At the moment, FSDP training is not implemented and the model is finetuned on a single process with parameters sharded across available devices.


### Zero-shot benchmarks via LM Evaluation Harness

To perform zero-shot evaluation, we adopt [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) framework. Our code works with models in standard `transformers`` format and may (optionally) load
the weights of a quantized model via `--aqlm_checkpoint_path` argument.

The evalution results in PV-Tuning were produced with `lm-eval=0.4.0`. 

To run evaluation make sure that proper version is installed or install it via:
`pip install lm-eval==0.4.0`. 

The main script for launching the evaluation procedure is `lmeval.py`.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # optional: select GPUs
export QUANTIZED_MODEL=<PATH_TO_SAVED_QUANTIZED_MODEL_FROM_MAIN.py>
export MODEL_PATH=<INSERT_PATH_TO_ORIINAL_MODEL_ON_HUB>
export DATASET=<INSERT DATASET NAME OR PATH TO CUSTOM DATA>
export WANDB_PROJECT=MY_AQLM_EVAL
export WANDB_NAME=COOL_EVAL_NAME

# for 0-shot evals
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks winogrande,piqa,hellaswag,arc_easy,arc_challenge \
    --batch_size <EVAL_BATCH_SIZE> \
    --aqlm_checkpoint_path QUANTIZED_MODEL # if evaluating quantized model

# for 5-shot MMLU
python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_PATH,dtype=float16,parallelize=True \
    --tasks mmlu \
    --batch_size <EVAL_BATCH_SIZE> \
    --num_fewshot 5 \
    --aqlm_checkpoint_path QUANTIZED_MODEL # if evaluating quantized model
```

### Preparing models for inference

To convert a model into a _Hugging Face_ compatible format, use `convert_to_hf.py model in_path out_path` with corresponding arguments:
 - `model` - the original pretrained model (corresponds to `MODEL_PATH` of `main.py`, e.g. `meta-llama/Llama-2-7b-hf`).
 - `in_path` - the folder containing an initially quantized model (corresponds to `--save` of `main.py`).
 - `out_path` - the folder to save `transformers` model to.

You may also specify flags such as `--save_safetensors` to control the saved model format (see `--help` for details).

Example command: `python convert_to_hf.py meta-llama/Llama-2-7b-hf ./path/to/saved/quantization ./converted-llama2-7b-hf  --save_safetensors`

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
@misc{malinovskii2024pvtuning,
      title={PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression}, 
      author={Vladimir Malinovskii and Denis Mazur and Ivan Ilin and Denis Kuznedelev and Konstantin Burlachenko and Kai Yi and Dan Alistarh and Peter Richtarik},
      year={2024},
      eprint={2405.14852},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
