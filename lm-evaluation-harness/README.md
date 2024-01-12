# Language Model Evaluation Harness

## Overview

The code, utilities and assets located in this directory are adapted from [LM Evaluation Harness benchmark suite](https://github.com/EleutherAI/lm-evaluation-harness) and customized to support quantization. The LICENSE and CODEOWNERS files inside lm-evaluation-harness refer to the original authors of lm-eval-harness and not the anonymous authors of this paper.

The workflow involves following steps:
- Model quantization
- Running tasks from the benchmarks for the quantized model

## Instructions
refer to `../README.md` in the project root folder for installation and launch instructions.

## Citation

BibTeX citation of the original lm-eval-harness repository.

```
@software{eval-harness,
  author       = {Gao, Leo and
                  Tow, Jonathan and
                  Biderman, Stella and
                  Black, Sid and
                  DiPofi, Anthony and
                  Foster, Charles and
                  Golding, Laurence and
                  Hsu, Jeffrey and
                  McDonell, Kyle and
                  Muennighoff, Niklas and
                  Phang, Jason and
                  Reynolds, Laria and
                  Tang, Eric and
                  Thite, Anish and
                  Wang, Ben and
                  Wang, Kevin and
                  Zou, Andy},
  title        = {A framework for few-shot language model evaluation},
  month        = sep,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v0.0.1},
  doi          = {10.5281/zenodo.5371628},
  url          = {https://doi.org/10.5281/zenodo.5371628}
}
```
