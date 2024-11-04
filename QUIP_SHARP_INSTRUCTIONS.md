This document contains instructions for running pv tuning with QIUP# algorithm.

All the described steps are performed in the `quip-sharp-patch` directory.
It contains `grids.py` file from quip-sharp repository, GNU license (since quip-sharp is under GNU license),
`grids.ipynb` notebook for getting grids, `convert.ipynb` notebook for checkpoint conversion, and modified `finetune.py` script.

To do this, you first need to convert quip# checkpoint into AQLM format.
1) Get grids by running `grids.ipynb` notebook.
2) Convert quip# checkpoint into AQLM format by running `convert.ipynb` notebook.

After second step you will have a checkpoint that is similar to the one you get after AQLM quantization.
You can run it with finetune.py script.

To make resulting checkpoint QUIP# compatible, you need to keep codes the same.
Edit your `finetune.py` arguments by omitting `--update_codebooks_and_scales` flag.
Note that the name is a bit misleading in described setup, as QUIP# scales will be actually updated.
