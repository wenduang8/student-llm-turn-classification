# finetuning/variants/: V1 and V2

V1 (focal loss with gamma 2.0 on all four heads) and V2 (six preceding interactions instead of three, 10,240 tokens in total) were trained on folds 3 and 4 of the final run's five folds (D-193, D-194; adoption rule D-202). Neither was adopted (D-234). Their results are in Table B.8 of the thesis.

- `variants.py`: the driver as it ran from `data/results/c1-variants-f34-20261028/`. Only its `ROOT` line was changed for this folder.
- `code-diff-vs-finetuning.patch`: the exact difference between `finetuning/` and the code copy the variants trained with. It adds the focal loss to `multihead.py` and `train_multihead.py`, and adds `variant_inputs.py` and `test_variants.py`. On 5 November 2026 it was checked against the code hashes the four runs recorded.

The driver expects the layout it ran in, with that code copy in a `finetuning/` folder beside it and its outputs written beside it. It is kept here as a record and is not meant to be run from this folder.
