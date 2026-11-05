# finetuning/: C1 training, inputs, labels and scoring

This folder holds the code of C1, the classifier the thesis reports. It covers:
- reading and checking labels;
- building model inputs;
- training and resuming the multi-head model;
- the five-fold cross-validation drivers;
- scoring saved predictions.

The reported results come from the final run D-191 (`final.py`). The variants V1 and V2 are recorded in `variants/`.

## The data is not in this repository

`data/` is excluded by `.gitignore`. It holds de-identified student coursework under an ethics protocol: the corpus, the label files, prepared inputs, splits, runs and results.

Every command below that reads real data needs those files in place under `data/`. The trainer refuses to write a run on real data, or its predictions, anywhere outside `data/`. The tests use synthetic records only.

## The model

C1 is Qwen/Qwen3-0.6B-Base (revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`) with LoRA (rank 16, alpha 32) and one classification head for each of the four dimensions.

- **Inputs**: it reads the target interaction with up to three preceding interactions, within a prompt cap of 1,536 tokens, a response cap of 3,072 and 7,168 tokens in total.
- **Training**: 8 epochs with AdamW, at a learning rate of 2e-4 for the adapter and 1e-3 for the heads, with class weights computed from each training part. Each optimizer step averages 8 interactions.
- **Decoding**: multi-select codes are chosen at a threshold of 0.5.
- **Evaluation**: five-fold cross-validation on the 485 researcher-labelled interactions, with folds grouped by student, so no student is in both training and test within a fold.

## Hardware

C1 was trained on Apple MPS (Apple M5 Pro, 48 GiB, macOS 27.0, Python 3.12.14) with the package versions in `../requirements-train.txt`.

The trainer supports MPS and CPU only. A config's `"device"` is `"mps"` or `"cpu"`, and the random number states saved for resuming are those of CPU and MPS. CUDA is not supported.

The trainer checks the SHA-256 of `model.safetensors` against the config. It loads the model offline from `.cache/huggingface` in the repository root, so download it there once beforehand, from the repository root:

    HF_HOME=.cache/huggingface python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B-Base', revision='da87bfb608c14b7cf20ba1ce41287e8de496c0cd')"

Keep `.cache/` out of git.

## Files

| File | What it does |
|---|---|
| `labels.py` | Reads labelled records and model predictions, and names the file and line in every reading error. Labelled records are human records, or researcher's records reviewed with AI assistance, each with its provenance. Checks every value against `codebook_schema.json`. Encodes targets for the four heads: a missing label is masked, never read as the negative code. Decodes scores into valid label sets. Writes predictions as their own record kind. |
| `data.py` | Reads the corpus and builds each interaction's context: up to N preceding interactions of the same conversation, stopping at a gap in Segment Order. |
| `prepare_data.py` | Joins one label file to the corpus and writes an unsplit input set (`examples.jsonl`, `manifest.json`) to a new folder under `data/training/`. |
| `grouped_split.py` | Writes, reads and checks a student-grouped split. No student or conversation is in both parts. |
| `model_inputs.py` | Builds the token sequence of one interaction within the config's budgets (template causal-two-pool-v2) and reports what was cut. |
| `multihead.py` | The model (the backbone with LoRA and the four heads) and the loss of one interaction with the class weights. |
| `train_multihead.py` | The trainer, with three commands: `train`, `resume` and `predict`. It saves complete checkpoints and evaluates the final step with `evaluation.py`. |
| `evaluation.py` | The evaluation module used by the trainer and the reports. It gives counts per dimension and per code and the joint distribution, and checks the provenance of a prediction file. |
| `student_bootstrap.py` | Scores saved out-of-fold predictions with intervals that resample students. Used by `comparison/score.py`. |
| `cv.py` | The student-grouped five-fold cross-validation of 22 October on 150 interactions. Its `assign_folds` assigns whole students to folds, and the later runs call it. |
| `c485.py` | The D-186 run on 485 interactions: folds from `cv.assign_folds`, configs, run and report. |
| `final.py` | The final run D-191 on 485 interactions with the D-190 labels. It reuses the D-186 folds and configs through `c485.py` and trains the five folds. |
| `final_report.py` | After the final run: `assemble` pools the five folds' test predictions into `predictions-C1-485-oof.jsonl`, the file `comparison/score.py` scores; `report` writes the run's reports. |
| `variants/` | V1 and V2, run on folds 3 and 4 and not adopted. See `variants/README.md`. |
| `configs/pilot-20261017.json` | The config of a pilot run on 17 October, kept as the base config of `test_training.py`. |
| `test_*.py` | Tests on synthetic records and a tiny model. They need no data. |

`cv.py`, `c485.py`, `final.py` and `final_report.py` are copies of the scripts as they ran from `data/results/<run>/`. The only changes are their `ROOT` and output folder lines, the import paths of `final.py` and `final_report.py`, and explicit text encodings. They therefore still write to `data/results/<run>/`. The usage lines in their docstrings still show the old location; run them as `python -I finetuning/<script>.py <command>`.

## Running

Run from the repository root, with the private data in place.

    # prepare inputs from one label file
    python finetuning/prepare_data.py --reviewed-direct data/labels/dw485-reviewed-20261028d.jsonl \
        --corpus data/processed/<corpus.csv> --output-dir data/training/<new folder> --context 3

    # the final five-fold run D-191, in this order; run trains every fold, and running it again resumes
    python -I finetuning/final.py check
    python -I finetuning/final.py prepare
    python -I finetuning/final.py plan
    python -I finetuning/final.py run
    python -I finetuning/final_report.py assemble     # the out-of-fold predictions and scores
    python -I finetuning/final_report.py report       # the run's reports

    # one training run, its resumption, and predictions from a saved checkpoint
    python finetuning/train_multihead.py train --config <config.json> --run-dir data/runs/<run id>
    python finetuning/train_multihead.py resume --run-dir data/runs/<run id>
    python finetuning/train_multihead.py predict --run-dir data/runs/<run id> --part validation \
        --checkpoint final --out data/<folder>/<predictions.jsonl>

    # scoring
    python finetuning/evaluation.py --prepared data/training/<folder> --predictions <predictions.jsonl> --out <report.json>
    python comparison/score.py --out-dir <new folder> --pred C1=<out-of-fold predictions.jsonl>

Without the data, a synthetic run checks the whole training path with the real model:

    python finetuning/train_multihead.py train --synthetic --config <config.json> --run-dir tmp/<run id> --stop-after-steps 1

`--stop-after-steps` stops with a checkpoint after that optimizer step, with exit code 3. `resume` continues it.

## Resuming

A run is resumable, not only restartable.

- **Checkpoints**: every `checkpoint.every_steps` optimizer steps (10 in the final run, keeping the last 3), the trainer writes a checkpoint under `checkpoints/<tag>/`. It holds:
  - the LoRA adapter and the heads;
  - the optimizer and scheduler states;
  - the random number states (Python, NumPy, torch CPU and MPS);
  - the progress.

  Each epoch's data order is recomputed from the seed.
- **Resuming one run**: `python finetuning/train_multihead.py resume --run-dir <run dir>` continues from the latest complete checkpoint. It refuses to resume if the config, data, model, inputs, packages, device or code changed since the run started.
- **The five-fold run**: `final.py run` resumes an unfinished fold and skips a finished one.

## Rerunning on finished runs

Tested on a copy of the finished runs: no command retrains a finished fold or changes anything under `data/runs/`, and `assemble` refuses when the out-of-fold files exist. Several commands still rewrite their own records under `data/results/<run>/`:

- `check` rewrites `check.json`.
- `run` rewrites the times in `run-status.json` and adds a line to `progress.md`.
- `report` rewrites the reports, with the same content except a hand-edited `summary.md`.
- `cv.py plan` and `plan-lc` rewrite `plan.json`, including the code version it records.
- `plan` in `final.py` and `c485.py` rewrites `code-diff-vs-head.patch` when `finetuning/` has uncommitted changes.

Work on a copy when those records must stay as they are.

## Outputs

- `data/runs/<run id>/`: `run.json` (config and fingerprint), `train_log.jsonl`, `inputs-report.json`, `checkpoints/`, and `outputs/` with the predictions, the per-code scores and the evaluation reports.
- `data/splits/<name>/` and `data/training/<name>/`: the split records and prepared inputs a driver writes.
- `data/results/<run name>/`: a driver's configs, plan, checks and reports. The final run's are in `data/results/dw485d-c1-final-20261028/`.

## Tests

    python -m unittest discover -s finetuning -p "test_*.py"
    python -m pytest finetuning

`test_student_bootstrap.py` holds plain test functions, which only pytest collects.
