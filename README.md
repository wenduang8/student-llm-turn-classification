# Classifying student-LLM programming interactions

Code for an Honours thesis at the University of Sydney (Bachelor of Advanced Computing, 2026) by Donghang Wen, supervised by Professor Mary Lou Maher. The project codes, turn by turn, how students and an AI assistant interact in programming conversations, and tests how closely a fine-tuned open-weights model reproduces that coding.

## Research questions

- **RQ1 (Taxonomy).** What turn-level taxonomy of student–LLM interaction roles can be derived from the literature and aligned with authentic student–LLM transcripts?
- **RQ2 (Automation).** How can an NLP/LLM pipeline operationalise this taxonomy to classify student–LLM interactions at the turn level?
- **RQ3 (Accuracy).** How closely do the fine-tuned and prompted LLM implementations reproduce the human-coded ground truth on authentic data, and which coding distinctions account for their disagreements with it?

## Data

The conversations are coursework from COMP2017 (Systems Programming, Semester 1 2025): records of LLM-assistant use that students submitted with their assignments. The analysis corpus holds 6,567 student interactions from 1,202 conversations and 361 students. The data are de-identified student coursework used under the CARE umbrella ethics approval (HREC 2025/HE001132, substudy 019).

**The data are not in this repository and cannot be shared.** Scripts that read data expect it under `data/`, which is excluded from version control. Every test runs on synthetic records.

## Coding scheme

The unit is a student interaction: one student prompt and the assistant response that follows it, read with the earlier turns of the conversation. Each interaction is coded on four dimensions.

| Dimension | Selection | Codes |
|---|---|---|
| Student interaction type | one | Providing material / responding; Information retrieval; Task automation; Guided generation |
| Topic | one or more | code; concept; tooling or environment; conversation management |
| AI delegation | one or more | ND; offer; hand-back: directive; hand-back: asks the student to supply; hand-back: Socratic |
| AI response type | one or more | Solve-Provide; Explain; Correct-Debug; Evaluate; none of the four applies |

ND and "none of the four applies" are used only on their own. The label sets are defined once, in `codebook_schema.json`. The coding guide (17 rules with examples) is in [docs/coding-guide.md](docs/coding-guide.md), the text of Appendix A of the thesis.

## Pipeline

| Step | Directory | What it does |
|---|---|---|
| 1. Corpus | `extraction/` | Extracts the conversations from the submitted files (pasted text, chat exports, screenshots, saved pages, share links) into one row per student interaction; repairs dropped openings; excludes two near-copy conversations. Frozen in September 2026; `extraction/README.md` gives the order in which it was built. |
| 2. Human coding | `coding/` | Builds de-identified coding packets with an identifier screen, stores returned records unchanged and checks them, validates records, and compares two coders. |
| 3. Model | `finetuning/` | Reads and validates labels, prepares model inputs with up to three preceding interactions (a gap in the record ends the context), trains C1 and runs the five-fold cross-validation grouped by student, and scores out-of-fold predictions with intervals that resample students. C1 is Qwen3-0.6B-Base with LoRA and one classification head per dimension, trained on Apple silicon (MPS); see `finetuning/README.md`. |
| 4. Comparison and scoring | `comparison/` | A prompted LLM comparison (inputs de-identified before sending; the model is given the coding guide), and per-code scoring of any set of predictions. |
| Corpus description | `analysis/` | Corpus figures (Table 3.1 of the thesis). |

## Measure

Agreement is reported per code as simple agreement C / max(A, B), where A and B are the number of interactions each side coded with the code and C the number both did. Solve-Provide and Explain are also reported on the absent side. The same measure is used between the two human coders and between a model and the ground truth. Intervals are 95% percentile intervals from 10,000 bootstrap replicates that resample students.

## Running

```
pip install -r requirements.txt
python -m pytest coding finetuning comparison
```

Training C1 needs the exact package versions of the reported runs, on Apple silicon (MPS) or CPU: `pip install -r requirements-train.txt`. Without them the training tests are skipped.

With the private data in place, for example:

```
python comparison/score.py --out-dir <new folder> --pred C1=<out-of-fold predictions .jsonl>
python extraction/validate_corpus_v2.py
python extraction/apply_d60_exclusion.py --check
```

## Notes

Comments in the code cite decisions as D-*n* (for example D-60). They refer to the project's internal decision register, which is not published.

### Corpus construction: known limitations

These do not affect the 485 labelled interactions or any reported figure, because the labels and the models read the same text.

- `Segment Order` is not renumbered after unpaired rows are excluded, so those exclusions leave a gap that the loader treats as a context boundary. Rows removed earlier, by the exact-duplicate pass in `extraction/combine_dedup.py`, are renumbered away and leave no gap. This affects 12 conversations (11 in the analysis file), none of them among the labelled interactions.
- Conversation ids of the screenshot groups segmented by `extraction/segment_needs_review.py` (`..._IMG####`) come from Python's `hash()`, which differs between runs. The 13 such ids in the corpus come from the run that built it and cannot be regenerated.
- When a segmentation anchor cannot be found verbatim, `segment_needs_review.py` falls back to an approximate position. Turns remain substrings of the source, but 41 of 42 fallback boundaries in 12 conversations fall 1 to 882 characters (median 4) early. One labelled interaction is affected, by at most 3 characters.

## License

The code is released under the MIT License ([LICENSE](LICENSE)). The coding guide, [docs/coding-guide.md](docs/coding-guide.md), is released under the Creative Commons Attribution 4.0 International licence ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
