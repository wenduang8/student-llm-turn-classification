# comparison/: prompted LLM comparison with C1 (D-192, D-196)

Plan: `docs/planning/llm-comparison-spec-20261028.md`. Model: OpenAI `gpt-6.1-terra`, reasoning effort medium (D-197), `store: false`,
structured output with the code lists as enums, one stateless request per interaction. The prompt gives the model the
27 October coding guide; runs and files carry the label A.

| File | Does |
|---|---|
| `deidentify.py` | Replacements before anything is sent, and the second screen that must find nothing |
| `llm_compare.py` | `build` (inputs, de-identification, prompt files, pilot ids), `reconfigure` (new config after a setting changes), `estimate`, `call`, `predictions` |
| `score.py` | Per-code C / max(A, B) (and the absent side for Solve-Provide and Explain), the C1 report's measures, student-resampled intervals; all 485 and each part |
| `test_comparison.py` | Synthetic tests; no data, no network |

Run on Windows from the repository root (PowerShell). The key stays in the session's environment; never put it in a file.

```powershell
$env:OPENAI_API_KEY = "<your key>"
python comparison\llm_compare.py call --set pilot          # 10 interactions; prints tokens, cost, projection
# DW reads data\results\llm-terra-20261028\pilot-summary.json and approves, then:
python comparison\llm_compare.py call --set all --variant A --pilot-approved
python comparison\llm_compare.py predictions --set all
python comparison\score.py --out-dir data\results\llm-terra-20261028\scores-final-c1 `
  --pred LLM-A=data\results\llm-terra-20261028\predictions\predictions-LLM-A-all.jsonl `
  --pred C1=data\results\dw485d-c1-final-20261028\oof-predictions\predictions-C1-485-oof.jsonl `
  --compare LLM-A:C1
# the same with --group 335=135+200 and --out-dir ...\scores-final-c1-150-335 gives the two batches of the thesis
```

The configuration hash that this version writes (`config_sha256` in `config.json` and in each prediction record)
differs from the one recorded in the stored predictions of 28 October to 2 November, because the list of prompt
files changed when the code was prepared for release. The requests themselves are unchanged.

A run that stops (network, budget, Ctrl+C) resumes where it stopped when the same command is given again: every attempt
is appended to `data/results/llm-terra-20261028/raw/`. The budget check counts all calls so far against US$20 and
stops before passing it. An HTTP error other than a rate limit or a server error stops the run and prints the API's
message (for example a parameter the model does not accept); nothing is retried in that case.
Output never contains interaction text: ids, counts, tokens, costs and verdicts only.

## Reruns of variant A (2 November 2026, D-250)

Two more runs of variant A, to compare repeated runs (Section 5.5 of the thesis). Each run has its own tag; set it
before every command. Inputs are built with the two conversation-scope rules that D-206 requires before any further
call (`deidentify.py`: a host after a replaced user, and every replaced token wherever else it appears in the same
conversation). The builds for r2 and r3 were made on 2 November (residual hits 0; the inputs of r2 and r3 are
identical and differ from the original run's in 23 interactions, all in P368_P2_03). Settings are unchanged; the
original pilot stands; the US$20 budget counts every run.

```powershell
$env:OPENAI_API_KEY = "<your key>"
$env:LLM_RUN_TAG = "r2"
python comparison\llm_compare.py call --set all --variant A --pilot-approved
python comparison\llm_compare.py predictions --set all
$env:LLM_RUN_TAG = "r3"
python comparison\llm_compare.py call --set all --variant A --pilot-approved
python comparison\llm_compare.py predictions --set all
Remove-Item Env:LLM_RUN_TAG
```

Outputs: `data/results/llm-terra-20261028-r2/` and `-r3/` (raw logs, predictions). Scoring is done afterwards with
`score.py` against the same labels.

Both reruns were made on 2 November (r2: 4 failed, r3: 5 failed; total spent US$7.35). Scores: `data/maintenance/20261102-assessment-fixes/rerun-scores/` (D-251).
