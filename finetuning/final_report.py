#!/usr/bin/env python3
"""Assembly and report of the final C1 run D-191 (README of mac-training-485-final-20261028.zip, steps 6 and 9). CPU only;
no model is run. Reuses the D-186 report code (data/results/dw485-c1-20261028/c485.py) for everything the D-186 report
gave, and adds the per-code simple agreement table and a descriptive paired comparison with the D-186 trial. Prints counts
only, never interaction text.

    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final_report.py assemble   # OOF predictions + OOF scores
    .venv/bin/python -I data/results/dw485d-c1-final-20261028/final_report.py report     # metrics.json, summary.md, ...

  agreement   per code and set: A = interactions whose reference has the code, B = whose prediction has it, C = both;
              simple agreement C / max(A, B). For Solve-Provide and Explain also the absent side (the same with "lacks the
              code"). Rows with a null reference on that dimension are left out, as everywhere.
  vs D-186    the D-186 trial's out-of-fold predictions and these, both scored against the D-190 labels on the same 485
              interactions (same inputs, folds, settings and code; only the training labels differ: 26 cells), paired by
              interaction with finetuning/student_bootstrap.py (10,000 replicates, seed 20261026). Descriptive.
  scores      oof-predictions/scores-C1-485-oof.jsonl: per test interaction, each head's per-code probabilities (the trainer's
              scores-validation.jsonl) with the predicted labels at 0.5, keyed by conversation_id and segment_order.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "data" / "results" / "dw485d-c1-final-20261028"   # outputs
sys.path.insert(0, str(Path(__file__).resolve().parent))         # final.py is beside this copy
import final as F  # noqa: E402  (the final run's driver; it imports the D-186 driver as F.B)

B, cv, EV, GS, L = F.B, F.cv, F.EV, F.GS, F.L
utc, sha, read_json, write_json, roundtrip = F.utc, F.sha, F.read_json, F.write_json, F.roundtrip
SPACE, ROOT = F.REAL, F.ROOT
PARTS, ALL, SETS = B.PARTS, B.ALL, B.SETS
OOF = SPACE.oof
SCORES = OOF.parent / "scores-C1-485-oof.jsonl"
D186_OOF = F.D186 / "oof-predictions" / "predictions-C1-485-oof.jsonl"
ABSENT_SIDE = (("AI response type", "Solve-Provide"), ("AI response type", "Explain"))
TARGETS = (("student interaction type", "Guided generation"), ("AI response type", "Evaluate"),
           ("AI delegation", "hand-back: directive"), ("AI delegation", "hand-back: asks the student to supply"))


# ----------------------------------------------------------------------------- assemble (steps 6 and 9)

def assemble() -> int:
    space = SPACE
    refs, fingerprint = EV.references_from_prepared(space.prepared)
    all_keys = sorted(refs)
    manifest_sha = sha(space.prepared / "manifest.json")
    folds = B.fold_keys(space)
    rosters = {k: set(f["test"]) for k, f in folds.items()}
    assert sorted(set().union(*rosters.values())) == all_keys and sum(map(len, rosters.values())) == len(all_keys)
    sources = {k: space.run_dir(k) / "outputs" / "predictions-validation.jsonl" for k in folds}
    score_sources = {k: space.run_dir(k) / "outputs" / "scores-validation.jsonl" for k in folds}
    before = {str(p): sha(p) for p in list(sources.values()) + list(score_sources.values())}
    origin, per_fold, scores = {}, {}, {}
    for k, path in sources.items():
        source = L.read_predictions(path)
        keys = {(r["conversation_id"], r["segment_order"]) for r in source}
        assert keys == rosters[k] and len(source) == len(keys), f"fold {k}: not exactly the fold's test interactions"
        assert len({tuple(r[f] for f in L.RUN_FIELDS) for r in source}) == 1, f"fold {k}: mixed provenance"
        first = source[0]
        assert first["dataset_manifest_sha256"] == manifest_sha, f"fold {k}: a different prepared dataset"
        route, _, tail = first["model_route"].partition("#")
        per_fold[k] = {"source": L.rel(path), "source_sha256": before[str(path)], "scores": L.rel(score_sources[k]),
                       "scores_sha256": before[str(score_sources[k])], "records": len(source), "original_run_id": first["run_id"],
                       "original_model_route": first["model_route"], "original_config_sha256": first["config_sha256"], "model": route,
                       "final_checkpoint": tail, "config": L.rel(space.config(k)), "config_file_sha256": sha(space.config(k)),
                       "split": L.rel(space.fold_split(k)), "split_sha256": sha(space.fold_split(k)), "decoding": "threshold 0.5"}
        for r in source:
            origin[(r["conversation_id"], r["segment_order"])] = (k, r)
        for _, line in L.jsonl_lines(score_sources[k]):
            s = json.loads(line)
            key = (s["conversation_id"], s["segment_order"])
            assert key in rosters[k] and key not in scores, f"fold {k}: scores do not match the fold's test interactions"
            assert all(s[f] == first[f] for f in L.RUN_FIELDS), f"fold {k}: scores from another run"
            assert s["codes"] == {a: list(L.DIRECT_CODES[a]) for a in L.AXES}, f"fold {k}: scores use another code order"
            scores[key] = (k, s)
    assert sorted(scores) == all_keys, "the scores do not cover every interaction once"
    models = {v["model"] for v in per_fold.values()}
    assert len(models) == 1, f"the folds name different models {models}"
    run = {"run_id": f"{space.name}-C1-oof", "model_route": f"{models.pop()}#out-of-fold-final-checkpoints-of-folds-1-5",
           "config_sha256": cv.canonical_sha([per_fold[k]["original_config_sha256"] for k in sorted(per_fold)]),
           "dataset_manifest_sha256": manifest_sha}
    records = [L.prediction_record(key[0], key[1], origin[key][1]["labels"], run, origin[key][1]["not_predicted"]) for key in all_keys]
    OOF.parent.mkdir(parents=True, exist_ok=True)
    L.write_predictions(OOF, records)                        # exclusive: never overwrites
    with SCORES.open("x", encoding="utf-8", newline="\n") as f:
        for key in all_keys:
            k, s = scores[key]
            f.write(json.dumps({"record_kind": "model_scores_with_labels", "schema_version": s["schema_version"],
                                "vocabulary_sha256": s["vocabulary_sha256"], **run, "fold": k,
                                "original_run_id": s["run_id"], "conversation_id": key[0], "segment_order": key[1],
                                "codes": s["codes"], "probabilities": s["probabilities"],
                                "predicted_labels": origin[key][1]["labels"], "threshold": 0.5}, ensure_ascii=False) + "\n")
    written = L.read_predictions(OOF)
    preds, info = EV.predictions_from_file(OOF, reference=fingerprint, reference_keys=set(refs))
    redecoded = sum(L.decode_scores(scores[k][1]["probabilities"], 0.5) == {a: L.canonical(a, v) for a, v in preds[k].items()}
                    for k in all_keys)
    same_as_run = {}
    for k in folds:
        mine = roundtrip(EV.evaluate(refs, preds, folds[k]["test"]))
        theirs = read_json(space.run_dir(k) / "outputs" / "evaluation-validation.json")
        same_as_run[k] = mine["axes"] == theirs["axes"] and mine["all_axis_exact_match"] == theirs["all_axis_exact_match"]
    checks = {"records": len(written), "covers_every_interaction": sorted((r["conversation_id"], r["segment_order"]) for r in written) == all_keys,
              "labels_identical_to_sources": all(r["labels"] == origin[(r["conversation_id"], r["segment_order"])][1]["labels"] for r in written),
              "each_from_the_fold_that_held_it_out": all((r["conversation_id"], r["segment_order"]) in rosters[origin[(r["conversation_id"], r["segment_order"])][0]]
                                                         for r in written),
              "passes_evaluation_provenance_check": info["provenance_consistent_records"] == len(all_keys) and info["dataset_manifest_matches_reference"],
              "per_fold_scores_equal_each_run_report": same_as_run,
              "scores_records": len(scores), "scores_redecoded_at_0.5_equal_the_predictions": redecoded,
              "sources_unchanged": all(sha(Path(p)) == h for p, h in before.items())}
    manifest = {"what": f"out-of-fold model_prediction file and per-code scores of the final C1 run {space.name} (D-191; threshold 0.5; "
                        f"five folds; {len(all_keys)} interactions; development validation against the D-190 labels)",
                "created_utc": utc(),
                "how": "the saved per-fold test predictions and scores (each run's outputs/predictions-validation.jsonl and "
                       "scores-validation.jsonl) were read and rewritten; no model was run",
                "reading_note": "the files pool five models; run_id, model_route and config_sha256 describe the pooled set (config_sha256 = "
                                "SHA-256 of the JSON list of the five per-fold config_sha256 values, in fold order); the origin of every "
                                "record is under 'folds' and 'interaction_fold' (and 'fold' / 'original_run_id' in the scores file)",
                "files": {"predictions": {"path": L.rel(OOF), "sha256": sha(OOF), "records": len(written), "run_fields": run},
                          "scores": {"path": L.rel(SCORES), "sha256": sha(SCORES), "records": len(scores),
                                     "fields": "codes, probabilities (per axis, per code, schema order), predicted_labels (threshold 0.5)"}},
                "dataset": {"prepared": L.rel(space.prepared), "manifest_sha256": manifest_sha, "examples_sha256": fingerprint["examples_sha256"],
                            "interactions": len(all_keys)},
                "reference_labels": {"path": L.rel(space.labels), "sha256": sha(space.labels)},
                "folds": per_fold, "interaction_fold": {GS.key_text(*k): origin[k][0] for k in all_keys}, "checks": checks}
    write_json(OOF.parent / "oof-predictions-manifest.json", manifest, exclusive=True)
    ok = (checks["records"] == len(all_keys) and checks["covers_every_interaction"] and checks["labels_identical_to_sources"]
          and checks["each_from_the_fold_that_held_it_out"] and checks["passes_evaluation_provenance_check"]
          and all(same_as_run.values()) and checks["scores_redecoded_at_0.5_equal_the_predictions"] == len(all_keys)
          and checks["sources_unchanged"])
    print(json.dumps({**checks, "predictions_sha256": sha(OOF), "scores_sha256": sha(SCORES), "all_checks_pass": ok}, indent=1))
    SPACE.note(f"assembled {L.rel(OOF)} and {L.rel(SCORES)} ({len(written)} records, checks pass: {ok})")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- report (step 9)

def agreement_table(report: dict) -> dict:
    """Per code: A (reference), B (model), C (both) and C / max(A, B), from evaluation.py's per-category counts; the absent
    side for Solve-Provide and Explain."""
    out = {}
    for axis in L.AXES:
        r = report["axes"][axis]
        n = r["n"]
        for code, c in r["per_category"].items():
            a, b, both = c["support"], c["predicted"], c["tp"]
            out[f"{axis} | {code}"] = {"A": a, "B": b, "C": both, "agreement": both / max(a, b) if max(a, b) else None, "n": n}
        for ax, code in ABSENT_SIDE:
            if ax == axis:
                c = r["per_category"][code]
                a, b, both = n - c["support"], n - c["predicted"], c["tn"]
                out[f"{axis} | {code} (absent)"] = {"A": a, "B": b, "C": both, "agreement": both / max(a, b) if max(a, b) else None, "n": n}
    return out


def compare_d186(plan_record: dict) -> dict:
    """Both trial and final OOF predictions against the D-190 labels, paired (student_bootstrap.py --compare)."""
    out_dir = HERE / "comparison-d186"
    out_dir.mkdir(exist_ok=True)
    d186_inputs = read_json(F.D186 / "plan.json")["dataset"]["manifest_sha256"]
    result = {"d186_predictions": {"path": L.rel(D186_OOF), "sha256": sha(D186_OOF), "prepared_manifest_sha256": d186_inputs},
              "inputs_identical_to_d186": plan_record["input_check"]["identical_inputs"], "sets": {}}
    for s in SETS:
        ref = SPACE.reference(s)
        out = out_dir / f"compare-{s}.json"
        if not out.exists():
            r = subprocess.run([str(B.PY), "-I", str(B.BOOTSTRAP), "--reference", str(ref), "--pred", f"D191={OOF}",
                                "--pred", f"D186={D186_OOF}", "--compare", "D191-D186", "--replicates", str(B.BOOT_REPLICATES),
                                "--seed", str(B.BOOT_SEED), "--out", str(out)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
            if r.returncode != 0:
                raise SystemExit(f"student_bootstrap.py failed for set {s}: {r.stderr[-800:]}")
            (out_dir / f"compare-{s}.txt").write_text(r.stdout, encoding="utf-8")
        res = read_json(out)
        result["sets"][s] = {"file": L.rel(out), "sha256": sha(out), "axes": {
            a: {"rows": v["rows"], "students": v["students"],
                "D191": {m: x["value"] for m, x in v["models"]["D191"].items()},
                "D186": {m: x["value"] for m, x in v["models"]["D186"].items()},
                "difference": {m: {"value": c["difference"], "ci": c["ci"]} for m, c in v["comparisons"]["D191 - D186"].items()
                               if m not in ("paired", "pooled_f1")},
                "pooled_f1": v["comparisons"]["D191 - D186"]["pooled_f1"],
                "paired_exact": v["comparisons"]["D191 - D186"]["paired"]} for a, v in res["axes"].items()}}
    return result


def report() -> int:
    space = SPACE
    refs, fingerprint = EV.references_from_prepared(space.prepared)
    statement = EV.statements(fingerprint)["validation"]
    plan_record = read_json(HERE / "plan.json")
    all_keys = sorted(refs)
    parts = {p: sorted(L.parse_key(t) for t in plan_record["parts"][p]) for p in PARTS}
    assert sorted(k for p in PARTS for k in parts[p]) == all_keys
    sets = {ALL: all_keys, **parts}
    folds = B.fold_keys(space)
    c1, info = EV.predictions_from_file(OOF, reference=fingerprint, reference_keys=set(refs))
    d186, _ = EV.predictions_from_file(D186_OOF)        # made for identical inputs (plan.json input_check); scored on D-190 labels
    assert set(d186) == set(refs)
    majority, majority_labels = {}, {}
    for k, f in folds.items():
        majority_labels[k] = cv.majority_labels([refs[x] for x in f["train"]])
        majority.update({x: majority_labels[k] for x in f["test"]})
    sources = {"C1": c1, "majority": majority}
    results = {s: {n: B.set_tables(refs, p, keys, folds) for n, p in sources.items()} for s, keys in sets.items()}
    d186_pooled = {s: roundtrip(EV.evaluate(refs, d186, keys)) for s, keys in sets.items()}
    out_dir = HERE / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for s in sets:
        for n in sources:
            full = {"title": f"{space.name} pooled out-of-fold, set {s}: {n}", "statement": statement,
                    "note": f"set {s} of the 485 (parts reported separately); pooled over the 5 test folds; every interaction predicted once"
                            + ("; majority = most frequent code / code set of each training fold" if n == "majority" else ""),
                    "reference": fingerprint, "predictions": info if n == "C1" else {"source": "majority baseline per training fold"},
                    "selection": {"part": f"set {s}, out of fold"}, **results[s][n]["pooled"]}
            for suffix, text in ((".json", json.dumps(full, ensure_ascii=False, indent=1) + "\n"), (".md", EV.markdown(full))):
                path = out_dir / f"pooled-{n}-{s}{suffix}"
                path.write_text(text, encoding="utf-8")
                files[L.rel(path)] = sha(path)
    boots, agree = {}, {}
    for s in sets:
        boots[s], paths = B.bootstrap(space, s, plan_record)
        for p in paths:
            files[L.rel(p)] = sha(p)
        pooled = results[s]["C1"]["pooled"]
        agree[s] = {}
        for axis in L.AXES:
            m, r = boots[s]["axes"][axis]["models"]["C1"], pooled["axes"][axis]
            pairs = [(m["exact"]["value"], r["row_accuracy"]), (m["macro_f1"]["value"], r["macro_f1"])]
            if axis not in L.SINGLE:
                pairs.append((m["micro_f1"]["value"], cv.multi_extra(r)["micro_f1"]))
            agree[s][axis] = boots[s]["axes"][axis]["rows"] == r["n"] and all(abs(a - b) < 1e-9 for a, b in pairs)
    agreement = {s: {"C1": agreement_table(results[s]["C1"]["pooled"]), "majority": agreement_table(results[s]["majority"]["pooled"]),
                     "D186": agreement_table(d186_pooled[s])} for s in sets}
    comparison = compare_d186(plan_record)
    for s in sets:      # the comparison's D-186 point values equal evaluation.py's scoring of the same predictions
        for axis in L.AXES:
            assert abs(comparison["sets"][s]["axes"][axis]["D186"]["exact"] - d186_pooled[s]["axes"][axis]["row_accuracy"]) < 1e-9
    for s in sets:
        files[comparison["sets"][s]["file"]] = comparison["sets"][s]["sha256"]
    status = read_json(HERE / "run-status.json")
    runs = {k: B.run_record(space, k, status, plan_record.get("estimate")) for k in folds}
    trainfit = {}
    for k, f in folds.items():
        preds, _ = EV.predictions_from_file(space.run_dir(k) / "outputs" / "predictions-train.jsonl")
        trainfit[k] = cv.headline(roundtrip(EV.evaluate(refs, preds, f["train"])))
    for p in (OOF, SCORES, OOF.parent / "oof-predictions-manifest.json"):
        files[L.rel(p)] = sha(p)
    manifest_note = read_json(F.LABELS_MANIFEST).get("note")
    metrics = {"what": f"{space.name}: the final C1 run D-191 (threshold 0.5), student-grouped 5-fold CV over the 485 interactions with "
                       "the D-190 training labels; development validation against the researcher's labels, not human reliability",
               "statement": statement,
               "caveats": ["the labels are the researcher's: the 150 after discussion (D-179: where disagreement remains, the researcher's "
                           "code), not joint adjudication; the 135 and the 200 single-coded", f"label manifest: {manifest_note}",
                           "C1's settings were chosen on the v9 folds of cv-20261022, which include the 150 interactions",
                           "README step 8 (one model on all 485) was skipped for now (master README step 1)",
                           "the 5 truncated responses (all in the 135) stay truncated, as in D-186"],
               "sets": {s: len(k) for s, k in sets.items()}, "fold_sizes": {s: results[s]["C1"]["fold_sizes"] for s in sets},
               "tables": {s: {n: results[s][n]["table"] for n in sources} for s in sets},
               "per_code": {s: {n: {a: {c: {kk: v[kk] for kk in ("support", "predicted", "tp", "precision", "recall", "f1")}
                                        for c, v in results[s][n]["pooled"]["axes"][a]["per_category"].items()} for a in L.AXES}
                                for n in sources} for s in sets},
               "agreement": agreement,
               "all_four_right": {s: {n: results[s][n]["pooled"]["all_axis_exact_match"] for n in sources} for s in sets},
               "bootstrap": {s: {"file": L.rel(space.results / "bootstrap" / f"bootstrap-{s}.json"), "reference": L.rel(space.reference(s)),
                                 "reference_sha256": sha(space.reference(s)), "replicates": boots[s]["bootstrap"]["replicates"],
                                 "seed": boots[s]["bootstrap"]["seed"],
                                 "axes": {a: {"rows": v["rows"], "students": v["students"], "C1": v["models"]["C1"]} for a, v in boots[s]["axes"].items()},
                                 "point_values_equal_evaluation_module": agree[s]} for s in sets},
               "comparison_with_d186": comparison, "majority_labels_per_training_fold": majority_labels,
               "training_fold_fit": trainfit, "runs": runs, "predictions": info,
               "per_fold_reports": {s: {n: results[s][n]["per_fold"] for n in sources} for s in sets},
               "pooled_reports": {s: {n: results[s][n]["pooled"] for n in sources} for s in sets}, "files": files}
    write_json(HERE / "metrics.json", metrics)
    (HERE / "summary.md").write_text(summary_md(metrics, plan_record), encoding="utf-8")
    print(json.dumps({s: {a: {m: round(v["pooled"], 3) for m, v in metrics["tables"][s]["C1"][a].items()
                              if m in ("accuracy", "macro_f1", "micro_f1", "rate") and v["pooled"] is not None}
                          for a in metrics["tables"][s]["C1"]} for s in sets}, ensure_ascii=False))
    print("bootstrap point values equal evaluation module:", {s: all(v.values()) for s, v in agree.items()})
    SPACE.note("report written: metrics.json, summary.md, bootstrap/, reports/, comparison-d186/")
    return 0


def f3(x) -> str:
    return "—" if x is None else f"{x:.3f}"


def summary_md(m: dict, plan_record: dict) -> str:
    t, fs, ic, est = m["tables"], m["fold_sizes"], plan_record["input_check"], plan_record["estimate"]
    zh = {"accuracy": "准确率（多选为整组一致率）", "macro_f1": "macro-F1", "micro_f1": "micro-F1"}
    boot_key = {"accuracy": "exact", "macro_f1": "macro_f1", "micro_f1": "micro_f1"}
    part_zh = {"150": "研究者讨论后的标签（v15）", "135": "单人编码（v4）", "200": "单人编码，经 AI 辅助检查（v1）"}
    out = [f"# {SPACE.name}：最终 C1 运行（D-191；485 条，D-190 训练标签；阈值 0.5；学生分组 5 折）", "",
           f"性质：{m['statement']}", "",
           "**说明（随结果一并陈述）：**",
           "- 标签是研究者的标签：150 部分是讨论后的标签（D-179：仍有分歧处采用研究者的代码），不是共同裁定；135 与 200 部分是单人编码。三部分分开报告。",
           "- 标签清单所列：150 中 P035_P2_06 #1 为暂定（附件缺失），P060_P2_MM1 #1 已标记；第二编码员的 30 条若返回，只用于一致性。",
           "- C1 的设置是在 cv-20261022 的 v9 折上选定的，那些折包含这 150 条互动。",
           "- 按总指令第 1 步，README 第 8 步（用全部 485 条训练一个模型）暂不执行。",
           "- 输入与 D-186 逐 token 相同（485/485），折、设置和代码也相同；与 D-186 只差训练标签（26 个单元格）。",
           "- 三部分：" + "；".join(f"{p} 为{part_zh[p]}" for p in PARTS) + "。485 合并不是同一种参照标签。", "",
           "每条互动只被预测一次（由不含其学生的那一折的模型预测）。各折测试集大小（485 / 150 / 135 / 200）："
           + "；".join(f"折{k} " + " / ".join(str(fs[s].get(k, 0)) for s in SETS) for k in fs[ALL]) + "。", "",
           "## 各维度结果", "", "每格：C1 合并值 [学生重抽样 95% 区间]（多数类基线；基线来自各训练折，仅作背景）。", "",
           "| 维度 | 指标 | " + " | ".join(SETS) + " |", "|---|---|" + "---|" * len(SETS)]
    for axis in L.AXES:
        for metric in ("accuracy", "macro_f1", "micro_f1"):
            if metric not in t[ALL]["C1"][axis] or (axis in L.SINGLE and metric == "micro_f1"):
                continue
            cells = []
            for s in SETS:
                ci = m["bootstrap"][s]["axes"][axis]["C1"][boot_key[metric]]["ci"]
                cells.append(f"{f3(t[s]['C1'][axis][metric]['pooled'])} [{f3(ci[0])}, {f3(ci[1])}]（{f3(t[s]['majority'][axis][metric]['pooled'])}）")
            out.append(f"| {axis} | {zh[metric]} | " + " | ".join(cells) + " |")
    a4 = m["all_four_right"]
    out.append("| 四维全对 | 条数 / n | " + " | ".join(f"{a4[s]['C1']['matches']}/{a4[s]['C1']['n']}（{a4[s]['majority']['matches']}）" for s in SETS) + " |")
    out += ["", "## 每个代码的简单一致率 C / max(A, B)", "",
            "A = 参照有该代码的互动数，B = 模型预测有该代码的互动数，C = 两者都有。Solve-Provide 与 Explain 另给“没有该代码”一侧（同一公式）。"
            "参照为空的行不计入该维度。", "",
            "| 维度 | 代码 | " + " | ".join(f"{s}：A / B / C → 一致率" for s in SETS) + " |", "|---|---|" + "---|" * len(SETS)]
    for name in m["agreement"][ALL]["C1"]:
        axis, code = name.split(" | ", 1)
        cells = []
        for s in SETS:
            v = m["agreement"][s]["C1"][name]
            cells.append(f"{v['A']} / {v['B']} / {v['C']} → {f3(v['agreement'])}")
        out.append(f"| {axis} | {code} | " + " | ".join(cells) + " |")
    out += ["", "## 每个代码的 F1（合并）", "",
            "| 维度 | 代码 | 支持数 " + " / ".join(SETS) + " | " + " | ".join(f"C1 F1 {s}" for s in SETS) + " | 多数类 F1 485 |",
            "|---|---|---|" + "---|" * len(SETS) + "---|"]
    pc = m["per_code"]
    for axis in L.AXES:
        for code in L.DIRECT_CODES[axis]:
            out.append(f"| {axis} | {code} | " + " / ".join(str(pc[s]["C1"][axis][code]["support"]) for s in SETS) + " | "
                       + " | ".join(f3(pc[s]["C1"][axis][code]["f1"]) for s in SETS) + f" | {f3(pc[ALL]['majority'][axis][code]['f1'])} |")
    out += ["", "支持数 < 5 的代码（稀有）的 F1 与一致率不稳定；macro-F1 只平均在该集合中有支持的代码，稀有代码在一次重抽样中缺席时其区间可能偏离点估计。", "",
            "## 485 的逐折结果（C1；均值 ± 标准差，以及各折值）", "",
            "| 维度 | 指标 | 均值 ± 标准差 | " + " | ".join(f"折{k}" for k in fs[ALL]) + " |", "|---|---|---|" + "---|" * len(fs[ALL])]
    for axis in list(L.AXES) + ["all_axis_exact_match"]:
        for metric, v in t[ALL]["C1"][axis].items():
            if metric == "hamming_accuracy" or (axis in L.SINGLE and metric == "micro_f1"):
                continue
            out.append(f"| {'四维全对' if axis == 'all_axis_exact_match' else axis} | {'比例' if metric == 'rate' else zh[metric]} | "
                       f"{f3(v['mean'])} ± {f3(v['sd'])} | " + " | ".join(f3(x) for x in v["per_fold"]) + " |")
    comp = m["comparison_with_d186"]
    out += ["", "## 与 D-186 试运行的描述性配对比较（同一 485 条，同一参照 D-190）", "",
            "两次运行的输入、折、设置和代码相同，只差训练标签（26 个单元格）；两组折外预测都按 D-190 标签计分，用 student_bootstrap.py 配对的学生重抽样"
            "（10,000 次，种子 20261026）。逐条配对的 p 把互动当作独立、未作多重比较校正，仅供参考。", "",
            "| 维度 | 集合 | 整组一致率 D-186 → D-191（差 [95% 区间]） | 逐条配对：D-191 对且 D-186 错 vs 反之（p） | 合并 F1 D-186 → D-191（差 [95% 区间]） |",
            "|---|---|---|---|---|"]
    for s in SETS:
        for axis, v in comp["sets"][s]["axes"].items():
            d, p, pf = v["difference"], v["paired_exact"], v["pooled_f1"]
            metric = pf["metric"]
            out.append(f"| {axis} | {s} | {f3(v['D186']['exact'])} → {f3(v['D191']['exact'])}（{d['exact']['value']:+.3f} "
                       f"[{d['exact']['ci'][0]:+.3f}, {d['exact']['ci'][1]:+.3f}]） | {p['D191 right, D186 wrong']} vs {p['D186 right, D191 wrong']}"
                       f"（{p['exact_binomial_p_per_interaction']:.3f}） | {metric} {f3(v['D186'][metric])} → {f3(v['D191'][metric])}（{pf['difference']:+.3f} "
                       f"[{pf['ci'][0]:+.3f}, {pf['ci'][1]:+.3f}]） |")
    out += ["", "合并 F1：学生互动类型用 macro-F1，其余三维用 micro-F1。", "",
            "四个目标代码的简单一致率（485）：" + "；".join(
                f"{c} D-186 {f3(m['agreement'][ALL]['D186'][f'{a} | {c}']['agreement'])} → D-191 {f3(m['agreement'][ALL]['C1'][f'{a} | {c}']['agreement'])}"
                for a, c in TARGETS) + "。", "",
            "## 训练拟合（诊断，不是表现）与运行记录", "",
            "| 折 | 训练 / 测试 | 优化步 | 恢复次数 | 训练折准确率（四维） | 训练折四维全对 | 用时（秒） | 估计（秒） | 每步结束时 MPS 驱动内存峰值（GiB） |",
            "|---|---|---|---|---|---|---|---|---|"]
    folds = {f["fold"]: f for f in plan_record["folds"]}
    for k, r in m["runs"].items():
        fit = m["training_fold_fit"][k]
        mem = r["mps_driver_gib_at_step_end"]["max"] if r["mps_driver_gib_at_step_end"] else "—"
        out.append(f"| {k} | {folds[int(k)]['train']} / {folds[int(k)]['test']} | {r['optimizer_steps']} | {r['resumes']} | "
                   + " / ".join(f3(fit[a]["accuracy"]) for a in L.AXES) + f" | {f3(fit['all_axis_exact_match']['rate'])} | {r['wall_seconds']} | "
                   f"{r['estimated_seconds']} | {mem} |")
    out += ["", "## 实现说明（设计未变；详见 plan.json）", "",
            f"- 输入（C1 的预算 1,536 / 3,072 / 7,168）：485 条中上下文被截 {ic[ALL]['loses_context']} 条、当前提示被截 {ic[ALL]['loses_prompt']} 条、"
            f"当前回复被截 {ic[ALL]['loses_response']} 条；" + "；".join(f"{p}：{ic[p]['loses_context']} / {ic[p]['loses_prompt']} / {ic[p]['loses_response']}" for p in PARTS)
            + f"。被截的 {len(ic['truncated_responses'])} 条回复都在 135 部分，保持截断（与 D-186 相同）。",
            f"- 用时：按 D-186 实测估计 {est['total_hours']} 小时，实际 {round(sum(r['wall_seconds'] for r in m['runs'].values()) / 3600, 2)} 小时。",
            "- 折：D-186 的名单不变（data/splits/cv485-20261028d，按新准备数据重新绑定哈希；cv.assign_folds 复现同样的测试学生）。",
            "- 配置：D-186 的配置，只改数据路径、标签说明、状态和验证用途。类别权重按同一规则由新训练标签算出，所以数值与 D-186 略有不同。",
            "- 每个代码的概率：`oof-predictions/scores-C1-485-oof.jsonl`（每条测试互动、每个头、每个代码的概率，附 0.5 下的预测标签）；训练器原本就保存这些，没有改代码。",
            "- 学生重抽样区间：finetuning/student_bootstrap.py，10,000 次，种子 20261026；485 的参照为标签文件，各部分的参照为该文件中该部分记录的逐字节子集（part-labels/）。"
            "点估计与评价模块一致：" + "，".join(f"{s} {all(m['bootstrap'][s]['point_values_equal_evaluation_module'].values())}" for s in SETS) + "。", "",
            "混淆矩阵、各代码的 precision/recall 与四维组合分布见 `reports/pooled-*.md` 与 `metrics.json`。", "", "## 文件", ""]
    out += [f"- `{k}` sha256 `{v}`" for k, v in m["files"].items() if "/pooled-" not in k]
    out += [f"- `reports/pooled-{{C1,majority}}-{{{','.join(SETS)}}}.json/.md`（{sum(1 for k in m['files'] if '/pooled-' in k)} 个文件；sha256 见 metrics.json files）", ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("assemble", "report"))
    a = p.parse_args(argv)
    return assemble() if a.command == "assemble" else report()


if __name__ == "__main__":
    raise SystemExit(main())
