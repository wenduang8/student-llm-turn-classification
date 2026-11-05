#!/usr/bin/env python3
"""The one evaluation entry for four-axis predictions against an explicitly chosen reference.

    python finetuning/evaluation.py --prepared data/training/<version> --predictions <file.jsonl> \
        [--split <split.json> --part validation] [--note "<extra context>"] --out <report.json>

Per axis it reports the rows considered, the rows with both a reference and a prediction (n) and
why others were excluded; row accuracy (single-select: same code; multi-select: identical code
set); macro-F1; per-category support, predicted count, precision, recall and F1; the confusion
matrix (single-select) or per-category TP/FP/FN/TN (multi-select); and the reference/model joint
label distribution (single-select: the confusion matrix; multi-select: joint code-set counts).
Across axes it reports all-axis exact match over rows whose four reference labels all exist, and
the cross-axis joint distribution: each interaction's four labels, in schema axis order with
multi-select codes in schema order, form one combination, counted for the reference and for the
model over the same interactions (those with four reference labels), shown for the union of
combinations with counts, shares of one stated denominator and consistency checks. Rare
categories (support below a threshold, including zero) are listed per axis in addition to the
macro-averages, so they are never visible only through an average.

Provenance: every record of a predictions file must carry the same run_id, model_route (which
names the checkpoint), config_sha256 and dataset_manifest_sha256, or the file is refused with
the line and field. The evaluation entry also requires the declared dataset manifest to be the
reference's and every prediction to be for an interaction of the reference. A predicted axis
left null with its not_predicted reason is legitimate, not a provenance error.

PROVISIONAL DEFINITIONS (multi-select details and missing/default handling are still pending):
  * a null reference cell (unavailable, truncated capture) is excluded from that axis; it is
    never counted as a negative and never as an error;
  * a null prediction cell is excluded from that axis's n (and counted under excluded), and it
    makes the row a non-match for all-axis exact match;
  * macro_f1 averages per-category F1 over categories with reference support > 0 in the rows
    evaluated; macro_f1_union also includes categories that were only predicted (F1 0 there);
    zero-support categories are listed with how often the model predicted them;
  * precision is null when a category is never predicted; F1 is 0 when support > 0 and TP = 0;
  * the joint distribution includes only interactions with all four reference labels; a model
    combination with a null axis is counted as prediction-incomplete, never filled in.
Every report states what it is, derived from the reference's own record kinds (statements()),
and for an AI-assisted review the review_provenance.decision read from the review file itself
(D-105 or D-144), checked against the hash the prepared data recorded; it is never fixed text.
For this project's pending-review labels, "development validation against pending-review
researcher labels; not final RQ3 performance and not human reliability". It is not a
reliability statistic, and a development check is not a final evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as L  # noqa: E402

RARE_THRESHOLD = 5
REFERENCE_KINDS = {   # (Chinese, English) description of a reference record kind; {decision} is review_decisions()
    "AI_assisted_development_review": ("{decision} AI辅助复核版（待复核；非独立编码；非共同裁定）",
                                       "the {decision} AI-assisted review of the researcher's labels (pending review; "
                                       "not independent coding; not joint adjudication)"),
    "human_independent": ("研究者原始标注（按收到原样；待复核）", "the researcher's original labels as received (pending review)"),
}
SYNTHETIC_STATEMENT = "Synthetic inputs: pipeline check only; no research meaning."


def review_decisions(reference: dict) -> str:
    """The review_provenance.decision of the reference's AI-assisted review sources (e.g. "D-105",
    "D-144"; several are joined with "/"), read from each review file named in the reference after
    checking it is the file the prepared data recorded (same SHA-256)."""
    found = set()
    for src in reference["sources"]:
        if src["record_kind"] != L.DEV_REVIEW_KIND:
            continue
        if not src.get("path") or not src.get("sha256"):
            raise ValueError("an AI-assisted review source must name its review file and SHA-256")
        path = Path(src["path"]) if Path(src["path"]).is_absolute() else L.ROOT / src["path"]
        if not path.is_file() or L.sha256_file(path) != src["sha256"]:
            raise ValueError(f"review file {src['path']} is missing or is not the file the reference was prepared "
                             "from (SHA-256 differs), so its review decision cannot be stated")
        found |= {item.source_record["review_provenance"]["decision"] for item in L.read_reviewed_direct_jsonl(path)}
    if not found:
        raise ValueError("the reference's AI-assisted review files record no review decision")
    return "/".join(sorted(found))


def statements(reference: dict) -> dict:
    """What each report is, derived from the reference's own record kinds and, for an AI-assisted
    review, the decision recorded in its review file; never fixed text."""
    kinds = sorted({src["record_kind"] for src in reference["sources"]})
    if not kinds:
        raise ValueError("the reference records no source record kinds")
    if kinds == ["synthetic"]:
        return {k: SYNTHETIC_STATEMENT for k in ("validation", "train", "prior", "all")}
    unknown = [k for k in kinds if k not in REFERENCE_KINDS]
    if unknown:
        raise ValueError(f"no report statement is defined for reference record kinds {unknown}")
    decision = review_decisions(reference) if L.DEV_REVIEW_KIND in kinds else ""
    zh = "；".join(REFERENCE_KINDS[k][0].format(decision=decision) for k in kinds)
    en = "; ".join(REFERENCE_KINDS[k][1].format(decision=decision) for k in kinds)
    return {"validation": f"基于待复核研究者标注的开发验证，参照为{zh}；不是最终RQ3表现，也不是人类信度。"
                          f"Development validation against {en}; not final RQ3 performance and not human reliability.",
            "train": f"训练集拟合诊断（参照为{zh}）：模型已在这些互动上训练，只说明拟合程度，不是表现。"
                     f"Training-fit diagnostic against {en} on interactions the model was trained on; not performance.",
            "prior": f"读数参照：按训练集标签先验的无模型预测（参照为{zh}），未训练、未调参、不参与任何选择。"
                     f"Reading aid only: label-prior predictions from training labels, compared with {en}; "
                     f"no model, no tuning, no selection.",
            "all": f"对全部准备互动的开发评价（可能包含训练用互动），参照为{zh}；不是最终RQ3表现，也不是人类信度。"
                   f"Development evaluation over all prepared interactions (may include training interactions) "
                   f"against {en}; not final RQ3 performance and not human reliability."}
DEFINITIONS = {
    "null_reference": "excluded from that axis; never a negative and never an error",
    "row_accuracy": "single-select: same code; multi-select: identical code set",
    "null_prediction": "excluded from that axis's n (reported under excluded); a non-match for all-axis exact match",
    "macro_f1": "mean per-category F1 over categories with reference support > 0 in the evaluated rows",
    "macro_f1_union": "mean per-category F1 over categories with reference support > 0 or at least one prediction",
    "precision": "null when the category is never predicted",
    "f1": "0 when support > 0 and there is no true positive; null when support = 0",
    "all_axis_exact_match": "denominator: rows whose four reference labels all exist; a match needs all four predictions present and equal",
    "rare": f"reference support < {RARE_THRESHOLD} in the evaluated rows (including 0)",
    "joint_distribution": ("four-axis combination per interaction (schema axis order; multi-select codes in schema "
                           "order); included: interactions with all four reference labels, the same set for reference "
                           "and model; every share uses the included count as denominator; a prediction with any null "
                           "axis is counted as prediction-incomplete and not as a combination"),
    "status": "provisional definitions; multi-select details and missing/default handling are pending",
}


def _prf(tp: int, fp: int, fn: int) -> dict:
    support, predicted = tp + fn, tp + fp
    precision = tp / predicted if predicted else None
    recall = tp / support if support else None
    if not support:
        f1 = None
    elif not tp:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"support": support, "predicted": predicted, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1}


def _set_text(axis: str, values) -> str:
    return " + ".join(L.canonical(axis, values))


def _combination(labels: dict) -> tuple:
    """One interaction's four labels as a hashable combination, axes and codes in schema order."""
    return tuple(labels[a] if a in L.SINGLE else tuple(L.canonical(a, labels[a])) for a in L.AXES)


def _combination_text(combination: tuple) -> str:
    return " | ".join(v if isinstance(v, str) else " + ".join(v) for v in combination)


def joint_distribution(references: dict, predictions: dict, keys: list) -> dict:
    """Reference and model counts of four-axis label combinations over the same interactions."""
    complete = [k for k in keys if all(references[k][a] is not None for a in L.AXES)]
    incomplete = {k for k in complete if any(predictions[k][a] is None for a in L.AXES)}
    scored = [k for k in complete if k not in incomplete]
    ref = Counter(_combination(references[k]) for k in complete)
    mod = Counter(_combination(predictions[k]) for k in scored)
    same = Counter(_combination(references[k]) for k in scored if _combination(references[k]) == _combination(predictions[k]))
    n = len(complete)
    share = (lambda c: c / n) if n else (lambda c: None)
    union = sorted(set(ref) | set(mod), key=lambda c: (-ref[c], -mod[c], _combination_text(c)))
    rows = [{"combination": _combination_text(c), "labels": {a: (v if isinstance(v, str) else list(v)) for a, v in zip(L.AXES, c)},
             "reference": ref[c], "reference_share": share(ref[c]), "model": mod[c], "model_share": share(mod[c]),
             "same_interaction_matches": same[c],
             "seen_in": "both" if ref[c] and mod[c] else ("reference only" if ref[c] else "model only")} for c in union]
    checks = {"reference_counts_sum_to_included": sum(ref.values()) == n,
              "model_counts_plus_prediction_incomplete_sum_to_included": sum(mod.values()) + len(incomplete) == n}
    if not all(checks.values()):                     # unreachable unless the counting above is wrong
        raise AssertionError(f"joint distribution checks failed: {checks}")
    return {"rows_considered": len(keys), "included": n, "excluded_reference_incomplete": len(keys) - n,
            "prediction_incomplete": len(incomplete), "prediction_incomplete_share": share(len(incomplete)),
            "denominator": n,
            "distinct": {"reference": len(ref), "model": len(mod), "union": len(union),
                         "both": sum(1 for c in union if ref[c] and mod[c]),
                         "reference_only": sum(1 for c in union if ref[c] and not mod[c]),
                         "model_only": sum(1 for c in union if mod[c] and not ref[c])},
            "reference_share_on_combinations_the_model_produced": share(sum(ref[c] for c in union if mod[c])),
            "model_share_on_combinations_in_the_reference": share(sum(mod[c] for c in union if ref[c])),
            "combinations": rows, "checks": checks}


def evaluate(references: dict, predictions: dict, keys=None, rare_threshold: int = RARE_THRESHOLD) -> dict:
    """references/predictions: {(conversation_id, segment_order): four-axis labels (null allowed)}."""
    keys = sorted(references if keys is None else keys)
    if not keys:
        raise ValueError("no interactions to evaluate")
    for name, table in (("reference", references), ("prediction", predictions)):
        absent = [k for k in keys if k not in table]
        if absent:
            raise ValueError(f"{len(absent)} interactions have no {name}, e.g. {absent[:3]}")
        for k in keys:
            errors = L.check_labels(table[k])
            if errors:
                raise ValueError(f"{name} {k}: " + "; ".join(errors))
    axes = {}
    for axis in L.AXES:
        codes = L.DIRECT_CODES[axis]
        ref_null = sum(references[k][axis] is None for k in keys)
        pred_null = sum(references[k][axis] is not None and predictions[k][axis] is None for k in keys)
        valid = [k for k in keys if references[k][axis] is not None and predictions[k][axis] is not None]
        n = len(valid)
        res = {"selection": "single" if axis in L.SINGLE else "multiple", "rows": len(keys), "n": n,
               "excluded": {"reference_null": ref_null, "prediction_null": pred_null}}
        if axis in L.SINGLE:
            cm = [[0] * len(codes) for _ in codes]
            for k in valid:
                cm[codes.index(references[k][axis])][codes.index(predictions[k][axis])] += 1
            per = {}
            for i, c in enumerate(codes):
                tp = cm[i][i]
                per[c] = _prf(tp, sum(row[i] for row in cm) - tp, sum(cm[i]) - tp)
            res["row_accuracy"] = sum(cm[i][i] for i in range(len(codes))) / n if n else None
            res["confusion_matrix"] = {"rows": "reference", "columns": "prediction", "codes": codes, "counts": cm}
        else:
            per = {}
            for c in codes:
                tp = sum(c in references[k][axis] and c in predictions[k][axis] for k in valid)
                fp = sum(c not in references[k][axis] and c in predictions[k][axis] for k in valid)
                fn = sum(c in references[k][axis] and c not in predictions[k][axis] for k in valid)
                per[c] = {**_prf(tp, fp, fn), "tn": n - tp - fp - fn}
            res["row_accuracy"] = (sum(set(references[k][axis]) == set(predictions[k][axis]) for k in valid) / n
                                   if n else None)
            joint = Counter((_set_text(axis, references[k][axis]), _set_text(axis, predictions[k][axis])) for k in valid)
            res["joint_set_distribution"] = [{"reference": r, "prediction": p, "count": c}
                                             for (r, p), c in sorted(joint.items(), key=lambda x: (-x[1], x[0]))]
            res["reference_set_distribution"] = dict(Counter(_set_text(axis, references[k][axis]) for k in valid).most_common())
            res["prediction_set_distribution"] = dict(Counter(_set_text(axis, predictions[k][axis]) for k in valid).most_common())
        present = [c for c in codes if per[c]["support"] > 0]
        union = [c for c in codes if per[c]["support"] > 0 or per[c]["predicted"] > 0]
        res["macro_f1"] = sum(per[c]["f1"] for c in present) / len(present) if present else None
        res["macro_f1_categories"] = present
        res["macro_f1_union"] = (sum(per[c]["f1"] or 0.0 for c in union) / len(union)) if union else None
        res["macro_f1_union_categories"] = union
        res["zero_support_categories"] = {c: {"predicted": per[c]["predicted"]} for c in codes if not per[c]["support"]}
        res["rare_categories"] = {c: {"support": per[c]["support"], "predicted": per[c]["predicted"]}
                                  for c in codes if per[c]["support"] < rare_threshold}
        res["per_category"] = per
        axes[axis] = res
    complete = [k for k in keys if all(references[k][a] is not None for a in L.AXES)]
    unpredicted = [k for k in complete if any(predictions[k][a] is None for a in L.AXES)]
    same = sum(all(predictions[k][a] is not None and L.canonical(a, references[k][a]) == L.canonical(a, predictions[k][a])
                   for a in L.AXES) for k in complete)
    joint = joint_distribution(references, predictions, keys)
    # Cross-checks against the all-axis computation above, which compares axis by axis rather than
    # through combinations; a disagreement means one of the two is wrong, so the report is refused.
    cross = {"included_equals_all_axis_n": joint["included"] == len(complete),
             "prediction_incomplete_equals_rows_with_null_prediction": joint["prediction_incomplete"] == len(unpredicted),
             "same_interaction_matches_equal_all_axis_matches":
                 sum(r["same_interaction_matches"] for r in joint["combinations"]) == same}
    if not all(cross.values()):
        raise AssertionError(f"joint distribution disagrees with the all-axis exact match: {cross}")
    joint["checks"].update(cross)
    return {"definitions": dict(DEFINITIONS, rare=f"reference support < {rare_threshold} in the evaluated rows (including 0)"),
            "evaluation_module_sha256": L.sha256_file(__file__),
            "rows": len(keys), "keys_sha256": L.sha256_bytes(json.dumps(keys).encode("utf-8")),
            "axes": axes,
            "all_axis_exact_match": {"n": len(complete), "matches": same, "rows_with_null_prediction": len(unpredicted),
                                     "rate": same / len(complete) if complete else None},
            "joint_distribution": joint}


def _fmt(x, digits=3) -> str:
    return "—" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def markdown(report: dict) -> str:
    """A readable summary of an evaluation report; the JSON remains the record."""
    out = [f"# {report.get('title', 'Evaluation')}", "", f"**{report.get('statement', '')}**", ""]
    for k in ("reference", "predictions", "selection"):
        if k in report:
            out += [f"- {k}: `{json.dumps(report[k], ensure_ascii=False)}`"]
    out += ["", f"Rows evaluated: {report['rows']}. All-axis exact match: "
            f"{report['all_axis_exact_match']['matches']}/{report['all_axis_exact_match']['n']} "
            f"({_fmt(report['all_axis_exact_match']['rate'])}) over rows with all four reference labels.", ""]
    out += ["| Axis | n | excluded (ref null / pred null) | row accuracy | macro-F1 (categories) | "
            "macro-F1 incl. predicted-only (categories) |", "|---|---|---|---|---|---|"]
    for axis, r in report["axes"].items():
        out.append(f"| {axis} | {r['n']} | {r['excluded']['reference_null']} / {r['excluded']['prediction_null']} | "
                   f"{_fmt(r['row_accuracy'])} | {_fmt(r['macro_f1'])} ({len(r['macro_f1_categories'])}"
                   f"/{len(r['per_category'])}) | {_fmt(r['macro_f1_union'])} ({len(r['macro_f1_union_categories'])}"
                   f"/{len(r['per_category'])}) |")
    for axis, r in report["axes"].items():
        out += ["", f"## {axis} ({r['selection']}-select)", "",
                "| Category | support | predicted | TP | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
        for c, m in r["per_category"].items():
            out.append(f"| {c} | {m['support']} | {m['predicted']} | {m['tp']} | {_fmt(m['precision'])} | "
                       f"{_fmt(m['recall'])} | {_fmt(m['f1'])} |")
        rare = ", ".join(f"{c} (support {v['support']}, predicted {v['predicted']})" for c, v in r["rare_categories"].items())
        out += ["", f"Rare categories: {rare or 'none'}."]
        if "confusion_matrix" in r:
            codes = r["confusion_matrix"]["codes"]
            out += ["", "Confusion matrix (rows reference, columns prediction):", "",
                    "| reference \\ prediction | " + " | ".join(codes) + " |", "|---" * (len(codes) + 1) + "|"]
            out += [f"| {c} | " + " | ".join(str(x) for x in row) + " |"
                    for c, row in zip(codes, r["confusion_matrix"]["counts"])]
        else:
            out += ["", "Joint reference/prediction code sets:", "", "| reference | prediction | count |", "|---|---|---|"]
            out += [f"| {j['reference']} | {j['prediction']} | {j['count']} |" for j in r["joint_set_distribution"]]
    j = report.get("joint_distribution")
    if j:
        d = j["distinct"]
        out += ["", "## Cross-axis joint distribution (four-axis label combinations)", "",
                f"Included: {j['included']} of {j['rows_considered']} interactions (all four reference labels present; "
                f"{j['excluded_reference_incomplete']} excluded for an incomplete reference). Prediction incomplete: "
                f"{j['prediction_incomplete']} ({_fmt(j['prediction_incomplete_share'])}). Every share uses the "
                f"denominator {j['denominator']}; reference counts sum to it, and model counts plus prediction-incomplete "
                f"rows sum to it. Distinct combinations: reference {d['reference']}, model {d['model']}, union {d['union']} "
                f"(both {d['both']}, reference only {d['reference_only']}, model only {d['model_only']}). Reference share on "
                f"combinations the model also produced: {_fmt(j['reference_share_on_combinations_the_model_produced'])}; "
                f"model share on combinations present in the reference: "
                f"{_fmt(j['model_share_on_combinations_in_the_reference'])}.", "",
                "| " + " | ".join(L.AXES) + " | reference n (share) | model n (share) |", "|---" * (len(L.AXES) + 2) + "|"]
        for row in j["combinations"]:
            cells = [row["labels"][a] if isinstance(row["labels"][a], str) else " + ".join(row["labels"][a]) for a in L.AXES]
            out.append("| " + " | ".join(cells) + f" | {row['reference']} ({_fmt(row['reference_share'])}) | "
                       f"{row['model']} ({_fmt(row['model_share'])}) |")
    out += ["", "Definitions (provisional):", ""] + [f"- {k}: {v}" for k, v in report["definitions"].items()]
    return "\n".join(out) + "\n"


def references_from_prepared(prepared_dir: Path) -> tuple[dict, dict]:
    import grouped_split
    fingerprint = grouped_split.prepared_fingerprint(Path(prepared_dir))
    refs = {(e["conversation_id"], e["segment_order"]): e["labels"]
            for e in grouped_split.load_examples(Path(prepared_dir))}
    return refs, fingerprint


def predictions_from_file(path: Path, reference: dict | None = None, reference_keys=None) -> tuple[dict, dict]:
    """One run's predictions and their provenance, refusing any record that disagrees.

    Every record must carry the same run_id, model_route (naming the checkpoint), config_sha256
    and dataset_manifest_sha256; the first disagreement is refused with its line and fields.
    With `reference` (a prepared fingerprint) and `reference_keys`, the declared dataset manifest
    must be the reference's and every prediction must be for an interaction of the reference."""
    path = Path(path)
    records = L.read_predictions(path)
    if not records:
        raise ValueError(f"{path.name}: no prediction records")
    lines = [n for n, _ in L.jsonl_lines(path)]       # the same physical lines read_predictions read
    first = records[0]
    for line_no, r in zip(lines, records):
        differ = [f for f in L.RUN_FIELDS if r[f] != first[f]]
        if differ:
            detail = "; ".join(f"{f} {r[f]!r} (line {lines[0]}: {first[f]!r})" for f in differ)
            raise ValueError(f"{path.name} line {line_no}: provenance differs from line {lines[0]} in {', '.join(differ)}: "
                             f"{detail}. One predictions file must come from one run, checkpoint, config and dataset.")
    preds = {(r["conversation_id"], r["segment_order"]): r["labels"] for r in records}
    info = {"path": L.rel(path), "sha256": L.sha256_file(path), "records": len(records),
            **{f: first[f] for f in L.RUN_FIELDS}, "provenance_consistent_records": len(records)}
    if (reference is None) != (reference_keys is None):
        raise ValueError("reference and reference_keys go together")
    if reference is not None:
        if first["dataset_manifest_sha256"] != reference["manifest_sha256"]:
            raise ValueError(f"{path.name} declares dataset manifest {first['dataset_manifest_sha256']!r}, but the "
                             f"reference {reference['dir']} has manifest {reference['manifest_sha256']!r}; these "
                             "predictions were not made for this evaluation input")
        stray = sorted(set(preds) - set(reference_keys))
        if stray:
            raise ValueError(f"{path.name}: {len(stray)} predictions are for interactions absent from the reference "
                             f"{reference['dir']}, e.g. {stray[:2]}")
        info["dataset_manifest_matches_reference"] = True
    return preds, info


def run(prepared: Path, predictions: Path, note: str | None, title: str, split: Path | None = None,
        part: str | None = None) -> dict:
    refs, fingerprint = references_from_prepared(prepared)
    statement = statements(fingerprint)[part or "all"]
    preds, pred_info = predictions_from_file(predictions, reference=fingerprint, reference_keys=set(refs))
    keys, selection = None, {"part": "all prepared interactions"}
    if split is not None:
        import grouped_split
        record, part_of = grouped_split.load_split(split, prepared)
        keys = [k for k, p in part_of.items() if p == part]
        selection = {"split": L.rel(split), "split_sha256": L.sha256_file(split), "part": part,
                     "split_status": record["status"]}
    report = evaluate(refs, preds, keys)
    return {"title": title, "statement": statement, "note": note, "reference": fingerprint, "predictions": pred_info,
            "selection": selection, **report}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prepared", type=Path, required=True, help="prepared examples holding the reference labels")
    p.add_argument("--predictions", type=Path, required=True, help="model_prediction JSONL from one run")
    p.add_argument("--split", type=Path)
    p.add_argument("--part", choices=("train", "validation"))
    p.add_argument("--note", help="optional extra context; the report's statement is derived from the reference")
    p.add_argument("--title", default="Four-axis evaluation")
    p.add_argument("--out", type=Path, required=True, help="report JSON (must not exist); a .md summary is written beside it")
    a = p.parse_args(argv)
    if (a.split is None) != (a.part is None):
        p.error("--split and --part go together")
    md = a.out.with_suffix(".md")
    if a.out.suffix != ".json" or md.exists() or a.out.exists():
        p.error("--out must be a new .json path whose .md companion does not exist either")
    private = (L.ROOT / "data").resolve()
    if a.prepared.resolve().is_relative_to(private) and not a.out.resolve().is_relative_to(private):
        p.error("a report on private labels must be written below repository data/")
    try:
        report = run(a.prepared, a.predictions, a.note, a.title, a.split, a.part)
        with a.out.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            f.write("\n")
        with md.open("x", encoding="utf-8", newline="\n") as f:
            f.write(markdown(report))
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
