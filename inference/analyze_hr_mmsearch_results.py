#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any


ANALYSIS_COLUMNS = [
    "sample_id",
    "correct",
    "question_type",
    "question",
    "ground_truth",
    "final_answer",
    "num_round",
    "finish_reason",
    "tools_used",
    "tool_sequence",
    "text_queries",
    "zoom_bboxes",
    "image_search_titles",
    "judge_decision",
    "judge_method",
    "judge_reason",
    "image_path",
    "raw_category",
    "raw_subcategory",
    "error_type_manual",
    "root_cause_manual",
    "data_improvement_suggestion",
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def is_correct(row: dict) -> bool:
    return bool(row.get("llm_score", 0) > 0.5 or row.get("em_score_mcq", 0) > 0.5)


def classify_question(question: str) -> str:
    """Coarse heuristic type for quick slicing before manual annotation."""
    q = question.lower()
    if any(x in q for x in ["how many", "number of", "count", "total"]):
        return "counting"
    if any(x in q for x in ["where", "location", "place", "city", "country"]):
        return "location"
    if any(x in q for x in ["when", "year", "date", "time"]):
        return "temporal"
    if any(x in q for x in ["who", "person", "people", "name of"]):
        return "entity_person"
    if any(x in q for x in ["what is the text", "read", "written", "sign", "label"]):
        return "ocr_text"
    if any(x in q for x in ["brand", "logo", "company", "product"]):
        return "brand_product"
    if any(x in q for x in ["same as", "similar", "identify", "object", "species", "landmark"]):
        return "visual_identification"
    if any(x in q for x in ["why", "reason", "cause"]):
        return "causal_reasoning"
    if any(x in q for x in ["compare", "difference", "larger", "smaller", "more than", "less than"]):
        return "comparison"
    return "other"


def extract_judge_reason(score_details: dict) -> tuple[str, str, str]:
    llm = score_details.get("llm_score", {}) if isinstance(score_details, dict) else {}
    decision = llm.get("decision", "")
    method = llm.get("method", "")
    attempts = llm.get("attempts", []) or []
    raw = attempts[-1].get("raw_response", "") if attempts else ""
    match = re.search(r"<reason>\s*(.*?)\s*</reason>", raw, re.DOTALL | re.IGNORECASE)
    reason = match.group(1).strip() if match else raw.strip()
    return decision, method, reason


def summarize_tools(tool_calls: list[dict]) -> dict:
    names = [c.get("name", "") for c in tool_calls]
    text_queries = [c.get("query", "") for c in tool_calls if c.get("name") == "text_search_tool"]
    zoom_bboxes = [c.get("bbox") for c in tool_calls if c.get("name") == "image_zoom_in_tool"]
    image_titles = []
    for c in tool_calls:
        if c.get("name") == "image_search_tool":
            image_titles.extend(c.get("titles", []) or [])
    return {
        "tools_used": ",".join(sorted(set(n for n in names if n))),
        "tool_sequence": " > ".join(n for n in names if n),
        "text_queries": " | ".join(text_queries),
        "zoom_bboxes": safe_str(zoom_bboxes),
        "image_search_titles": " | ".join(image_titles),
    }


def get_raw_label(metadata: dict, candidates: list[str]) -> str:
    if not isinstance(metadata, dict):
        return ""
    lower_map = {k.lower(): k for k in metadata.keys()}
    for cand in candidates:
        key = lower_map.get(cand.lower())
        if key:
            return safe_str(metadata.get(key))
    for k, v in metadata.items():
        lk = k.lower()
        if any(c.lower() in lk for c in candidates):
            return safe_str(v)
    return ""


def make_case_row(row: dict) -> dict:
    metadata = row.get("metadata", {}) or {}
    tool_info = summarize_tools(row.get("tool_calls", []) or [])
    decision, method, reason = extract_judge_reason(row.get("score_details", {}) or {})
    question = row.get("question", "")
    return {
        "sample_id": row.get("sample_id", ""),
        "correct": int(is_correct(row)),
        "question_type": classify_question(question),
        "question": question,
        "ground_truth": safe_str(row.get("gts", "")),
        "final_answer": row.get("final_answer", ""),
        "num_round": row.get("num_round", ""),
        "finish_reason": row.get("finish_reason", ""),
        "tools_used": tool_info["tools_used"],
        "tool_sequence": tool_info["tool_sequence"],
        "text_queries": tool_info["text_queries"],
        "zoom_bboxes": tool_info["zoom_bboxes"],
        "image_search_titles": tool_info["image_search_titles"],
        "judge_decision": decision,
        "judge_method": method,
        "judge_reason": reason,
        "image_path": row.get("image_path", ""),
        "raw_category": get_raw_label(metadata, ["category", "question_category", "type", "question_type"]),
        "raw_subcategory": get_raw_label(metadata, ["subcategory", "sub_category", "domain", "source"]),
        "error_type_manual": "",
        "root_cause_manual": "",
        "data_improvement_suggestion": "",
    }


def rate(counter: Counter, correct_counter: Counter) -> dict:
    out = {}
    for key, total in sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        correct = correct_counter.get(key, 0)
        out[str(key)] = {
            "total": total,
            "correct": correct,
            "wrong": total - correct,
            "accuracy": correct / total if total else 0.0,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze HR-MMSearch eval results for bad-case mining.")
    parser.add_argument("results_jsonl", help="Path to results.jsonl")
    parser.add_argument("--output-dir", default="", help="Directory for analysis outputs; defaults to results dir/analysis")
    args = parser.parse_args()

    rows = load_jsonl(args.results_jsonl)
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.results_jsonl), "analysis")
    os.makedirs(output_dir, exist_ok=True)

    case_rows = [make_case_row(r) for r in rows]
    wrong_rows = [r for r in case_rows if not int(r["correct"])]

    total = len(rows)
    correct = sum(1 for r in rows if is_correct(r))

    by_qtype = Counter()
    by_qtype_correct = Counter()
    by_toolset = Counter()
    by_toolset_correct = Counter()
    by_finish = Counter()
    by_finish_correct = Counter()

    tool_single_total = Counter()
    tool_single_correct = Counter()
    rounds_wrong = []

    for raw, case in zip(rows, case_rows):
        ok = bool(int(case["correct"]))
        qtype = case["question_type"]
        toolset = case["tools_used"] or "no_tool"
        finish = case["finish_reason"] or "unknown"

        by_qtype[qtype] += 1
        by_toolset[toolset] += 1
        by_finish[finish] += 1
        if ok:
            by_qtype_correct[qtype] += 1
            by_toolset_correct[toolset] += 1
            by_finish_correct[finish] += 1
        else:
            rounds_wrong.append(raw.get("num_round", 0))

        used_names = {c.get("name", "") for c in raw.get("tool_calls", []) or []}
        if not used_names:
            used_names = {"no_tool"}
        for name in used_names:
            tool_single_total[name] += 1
            if ok:
                tool_single_correct[name] += 1

    summary = {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": correct / total if total else 0.0,
        "by_question_type": rate(by_qtype, by_qtype_correct),
        "by_toolset": rate(by_toolset, by_toolset_correct),
        "by_individual_tool": rate(tool_single_total, tool_single_correct),
        "by_finish_reason": rate(by_finish, by_finish_correct),
        "wrong_rounds": {
            "count": len(rounds_wrong),
            "avg": sum(rounds_wrong) / len(rounds_wrong) if rounds_wrong else 0.0,
            "max": max(rounds_wrong) if rounds_wrong else 0,
        },
    }

    with open(os.path.join(output_dir, "analysis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "bad_cases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(wrong_rows)

    with open(os.path.join(output_dir, "all_cases.csv"), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANALYSIS_COLUMNS)
        writer.writeheader()
        writer.writerows(case_rows)

    with open(os.path.join(output_dir, "error_cases.jsonl"), "w", encoding="utf-8") as f:
        for raw in rows:
            if not is_correct(raw):
                f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    print(f"Analyzed {total} cases: {correct} correct, {total - correct} wrong")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
