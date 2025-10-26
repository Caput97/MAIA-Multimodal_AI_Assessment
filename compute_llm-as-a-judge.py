#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Task 2 (gen_*): LLM-as-a-judge + per-item accuracy (no aggregate results).

- Scans the root (MAIA_big_model_results) and considers ONLY 'gen_*' folders
- If --skip-judge is not passed, invokes the LLM and saves JSONL files "judged" with field 'llm-as-a-judge'
  in a subfolder 'judged_jsonl' next to the original files, streaming (checkpoint + resume)
- Computes and saves:
    * gen_accuracy_by_category.csv  (per-item; row 'Overall accuracy' on top of each setting)
  Columns: question_category, n, correct, accuracy, setting, task='gen'

OpenAI SDK Compatibility:
- Works with both openai>=1.x (OpenAI client) and openai==0.x (ChatCompletion)
"""


import os
import re
import json
import glob
import argparse
import time
import random
from typing import List, Union, Optional
import pandas as pd

# ---------- SDK compatibility: new (>=1.x) or legacy (0.x) ----------
def make_chat_complete():
    """
    Returns a chat_complete(model, system_prompt, user_prompt) function that is compatible
    with either the new OpenAI SDK (>=1.x) or the legacy one (0.x).
    """
    try:
        # New SDK
        from openai import OpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable missing.")
        client = OpenAI(api_key=api_key)

        def chat_complete(model: str, system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content or ""
            return content.strip()

        return chat_complete

    except Exception:
        # Legacy SDK
        import openai  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable missing.")
        openai.api_key = api_key

        def chat_complete(model: str, system: str, user: str) -> str:
            resp = openai.ChatCompletion.create(
                model=model,
                temperature=0,
                max_tokens=4,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp["choices"][0]["message"]["content"] or ""
            return content.strip()

        return chat_complete


# ---------- IO ----------
def load_jsonlike_file(path: str) -> List[dict]:
    _, ext = os.path.splitext(path.lower())
    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
        else:
            raise ValueError(f"JSON Structure not supported in {path}")
    else:
        raise ValueError(f"Extension not supported for {path}")

def find_files(folder: str, patterns: Union[str, List[str]] = ("*.json", "*.jsonl")) -> List[str]:
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(folder, "**", pat), recursive=True))
    return sorted(files)

# ---------- Helpers ----------
def get_first_present(d: dict, candidates: List[str], default: Optional[str] = "") -> str:
    for c in candidates:
        if c in d and d[c] is not None:
            return d[c]
    return default


def normalize_question_category_str(s: str) -> str:
    s = str(s)
    s = re.sub(r"Causale(?:Implicita|Esplicita)_(A|B)", r"Causale_\1", s)
    s = re.sub(r"_[AB]$", "", s)
    return s


def infer_setting_from_dirname(dirname: str) -> str:
    name = dirname.lower()
    if name.endswith("allvideo_fr32"):
        return "standard"
    if name.endswith("black_fr1"):
        return "black_video"
    if name.endswith("allvideo_fr1"):
        return "one_frame"
    return "unknown"

def to_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "question_category" in df.columns:
        df["question_category"] = df["question_category"].astype(str).map(normalize_question_category_str)
    else:
        raise KeyError(" 'question_category' is missing within the records.")

    if "llm-as-a-judge" not in df.columns:
        raise KeyError(" 'llm-as-a-judge' is missing (you need --judge or file already judged).")
    judge = df["llm-as-a-judge"].astype(str).str.lower().str.strip()
    df["correct_bool"] = judge.eq("correct")
    return df

# ---------- Accuracy (per-item) ----------
def per_item_table(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("question_category", dropna=False)["correct_bool"]
    out = grp.agg(n="size", correct="sum").reset_index()
    out["accuracy"] = out["correct"] / out["n"]
    return out.sort_values("question_category").reset_index(drop=True)

# ---------- LLM-as-a-judge with retry ----------
def chat_complete_with_retry(chat_complete, model: str, system: str, user: str,
                             max_retries: int = 8, base_sleep: float = 1.0) -> str:
    for attempt in range(max_retries):
        try:
            return chat_complete(model, system, user)
        except Exception as e:
            sleep_s = min(base_sleep * (2 ** attempt), 30.0) + random.random() * 0.25
            print(f"[retry {attempt+1}/{max_retries}] {type(e).__name__}: {e} -> sleep {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)
    return chat_complete(model, system, user)

def llm_judge(chat_complete, model: str, question: str, candidate: str, refs: List[str]) -> str:
    refs = (refs + [""]*8)[:8]
    system = (
        "You are an assistant that evaluates the correctness of a candidate answer "
        "based on a set of 8 reference answers. "
        "A is considered correct if it aligns with at least one reference answer. "
        "Return exactly one label: Correct or Incorrect."
    )
    user = (
        f"Question: {question}\n"
        f"Candidate Answer: {candidate}\n"
        + "\n".join([f"R{i+1}: {r}" for i, r in enumerate(refs)])
    )
    out = chat_complete_with_retry(chat_complete, model, system, user)
    out = (out or "").strip()
    out = out.split()[0] if out else ""
    return "Correct" if out.lower().startswith("correct") else "Incorrect"

# ---------- Streaming + checkpoint ----------
def count_valid_json_lines(path: str) -> int:
    """Counts only JSON-parseable lines (skips a possible truncated last line)."""
    if not os.path.exists(path):
        return 0
    ok = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            if not line_strip:
                continue
            try:
                json.loads(line_strip)
                ok += 1
            except Exception:
                break
    return ok

def judge_records_streaming(
    records: List[dict],
    chat_complete,
    model: str,
    judged_path: str,
    overwrite: bool = False,
    log_every: int = 25,
    file_label: str = ""
) -> int:
    """
    It writes to judged_path in streaming mode: resumes from where it left off.
    Returns the total number of records in the resulting file.
    """
    os.makedirs(os.path.dirname(judged_path), exist_ok=True)

    start = 0
    if not overwrite:
        start = count_valid_json_lines(judged_path)
        mode = "a"
    else:
        mode = "w"

    total = len(records)
    if start > total:
        start = 0
        mode = "w"

    if start:
        print(f"[{file_label}] resume from {start}/{total} (checkpoint trovato).", flush=True)

    with open(judged_path, mode, encoding="utf-8") as f:
        for i in range(start, total):
            r = records[i]
            if (not overwrite) and ("llm-as-a-judge" in r) and str(r["llm-as-a-judge"]).strip():
                r2 = r
            else:
                q = get_first_present(r, ["question", "Question"], "")
                candidate = get_first_present(r, ["filtered_resps", "generation", "response", "model_generation"], "")
                refs = [get_first_present(r, [f"answer{i}", f"Answer{i}"], "") for i in range(1, 9)]
                label = llm_judge(chat_complete, model, q, candidate, refs)
                r2 = dict(r); r2["llm-as-a-judge"] = label

            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
            f.flush()

            done = i + 1
            if log_every and (done % log_every == 0):
                print(f"[{file_label}] judged {done}/{total}...", flush=True)

    if log_every and (total % log_every != 0):
        print(f"[{file_label}] judged {total}/{total}.", flush=True)

    return total

# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Task2 (gen_*): LLM-as-a-judge + accuracy (per-item, no aggregation).")
    parser.add_argument("jsonl_path", type=str, help="Path to the results.jsonl file produced by eval_model.py")
    parser.add_argument("--outdir", type=str, default=None, help="Folder for CSV outputs. Default: input folder")
    parser.add_argument("--judge-model", default="gpt-4o-mini-2024-07-18", help="Model used for LLM-as-a-judge")
    parser.add_argument("--skip-judge", action="store_true", help="Do not call the API; use only pre-judged JSON/JSONL files")
    parser.add_argument("--overwrite-judged", action="store_true", help="Always regenerate 'llm-as-a-judge' even if already present")
    parser.add_argument("--log-every", type=int, default=25, help="Log every N judged examples (default: 25)")
    args = parser.parse_args()

    # Source
    src_path = args.jsonl_path
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"File not found: {src_path}")

    # Output dir (default: input file folder)
    outdir = args.outdir or os.path.dirname(src_path)
    os.makedirs(outdir, exist_ok=True)
    out_csv = os.path.join(outdir, "gen_accuracy_by_category.csv")

    # folder for judged files next to the original one
    judged_dir = os.path.join(os.path.dirname(src_path), "judged_jsonl")
    os.makedirs(judged_dir, exist_ok=True)
    judged_path = os.path.join(judged_dir, os.path.splitext(os.path.basename(src_path))[0] + ".jsonl")

    # prepare chat function
    chat_complete = None
    if not args.skip_judge:
        chat_complete = make_chat_complete()

    # load records
    recs = load_jsonlike_file(src_path)
    file_label = os.path.basename(src_path)

    # judge/fill column
    total = judge_records_streaming(
        records=recs,
        chat_complete=chat_complete,
        model=args.judge_model,
        judged_path=judged_path,
        overwrite=args.overwrite_judged,
        log_every=args.log_every,
        file_label=file_label
    )

    # reload from judged_path (or use recs) and build dataframe
    # Note: to_dataframe expects standard fields, and ensuring the 'llm-as-a-judge' column is handled upstream
    # We choose to load directly from judged_path when available
    src_for_df = judged_path if os.path.exists(judged_path) else src_path
    final_records = load_jsonlike_file(src_for_df)
    df = to_dataframe(final_records)

    # infer setting from the name of the folder containing the input file (e.g., gen_abcd_*)
    parent_dirname = os.path.basename(os.path.dirname(src_path))
    setting = infer_setting_from_dirname(parent_dirname)

    # per-item
    t_items = per_item_table(df)
    n = int(t_items["n"].sum())
    c = int(t_items["correct"].sum())
    overall = c / n if n > 0 else float("nan")
    head_items = pd.DataFrame([{
        "question_category": "Overall accuracy",
        "n": n, "correct": c, "accuracy": overall, "setting": setting, "task": "gen"
    }])
    t_items["setting"] = setting
    t_items["task"] = "gen"
    final = pd.concat([head_items, t_items], ignore_index=True)
    final.to_csv(out_csv, index=False)

    print(f"Per-item CSV saved  -> {out_csv}")
    print(f"Judged items saved in -> {judged_path}")


if __name__ == "__main__":
    main()
