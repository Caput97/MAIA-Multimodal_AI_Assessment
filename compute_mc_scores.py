#!/usr/bin/env python3
import json, argparse, re
from collections import defaultdict

AB_REGEX = re.compile(r"\b([A-B])\b")

def main():
    ap = argparse.ArgumentParser(description="Accuracy totale e per categoria (A/B) da JSONL.")
    ap.add_argument("jsonl_path")
    args = ap.parse_args()

    total = correct = out_of_range = 0
    by_cat = defaultdict(lambda: [0, 0])  # [correct, total]

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # linea malformata: ignora

            correct_opt = "A" if obj["target"] == 0 else "B"
            m = AB_REGEX.search(str(obj.get("model_generation", "")))
            chosen = m.group(1) if m else None

            cat = obj["question_category"].split("_", 1)[0]

            total += 1
            by_cat[cat][1] += 1
            if chosen is None:
                out_of_range += 1
            if chosen == correct_opt:
                correct += 1
                by_cat[cat][0] += 1

    print(f"Accuracy totale: {correct/total:.4f} ({correct}/{total})")
    print(f"Risposte out-of-range (non A/B): {out_of_range}")
    print("Accuracy per categoria:")
    for cat in sorted(by_cat):
        c, n = by_cat[cat]
        print(f"  - {cat}: {c/n:.4f} ({c}/{n})")

if __name__ == "__main__":
    main()
