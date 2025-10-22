import json
import os
import tempfile
import subprocess
import argparse
from evaluate import load
from pycocoevalcap.cider.cider import CiderScorer

def tokenize(predictions, references):
    PUNCTUATIONS = [
        "''", "'", "``", "`", "-LRB-", "-RRB-", "-LCB-", "-RCB-", ".", "?", "!", ",", ":", "-", "--", "...", ";"
    ]

    cmd = [
        "java", "-cp", "/leonardo_scratch/fast/FBKLM_prj1/PROJECTS/lmms-eval-giovanni/evaluation_maia/stanford-corenlp-3.4.1.jar",
        "edu.stanford.nlp.process.PTBTokenizer", "-preserveLines", "-lowerCase"
    ]

    sentences = "\n".join([
        s.replace("\n", " ") for s in predictions + [ref for refs in references for ref in refs]
    ])

    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(sentences.encode())
        temp_file_name = f.name

    cmd.append(temp_file_name)
    p_tokenizer = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    token_lines = p_tokenizer.communicate(input=sentences.rstrip())
    token_lines = token_lines[0].decode()
    os.remove(temp_file_name)

    lines = [
        " ".join([w for w in line.rstrip().split(" ") if w not in PUNCTUATIONS])
        for line in token_lines.split("\n")
    ]

    pred_size = len(predictions)
    ref_sizes = [len(ref) for ref in references]

    predictions = lines[:pred_size]
    start = pred_size
    new_references = []
    for size in ref_sizes:
        new_references.append(lines[start:start + size])
        start += size

    return predictions, new_references

def load_log_data(jsonl_path):
    """
    Supporta due formati:
    1) NUOVO (piatto), ad es.:
       {"id":0, "video_id":"Video1", ..., "answer1":"...", ..., "answer8":"...", "model_generation":"..."}
    2) VECCHIO (annidato) con entry["doc"]["generation"]["model_generation"][0] e answer1..answer8 in entry["doc"].
    """
    references = []
    predictions = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)

            if isinstance(entry.get("doc"), dict):
                # Compatibilità con il formato precedente
                model_generation = entry["doc"]["generation"]["model_generation"][0]
                possible_references = [entry["doc"][f"answer{i}"] for i in range(1, 9)]
            else:
                # NUOVO formato piatto (come nel tuo esempio)
                model_generation = entry["model_generation"]
                possible_references = [entry[f"answer{i}"] for i in range(1, 9)]

            predictions.append(model_generation)
            references.append(possible_references)

    return references, predictions

def compute_rouge(references, predictions):
    print("computing rouge")
    rouge_scorer = load("rouge", keep_in_memory=True)
    rouge_scores = rouge_scorer.compute(predictions=predictions, references=references)["rouge1"]
    return rouge_scores

def compute_bertscore(references, predictions):
    print("computing BERTScore")
    bertscore_scorer = load("bertscore")
    bertscore_results = bertscore_scorer.compute(predictions=predictions, references=references, lang="it")["f1"]
    return sum(bertscore_results) / len(predictions)

def compute_bleu(references, predictions):
    print("computing bleu")
    bleu_scorer = load("bleu")
    bleu_scores = bleu_scorer.compute(predictions=predictions, references=references)['bleu']
    return bleu_scores

def compute_meteor(references, predictions):
    print("computing meteor")
    meteor_scorer = load('meteor', keep_in_memory=True)
    meteor_score = meteor_scorer.compute(predictions=predictions, references=references)['meteor']
    return meteor_score

def compute_cider(references, predictions, n=4, sigma=6.0):
    print("computing CIDEr")
    predictions_tok, references_tok = tokenize(predictions, references)

    scorer = CiderScorer(n=n, sigma=sigma)
    for pred, refs in zip(predictions_tok, references_tok):
        scorer += (pred, refs)
    score, _ = scorer.compute_score()

    return score

def compute_metrics(jsonl_path):
    references, predictions = load_log_data(jsonl_path)

    print(f"references are {len(references)} and predictions are {len(predictions)}")
    print(f"reference examples {references[0]} and prediction {predictions[0]}")

    return {
        "ROUGE-1": compute_rouge(references, predictions),
        "BERTScore": compute_bertscore(references, predictions),
        "BLEU": compute_bleu(references, predictions),
        "CIDEr": compute_cider(references, predictions),
        "METEOR": compute_meteor(references, predictions)
    }

def main():
    parser = argparse.ArgumentParser(description="Compute metrics for model-generated text.")
    parser.add_argument("jsonl_path", type=str, help="Path to the input JSONL file.")

    args = parser.parse_args()
    print(f"computing scores for {args.jsonl_path}")

    metrics = compute_metrics(args.jsonl_path)
    print(metrics)

    # Salva i risultati
    results_filename = os.path.splitext(args.jsonl_path)[0] + "_gen_metrics.txt"
    with open(results_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"Results saved to {results_filename}")

if __name__ == "__main__":
    main()
