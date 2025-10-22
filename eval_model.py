#!/usr/bin/env python
"""
Main entry point to evaluate a VLM on the
`giobin/MAIA` dataset (config `gen`, split `train`).

Usage example:

  python eval_model.py \
    --model OpenGVLab/InternVL3-8B \
    --videos-dir ./Videos \
    --out-json maia_gen_internvl3_answers.json \
    --num-frames 12 --batch-size 4 --max-new-tokens 128 --limit 50

Dependencies:
  pip install -U vllm transformers datasets pillow decord tqdm
"""

from __future__ import annotations

import argparse
import json
import os
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from vllm_engine import VllmModel

try:
    from decord import VideoReader, cpu
except Exception as e:
    raise RuntimeError(
        "decord is required for video frame extraction. Install with `pip install decord`."
    ) from e

def format_prompt(prompt_template: str, injecting_elements: List[str]) -> str:
    """
    If the injecting_elements is a list with one element we assume it to be the question and espect prompt_template to have one {} to be replaced
    If the injecting_elements is a list with two elements we assume them to be the choices A and B and expect prompt_template to have two {} to be replaced
    """
    try:
        return prompt_template.format(*injecting_elements)
    except Exception as e:
        raise ValueError(f"Error formatting prompt with template: {prompt_template} and elements: {injecting_elements}") from e
    
# ---------------------------
# Helpers
# ---------------------------

def chunks(seq: Sequence[Any], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _resize_max_dim(img: Image.Image, max_dim: int) -> Image.Image:
    """Resize keeping aspect ratio so that max(width, height) <= max_dim."""
    w, h = img.size
    scale = min(1.0, float(max_dim) / float(max(w, h)))
    if scale == 1.0:
        return img
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), Image.BICUBIC)


def _sample_video_frames(
    video_path: str,
    num_frames: int,
    max_image_dimension: int,
) -> List[Image.Image]:
    """
    Uniformly sample `num_frames` frames from the video and return as PIL Images,
    resized to fit within `max_image_dimension`.
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    n = len(vr)
    if n == 0:
        raise ValueError(f"Video has 0 frames: {video_path}")

    k = max(1, min(num_frames, n))
    indices = np.linspace(0, n - 1, num=k, dtype=np.int64)

    frames = vr.get_batch(indices).asnumpy()  # shape: (k, H, W, 3), uint8
    pil_images: List[Image.Image] = []
    for arr in frames:
        img = Image.fromarray(arr, mode="RGB")
        img = _resize_max_dim(img, max_image_dimension)
        pil_images.append(img)
    return pil_images


def _prepare_rows_for_batch(
    batch_rows: List[Dict[str, Any]],
    processed_videos: Dict[str, Any],
    prompt_template: str,
) -> Tuple[List[str], List[List[Image.Image]], List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """
    For a batch of dataset rows, build:
      - prompts: List[str]
      - images:  List[List[PIL.Image]]
      - ok_rows: rows that produced a prompt+images
      - errors:  list of (row, error_msg)
    """
    prompts: List[str] = []
    images: List[List[Image.Image]] = []
    ok_rows: List[Dict[str, Any]] = []

    for row in batch_rows:
        frames = processed_videos[row["video_id"]] if row["video_id"] in processed_videos else processed_videos['single_video_for_all_rows']
        injecting_elements = []
        if "question" not in row:
            #mc case
            injecting_elements.append(row['answer1'])
            injecting_elements.append(row['answer2'])
        else:
            #gen case
            injecting_elements.append(row["question"])
        prompt = format_prompt(prompt_template, injecting_elements) # this automatically handles both gen and mc cases
        prompts.append(prompt)
        images.append(frames)
        ok_rows.append(row)

    return prompts, images, ok_rows


# ---------------------------
# Main
# ---------------------------

PROMPT_GEN = "Answer with a single sentence the following question related to the video in Italian:\n{}"
PROMPT_MC = "Scegli la descrizione corretta:\nA. {}\nB. {}\nRispondi solo A o B\n"

def run(
    model_id: str,
    videos_dir: str,
    out_dir_path: str,
    *,
    dataset_name: str = "giobin/MAIA",
    dataset_config: str = "gen",
    dataset_split: str = "train",
    batch_size: int = 4,
    num_frames: int = 12,
    max_image_dimension: int = 900,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 128,
    gpu_mem_util: float = 0.95,
    max_model_len: int | None = None,  # forwarded but may be ignored by your utility
    limit: int | None = None,
    enforce_eager: bool = False,
) -> None:

    # 1) Load dataset
    ds = load_dataset(dataset_name, dataset_config, split=dataset_split)
    rows: List[Dict[str, Any]] = [ds[i] for i in range(len(ds))]
    if limit is not None:
        rows = rows[: int(limit)]
    
    os.makedirs(out_dir_path, exist_ok=True)
    jsonl_path = os.path.join(out_dir_path, "results.jsonl")
    
    if os.path.exists(jsonl_path):
        print(f"Found existing {jsonl_path} containing previous results. Loading to skip already processed rows.")
        existing_results_ids = set()
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                existing_results_ids.add(item["id"])
        # filter out rows that are already in existing_results_ids
        print(f"Skipping {len(existing_results_ids)} already processed rows.")
        rows = [row for row in rows if row["id"] not in existing_results_ids]

    engine_kwargs = {}
    if max_model_len is not None:
        # Your VllmModel may ignore this unless you wire it through internally,
        # but passing it is harmless.
        engine_kwargs["max_model_len"] = max_model_len

    vm = VllmModel(
        model_id=model_id,
        gpu_memory_utilization=gpu_mem_util,
        verbose=False,
        enforce_eager=enforce_eager,
        **engine_kwargs
    )

    # 3) Iterate in batches
    with open(jsonl_path, "a", encoding="utf-8") as f:
        # Preprocess video_id -> frames to avoid reloading the same video multiple times
        processed_videos = dict()
        # if video_dir is a directory then process all .mp4 files in it
        if os.path.isdir(videos_dir):
            available_videos = {f[:-4] for f in os.listdir(videos_dir) if f.endswith(".mp4")}
            print(f"preprocessing videos to sample {num_frames} frames each...")
            for row in tqdm(rows):
                if row["video_id"] not in available_videos:
                    raise ValueError(f"Video file for video_id {row['video_id']} not found in {videos_dir}")
                if row["video_id"] not in processed_videos:
                    processed_videos[row["video_id"]] = _sample_video_frames(
                        os.path.join(videos_dir, f"{row['video_id']}.mp4"),
                        num_frames=num_frames,
                        max_image_dimension=max_image_dimension,
                    )
        else:
            processed_videos['single_video_for_all_rows'] = _sample_video_frames(
                videos_dir,
                num_frames=num_frames,
                max_image_dimension=max_image_dimension,
            )

        for ix, batch_rows in tqdm(enumerate(list(chunks(rows, batch_size))), desc="Batches"):
            # Build (prompts, images) for vLLM and capture per-row errors
            prompts, image_batches, ok_rows = _prepare_rows_for_batch(
                batch_rows,
                processed_videos=processed_videos,
                prompt_template=PROMPT_GEN if dataset_config == "gen" else PROMPT_MC,
            )

            if not prompts:
                continue

            # 4) Generate — prompts are strings; images = list[list[PIL.Image]]
            texts: List[str] = vm.generate_continuation(
                prompts,
                images=image_batches,
                max_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                iteration=ix,
                experiment_dir=out_dir_path,
            )

            # Attach answers
            for row, text in zip(ok_rows, texts):
                tmp = dict(row)
                tmp["model_generation"] = text
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
                f.flush()
                #os.fsync(f.fileno())


    if torch.distributed.is_available() and torch.distributed.is_initialized():
        import torch.distributed as dist
        dist.destroy_process_group()
    
    return


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate InternVL3-8B on MAIA gen split (vLLM)")
    p.add_argument("--model", default="OpenGVLab/InternVL3-8B", help="HF model id")
    p.add_argument("--videos-dir", default="./Videos", help="Directory containing videos")
    p.add_argument("--out-dir", type=str, required=True, help="Directory to save results")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-frames", type=int, default=12, help="Frames sampled per video")
    p.add_argument("--max-image-dimension", type=int, default=900, help="Max width/height of decoded frames")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--gpu-mem-util", type=float, default=0.95)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N rows (debug)")
    p.add_argument("--enforce-eager", action="store_true", help="Enforce eager mode in vLLM (may be slower)")
    p.add_argument("--dataset-config", type=str, default="gen", choices=["gen", "mc"], help="Dataset config: 'gen' for open-ended, 'mc' for multiple-choice")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for k,v in vars(args).items():
        print(f"{k}: {v}")
    run(
        model_id=args.model,
        videos_dir=args.videos_dir,
        out_dir_path=args.out_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        gpu_mem_util=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        limit=args.limit,
        max_image_dimension=args.max_image_dimension,
        enforce_eager=args.enforce_eager,
        dataset_config=args.dataset_config,
    )

# Prompts gen:
# "Answer with a single sentence the following question related to the video in Italian:\n'{{question}}'"
# "You need to perform a Visual Question Answering task. Answer with a single sentence the following question in Italian:\n'{{question}}'"

# Prompts mc:
# "Scegli la descrizione corretta:\nA. {}\nB. {}\nRispondi solo A o B\n"