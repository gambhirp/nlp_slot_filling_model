# src/generate_skeletons.py

import os
import json
import argparse

def generate_skeletons(transcript_dir: str, output_dir: str):
    """
    For every .txt in transcript_dir, write a JSON:
      {
        "transcript_id": "<basename>",
        "text": "<full transcript…>",
        "labels": []
      }
    into output_dir alongside the .txt files.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fn in os.listdir(transcript_dir):
        if not fn.lower().endswith(".txt"):
            continue
        transcript_id = fn[:-4]  # strip “.txt”
        txt_path = os.path.join(transcript_dir, fn)
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()
        skeleton = {
            "transcript_id": transcript_id,
            "text": text,
            "labels": []
        }
        out_path = os.path.join(output_dir, f"{transcript_id}.json")
        with open(out_path, "w", encoding="utf-8") as outf:
            json.dump(skeleton, outf, indent=2, ensure_ascii=False)
        print(f"Written: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate empty-‘labels’ JSONs for annotation"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Folder containing transcript_*.txt files"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Where to write transcript_*.json skeletons"
    )
    args = parser.parse_args()
    generate_skeletons(args.input_dir, args.output_dir)
