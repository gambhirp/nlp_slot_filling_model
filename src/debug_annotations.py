# src/debug_annotations.py

import os
import json
import spacy

def count_entities(data_dir):
    nlp = spacy.blank("en")
    total_docs = 0
    total_ents = 0
    for fn in os.listdir(data_dir):
        if not fn.endswith(".json"):
            continue
        total_docs += 1
        obj = json.load(open(os.path.join(data_dir, fn), encoding="utf-8"))
        text = obj["text"]
        ents = obj.get("labels", [])
        print(f"{fn}: {len(ents)} annotated entities")
        total_ents += len(ents)
    print(f"→ {total_docs} documents, {total_ents} total entities")

if __name__ == "__main__":
    print("TRAIN:")
    count_entities("data/train")
    print("\nVALIDATION:")
    count_entities("data/val")
    print("\nTEST:")
    count_entities("data/test")
