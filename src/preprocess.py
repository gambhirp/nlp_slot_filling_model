import os
import json
import spacy
from spacy.tokens import DocBin

def convert_annotations_to_spacy(data_dir: str, output_path: str):
    """
    Scans data_dir for paired .txt/.json files, reads the JSON "labels"
    (with start/end offsets and slot names), and writes a DocBin to output_path.
    """
    nlp = spacy.blank("en")
    doc_bin = DocBin()

    for fn in os.listdir(data_dir):
        if not fn.endswith(".json"):
            continue
        base = fn[:-5]  # strip ".json"
        txt_path = os.path.join(data_dir, base + ".txt")
        json_path = os.path.join(data_dir, fn)

        # Load transcript text
        with open(txt_path, encoding="utf-8") as f:
            text = f.read()

        # Load annotations
        ann = json.load(open(json_path, encoding="utf-8"))
        ents = []
        for label in ann.get("labels", []):
            start, end, slot = label["start"], label["end"], label["slot"]
            span = nlp.make_doc(text).char_span(start, end, label=slot)
            if span is not None:
                ents.append((start, end, slot))

        # Build Doc with entities
        doc = nlp(text)
        doc.ents = [doc.char_span(s, e, label=l) for s, e, l in ents]
        doc_bin.add(doc)

    # Write the .spacy file
    doc_bin.to_disk(output_path)
    print(f"✅ Wrote {output_path}")
