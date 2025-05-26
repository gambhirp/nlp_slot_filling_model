import plac
import random
import spacy
from spacy.util import minibatch, compounding
from spacy.training.example import Example

@plac.annotations(
    train_path=("Path to train.spacy", "option", "T", str),
    dev_path=("Path to val.spacy",   "option", "V", str),
    output_dir=("Directory to save best model", "option", "O", str),
    n_iter=("Number of training epochs",   "option", "N", int)
)
def main(train_path="data/train.spacy",
         dev_path="data/val.spacy",
         output_dir="models/best_model",
         n_iter=20):

    # 1. Load empty English pipeline and add NER
    nlp = spacy.blank("en")
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # 2. Read DocBins
    train_db = spacy.tokens.DocBin().from_disk(train_path)
    dev_db   = spacy.tokens.DocBin().from_disk(dev_path)
    train_docs = list(train_db.get_docs(nlp.vocab))
    dev_docs   = list(dev_db.get_docs(nlp.vocab))

    # 3. Add labels to NER
    for doc in train_docs:
        for ent in doc.ents:
            ner.add_label(ent.label_)

    # 4. Disable other pipes during training
    pipe_exceptions = ["ner"]
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

    with nlp.disable_pipes(*unaffected_pipes):
        optimizer = nlp.begin_training()
        print(f"Training for {n_iter} epochs…")
        best_f1 = 0.0

        for epoch in range(n_iter):
            random.shuffle(train_docs)
            losses = {}
            batches = minibatch(train_docs, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                examples = [Example.from_dict(doc, {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}) for doc in batch]
                nlp.update(examples, sgd=optimizer, drop=0.2, losses=losses)
            print(f"Epoch {epoch+1}/{n_iter} — Loss: {losses['ner']:.3f}")

            # 5. Evaluate on dev set
            examples_dev = [
                Example.from_dict(
                    doc,
                    {"entities": [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]}
                )
                for doc in dev_docs
            ]
            scores = nlp.evaluate(examples_dev)
            # ents_f, ents_p, and ents_r may be None if no entities were evaluated—default to 0.0
            precision = scores.get("ents_p", 0.0) or 0.0
            recall = scores.get("ents_r", 0.0) or 0.0
            f1 = scores.get("ents_f", 0.0) or 0.0
            print(f"Validation Precision: {precision:.3f}, "
                  f"Recall: {recall:.3f}, "
                  f"F1: {f1:.3f}")

            # 6. Save best model
            if f1 > best_f1:
                best_f1 = f1
                nlp.to_disk(output_dir)
                print(f"✔ Saved new best model (F1 {best_f1:.3f}) to {output_dir}")

    print("Training complete.")

if __name__ == "__main__":
    plac.call(main)
