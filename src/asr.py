import os
import argparse
import whisper  # or 'from google.cloud import speech' etc.

def transcribe_file(input_path, output_dir, model_name="base"):
    os.makedirs(output_dir, exist_ok=True)
    model = whisper.load_model(model_name)
    result = model.transcribe(input_path)
    txt_name = os.path.splitext(os.path.basename(input_path))[0] + ".txt"
    out_path = os.path.join(output_dir, txt_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    return out_path

def batch_transcribe(input_dir, output_dir, model_name):
    for fn in os.listdir(input_dir):
        if not fn.lower().endswith((".mp3", ".mpeg", ".wav")):
            continue
        in_path = os.path.join(input_dir, fn)
        try:
            out = transcribe_file(in_path, output_dir, model_name)
            print(f"Wrote: {out}")
        except Exception as e:
            print(f"Error on {fn}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  required=True, help="Where your raw audio lives")
    parser.add_argument("--output_dir", required=True, help="Where to save .txt transcripts")
    parser.add_argument("--model",      default="base", help="Whisper model size (tiny, base, etc.)")
    args = parser.parse_args()
    batch_transcribe(args.input_dir, args.output_dir, args.model)
