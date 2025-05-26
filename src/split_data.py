import os
import glob
import random
import shutil

def split_transcripts(input_dir="data/transcripts",
                      output_base="data",
                      train_frac=0.8,
                      val_frac=0.1,
                      seed=42):
    # 1. Gather all transcript paths
    files = glob.glob(os.path.join(input_dir, "*.txt"))
    files.sort()
    random.seed(seed)
    random.shuffle(files)

    n = len(files)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    # Remaining goes to test
    n_test  = n - n_train - n_val

    splits = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    # 2. Create output dirs
    for split in splits:
        out_dir = os.path.join(output_base, split)
        os.makedirs(out_dir, exist_ok=True)

    # 3. Copy files
    for split, paths in splits.items():
        out_dir = os.path.join(output_base, split)
        for src_path in paths:
            fname = os.path.basename(src_path)
            dst_path = os.path.join(out_dir, fname)
            shutil.copy(src_path, dst_path)
            print(f"[{split}] copied {fname}")

if __name__ == "__main__":
    split_transcripts()
