"""
create_temporal_split.py
------------------------
Creates a clean train/validation split from the existing MINDsmall_train data,
since the canonical MINDsmall_dev file is unavailable (Microsoft pulled public
access to the Azure blob in 2024; public mirrors contain train data mislabeled
as dev).

Strategy: temporal 75/25 split. MIND impressions are already time-ordered
(Nov 9–14 in the train file). We take the first 75% as training data and the
last 25% as held-out validation data. This matches MIND's original "train on
earlier, evaluate on later" design philosophy.

Usage (from project root):
    python create_temporal_split.py

This overwrites data/MINDsmall_train/behaviors.tsv and
data/MINDsmall_dev/behaviors.tsv. news.tsv and the .vec files are left alone —
they're identical in both splits and serve the union of news articles.

A backup of the original train file is saved to
data/MINDsmall_train/behaviors_full.tsv.backup before splitting.
"""

import os
import shutil
import sys

SPLIT_RATIO = 0.75  # 75% train / 25% validation

TRAIN_DIR = "data/MINDsmall_train"
DEV_DIR   = "data/MINDsmall_dev"

TRAIN_BEH = os.path.join(TRAIN_DIR, "behaviors.tsv")
DEV_BEH   = os.path.join(DEV_DIR,   "behaviors.tsv")
BACKUP    = os.path.join(TRAIN_DIR, "behaviors_full.tsv.backup")


def main():
    # Sanity checks
    if not os.path.exists(TRAIN_BEH):
        print(f"ERROR: {TRAIN_BEH} not found. Run this from the project root.")
        sys.exit(1)

    # Back up the original train file if we haven't already
    if not os.path.exists(BACKUP):
        print(f"[backup] {TRAIN_BEH} -> {BACKUP}")
        shutil.copy2(TRAIN_BEH, BACKUP)
    else:
        print(f"[backup] already exists at {BACKUP}, using it as source of truth")
        # Restore from backup so re-running the script produces a consistent result
        shutil.copy2(BACKUP, TRAIN_BEH)

    # Count lines
    print(f"[count] reading {TRAIN_BEH}...")
    with open(TRAIN_BEH, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    total = len(all_lines)
    split_idx = int(total * SPLIT_RATIO)
    print(f"[count] total impressions: {total:,}")
    print(f"[count] split at line {split_idx:,} ({SPLIT_RATIO*100:.0f}/"
          f"{(1-SPLIT_RATIO)*100:.0f})")

    train_lines = all_lines[:split_idx]
    val_lines   = all_lines[split_idx:]
    print(f"[count] new train: {len(train_lines):,} impressions")
    print(f"[count] new val:   {len(val_lines):,} impressions")

    # Report the temporal boundary
    def first_date(line):
        parts = line.rstrip("\n").split("\t")
        return parts[2] if len(parts) > 2 else "?"

    print(f"[boundary] last train impression time : {first_date(train_lines[-1])}")
    print(f"[boundary] first val   impression time: {first_date(val_lines[0])}")

    # Ensure dev directory exists. If the .vec and news.tsv files aren't there,
    # copy them from train (they're the same files per MIND convention).
    os.makedirs(DEV_DIR, exist_ok=True)
    for fn in ("news.tsv", "entity_embedding.vec", "relation_embedding.vec"):
        src = os.path.join(TRAIN_DIR, fn)
        dst = os.path.join(DEV_DIR, fn)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"[copy] {src} -> {dst}")
            shutil.copy2(src, dst)

    # Write the split
    print(f"[write] {TRAIN_BEH} ({len(train_lines):,} rows)")
    with open(TRAIN_BEH, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print(f"[write] {DEV_BEH} ({len(val_lines):,} rows)")
    with open(DEV_BEH, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    # Verify sizes differ
    train_size = os.path.getsize(TRAIN_BEH)
    dev_size   = os.path.getsize(DEV_BEH)
    print(f"\n[verify] train size: {train_size:,} bytes")
    print(f"[verify] dev   size: {dev_size:,} bytes")
    if train_size == dev_size:
        print("WARNING: sizes are equal — split may not have worked.")
    else:
        print("[ok] split successful, files differ in size.\n")

    print("Next steps:")
    print("  1. Clear the old cache:  rm -rf cache/*  (PowerShell: Remove-Item -Recurse -Force cache\\*)")
    print("  2. Clear old checkpoints: rm -rf models/* and old results/*.json, *.csv")
    print("  3. Re-run notebook 02 (preprocessing)")
    print("  4. Re-run notebook 03 (training)")


if __name__ == "__main__":
    main()