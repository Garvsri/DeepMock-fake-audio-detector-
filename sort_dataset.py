import os, shutil
from pathlib import Path

# Path to your LA folder inside archive
LA_ROOT = Path("archive/LA/LA")

PROTOCOL_FILE = LA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
AUDIO_DIR     = LA_ROOT / "ASVspoof2019_LA_train" / "flac"

REAL_OUT = Path("dataset/real")
FAKE_OUT = Path("dataset/fake")
REAL_OUT.mkdir(parents=True, exist_ok=True)
FAKE_OUT.mkdir(parents=True, exist_ok=True)

real_count = 0
fake_count = 0
LIMIT = 500  # 500 real + 500 fake files

print("Reading protocol file...")
print(f"Audio folder: {AUDIO_DIR}")
print(f"Protocol file: {PROTOCOL_FILE}")

if not PROTOCOL_FILE.exists():
    print("❌ Protocol file not found! Check archive/LA folder structure.")
    exit()

if not AUDIO_DIR.exists():
    print("❌ Audio folder not found! Check archive/LA folder structure.")
    exit()

with open(PROTOCOL_FILE, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        file_id = parts[1]
        label   = parts[4]  # 'genuine' or 'spoof'

        src = AUDIO_DIR / (file_id + ".flac")
        if not src.exists():
            continue

        if label in ["genuine", "bonafide"] and real_count < LIMIT:
            shutil.copy(src, REAL_OUT / (file_id + ".flac"))
            real_count += 1
            if real_count % 100 == 0:
                print(f"  Copied {real_count} real files...")

        elif label == "spoof" and fake_count < LIMIT:
            shutil.copy(src, FAKE_OUT / (file_id + ".flac"))
            fake_count += 1
            if fake_count % 100 == 0:
                print(f"  Copied {fake_count} fake files...")

        if real_count >= LIMIT and fake_count >= LIMIT:
            break

print(f"\n✅ Done!")
print(f"   dataset/real/ → {real_count} files")
print(f"   dataset/fake/ → {fake_count} files")