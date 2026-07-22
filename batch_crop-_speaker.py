#!/usr/bin/env python3
"""
batch_crop_speaker.py

Local (non-Colab) batch version of the sentence-cropping pipeline.

Fixes the risky part of the Colab notebook: instead of pairing .wav and
.TextGrid files by their position after alphabetical sorting (which
silently breaks if the two folders don't sort into the same order), this
version pairs files by matching BASENAME. A .wav and a .TextGrid are
only paired if they share the same filename (minus extension).

HOW TO USE:
1. Put all your .wav files in WAV_DIR and all your .TextGrid files in
   TG_DIR (they can be the same folder too -- it just filters by extension).
2. Set OUTPUT_DIR to where you want the cropped clips to go.
3. Run: python batch_crop_speaker.py

Matching files must have the same base name, e.g.:
    ASS_S1_SEN__5_.wav   <-->   ASS_S1_SEN__5_.TextGrid

Any .wav without a matching .TextGrid (or vice versa) is reported and
skipped, rather than silently mismatched.
"""

import os
import sys
import glob
import numpy as np
import soundfile as sf
import textgrid

# ----------------------------- CONFIG -----------------------------
WAV_DIR = "Speech/Speaker6"        # <-- folder containing .wav files
TG_DIR = "Speech/Textgrid"         # <-- folder containing .TextGrid files
OUTPUT_DIR = "Processed_Speaker6"  # <-- output folder for cropped clips
REMOVE_LABEL = "p"                 # label marking segments to discard
GAP_MERGE = 0.0                    # seconds; >0 merges close kept intervals
                                    # into a single output file (use only if
                                    # "p" also marks small internal noise
                                    # within one sentence)
EXPECTED_SEGMENTS = 3               # expected repetitions per sentence (for warnings)
# --------------------------------------------------------------------


def find_annotated_tier(tg, remove_label):
    for tier in tg.tiers:
        labels = {(iv.mark or "").strip().lower() for iv in tier}
        if remove_label.lower() in labels:
            return tier
    for tier in tg.tiers:
        if any((iv.mark or "").strip() for iv in tier):
            return tier
    return None


def get_kept_intervals(tier, remove_label):
    kept = []
    for interval in tier:
        label = (interval.mark or "").strip()
        if label == "" or label.lower() == remove_label.lower():
            continue
        kept.append((interval.minTime, interval.maxTime, label))
    return kept


def group_intervals(kept, gap_merge):
    if not kept:
        return []
    groups = [[kept[0]]]
    for prev, curr in zip(kept, kept[1:]):
        gap = curr[0] - prev[1]
        if gap_merge > 0 and gap <= gap_merge:
            groups[-1].append(curr)
        else:
            groups.append([curr])
    return groups


def crop_and_concat(audio, sr, intervals):
    chunks = []
    for start, end, _ in intervals:
        i0 = int(round(start * sr))
        i1 = int(round(end * sr))
        chunks.append(audio[i0:i1])
    return np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]


def collect_files(directory, ext):
    return {
        os.path.splitext(os.path.basename(f))[0]: f
        for f in glob.glob(os.path.join(directory, f"*{ext}"))
    }


def main():
    if not os.path.isdir(WAV_DIR):
        sys.exit(f"WAV_DIR not found: {WAV_DIR}")
    if not os.path.isdir(TG_DIR):
        sys.exit(f"TG_DIR not found: {TG_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_map = collect_files(WAV_DIR, ".wav")
    tg_map = {}
    for ext in (".TextGrid", ".textgrid"):
        tg_map.update(collect_files(TG_DIR, ext))

    wav_names = set(wav_map.keys())
    tg_names = set(tg_map.keys())

    matched = sorted(wav_names & tg_names)
    only_wav = sorted(wav_names - tg_names)
    only_tg = sorted(tg_names - wav_names)

    print(f"🎧 WAV files found      : {len(wav_map)}")
    print(f"📝 TextGrid files found : {len(tg_map)}")
    print(f"🔗 Matched pairs        : {len(matched)}")

    if only_wav:
        print(f"⚠️  {len(only_wav)} wav file(s) with NO matching TextGrid (skipped): {only_wav}")
    if only_tg:
        print(f"⚠️  {len(only_tg)} TextGrid file(s) with NO matching wav (skipped): {only_tg}")

    if not matched:
        sys.exit("No matched pairs found -- check WAV_DIR/TG_DIR and filenames.")

    print("\n🎙️ Processing matched pairs...\n")

    total_ok, total_fail = 0, 0

    for base in matched:
        wav_path = os.path.join(WAV_DIR, wav_map[base])
        tg_path = os.path.join(TG_DIR, tg_map[base])
        print(f"🔗 {base}")

        try:
            audio, sr = sf.read(wav_path)
            tg = textgrid.TextGrid.fromFile(tg_path)
            tier = find_annotated_tier(tg, REMOVE_LABEL)
            if tier is None:
                print(f"   ❌ No usable tier found in {tg_map[base]}")
                total_fail += 1
                continue

            kept = get_kept_intervals(tier, REMOVE_LABEL)
            if not kept:
                print(f"   ❌ No kept intervals in {tg_map[base]}")
                total_fail += 1
                continue

            groups = group_intervals(kept, GAP_MERGE)

            if len(groups) != EXPECTED_SEGMENTS:
                print(f"   ⚠️  Expected {EXPECTED_SEGMENTS} segments, got {len(groups)}")

            for i, group in enumerate(groups, start=1):
                cropped = crop_and_concat(audio, sr, group)
                out_name = f"{base}_part{i}.wav"
                sf.write(os.path.join(OUTPUT_DIR, out_name), cropped, sr)
                print(f"   ✅ {out_name}")

            total_ok += 1

        except Exception as e:
            print(f"   ❌ Error processing {base}: {e}")
            total_fail += 1

    print(f"\nDone. {total_ok} pair(s) processed successfully, {total_fail} failed.")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()