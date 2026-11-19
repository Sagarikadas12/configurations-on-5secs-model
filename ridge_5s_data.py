#!/usr/bin/env python
"""
Data of the 5 s slice ridge lie detector: reading the LibreFace results, the train / test split and the z-scores.
The model, its training (mean, spread, penalty, weights) and the test are in ridge_5s_train.py, which imports this file.
Same steps and numbers as ridge_5s_model.py (kept as it is: the other scripts of the project import it).

Data: 121 people, each with one truth video (S10) and one lie video (S11) of the same 240 s picture-description task, analysed by LibreFace:
    D:/librefaceresults/<NNN>truth.csv and <NNN>lie.csv, every 2nd frame of the 35 fps video (17.5 values per second, 4,200 rows).
    Person 102 has CSVs but is not part of the dataset (as in all earlier models) and is left out.

Steps (nothing is learned here)
    1. per video 48 signals per frame -> [4,200 frames x 48]: 12 AUs on/off, 12 AU intensities, 8 expressions (one-hot), head pitch / yaw / roll,
       12 measures from the MediaPipe face landmarks, head rotation speed. Frames without a face take the last value before them.
       (read_csv_values, fill_gaps, video_series; kept in cache/)
    2. 47 slices of 5 s (88 frames); per slice and signal  (slice average - video average) / video spread           -> 47 x 48 = 2,256 numbers
       (slice_levels)
    3. split (split_90_31.csv, make_split / read_split): the 121 people shuffled with seed 0; the first 90 people (180 videos) train, the other
       31 people (62 videos) test. Both videos of a person are always on the same side. These are the same people as the hold-out of
       libreface_clock_aware_5s_slice.py in D:/liedtection2.
    4. z-score: (number - training mean) / training spread, cut at +-4 (zscore). The training mean and spread are computed in ridge_5s_train.py,
       from the training videos only.

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_data.py --make_split     # write split_90_31.csv (once)
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_data.py --make_split --train_people 91 --in_order --split split_91_30.csv
                                                                                       # P000-P090 train, the next 30 people test
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_data.py                  # read the split's CSVs into cache/, show the data

The first run reads the 242 raw CSVs (28 GB) once and keeps the 48 signals of every video in cache/ (rebuilt by itself when a CSV changes),
together with what was missing before the gap filling (frames without a face, NaN values); check_cache_nans.py lists them.
The cache is the same as that of ridge_5s_model.py. Needs numpy and pandas.
"""
import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LIBREFACE_DIR = "D:/librefaceresults"
SPLIT_FILE = os.path.join(HERE, "split_90_31.csv")
CACHE_DIR = os.path.join(HERE, "cache")
CACHE_VERSION = 2                               # 2: the cache also keeps has_row and missing (before the gap filling)

EXCLUDED_PEOPLE = {"P102"}                      # has CSVs, but is not in the dataset of the earlier models
TRAIN_PEOPLE, SPLIT_SEED = 90, 0

VIDEO_FPS = 35                                  # frame numbers in the CSVs are frames of the 35 fps video
RATE = 17.5                                     # every 2nd frame -> 17.5 values per second
FRAMES = 4200                                   # 240 s
SECONDS = 5
WIDTH = int(round(SECONDS * RATE))              # 88 frames per slice
SLICES = FRAMES // WIDTH                        # 47 slices
CLIP = 4.0                                      # z-scores are cut at +-4

DETECTED_AUS = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
INTENSITY_AUS = [1, 2, 4, 5, 6, 9, 12, 15, 17, 20, 25, 26]
EXPRESSIONS = ["Neutral", "Happiness", "Sadness", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
AU_VALUE = re.compile(r"'au_(\d+)(?:_intensity)?':\s*([-\d.eE]+)")
LM_IDX = [1, 10, 13, 14, 33, 61, 105, 133, 145, 152, 159, 263, 291, 334, 362, 374, 386, 468, 473]
COLUMNS = (["file_name", "detected_aus", "au_intensities", "facial_expression", "pitch", "yaw", "roll"]
           + [f"lm_mp_{i}_{a}" for i in LM_IDX for a in "xy"])
SIGNALS = ([f"AU{a:02d} on/off" for a in DETECTED_AUS] + [f"AU{a:02d} intensity" for a in INTENSITY_AUS] + [f"expression {e}" for e in EXPRESSIONS]
           + ["head pitch", "head yaw", "head roll", "eye opening 1", "eye opening 2", "mouth opening", "mouth width", "brow height 1", "brow height 2",
              "iris horizontal 1", "iris vertical 1", "iris horizontal 2", "iris vertical 2", "head speed", "face size", "head rotation speed"])


# =====================================================================================
# 1. LibreFace CSV -> [4,200 frames x 48 signals]
# =====================================================================================
def read_csv_values(path):
    """Frame numbers (of the 35 fps video) and 47 values per row of one LibreFace CSV. Only even frames are kept (17.5 per second), as for the
    training videos; a CSV of every frame of a 35 fps video therefore gives the same frames as the training CSVs."""
    df = pd.read_csv(path, usecols=COLUMNS)
    frames = df["file_name"].str.extract(r"(\d+)", expand=False).astype(int).to_numpy()
    keep = frames % 2 == 0
    df, frames = df[keep].reset_index(drop=True), frames[keep]

    # 35 LibreFace values: 12 AUs on/off, 12 AU intensities, 8 expressions (one-hot), pitch / yaw / roll
    lf = np.full((len(df), 35), np.nan, np.float32)
    for i, (det, inten) in enumerate(zip(df["detected_aus"].astype(str), df["au_intensities"].astype(str))):
        det, inten = dict(AU_VALUE.findall(det)), dict(AU_VALUE.findall(inten))
        lf[i, :12] = [float(det.get(str(a), "nan")) for a in DETECTED_AUS]
        lf[i, 12:24] = [float(inten.get(str(a), "nan")) for a in INTENSITY_AUS]
    expr = df["facial_expression"].fillna("").to_numpy()
    unknown = set(expr) - set(EXPRESSIONS) - {""}
    if unknown:
        raise ValueError(f"{path}: unknown facial expressions {unknown}")
    for j, e in enumerate(EXPRESSIONS):
        lf[:, 24 + j] = np.where(expr == "", np.nan, expr == e)
    lf[:, 32:35] = df[["pitch", "yaw", "roll"]].to_numpy(np.float32)

    # 12 measures from MediaPipe's face landmarks (x, y in 0..1 image coordinates); speeds are between consecutive rows
    P = {i: df[[f"lm_mp_{i}_x", f"lm_mp_{i}_y"]].to_numpy(np.float32) for i in LM_IDX}
    d = lambda a, b: np.linalg.norm(P[a] - P[b], axis=1)
    ratio = lambda num, den: np.clip(num / np.where(np.abs(den) < 1e-6, 1e-6, den), -1, 3)
    iod = d(33, 263)                                                       # outer eye corners = face scale
    nose_step = np.linalg.norm(np.diff(P[1], axis=0), axis=1) / np.maximum(iod[1:], 1e-6)
    lm = np.stack([
        ratio(d(159, 145), d(33, 133)),                                    # eye opening, eye 1
        ratio(d(386, 374), d(362, 263)),                                   # eye opening, eye 2
        ratio(d(13, 14), d(61, 291)),                                      # mouth opening
        ratio(d(61, 291), iod),                                            # mouth width (smile)
        ratio(d(105, 159), iod),                                           # eyebrow height, eye 1
        ratio(d(334, 386), iod),                                           # eyebrow height, eye 2
        ratio(P[468][:, 0] - P[33][:, 0], P[133][:, 0] - P[33][:, 0]),     # iris horizontal, eye 1
        ratio(P[468][:, 1] - P[159][:, 1], P[145][:, 1] - P[159][:, 1]),   # iris vertical, eye 1
        ratio(P[473][:, 0] - P[362][:, 0], P[263][:, 0] - P[362][:, 0]),   # iris horizontal, eye 2
        ratio(P[473][:, 1] - P[386][:, 1], P[374][:, 1] - P[386][:, 1]),   # iris vertical, eye 2
        np.r_[0.0, nose_step],                                             # head speed
        iod / np.median(iod),                                              # face size (leaning in / out)
    ], 1).astype(np.float32)

    pose_speed = np.r_[0.0, np.abs(np.diff(lf[:, 32:35], axis=0)).sum(1)][:, None]   # head rotation speed
    return frames, np.concatenate([lf, lm, pose_speed], 1).astype(np.float32)


def fill_gaps(x):
    """Frames without a face take the last value before them (the first value after them, at the start)."""
    x = x.copy()
    for c in range(x.shape[1]):
        v = x[:, c]
        bad = ~np.isfinite(v)
        if bad.any():
            last = np.where(~bad, np.arange(len(v)), 0)
            np.maximum.accumulate(last, out=last)
            v = v[last]
            bad = ~np.isfinite(v)
            if bad.any() and (~bad).any():
                v[bad] = v[~bad][0]
            x[:, c] = np.nan_to_num(v)
    return x


def video_series(path):
    """One video:
        series      [4,200, 48]  every kept frame at its true time position, gaps filled
        last_frame  the last frame number in the CSV;  rows: the number of kept (even) rows
        has_row     [4,200]      False = no CSV row for that frame (LibreFace writes no row when it finds no face)
        missing     [4,200, 48]  True = no value (no row, or NaN) before the gap filling"""
    frames, values = read_csv_values(path)
    pos = frames // 2                                                      # every 2nd frame of 35 fps -> position at 17.5 per second
    x = np.full((FRAMES, values.shape[1]), np.nan, np.float32)
    inside = pos < FRAMES
    x[pos[inside]] = values[inside]
    has_row = np.zeros(FRAMES, bool)
    has_row[pos[inside]] = True
    return {"series": fill_gaps(x), "last_frame": int(frames.max()), "rows": len(frames), "has_row": has_row, "missing": ~np.isfinite(x)}


def seconds_of(last_frame):
    return (last_frame + 2) / VIDEO_FPS


# ---- cache/: the raw CSVs are read only once
def cache_path(path):
    return os.path.join(CACHE_DIR, os.path.splitext(os.path.basename(path))[0] + ".npz")


def source_key(path):
    st = os.stat(path)
    return np.array([CACHE_VERSION, st.st_size, st.st_mtime_ns], dtype=np.int64)


def cache_ok(path):
    c = cache_path(path)
    if not os.path.exists(c):
        return False
    with np.load(c) as d:
        return "source" in d.files and np.array_equal(d["source"], source_key(path))


def build_cache(path):
    """Read one raw CSV and keep its signals, and what was missing before the gap filling, in cache/ (write, then rename)."""
    info = video_series(path)
    c = cache_path(path)
    tmp = c[:-4] + ".tmp.npz"
    np.savez_compressed(tmp, **info, source=source_key(path), csv=os.path.abspath(path))
    os.replace(tmp, c)
    return path


def build_missing(paths, workers, log):
    """Build the cache files that are missing or stale (raw CSVs are read only then)."""
    todo = [p for p in paths if not cache_ok(p)]
    if todo:
        os.makedirs(CACHE_DIR, exist_ok=True)
        log(f"reading {len(todo)} raw LibreFace CSVs into {CACHE_DIR} ({workers} workers) ...")
        t0 = time.time()
        if workers > 1:
            with ProcessPoolExecutor(min(workers, len(todo))) as ex:
                for k, _ in enumerate(ex.map(build_cache, todo), 1):
                    if k % 20 == 0 or k == len(todo):
                        print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
        else:
            for k, p in enumerate(todo, 1):
                build_cache(p)
                if k % 20 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


def load_series(paths, workers, log):
    """[4,200, 48] signals, the last frame number and the number of frames without a face of every video, from cache/."""
    build_missing(paths, workers, log)
    xs, lasts, no_face = [], [], []
    for p in paths:
        with np.load(cache_path(p)) as d:
            xs.append(d["series"])
            lasts.append(int(d["last_frame"]))
            no_face.append(int((~d["has_row"]).sum()))
    return xs, np.array(lasts), np.array(no_face)


# =====================================================================================
# 2. [4,200 x 48] -> 47 slices x 48 signals = 2,256 numbers per video
# =====================================================================================
def slice_shape(seconds=SECONDS):
    """Frames per slice and number of whole slices in the 240 s, for slices of `seconds` (5 s: 88 frames, 47 slices). Frames after the last
    whole slice are not used."""
    width = int(round(seconds * RATE))
    return width, FRAMES // width


def slice_levels(x, seconds=SECONDS):
    """slices x 48 numbers of one video (47 x 48 for 5 s): level of every signal in each slice, relative to the video's own average and spread."""
    width, slices = slice_shape(seconds)
    lv = x[:slices * width].reshape(slices, width, x.shape[1]).mean(1)
    return ((lv - x.mean(0)) / np.maximum(x.std(0), 1e-6)).ravel()


def number_names(seconds=SECONDS):
    """(slice, seconds, signal) of each number (2,256 for 5 s), in the order of slice_levels."""
    width, slices = slice_shape(seconds)
    out = []
    for k in range(slices):
        for s in SIGNALS:
            out.append((k + 1, f"{k * width / RATE:.1f}-{(k + 1) * width / RATE:.1f}", s))
    return out


# =====================================================================================
# 3. Split
# =====================================================================================
def write_csv(path, rows, fields, fmt="{:.4f}"):
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (fmt.format(v) if isinstance(v, float) else v) for k, v in r.items()})


def make_split(libreface_dir, split_file, n_train=TRAIN_PEOPLE, shuffle=True):
    """split_90_31.csv: every video of the dataset with its person, label (lie = 1), side (train / test) and LibreFace CSV.
    The people are shuffled with seed 0; the first n_train (90) train, the rest test.
    shuffle=False: the people in number order (P000, P001, ...); the first n_train train, the rest test."""
    if os.path.exists(split_file):
        raise SystemExit(f"[error] {split_file} already exists; delete it first to make a new one")
    name = re.compile(r"^(\d{3})(truth|lie)\.csv$")
    found = {}
    for f in os.listdir(libreface_dir):
        m = name.match(f)
        if m:
            found.setdefault(f"P{m.group(1)}", {})[m.group(2)] = os.path.join(libreface_dir, f).replace("\\", "/")
    incomplete = sorted(p for p, v in found.items() if len(v) != 2)
    people = sorted(p for p, v in found.items() if len(v) == 2 and p not in EXCLUDED_PEOPLE)
    if not 0 < n_train < len(people):
        raise SystemExit(f"[error] {n_train} training people, but {len(people)} people with a truth and a lie CSV in {libreface_dir}")
    order = np.random.default_rng(SPLIT_SEED).permutation(np.array(people)) if shuffle else np.array(people)
    train_people = set(order[:n_train])
    rows = [{"video_id": f"{cls}_{p}", "person": p, "label": int(cls == "lie"), "split": "train" if p in train_people else "test",
             "libreface_csv": found[p][cls]} for p in people for cls in ("truth", "lie")]
    write_csv(split_file, rows, ["video_id", "person", "label", "split", "libreface_csv"])
    n_test = len(people) - n_train
    print(f"{split_file}: {len(people)} people, {len(rows)} videos")
    print(f"  train {n_train} people ({2 * n_train} videos), test {n_test} people ({2 * n_test} videos), "
          + (f"shuffle seed {SPLIT_SEED}" if shuffle else "people in number order (no shuffle)"))
    print(f"  left out: {', '.join(sorted(EXCLUDED_PEOPLE & set(found)))} (not in the dataset)"
          + (f"; without both videos: {', '.join(incomplete)}" if incomplete else ""))
    print(f"  test people: {', '.join(sorted(set(people) - train_people))}")


def read_split(split_file):
    if not os.path.exists(split_file):
        raise SystemExit(f"[error] no {split_file}; make it with ridge_5s_data.py --make_split")
    with open(split_file, newline="", encoding="utf8") as fh:
        rows = list(csv.DictReader(fh))
    need = {"video_id", "person", "label", "split", "libreface_csv"}
    if not rows or need - set(rows[0]):
        raise SystemExit(f"[error] {split_file} needs the columns {sorted(need)}")
    base = os.path.dirname(os.path.abspath(split_file))
    ids = [r["video_id"] for r in rows]
    persons = np.array([r["person"] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    part = np.array([r["split"].strip().lower() for r in rows])
    paths = [r["libreface_csv"] if os.path.isabs(r["libreface_csv"]) else os.path.join(base, r["libreface_csv"]) for r in rows]
    if set(part) - {"train", "test"}:
        raise SystemExit(f"[error] {split_file}: split must be train or test, found {sorted(set(part) - {'train', 'test'})}")
    if set(y) - {0, 1}:
        raise SystemExit(f"[error] {split_file}: label must be 0 (truth) or 1 (lie)")
    both = set(persons[part == "train"]) & set(persons[part == "test"])
    if both:
        raise SystemExit(f"[error] {split_file}: people in both train and test: {sorted(both)}")
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"[error] {len(missing)} LibreFace CSVs not found, e.g. {missing[0]}")
    return ids, persons, y, part, paths


def load_dataset(split_file=SPLIT_FILE, workers=4, log=print):
    """Every video of the split as its 2,256 numbers (steps 1-2), with id, person, label (lie = 1) and side (train / test):
        ids, persons, y, part, paths   one entry per video, in the order of the split file
        X          [videos, 2,256]   slice levels, not yet z-scored with the training mean and spread
        last_frame, no_face           last frame number and number of frames without a face of every video"""
    ids, persons, y, part, paths = read_split(split_file)
    xs, lasts, no_face = load_series(paths, workers, log)
    too_long = [ids[k] for k in np.where(lasts // 2 >= FRAMES)[0]]
    if too_long:
        raise SystemExit(f"[error] longer than {FRAMES / RATE:.0f} s: {', '.join(too_long)}")
    X = np.stack([slice_levels(x) for x in xs]).astype(np.float32)
    return {"ids": ids, "persons": persons, "y": y, "part": part, "paths": paths, "X": X, "last_frame": lasts, "no_face": no_face}


# =====================================================================================
# 4. z-score with the training mean and spread (computed in ridge_5s_train.py)
# =====================================================================================
def zscore(X, mean, spread):
    """(number - training mean) / training spread, cut at +-4; missing values become 0 (= the training mean)."""
    return np.clip(np.nan_to_num((X - mean) / spread), -CLIP, CLIP).astype(np.float64)


# =====================================================================================
# 5. Show what is in the data
# =====================================================================================
def show(split_file, workers):
    t0 = time.time()
    d = load_dataset(split_file, workers)
    persons, y, part, X, no_face = d["persons"], d["y"], d["part"], d["X"], d["no_face"]
    print(f"{len(d['ids'])} videos, {len(set(persons))} people; {SLICES} slices x {len(SIGNALS)} signals = {X.shape[1]} numbers per video "
          f"({time.time() - t0:.0f}s)")
    print(f"video lengths {seconds_of(d['last_frame'].min()):.1f}-{seconds_of(d['last_frame'].max()):.1f} s; frames without a face (filled with the "
          f"last value before): {no_face.sum()} of {len(d['ids']) * FRAMES} in {(no_face > 0).sum()} videos (check_cache_nans.py lists them)")
    print(f"split {os.path.basename(split_file)}:")
    for side in ("train", "test"):
        k = part == side
        print(f"  {side:5s} {len(set(persons[k])):3d} people, {k.sum():3d} videos ({int(y[k].sum())} lies)")
    print(f"slice levels: {np.isfinite(X).all(1).sum()} of {len(X)} videos without missing values, "
          f"range {np.nanmin(X):.2f} to {np.nanmax(X):.2f}")


def main():
    p = argparse.ArgumentParser(description="Data of the 5 s slice ridge: make the split, or read the split's LibreFace CSVs into cache/ and show them")
    p.add_argument("--make_split", action="store_true", help="write the split file (--train_people / rest people) and stop")
    p.add_argument("--train_people", type=int, default=TRAIN_PEOPLE, help=f"for --make_split: training people, the first of the shuffled "
                                                                          f"order (default {TRAIN_PEOPLE})")
    p.add_argument("--in_order", action="store_true", help="for --make_split: no shuffle, the people in number order (P000, P001, ...)")
    p.add_argument("--libreface_dir", default=LIBREFACE_DIR, help=f"raw LibreFace CSVs, for --make_split (default {LIBREFACE_DIR})")
    p.add_argument("--split", default=SPLIT_FILE, help=f"split file (default {SPLIT_FILE})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    if args.make_split:
        make_split(args.libreface_dir, args.split, args.train_people, shuffle=not args.in_order)
    else:
        show(args.split, max(1, args.workers))


if __name__ == "__main__":
    sys.exit(main())
