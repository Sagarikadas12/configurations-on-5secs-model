#!/usr/bin/env python
"""
Ridge lie detector on the FIRST MINUTE of every video only, in 5 s slices, tested by 10-fold person-grouped cross-validation.
Same architecture as the 5 s model of ridge_slices_cv10.py; the only difference is that everything after the first minute is thrown away
before anything is computed.

Numbers of one video (nothing is learned here)
    1. its [4,200 frames x 48 signals] (ridge_5s_data.py, from cache/) cut to exactly the first minute: 60 s x 17.5 = 1,050 frames, in 12 slices
       of 5 s. 5 s is 87.5 frames, so the slices alternate 88 and 87 frames (frames 0-87, 88-174, 175-262, ...; slice k ends at frame
       ceil(k x 87.5)): 6 x 88 + 6 x 87 = 1,050 frames = 60.0 s
    2. per slice and signal  (slice average - average of the first minute) / spread of the first minute    -> 12 x 48 = 576 numbers
       (average and spread over the same 1,050 frames: no frame after the first minute is used anywhere)

Model, a completely new one for every fold, trained on the other 9 folds only (ridge_5s_train.py)
    mean and spread of every number over its training videos -> z-scores, cut at +-4
    -> penalty strength by an inner person-grouped 5-fold CV over its training people (best held-out AUC)
    -> ridge weights fitted on all its training videos (one weight per number, no bias; target lie +1, truth -1)
    -> tested once on the people of its own fold (score > 0 = lie)

Folds: the 121 people of split_90_31.csv in 10 folds of 12 or 13 people, both videos of a person in the same fold (sklearn GroupKFold over the
video order of the split file: the same folds as ridge_5s_cv10.py and ridge_slices_cv10.py, so the results compare on the same people).
The train / test column of the split file is not used here.

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_first_minute_cv10.py
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_first_minute_cv10.py --first_seconds 120 --out_dir results_ridge_first_2min_cv10
                                                                       # the first 2 minutes instead: 24 slices, 2,100 frames

Results in results_ridge_first_minute_cv10/:
    summary.csv            all 242 out-of-fold scores together: every metric with person-resampled 95% intervals; per-fold accuracy mean, sd
    per_fold.csv           per fold: people and videos, chosen strength, inner-CV AUC, training and test accuracy, test AUC, pair accuracy,
                           confusion counts, the wrong videos
    predictions.csv        every video's score from the model of its fold (no video is scored by a model that saw its person)
    folds.csv              every person's fold
    fold_01 ... fold_10/   weights.npz          the fold's model: mean, spread, weight (576 each), penalty, first_frames, slice_bounds
                           penalty_choice.csv   its inner CV: held-out AUC, accuracy, error per strength; fit per strength
    weights_all_folds.csv  one row per number (576): the weight of every fold's model, their average, spread and sign agreement
    log.txt
"""
import argparse
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_5s_train as model                  # noqa: E402

OUT_DIR = os.path.join(HERE, "results_ridge_first_minute_cv10")
FIRST_SECONDS = 60                              # use only this start of every video
FOLDS = 10


def slice_bounds(first_seconds):
    """Frame boundaries of the 5 s slices of the first first_seconds of the video (60 s -> 0, 88, 175, 263, 350, ..., 1,050: 12 slices).
    A 5 s slice is 87.5 frames at 17.5 frames per second, so slice k ends at frame ceil(k x 87.5): the slices alternate 88 and 87 frames and
    together cover exactly first_seconds (60 s = 1,050 frames)."""
    if first_seconds % data.SECONDS or not data.SECONDS <= first_seconds <= data.FRAMES / data.RATE:
        raise SystemExit(f"[error] --first_seconds must be a multiple of {data.SECONDS} between {data.SECONDS} and {data.FRAMES / data.RATE:.0f}")
    step = data.SECONDS * data.RATE                                         # 87.5 frames
    return np.ceil(np.arange(first_seconds // data.SECONDS + 1) * step).astype(int)


def first_part_levels(x, bounds):
    """slices x 48 numbers of one video from its first bounds[-1] frames only (the first minute): level of every signal in each 5 s slice,
    relative to the average and spread of those first frames (the formula of ridge_5s_data.slice_levels, on the start of the video only)."""
    part = x[:bounds[-1]]
    lv = np.stack([part[a:b].mean(0) for a, b in zip(bounds[:-1], bounds[1:])])
    return ((lv - part.mean(0)) / np.maximum(part.std(0), 1e-6)).ravel()


def first_part_names(bounds):
    """(slice, seconds, signal) of each number, in the order of first_part_levels."""
    return [(k + 1, f"{a / data.RATE:.1f}-{b / data.RATE:.1f}", s)
            for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])) for s in data.SIGNALS]


def run(split_file, out_dir, workers, first_seconds):
    os.makedirs(out_dir, exist_ok=True)
    log = model.Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    ids, persons, y, _, paths = data.read_split(split_file)
    xs, lasts, no_face = data.load_series(paths, workers, log)
    too_long = [ids[k] for k in np.where(lasts // 2 >= data.FRAMES)[0]]
    if too_long:
        raise SystemExit(f"[error] longer than {data.FRAMES / data.RATE:.0f} s: {', '.join(too_long)}")

    bounds = slice_bounds(first_seconds)
    n_frames, slices, widths = int(bounds[-1]), len(bounds) - 1, np.diff(bounds)
    X = np.stack([first_part_levels(x, bounds) for x in xs]).astype(np.float32)
    folds = list(GroupKFold(FOLDS).split(np.zeros((len(y), 1)), groups=persons))
    fold_of = np.zeros(len(y), int)
    for f, (_, te) in enumerate(folds, 1):
        fold_of[te] = f

    log(f"ridge on the first {n_frames / data.RATE:.1f} s of every video ({slices} slices of {data.SECONDS} s, {widths.min()}-{widths.max()} frames each, = {n_frames} of {data.FRAMES} frames; "
        f"the rest of the video is not used), {FOLDS}-fold person-grouped CV")
    log(f"{len(ids)} videos, {len(set(persons))} people; {slices} slices x {len(data.SIGNALS)} signals = {X.shape[1]} numbers per video; folds of "
        f"{min(len(set(persons[te])) for _, te in folds)}-{max(len(set(persons[te])) for _, te in folds)} people; every fold trains a new model on "
        f"the other {FOLDS - 1} folds and tests it on its own people ({time.time() - t0:.0f}s)\n")

    S = np.zeros(len(y))
    per_fold, weights, names = [], [], first_part_names(bounds)
    for f, (tr, te) in enumerate(folds, 1):
        m = model.fit(X, y, persons, tr)                                   # new model: mean, spread, inner CV for the strength, weights
        table = model.strength_table(X, y, tr, m)
        s, s_tr = model.score(X[te], m), model.score(X[tr], m)
        S[te] = s
        weights.append(m["weight"])
        q = model.point_metrics(y[te], s)
        test_people = sorted(set(persons[te]))
        wrong = [ids[i] for k, i in enumerate(te) if (s[k] > 0) != (y[i] == 1)]
        row = {"fold": f, "train_people": len(set(persons[tr])), "train_videos": len(tr), "test_people": len(test_people), "test_videos": len(te),
               "penalty": m["penalty"], "inner_cv_auc": m["inner_auc"], "train_acc": float(np.mean((s_tr > 0) == y[tr])),
               "test_acc": q["acc"], "test_right": int(q["tp"] + q["tn"]), "test_auc": q["auc"],
               "pair_acc": model.pair_accuracy(persons[te], y[te], s),
               "tp": q["tp"], "fp": q["fp"], "tn": q["tn"], "fn": q["fn"], "wrong_videos": "; ".join(wrong), "people": " ".join(test_people)}
        per_fold.append(row)
        log(f"fold {f:2d}: train {row['train_people']} people ({len(tr)} videos), test {row['test_people']} people ({len(te)} videos): "
            f"{' '.join(test_people)}")
        log(f"         strength {m['penalty']:g} (inner-CV AUC {m['inner_auc']:.3f}), training acc {row['train_acc']:.3f}  ->  test acc {q['acc']:.3f} "
            f"({row['test_right']}/{len(te)})  AUC {q['auc']:.3f}  pair {row['pair_acc']:.3f}  [TP {q['tp']} FP {q['fp']} TN {q['tn']} FN {q['fn']}]"
            + (f"  wrong: {', '.join(wrong)}" if wrong else ""))

        d = os.path.join(out_dir, f"fold_{f:02d}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                            signals=np.array(data.SIGNALS), seconds=data.SECONDS, first_frames=n_frames, slice_bounds=bounds, rate=data.RATE, frames=data.FRAMES,
                            clip=data.CLIP, fold=f, train_people=np.array(sorted(set(persons[tr]))), test_people=np.array(test_people))
        data.write_csv(os.path.join(d, "penalty_choice.csv"), table, list(table[0]), fmt="{:.6g}")

    # ---- all folds together
    accs, aucs = np.array([r["test_acc"] for r in per_fold]), np.array([r["test_auc"] for r in per_fold])
    res = model.all_metrics(y, S, persons)
    chosen = " ".join(f"{r['penalty']:g}" for r in per_fold)
    log(f"\nper fold: test acc {accs.mean():.3f} on average (sd {accs.std(ddof=1):.3f}, {accs.min():.3f}-{accs.max():.3f}), "
        f"test AUC {aucs.mean():.3f} (sd {aucs.std(ddof=1):.3f}, {aucs.min():.3f}-{aucs.max():.3f}); strengths chosen: {chosen}")
    log(f"all {len(y)} videos, each scored by the model of its fold (95% intervals resample people, {model.BOOTSTRAP} times):")
    log(model.metrics_line(f"first {n_frames / data.RATE:.0f} s, {FOLDS}-fold", res))

    Wf = np.array(weights)
    w_mean, w_sd = Wf.mean(0), Wf.std(0, ddof=1)
    agree = np.maximum((Wf > 0).sum(0), (Wf < 0).sum(0))
    C = np.corrcoef(Wf)
    off = C[~np.eye(FOLDS, dtype=bool)]
    log(f"\nthe {FOLDS} models' weights: correlation between two folds {off.min():.3f}-{off.max():.3f} (average {off.mean():.3f}); "
        f"numbers whose weight has the same sign in all {FOLDS} folds: {int((agree == FOLDS).sum())} of {Wf.shape[1]}")
    log("largest average weights (+ pushes towards lie), with their spread over the folds:")
    for k in np.argsort(-np.abs(w_mean))[:10]:
        log(f"  {w_mean[k]:+.5f} (sd {w_sd[k]:.5f}, same sign in {agree[k]}/{FOLDS} folds)  slice {names[k][0]:2d} ({names[k][1]} s)  {names[k][2]}")

    # ---- save
    data.write_csv(os.path.join(out_dir, "summary.csv"),
                   [{"test": f"{FOLDS}-fold CV, first {n_frames / data.RATE:.1f} s", "first_frames": n_frames, "slices": slices,
                     "numbers": X.shape[1], "people": len(set(persons)), "videos": len(y), "fold_acc_mean": float(accs.mean()),
                     "fold_acc_sd": float(accs.std(ddof=1)), "fold_auc_mean": float(aucs.mean()), "fold_auc_sd": float(aucs.std(ddof=1)),
                     "penalties": chosen, **res}],
                   ["test", "first_frames", "slices", "numbers", "people", "videos", "fold_acc_mean", "fold_acc_sd", "fold_auc_mean", "fold_auc_sd",
                    "penalties"] + model.FIELDS)
    data.write_csv(os.path.join(out_dir, "per_fold.csv"), per_fold, list(per_fold[0]))
    data.write_csv(os.path.join(out_dir, "predictions.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "fold": int(fold_of[i]), "score": float(S[i]),
                     "pred": int(S[i] > 0), "correct": int((S[i] > 0) == y[i])} for i in range(len(ids))],
                   ["video_id", "person", "label", "fold", "score", "pred", "correct"])
    data.write_csv(os.path.join(out_dir, "folds.csv"), [{"person": p, "fold": int(fold_of[persons == p][0])} for p in sorted(set(persons))],
                   ["person", "fold"])
    wcols = [f"weight_fold{f:02d}" for f in range(1, FOLDS + 1)]
    data.write_csv(os.path.join(out_dir, "weights_all_folds.csv"),
                   [{"position": k, "slice": names[k][0], "seconds": names[k][1], "signal": names[k][2],
                     **{c: float(Wf[j, k]) for j, c in enumerate(wcols)}, "weight_mean": float(w_mean[k]), "weight_sd": float(w_sd[k]),
                     "same_sign_folds": int(agree[k])} for k in range(Wf.shape[1])],
                   ["position", "slice", "seconds", "signal"] + wcols + ["weight_mean", "weight_sd", "same_sign_folds"], fmt="{:.8g}")
    log(f"\nresults in {out_dir}: summary.csv, per_fold.csv, predictions.csv, folds.csv, fold_01 ... fold_{FOLDS:02d}/ (weights.npz, "
        f"penalty_choice.csv), weights_all_folds.csv, log.txt ({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description="Ridge on the first minute of every video, 5 s slices, 10-fold person-grouped CV")
    p.add_argument("--first_seconds", type=int, default=FIRST_SECONDS,
                   help=f"use only this start of every video, a multiple of 5 s (default {FIRST_SECONDS}: 12 slices of 87-88 frames, 1,050 frames)")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    run(args.split, args.out_dir, max(1, args.workers), args.first_seconds)


if __name__ == "__main__":
    sys.exit(main())
