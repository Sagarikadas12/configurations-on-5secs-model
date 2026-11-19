#!/usr/bin/env python
"""
Six ridge lie detectors, one per slice length (5, 10, 15, 20, 25 and 30 s), each tested by 10-fold person-grouped cross-validation.

Every slice length is its own model. The 240 s video is cut into slices of that length and becomes  slices x 48 signals  numbers: per slice and
signal  (slice average - video average) / video spread  (ridge_5s_data.slice_levels). For every fold a completely new model is trained on the
other 9 folds only, nothing is carried over from another fold or another slice length:
    mean and spread of every number over its training videos -> z-scores, cut at +-4
    -> penalty strength by an inner person-grouped 5-fold CV over its training people (best held-out AUC)
    -> ridge weights fitted on all its training videos (one weight per number, no bias; target lie +1, truth -1)
and it is tested once on the people of its own fold (score > 0 = lie). 6 slice lengths x 10 folds = 60 models, each with its own mean,
spread, penalty and weights. The 5 s model is the one of ridge_5s_train.py (and its 10-fold CV the one of ridge_5s_cv10.py).

Slices are whole numbers of frames (17.5 per second): 5 s = 88 frames, 10 s = 175, 15 s = 262, 20 s = 350, 25 s = 438, 30 s = 525; frames
after the last whole slice are not used (5 s: the last 3.7 s, 15 s: 0.5 s, 25 s: 14.7 s; 10, 20 and 30 s use all 240 s).

Folds: the 121 people of split_90_31.csv in 10 folds of 12 or 13 people, both videos of a person in the same fold (sklearn GroupKFold over the
video order of the split file: the same folds as ridge_5s_cv10.py and the 10-fold CV of the earlier models). The same folds for all six slice
lengths, so they are compared on the same people. The train / test column of the split file is not used here.

Data and model come from ridge_5s_data.py and ridge_5s_train.py (same folder).

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_slices_cv10.py

Results in results_ridge_slices_cv10/:
    summary.csv           one row per slice length: numbers per video, per-fold test accuracy (mean, sd), and all 242 out-of-fold scores
                          together: every metric with person-resampled 95% intervals
    per_fold.csv          one row per slice length and fold: chosen strength, inner-CV AUC, training and test accuracy, test AUC, pair
                          accuracy, confusion counts, the wrong videos
    predictions.csv       one row per video: its fold and its out-of-fold score from each of the six models
    folds.csv             every person's fold
    slices_05s/ ... slices_30s/
        fold_01 ... fold_10/weights.npz   that fold's model: mean, spread, weight, penalty, seconds
                                          (to score new videos: ridge_5s_train.py --predict a.csv --out_dir results_ridge_slices_cv10/slices_10s/fold_01)
        weights_all_folds.csv             one row per number: the weight of every fold's model, their average and spread
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

OUT_DIR = os.path.join(HERE, "results_ridge_slices_cv10")
SLICE_SECONDS = (5, 10, 15, 20, 25, 30)
FOLDS = 10


def cv_one_length(sec, xs, ids, persons, y, folds, out_dir):
    """10 new models for slices of sec seconds, one per fold, each saved. Returns the numbers per video, every video's out-of-fold score and
    one row per fold."""
    X = np.stack([data.slice_levels(x, sec) for x in xs]).astype(np.float32)
    sub = os.path.join(out_dir, f"slices_{sec:02d}s")
    S = np.zeros(len(y))
    rows, weights = [], []
    for f, (tr, te) in enumerate(folds, 1):
        m = model.fit(X, y, persons, tr)                               # new model: mean, spread, inner CV for the strength, weights
        s, s_tr = model.score(X[te], m), model.score(X[tr], m)
        S[te] = s
        weights.append(m["weight"])
        q = model.point_metrics(y[te], s)
        test_people = sorted(set(persons[te]))
        wrong = [ids[i] for k, i in enumerate(te) if (s[k] > 0) != (y[i] == 1)]
        rows.append({"seconds": sec, "fold": f, "train_people": len(set(persons[tr])), "train_videos": len(tr), "test_people": len(test_people),
                     "test_videos": len(te), "penalty": m["penalty"], "inner_cv_auc": m["inner_auc"],
                     "train_acc": float(np.mean((s_tr > 0) == y[tr])), "test_acc": q["acc"], "test_right": int(q["tp"] + q["tn"]),
                     "test_auc": q["auc"], "pair_acc": model.pair_accuracy(persons[te], y[te], s),
                     "tp": q["tp"], "fp": q["fp"], "tn": q["tn"], "fn": q["fn"], "wrong_videos": "; ".join(wrong), "people": " ".join(test_people)})
        d = os.path.join(sub, f"fold_{f:02d}")
        os.makedirs(d, exist_ok=True)
        np.savez_compressed(os.path.join(d, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                            signals=np.array(data.SIGNALS), seconds=sec, rate=data.RATE, frames=data.FRAMES, clip=data.CLIP, fold=f,
                            train_people=np.array(sorted(set(persons[tr]))), test_people=np.array(test_people))

    names, Wf = data.number_names(sec), np.array(weights)
    wcols = [f"weight_fold{f:02d}" for f in range(1, len(folds) + 1)]
    data.write_csv(os.path.join(sub, "weights_all_folds.csv"),
                   [{"position": k, "slice": names[k][0], "seconds": names[k][1], "signal": names[k][2],
                     **{c: float(Wf[j, k]) for j, c in enumerate(wcols)}, "weight_mean": float(Wf[:, k].mean()),
                     "weight_sd": float(Wf[:, k].std(ddof=1))} for k in range(Wf.shape[1])],
                   ["position", "slice", "seconds", "signal"] + wcols + ["weight_mean", "weight_sd"], fmt="{:.8g}")
    return X.shape[1], S, rows


def run(split_file, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    log = model.Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    ids, persons, y, _, paths = data.read_split(split_file)
    xs, lasts, _ = data.load_series(paths, workers, log)
    too_long = [ids[k] for k in np.where(lasts // 2 >= data.FRAMES)[0]]
    if too_long:
        raise SystemExit(f"[error] longer than {data.FRAMES / data.RATE:.0f} s: {', '.join(too_long)}")
    folds = list(GroupKFold(FOLDS).split(np.zeros((len(y), 1)), groups=persons))
    fold_of = np.zeros(len(y), int)
    for f, (_, te) in enumerate(folds, 1):
        fold_of[te] = f

    log(f"ridge per slice length, {FOLDS}-fold person-grouped CV: {len(ids)} videos, {len(set(persons))} people, folds of "
        f"{min(len(set(persons[te])) for _, te in folds)}-{max(len(set(persons[te])) for _, te in folds)} people; every fold of every slice "
        f"length trains a new model on the other {FOLDS - 1} folds and tests it on its own people ({time.time() - t0:.0f}s)")
    log(f"  {'slice':>6s} {'frames':>7s} {'slices':>7s} {'numbers':>8s} {'seconds used':>13s}")
    for sec in SLICE_SECONDS:
        width, slices = data.slice_shape(sec)
        log(f"  {sec:>4d} s {width:>7d} {slices:>7d} {slices * len(data.SIGNALS):>8d} {slices * width / data.RATE:>13.1f}")

    summary, per_fold, scores = [], [], {}
    for sec in SLICE_SECONDS:
        width, slices = data.slice_shape(sec)
        n, S, rows = cv_one_length(sec, xs, ids, persons, y, folds, out_dir)
        accs, aucs = np.array([r["test_acc"] for r in rows]), np.array([r["test_auc"] for r in rows])
        res = model.all_metrics(y, S, persons)
        chosen = " ".join(f"{r['penalty']:g}" for r in rows)
        log(f"\n{sec} s slices: {slices} slices x {len(data.SIGNALS)} signals = {n} numbers per video ({time.time() - t0:.0f}s)")
        log(f"  test acc per fold: {' '.join(f'{a:.3f}' for a in accs)}")
        log(f"  strengths chosen:  {chosen}")
        log(f"  per fold: test acc {accs.mean():.3f} on average (sd {accs.std(ddof=1):.3f}, {accs.min():.3f}-{accs.max():.3f}), "
            f"test AUC {aucs.mean():.3f} (sd {aucs.std(ddof=1):.3f})")
        log(model.metrics_line(f"all {len(y)} videos", res))
        summary.append({"seconds": sec, "frames_per_slice": width, "slices": slices, "numbers": n, "seconds_used": slices * width / data.RATE,
                        "people": len(set(persons)), "videos": len(y), "fold_acc_mean": float(accs.mean()), "fold_acc_sd": float(accs.std(ddof=1)),
                        "fold_auc_mean": float(aucs.mean()), "fold_auc_sd": float(aucs.std(ddof=1)),
                        "penalties": chosen, **res})
        per_fold += rows
        scores[sec] = S

    log(f"\nsummary: {FOLDS}-fold person-grouped CV, {len(set(persons))} people, the same folds for every slice length. accuracy and AUC of all "
        f"{len(y)} out-of-fold scores (each video scored by the model of its fold), 95% intervals resample people ({model.BOOTSTRAP} times)")
    log(f"  {'slice':>6s} {'numbers':>8s} | {'accuracy (95% interval)':>24s} | {'fold acc mean, sd':>17s} | {'AUC (95% interval)':>20s} | "
        f"{'pair':>5s} | TP FP TN FN")
    for r in summary:
        log(f"  {r['seconds']:>4d} s {r['numbers']:>8d} | {r['acc']:>7.3f} ({r['acc_low']:.3f}-{r['acc_high']:.3f})     | "
            f"{r['fold_acc_mean']:>9.3f}, {r['fold_acc_sd']:.3f} | {r['auc']:>7.3f} ({r['auc_low']:.3f}-{r['auc_high']:.3f}) | "
            f"{r['pair_acc']:>5.3f} | {r['tp']:>3d} {r['fp']:>2d} {r['tn']:>3d} {r['fn']:>2d}")

    # ---- save
    data.write_csv(os.path.join(out_dir, "summary.csv"), summary,
                   ["seconds", "frames_per_slice", "slices", "numbers", "seconds_used", "people", "videos", "fold_acc_mean", "fold_acc_sd",
                    "fold_auc_mean", "fold_auc_sd", "penalties"] + model.FIELDS)
    data.write_csv(os.path.join(out_dir, "per_fold.csv"), per_fold, list(per_fold[0]))
    cols = [c for sec in SLICE_SECONDS for c in (f"score_{sec:02d}s", f"correct_{sec:02d}s")]
    data.write_csv(os.path.join(out_dir, "predictions.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "fold": int(fold_of[i]),
                     **{k: v for sec in SLICE_SECONDS for k, v in ((f"score_{sec:02d}s", float(scores[sec][i])),
                                                                   (f"correct_{sec:02d}s", int((scores[sec][i] > 0) == y[i])))}}
                    for i in range(len(ids))],
                   ["video_id", "person", "label", "fold"] + cols)
    data.write_csv(os.path.join(out_dir, "folds.csv"), [{"person": p, "fold": int(fold_of[persons == p][0])} for p in sorted(set(persons))],
                   ["person", "fold"])
    log(f"\nresults in {out_dir}: summary.csv, per_fold.csv, predictions.csv, folds.csv, slices_05s ... slices_30s/ (fold_01 ... fold_10/weights.npz, "
        f"weights_all_folds.csv), log.txt ({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description="Ridge per slice length (5, 10, 15, 20, 25, 30 s), each by 10-fold person-grouped CV")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    run(args.split, args.out_dir, max(1, args.workers))


if __name__ == "__main__":
    sys.exit(main())
