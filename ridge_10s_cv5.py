#!/usr/bin/env python
"""
Ridge lie detector on the FULL 240 s of every video, in 10 s slices, tested by 5-fold person-grouped cross-validation.
Same as ridge_5s_cv5.py; the only difference is the slice length: 10 s instead of 5 s.

Numbers of one video (ridge_5s_data.slice_levels with seconds=10, nothing is learned here)
    its [4,200 frames x 48 signals] (ridge_5s_data.py, from cache/) in 24 slices of 175 frames (10.0 s; 24 x 175 = 4,200 frames = the whole
    240 s, no frame left over); per slice and signal  (slice average - video average) / video spread, average and spread over the whole 240 s
    -> 24 x 48 = 1,152 numbers per video (the 5 s model: 47 x 48 = 2,256)

Folds: the 121 people of split_90_31.csv in 5 folds of 24 or 25 people (48-50 videos), both videos of a person in the same fold (sklearn GroupKFold
over the video order of the split file: the same 5 folds as ridge_5s_cv5.py and the minute scripts, so the results compare on the same people).
Every person is tested exactly once. The train / test column of the split file is not used.

Model: per fold a completely new model from the other 4 folds only (about 97 people, ridge_5s_train.py):
    mean and spread of every number over its training videos -> z-scores, cut at +-4
    -> penalty strength by an inner person-grouped 5-fold CV over its training people (best held-out AUC)
    -> ridge weights fitted on all its training videos (one weight per number, no bias; target lie +1, truth -1)
    -> tested once on the people of its own fold (score > 0 = lie)

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_10s_cv5.py
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_10s_cv5.py --seconds 15 --out_dir results_ridge_15s_cv5   # other slice lengths

Results in results_ridge_10s_cv5/:
    summary.csv            all 242 out-of-fold scores together: every metric with person-resampled 95% intervals; per-fold accuracy mean, sd
    per_fold.csv           per fold: people and videos, chosen strength, inner-CV AUC, training and test accuracy, test AUC, pair accuracy,
                           confusion counts, the wrong videos
    predictions.csv        every video's score from the model of its fold (no video is scored by a model that saw its person)
    folds.csv              every person's fold
    fold_01 ... fold_05/   weights.npz          the fold's model: mean, spread, weight (1,152 each), penalty, seconds = 10 (format of
                                                ridge_5s_train.py; to score new videos with it:
                                                ridge_5s_train.py --predict a.csv --out_dir results_ridge_10s_cv5/fold_01)
                           penalty_choice.csv   its inner CV: held-out AUC, accuracy, error per strength; fit per strength
    weights_all_folds.csv  one row per number (1,152): the weight of every fold's model, their average, spread and sign agreement
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

SECONDS = 10                                    # slice length; the only difference to ridge_5s_cv5.py
FOLDS = 5


def run(split_file, out_dir, workers, seconds, folds_n):
    os.makedirs(out_dir, exist_ok=True)
    log = model.Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    ids, persons, y, _, paths = data.read_split(split_file)
    xs, lasts, _ = data.load_series(paths, workers, log)
    too_long = [ids[k] for k in np.where(lasts // 2 >= data.FRAMES)[0]]
    if too_long:
        raise SystemExit(f"[error] longer than {data.FRAMES / data.RATE:.0f} s: {', '.join(too_long)}")

    width, slices = data.slice_shape(seconds)
    X = np.stack([data.slice_levels(x, seconds) for x in xs]).astype(np.float32)   # slices x 48 numbers per video
    names = data.number_names(seconds)
    assert X.shape[1] == len(names) == slices * len(data.SIGNALS)
    folds = list(GroupKFold(folds_n).split(np.zeros((len(y), 1)), groups=persons))
    fold_of = np.zeros(len(y), int)
    for f, (_, te) in enumerate(folds, 1):
        fold_of[te] = f

    log(f"ridge on the full 240 s of every video ({slices} slices of {seconds} s, {width} frames each = {slices * width / data.RATE:.1f} s of "
        f"{data.FRAMES / data.RATE:.0f} s), {folds_n}-fold person-grouped CV")
    log(f"{len(ids)} videos, {len(set(persons))} people; {slices} slices x {len(data.SIGNALS)} signals = {X.shape[1]} numbers per video; "
        f"folds of {min(len(set(persons[te])) for _, te in folds)}-{max(len(set(persons[te])) for _, te in folds)} people; every fold trains a "
        f"new model on the other {folds_n - 1} folds and tests it on its own people ({time.time() - t0:.0f}s)\n")

    S = np.zeros(len(y))
    per_fold, weights = [], []
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

        fd = os.path.join(out_dir, f"fold_{f:02d}")
        os.makedirs(fd, exist_ok=True)
        np.savez_compressed(os.path.join(fd, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                            signals=np.array(data.SIGNALS), seconds=seconds, rate=data.RATE, frames=data.FRAMES, clip=data.CLIP, fold=f,
                            train_people=np.array(sorted(set(persons[tr]))), test_people=np.array(test_people))
        data.write_csv(os.path.join(fd, "penalty_choice.csv"), table, list(table[0]), fmt="{:.6g}")

    # ---- all folds together
    accs, aucs = np.array([r["test_acc"] for r in per_fold]), np.array([r["test_auc"] for r in per_fold])
    res = model.all_metrics(y, S, persons)
    chosen = " ".join(f"{r['penalty']:g}" for r in per_fold)
    log(f"\nper fold: test acc {accs.mean():.3f} on average (sd {accs.std(ddof=1):.3f}, {accs.min():.3f}-{accs.max():.3f}), "
        f"test AUC {aucs.mean():.3f} (sd {aucs.std(ddof=1):.3f}, {aucs.min():.3f}-{aucs.max():.3f}); strengths chosen: {chosen}")
    log(f"all {len(y)} videos, each scored by the model of its fold (95% intervals resample people, {model.BOOTSTRAP} times):")
    log(model.metrics_line(f"full 240 s, {seconds} s slices, {folds_n}-fold", res))

    Wf = np.array(weights)
    w_mean, w_sd = Wf.mean(0), Wf.std(0, ddof=1)
    agree = np.maximum((Wf > 0).sum(0), (Wf < 0).sum(0))
    off = np.corrcoef(Wf)[~np.eye(folds_n, dtype=bool)]
    log(f"\nthe {folds_n} models' weights: correlation between two folds {off.min():.3f}-{off.max():.3f} (average {off.mean():.3f}); "
        f"numbers whose weight has the same sign in all {folds_n} folds: {int((agree == folds_n).sum())} of {Wf.shape[1]}")
    log("largest average weights (+ pushes towards lie), with their spread over the folds:")
    for k in np.argsort(-np.abs(w_mean))[:10]:
        log(f"  {w_mean[k]:+.5f} (sd {w_sd[k]:.5f}, same sign in {agree[k]}/{folds_n} folds)  slice {names[k][0]:2d} ({names[k][1]} s)  {names[k][2]}")

    # ---- save
    data.write_csv(os.path.join(out_dir, "summary.csv"),
                   [{"test": f"{folds_n}-fold CV, full 240 s, {seconds} s slices", "seconds": seconds, "frames_per_slice": width,
                     "slices": slices, "numbers": X.shape[1], "seconds_used": slices * width / data.RATE, "people": len(set(persons)),
                     "videos": len(y), "fold_acc_mean": float(accs.mean()), "fold_acc_sd": float(accs.std(ddof=1)),
                     "fold_auc_mean": float(aucs.mean()), "fold_auc_sd": float(aucs.std(ddof=1)), "penalties": chosen, **res}],
                   ["test", "seconds", "frames_per_slice", "slices", "numbers", "seconds_used", "people", "videos",
                    "fold_acc_mean", "fold_acc_sd", "fold_auc_mean", "fold_auc_sd", "penalties"] + model.FIELDS)
    data.write_csv(os.path.join(out_dir, "per_fold.csv"), per_fold, list(per_fold[0]))
    data.write_csv(os.path.join(out_dir, "predictions.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "fold": int(fold_of[i]), "score": float(S[i]),
                     "pred": int(S[i] > 0), "correct": int((S[i] > 0) == y[i])} for i in range(len(ids))],
                   ["video_id", "person", "label", "fold", "score", "pred", "correct"])
    data.write_csv(os.path.join(out_dir, "folds.csv"), [{"person": p, "fold": int(fold_of[persons == p][0])} for p in sorted(set(persons))],
                   ["person", "fold"])
    wcols = [f"weight_fold{f:02d}" for f in range(1, folds_n + 1)]
    data.write_csv(os.path.join(out_dir, "weights_all_folds.csv"),
                   [{"position": k, "slice": names[k][0], "seconds": names[k][1], "signal": names[k][2],
                     **{c: float(Wf[j, k]) for j, c in enumerate(wcols)}, "weight_mean": float(w_mean[k]), "weight_sd": float(w_sd[k]),
                     "same_sign_folds": int(agree[k])} for k in range(Wf.shape[1])],
                   ["position", "slice", "seconds", "signal"] + wcols + ["weight_mean", "weight_sd", "same_sign_folds"], fmt="{:.8g}")
    log(f"\nresults in {out_dir}: summary.csv, per_fold.csv, predictions.csv, folds.csv, fold_01 ... fold_{folds_n:02d}/ (weights.npz, "
        f"penalty_choice.csv), weights_all_folds.csv, log.txt ({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description="Ridge on the full 240 s of every video, 10 s slices, 5-fold person-grouped CV")
    p.add_argument("--seconds", type=int, default=SECONDS, help=f"slice length in seconds (default {SECONDS}: 24 slices of 175 frames)")
    p.add_argument("--folds", type=int, default=FOLDS, help=f"number of folds (default {FOLDS})")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", help="results folder (default results_ridge_<seconds>s_cv<folds>)")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    out_dir = args.out_dir or os.path.join(HERE, f"results_ridge_{args.seconds}s_cv{args.folds}")
    run(args.split, out_dir, max(1, args.workers), args.seconds, args.folds)


if __name__ == "__main__":
    sys.exit(main())
