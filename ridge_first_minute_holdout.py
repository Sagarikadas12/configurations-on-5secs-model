#!/usr/bin/env python
"""
Ridge lie detector on the FIRST MINUTE of every video only, in 5 s slices: trained once on the training people of a held-out split and tested
once on its test people (default split_91_30.csv: P000-P090 train, the next 30 people P091-P121 test).

Numbers of one video: the same as ridge_first_minute_cv10.py (its functions are used): exactly the first 60 s = 1,050 frames, 12 slices of
87-88 frames; per slice and signal  (slice average - first-minute average) / first-minute spread  -> 12 x 48 = 576 numbers. Nothing after the
first minute is used.

Model: the same as ridge_5s_train.py, fitted on the training videos only (the test videos are not used until the end):
    mean and spread of every number over the training videos -> z-scores, cut at +-4
    -> penalty strength by an inner person-grouped 5-fold CV over the training people (best held-out AUC)
    -> ridge weights fitted on all training videos (one weight per number, no bias; target lie +1, truth -1)
    -> tested once on the test people (score > 0 = lie)

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_first_minute_holdout.py
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_first_minute_holdout.py --split split_90_31.csv --out_dir results_ridge_first_minute_90_31

Results in results_ridge_first_minute_91_30/:
    weights.npz           mean, spread, weight (576 each), penalty, first_frames, slice_bounds
    weights.csv           one row per number: slice, seconds, signal, training mean, spread, weight
    penalty_choice.csv    per penalty strength: inner-CV AUC, accuracy and error; fit on all training videos
    predictions_test.csv  score and prediction of every test video
    predictions_all.csv   score and prediction of every video (training videos in-sample)
    summary.csv           test and training metrics (lie = positive class) with person-resampled 95% intervals
    log.txt
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_5s_train as model                  # noqa: E402
import ridge_first_minute_cv10 as fm            # noqa: E402  (first-minute slices: slice_bounds, first_part_levels, first_part_names)

SPLIT_FILE = os.path.join(HERE, "split_91_30.csv")
OUT_DIR = os.path.join(HERE, "results_ridge_first_minute_91_30")


def run(split_file, out_dir, workers, first_seconds):
    os.makedirs(out_dir, exist_ok=True)
    log = model.Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    ids, persons, y, part, paths = data.read_split(split_file)
    xs, lasts, _ = data.load_series(paths, workers, log)
    too_long = [ids[k] for k in np.where(lasts // 2 >= data.FRAMES)[0]]
    if too_long:
        raise SystemExit(f"[error] longer than {data.FRAMES / data.RATE:.0f} s: {', '.join(too_long)}")

    bounds = fm.slice_bounds(first_seconds)
    n_frames, slices, widths = int(bounds[-1]), len(bounds) - 1, np.diff(bounds)
    X = np.stack([fm.first_part_levels(x, bounds) for x in xs]).astype(np.float32)
    tr, te = np.where(part == "train")[0], np.where(part == "test")[0]

    log(f"ridge on the first {n_frames / data.RATE:.1f} s of every video ({slices} slices of {data.SECONDS} s, {widths.min()}-{widths.max()} frames "
        f"each, = {n_frames} of {data.FRAMES} frames; the rest of the video is not used), held-out split")
    log(f"{len(ids)} videos, {len(set(persons))} people; {slices} slices x {len(data.SIGNALS)} signals = {X.shape[1]} numbers per video "
        f"({time.time() - t0:.0f}s)")
    log(f"split {os.path.basename(split_file)}: training {len(set(persons[tr]))} people ({len(tr)} videos, {int(y[tr].sum())} lies; "
        f"{min(persons[tr])}-{max(persons[tr])}), test {len(set(persons[te]))} people ({len(te)} videos, {int(y[te].sum())} lies; "
        f"{min(persons[te])}-{max(persons[te])})")

    # ---- mean, spread, penalty and weights from the training videos only
    m = model.fit(X, y, persons, tr)
    table = model.strength_table(X, y, tr, m)
    log(f"\npenalty strength: each of the {len(model.STRENGTHS)} strengths gets an inner person-grouped {model.INNER_FOLDS}-fold CV over the "
        f"{len(set(persons[tr]))} training people; the best held-out AUC is chosen, then the weights are fitted on all {len(tr)} training videos.")
    log(f"loss = error + lambda / {len(tr)} x sum of weights^2, lambda = strength x {table[0]['lambda'] / model.STRENGTHS[0]:.0f}. "
        f"Test videos are not used here.")
    log(f"  {'':19s}|{'held-out (inner CV)':^22s}|{'fit on all ' + str(len(tr)) + ' training videos':^30s}|")
    log(f"  {'strength':>8s} {'lambda':>9s} |{'AUC':>7s}{'acc':>7s}{'error':>8s}|{'error':>9s}{'sum w^2':>11s}{'loss':>10s}|")
    for r in table:
        log(f"  {r['penalty']:>8g} {r['lambda']:>9.1f} |{r['inner_cv_auc']:>7.3f}{r['inner_cv_acc']:>7.3f}{r['inner_cv_error']:>8.3f}|"
            f"{r['train_error']:>9.4f}{r['weights_sum_sq']:>11.5f}{r['train_loss']:>10.4f}|" + ("  <- chosen" if r["chosen"] else ""))

    # ---- test once
    s_tr, s_te = model.score(X[tr], m), model.score(X[te], m)
    res_te, res_tr = model.all_metrics(y[te], s_te, persons[te]), model.all_metrics(y[tr], s_tr, persons[tr])
    log(f"\nresults (lie = positive class; 95% intervals resample people, {model.BOOTSTRAP} times):")
    log(model.metrics_line(f"test, {len(set(persons[te]))} people", res_te))
    log(model.metrics_line(f"training, {len(set(persons[tr]))} people", res_tr) + "   (videos the weights were fitted on)")

    s_all = np.zeros(len(ids))
    s_all[tr], s_all[te] = s_tr, s_te
    log(f"\nscores of the test videos (score = sum of the {X.shape[1]} z-scored numbers x their weights; > 0 = lie):")
    for p in sorted(set(persons[te])):
        st, sl = (s_all[(persons == p) & (part == "test") & (y == c)][0] for c in (0, 1))
        log(f"  {p}   truth video {st:+.3f} {'right' if st <= 0 else 'WRONG'}   lie video {sl:+.3f} {'right' if sl > 0 else 'WRONG'}"
            f"   lie scores higher: {'yes' if sl > st else 'NO'}")

    names = fm.first_part_names(bounds)
    log("\nlargest weights (on z-scored numbers; + pushes towards lie):")
    for k in np.argsort(-np.abs(m["weight"]))[:10]:
        sl, sec, sig = names[k]
        log(f"  {m['weight'][k]:+.5f}  slice {sl:2d} ({sec} s)  {sig}")

    # ---- save
    np.savez_compressed(os.path.join(out_dir, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                        signals=np.array(data.SIGNALS), seconds=data.SECONDS, first_frames=n_frames, slice_bounds=bounds, rate=data.RATE,
                        frames=data.FRAMES, clip=data.CLIP, train_people=np.array(sorted(set(persons[tr]))),
                        test_people=np.array(sorted(set(persons[te]))))
    data.write_csv(os.path.join(out_dir, "weights.csv"),
                   [{"position": k, "slice": names[k][0], "seconds": names[k][1], "signal": names[k][2], "train_mean": float(m["mean"][k]),
                     "train_spread": float(m["spread"][k]), "weight": float(m["weight"][k])} for k in range(len(names))],
                   ["position", "slice", "seconds", "signal", "train_mean", "train_spread", "weight"], fmt="{:.8g}")
    data.write_csv(os.path.join(out_dir, "penalty_choice.csv"), table, list(table[0]), fmt="{:.6g}")
    data.write_csv(os.path.join(out_dir, "predictions_all.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "split": part[i], "score": float(s_all[i]),
                     "pred": int(s_all[i] > 0), "correct": int((s_all[i] > 0) == y[i])} for i in range(len(ids))],
                   ["video_id", "person", "label", "split", "score", "pred", "correct"])
    data.write_csv(os.path.join(out_dir, "predictions_test.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "score": float(s_te[k]), "pred": int(s_te[k] > 0),
                     "correct": int((s_te[k] > 0) == y[i])} for k, i in enumerate(te)],
                   ["video_id", "person", "label", "score", "pred", "correct"])
    fields = ["set", "people", "videos", "first_frames", "numbers", "penalty"] + model.FIELDS
    data.write_csv(os.path.join(out_dir, "summary.csv"),
                   [{"set": "test", "people": len(set(persons[te])), "videos": len(te), "first_frames": n_frames, "numbers": X.shape[1],
                     "penalty": m["penalty"], **res_te},
                    {"set": "training (in-sample)", "people": len(set(persons[tr])), "videos": len(tr), "first_frames": n_frames,
                     "numbers": X.shape[1], "penalty": m["penalty"], **res_tr}],
                   fields)
    log(f"\nresults in {out_dir}: weights.npz, weights.csv, penalty_choice.csv, predictions_test.csv, predictions_all.csv, summary.csv, log.txt "
        f"({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description="Ridge on the first minute of every video, 5 s slices: train on a split's training people, test once")
    p.add_argument("--split", default=SPLIT_FILE, help=f"split file (default {SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--first_seconds", type=int, default=fm.FIRST_SECONDS,
                   help=f"use only this start of every video, a multiple of 5 s (default {fm.FIRST_SECONDS})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    run(args.split, args.out_dir, max(1, args.workers), args.first_seconds)


if __name__ == "__main__":
    sys.exit(main())
