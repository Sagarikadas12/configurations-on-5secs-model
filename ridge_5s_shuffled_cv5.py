#!/usr/bin/env python
"""
5 s slice ridge on the full video with the ORDER OF THE 5 s SLICES RANDOMLY SHUFFLED inside every video, tested by 5-fold person-grouped
cross-validation. Question: does the model need to know WHEN in the 240 s task something happens, or only WHAT happens?

Numbers of one video: exactly those of ridge_5s_cv5.py (ridge_5s_data.slice_levels: 47 slices of 88 frames, per slice and signal
(slice average - video average) / video spread -> 47 x 48 = 2,256 numbers), then the 47 slices are put in a random order, every slice keeping its
48 numbers together: e.g. the slice of 0-5 s goes to the place of 40-45 s, the slice of 40-45 s to the place of 115-120 s, ... Moving a slice does
not change its numbers (average and spread are those of the whole video, which do not depend on the order), so this is the same as cutting the
video into its 5 s pieces and putting them back in a random order.

    --mode per_video   (default) every video gets its OWN random order: slice position k no longer means the same time in different videos
    --mode same        one random order for ALL videos (a check: ridge does not care about the order of its numbers, so this must give the
                       same results as ridge_5s_cv5.py, only with the weights in the new order)
    --mode none        no shuffle (a check: must reproduce ridge_5s_cv5.py exactly)

Every seed (--seeds 10: seeds 0-9) is a new set of random orders; everything else is fixed. Folds: the same 5 person-grouped folds as
ridge_5s_cv5.py and the other 5-fold scripts. Model per fold: exactly that of ridge_5s_cv5.py (ridge_5s_train.py): mean and spread -> z-scores, cut
at +-4 -> penalty strength by an inner person-grouped 5-fold CV (best held-out AUC) -> ridge weights -> tested once on the fold's people.

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_shuffled_cv5.py
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_shuffled_cv5.py --mode same --seeds 1 --out_dir <folder>

Results in results_ridge_5s_shuffled_cv5/:
    summary_seeds.csv      one row per seed: every metric of the 242 out-of-fold scores with person-resampled 95% intervals, per-fold accuracy
                           mean and sd, penalties; then the mean and sd over the seeds
    log.txt
    seed_00 ... seed_09/   the same files as ridge_5s_cv5.py: summary.csv, per_fold.csv, predictions.csv, folds.csv,
                           fold_01 ... fold_05/ (weights.npz, penalty_choice.csv), weights_all_folds.csv (weights per position in the shuffled
                           order: a position is a different time in every video), and
                           permutations.csv    every video's order: pos_01 ... pos_47 = the original slice (1 = 0-5 s, ..., 47 = 231-236 s)
                                               that was put at that position
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

FOLDS, SEEDS = 5, 10
OUT_DIR = os.path.join(HERE, "results_ridge_5s_shuffled_cv5")
SUMMARY_KEYS = ["acc", "auc", "balanced_acc", "precision", "recall", "specificity", "f1", "mcc", "pair_acc"]


def shuffle_slices(X, mode, seed):
    """X [videos, slices x 48] -> the same numbers with the slices of every video in a random order, and the orders [videos, slices]
    (order[i, k] = the original slice put at position k of video i)."""
    n, slices, signals = len(X), data.SLICES, len(data.SIGNALS)
    X3 = X.reshape(n, slices, signals)
    rng = np.random.default_rng(seed)
    if mode == "per_video":
        order = np.stack([rng.permutation(slices) for _ in range(n)])
    elif mode == "same":
        order = np.tile(rng.permutation(slices), (n, 1))
    else:
        order = np.tile(np.arange(slices), (n, 1))
    return X3[np.arange(n)[:, None], order].reshape(n, slices * signals), order


def cross_validate(X, y, persons, ids, folds, out_dir, label, log):
    """ridge_5s_cv5.py's fold loop and files on the numbers X."""
    os.makedirs(out_dir, exist_ok=True)
    folds_n = len(folds)
    fold_of = np.zeros(len(y), int)
    for f, (_, te) in enumerate(folds, 1):
        fold_of[te] = f
    S = np.zeros(len(y))
    per_fold, weights = [], []
    for f, (tr, te) in enumerate(folds, 1):
        m = model.fit(X, y, persons, tr)
        table = model.strength_table(X, y, tr, m)
        s, s_tr = model.score(X[te], m), model.score(X[tr], m)
        S[te] = s
        weights.append(m["weight"])
        q = model.point_metrics(y[te], s)
        test_people = sorted(set(persons[te]))
        wrong = [ids[i] for k, i in enumerate(te) if (s[k] > 0) != (y[i] == 1)]
        per_fold.append({"fold": f, "train_people": len(set(persons[tr])), "train_videos": len(tr), "test_people": len(test_people),
                         "test_videos": len(te), "penalty": m["penalty"], "inner_cv_auc": m["inner_auc"],
                         "train_acc": float(np.mean((s_tr > 0) == y[tr])), "test_acc": q["acc"], "test_right": int(q["tp"] + q["tn"]),
                         "test_auc": q["auc"], "pair_acc": model.pair_accuracy(persons[te], y[te], s),
                         "tp": q["tp"], "fp": q["fp"], "tn": q["tn"], "fn": q["fn"], "wrong_videos": "; ".join(wrong), "people": " ".join(test_people)})
        fd = os.path.join(out_dir, f"fold_{f:02d}")
        os.makedirs(fd, exist_ok=True)
        np.savez_compressed(os.path.join(fd, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                            signals=np.array(data.SIGNALS), seconds=data.SECONDS, rate=data.RATE, frames=data.FRAMES, clip=data.CLIP, fold=f,
                            train_people=np.array(sorted(set(persons[tr]))), test_people=np.array(test_people))
        data.write_csv(os.path.join(fd, "penalty_choice.csv"), table, list(table[0]), fmt="{:.6g}")

    accs, aucs = np.array([r["test_acc"] for r in per_fold]), np.array([r["test_auc"] for r in per_fold])
    res = model.all_metrics(y, S, persons)
    chosen = " ".join(f"{r['penalty']:g}" for r in per_fold)
    log(f"  {label}: per fold acc {accs.mean():.3f} (sd {accs.std(ddof=1):.3f}), AUC {aucs.mean():.3f}; strengths {chosen}")
    log("  " + model.metrics_line(label, res).strip())
    row = {"test": f"{folds_n}-fold CV, full 240 s, {label}", "slices": data.SLICES, "numbers": X.shape[1], "people": len(set(persons)),
           "videos": len(y), "fold_acc_mean": float(accs.mean()), "fold_acc_sd": float(accs.std(ddof=1)),
           "fold_auc_mean": float(aucs.mean()), "fold_auc_sd": float(aucs.std(ddof=1)), "penalties": chosen, **res}
    data.write_csv(os.path.join(out_dir, "summary.csv"), [row],
                   ["test", "slices", "numbers", "people", "videos", "fold_acc_mean", "fold_acc_sd", "fold_auc_mean", "fold_auc_sd", "penalties"]
                   + model.FIELDS)
    data.write_csv(os.path.join(out_dir, "per_fold.csv"), per_fold, list(per_fold[0]))
    data.write_csv(os.path.join(out_dir, "predictions.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "fold": int(fold_of[i]), "score": float(S[i]),
                     "pred": int(S[i] > 0), "correct": int((S[i] > 0) == y[i])} for i in range(len(ids))],
                   ["video_id", "person", "label", "fold", "score", "pred", "correct"])
    data.write_csv(os.path.join(out_dir, "folds.csv"), [{"person": p, "fold": int(fold_of[persons == p][0])} for p in sorted(set(persons))],
                   ["person", "fold"])
    Wf = np.array(weights)
    w_mean, w_sd = Wf.mean(0), Wf.std(0, ddof=1)
    agree = np.maximum((Wf > 0).sum(0), (Wf < 0).sum(0))
    names = data.number_names()                  # position k of the shuffled order (a different time in every video when shuffled per video)
    wcols = [f"weight_fold{f:02d}" for f in range(1, folds_n + 1)]
    data.write_csv(os.path.join(out_dir, "weights_all_folds.csv"),
                   [{"position": k, "slot": names[k][0], "slot_seconds": names[k][1], "signal": names[k][2],
                     **{c: float(Wf[j, k]) for j, c in enumerate(wcols)}, "weight_mean": float(w_mean[k]), "weight_sd": float(w_sd[k]),
                     "same_sign_folds": int(agree[k])} for k in range(Wf.shape[1])],
                   ["position", "slot", "slot_seconds", "signal"] + wcols + ["weight_mean", "weight_sd", "same_sign_folds"], fmt="{:.8g}")
    return row


def run(split_file, out_dir, workers, mode, seeds):
    os.makedirs(out_dir, exist_ok=True)
    log = model.Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    d = data.load_dataset(split_file, workers, log)                        # 2,256 numbers per video, as ridge_5s_cv5.py
    ids, persons, y, X = d["ids"], d["persons"], d["y"], d["X"]
    folds = list(GroupKFold(FOLDS).split(np.zeros((len(y), 1)), groups=persons))
    log(f"5 s slice ridge on the full 240 s, slices shuffled: mode {mode}, seeds 0-{seeds - 1}; {len(ids)} videos, {len(set(persons))} people, "
        f"{data.SLICES} slices x {len(data.SIGNALS)} signals = {X.shape[1]} numbers per video; {FOLDS}-fold person-grouped CV "
        f"(the folds of ridge_5s_cv5.py) ({time.time() - t0:.0f}s)\n")
    rows = []
    for seed in range(seeds):
        Xs, order = shuffle_slices(X, mode, seed)
        sd = os.path.join(out_dir, f"seed_{seed:02d}")
        moved = float(np.mean(order != np.arange(data.SLICES)))
        log(f"seed {seed}: {moved:.0%} of the slices are at another position than in the video")
        row = cross_validate(Xs.astype(np.float32), y, persons, ids, folds, sd, f"shuffled {mode}, seed {seed}", log)
        data.write_csv(os.path.join(sd, "permutations.csv"),
                       [{"video_id": ids[i], **{f"pos_{k + 1:02d}": int(order[i, k]) + 1 for k in range(data.SLICES)}} for i in range(len(ids))],
                       ["video_id"] + [f"pos_{k + 1:02d}" for k in range(data.SLICES)])
        rows.append({"seed": seed, "mode": mode, **{k: row[k] for k in ("fold_acc_mean", "fold_acc_sd", "penalties")},
                     **{f: row[f] for k in SUMMARY_KEYS for f in ((k, f"{k}_low", f"{k}_high") if f"{k}_low" in row else (k,))}})
    fields = list(rows[0])
    num = [f for f in fields if f not in ("seed", "mode", "penalties")]
    mean = {"seed": "mean", "mode": mode, "penalties": "", **{f: float(np.mean([r[f] for r in rows])) for f in num}}
    sdv = {"seed": "sd", "mode": mode, "penalties": "", **{f: float(np.std([r[f] for r in rows], ddof=1)) if len(rows) > 1 else float("nan")
                                                          for f in num}}
    data.write_csv(os.path.join(out_dir, "summary_seeds.csv"), rows + [mean, sdv], fields)
    log(f"\nover {seeds} seeds: acc {mean['acc']:.3f} (sd {sdv['acc']:.3f}, {min(r['acc'] for r in rows):.3f}-{max(r['acc'] for r in rows):.3f}), "
        f"AUC {mean['auc']:.3f} (sd {sdv['auc']:.3f}, {min(r['auc'] for r in rows):.3f}-{max(r['auc'] for r in rows):.3f}), "
        f"MCC {mean['mcc']:.3f}, pair {mean['pair_acc']:.3f}")
    log(f"results in {out_dir}: summary_seeds.csv, log.txt, seed_00 ... seed_{seeds - 1:02d}/ ({time.time() - t0:.0f}s)")


def main():
    p = argparse.ArgumentParser(description="5 s slice ridge, full video, slice order randomly shuffled, 5-fold person-grouped CV")
    p.add_argument("--mode", choices=("per_video", "same", "none"), default="per_video",
                   help="per_video: own random order for every video (default); same: one order for all videos; none: no shuffle")
    p.add_argument("--seeds", type=int, default=SEEDS, help=f"number of random seeds, 0 ... seeds-1 (default {SEEDS})")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    run(args.split, args.out_dir, max(1, args.workers), args.mode, max(1, args.seeds))


if __name__ == "__main__":
    sys.exit(main())
