#!/usr/bin/env python
"""
Ridge lie detector on 5 s slices of LibreFace results: the model, its training (mean, spread, penalty and weights), the test, and scoring new
videos. The data (LibreFace CSVs -> 2,256 numbers per video, the train / test split, the z-scoring) come from ridge_5s_data.py in this folder.
Same model and results as ridge_5s_model.py.

Model, everything fitted on the training videos only
    mean, spread   of every one of the 2,256 numbers over the training videos (mean_spread); every video's numbers are z-scored with them
                   and cut at +-4 (ridge_5s_data.zscore)
    weights        ridge: one weight per number, no bias; target lie +1, truth -1; the weights are the exact minimiser of
                   squared error + penalty x sum of weights^2 (ridge_weights)
    penalty        the strength (10 candidates) with the best AUC in an inner person-grouped 5-fold CV over the training people
                   (pick_strength); then the weights are fitted on all training videos with it (fit)
    score          sum of z-scored number x weight; score > 0 = lie

Split (split_90_31.csv): the first 90 shuffled people (180 videos) train, the other 31 people (62 videos) test, once.

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_data.py --make_split            # write split_90_31.csv (once)
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_train.py                         # train on the 90, test on the 31, save
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_5s_train.py --predict a.csv b.csv   # score new 240 s videos

Needs numpy, pandas and scikit-learn.

Results in results_ridge_5s/:
    weights.npz           mean, spread, weight (2,256 each), penalty and settings: everything --predict needs
    weights.csv           one row per number: slice, seconds, signal, training mean, spread, weight
    penalty_choice.csv    per penalty strength: held-out (inner-CV) AUC, accuracy and error, the numbers the strength is chosen by; and its fit
                          on all training videos: training error, sum of weights^2, loss = error + lambda / videos x sum of weights^2
    predictions_test.csv  score and prediction of every test video
    predictions_all.csv   score and prediction of every video, training videos too (theirs are in-sample: the weights were fitted on them)
    summary.csv           test and training metrics (lie = positive class) with person-resampled 95% intervals, confusion counts, pair accuracy
    log.txt
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402

OUT_DIR = os.path.join(HERE, "results_ridge_5s")
STRENGTHS = (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300)
INNER_FOLDS = 5
BOOTSTRAP, BOOT_SEED = 2000, 0


# =====================================================================================
# 1. Model: mean, spread, ridge weights, penalty
# =====================================================================================
def mean_spread(X):
    """Mean and spread of every number over the videos X (the training videos); a number without spread gets spread 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    return np.nan_to_num(mu), np.where(np.isfinite(sd) & (sd > 1e-8), sd, 1.0)


def ridge_weights(A, t, strengths=STRENGTHS):
    """Ridge weights for every penalty strength on the z-scored numbers A (videos x numbers), target t, no bias. Dual form:
    w = A^T (A A^T + r I)^-1 t, exact, and fast for many more numbers (2,256) than videos (180)."""
    K = A @ A.T
    w, U = np.linalg.eigh(K)
    w = np.maximum(w, 0)
    Ut = U.T @ t
    scale = max(w.sum() / len(A), 1e-9)                 # the penalty is relative to the average squared length of a row
    return [A.T @ (U @ (Ut / (w + r * scale))) for r in strengths]


def pick_strength(X, y, persons, tr):
    """Inner person-grouped 5-fold CV over the training videos tr: AUC of every penalty strength; returns the best strength, all AUCs and the
    held-out scores [strengths, training videos] (every training video scored by weights fitted without its person). Each fold z-scores with
    the mean and spread of its own 4/5."""
    scores = np.zeros((len(STRENGTHS), len(tr)))
    for a, b in GroupKFold(INNER_FOLDS).split(np.zeros((len(tr), 1)), groups=persons[tr]):
        mu, sd = mean_spread(X[tr[a]])
        A = data.zscore(X[tr[a]], mu, sd)
        Zb = data.zscore(X[tr[b]], mu, sd)
        for k, coef in enumerate(ridge_weights(A, 2.0 * y[tr[a]] - 1)):
            scores[k, b] = Zb @ coef
    aucs = [float(roc_auc_score(y[tr], s)) for s in scores]
    best = int(np.argmax(aucs))
    return STRENGTHS[best], aucs, scores


def fit(X, y, persons, tr):
    """Mean, spread, penalty and weights, all from the training videos tr only."""
    strength, aucs, inner_scores = pick_strength(X, y, persons, tr)
    mu, sd = mean_spread(X[tr])
    coef = ridge_weights(data.zscore(X[tr], mu, sd), 2.0 * y[tr] - 1, (strength,))[0]
    return {"mean": mu, "spread": sd, "weight": coef, "penalty": strength, "inner_auc": max(aucs), "inner_aucs": aucs, "inner_scores": inner_scores}


def strength_table(X, y, tr, m):
    """One row per penalty strength: the held-out numbers of the inner CV (AUC is what the strength is chosen by; accuracy and error are shown
    too) and the fit of that strength on all training videos. The ridge minimises  sum (score - target)^2 + lambda x sum of weights^2  with
    lambda = strength x the average squared length of a video's z-scored numbers; shown per video:  loss = error + lambda / videos x sum w^2."""
    t = 2.0 * y[tr] - 1
    A = data.zscore(X[tr], m["mean"], m["spread"])
    scale = float((A ** 2).sum(1).mean())
    rows = []
    for r, auc, held_out, w in zip(STRENGTHS, m["inner_aucs"], m["inner_scores"], ridge_weights(A, t)):
        lam, sum_sq = r * scale, float((w ** 2).sum())
        err = float(np.mean((A @ w - t) ** 2))
        rows.append({"penalty": r, "lambda": lam, "inner_cv_auc": auc, "inner_cv_acc": float(np.mean((held_out > 0) == y[tr])),
                     "inner_cv_error": float(np.mean((held_out - t) ** 2)), "train_error": err, "weights_sum_sq": sum_sq,
                     "train_loss": err + lam / len(tr) * sum_sq, "chosen": int(r == m["penalty"])})
    return rows


def score(X, m):
    """Score of every video: its numbers z-scored with the training mean and spread, times the weights; > 0 = lie."""
    return data.zscore(X, m["mean"], m["spread"]) @ m["weight"]


def save_model(out_dir, m, persons, tr, te):
    """weights.npz (what --predict needs) and weights.csv (mean, spread and weight of every number, readable)."""
    np.savez_compressed(os.path.join(out_dir, "weights.npz"), mean=m["mean"], spread=m["spread"], weight=m["weight"], penalty=m["penalty"],
                        signals=np.array(data.SIGNALS), seconds=data.SECONDS, rate=data.RATE, frames=data.FRAMES, clip=data.CLIP,
                        train_people=np.array(sorted(set(persons[tr]))), test_people=np.array(sorted(set(persons[te]))))
    names = data.number_names()
    data.write_csv(os.path.join(out_dir, "weights.csv"),
                   [{"position": k, "slice": names[k][0], "seconds": names[k][1], "signal": names[k][2], "train_mean": float(m["mean"][k]),
                     "train_spread": float(m["spread"][k]), "weight": float(m["weight"][k])} for k in range(len(names))],
                   ["position", "slice", "seconds", "signal", "train_mean", "train_spread", "weight"], fmt="{:.8g}")


# =====================================================================================
# 2. Metrics (lie = positive class), 95% intervals resampling people
# =====================================================================================
INTERVAL_KEYS = ("acc", "balanced_acc", "precision", "recall", "specificity", "npv", "f1", "macro_f1", "mcc", "kappa", "auc", "avg_precision")
FIELDS = (["tp", "fp", "tn", "fn"] + [f for k in ("acc", "balanced_acc", "precision", "recall", "specificity", "npv", "f1") for f in (k, f"{k}_low", f"{k}_high")]
          + ["f1_truth", "macro_f1", "macro_f1_low", "macro_f1_high", "mcc", "mcc_low", "mcc_high", "kappa", "kappa_low", "kappa_high",
             "auc", "auc_low", "auc_high", "avg_precision", "avg_precision_low", "avg_precision_high", "pair_acc"])


def _div(a, b):
    return a / b if b else float("nan")


def point_metrics(y, s):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(s) > 0).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    recall, spec = _div(tp, tp + fn), _div(tn, tn + fp)
    f1, f1_truth = _div(2 * tp, 2 * tp + fp + fn), _div(2 * tn, 2 * tn + fn + fp)
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    both = 0 < y.mean() < 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "acc": po, "balanced_acc": (recall + spec) / 2, "precision": _div(tp, tp + fp), "recall": recall,
            "specificity": spec, "npv": _div(tn, tn + fn), "f1": f1, "f1_truth": f1_truth, "macro_f1": (f1 + f1_truth) / 2,
            "mcc": _div(tp * tn - fp * fn, np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))),
            "kappa": _div(po - pe, 1 - pe),
            "auc": float(roc_auc_score(y, s)) if both else float("nan"),
            "avg_precision": float(average_precision_score(y, s)) if both else float("nan")}


def pair_accuracy(persons, y, s):
    """Share of people whose lie video scores above their truth video."""
    d = {}
    for p, label, v in zip(persons, y, s):
        d.setdefault(p, {})[int(label)] = v
    pairs = [v[1] > v[0] for v in d.values() if len(v) == 2]
    return float(np.mean(pairs)) if pairs else float("nan")


def all_metrics(y, s, persons):
    """Point value of every metric plus <metric>_low / <metric>_high: 95% interval from resampling people (both videos of a person together)."""
    y, s, persons = np.asarray(y).astype(int), np.asarray(s, dtype=float), np.asarray(persons)
    r = point_metrics(y, s)
    rng = np.random.default_rng(BOOT_SEED)
    names = np.unique(persons)
    idx_of = {g: np.where(persons == g)[0] for g in names}
    boot = {k: [] for k in INTERVAL_KEYS}
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([idx_of[g] for g in rng.choice(names, len(names))])
        q = point_metrics(y[idx], s[idx])
        for k in INTERVAL_KEYS:
            boot[k].append(q[k])
    for k in INTERVAL_KEYS:
        v = np.array(boot[k], dtype=float)
        lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).any() else (np.nan, np.nan)
        r[f"{k}_low"], r[f"{k}_high"] = float(lo), float(hi)
    r["pair_acc"] = pair_accuracy(persons, y, s)
    return r


def metrics_line(name, r):
    return (f"  {name:24s} acc {r['acc']:.3f} ({r['acc_low']:.3f}-{r['acc_high']:.3f})  AUC {r['auc']:.3f} ({r['auc_low']:.3f}-{r['auc_high']:.3f})  "
            f"precision {r['precision']:.3f}  recall {r['recall']:.3f}  specificity {r['specificity']:.3f}  F1 {r['f1']:.3f}  MCC {r['mcc']:.3f}  "
            f"pair {r['pair_acc']:.3f}  [TP {r['tp']} FP {r['fp']} TN {r['tn']} FN {r['fn']}]")


class Log:
    def __init__(self, path):
        self.fh = open(path, "w", encoding="utf8")

    def __call__(self, text=""):
        print(text, flush=True)
        self.fh.write(text + "\n")
        self.fh.flush()


# =====================================================================================
# 3. Train on the training people, test once on the test people, save
# =====================================================================================
def train(split_file, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    log = Log(os.path.join(out_dir, "log.txt"))
    t0 = time.time()
    d = data.load_dataset(split_file, workers, log)
    ids, persons, y, part, X, lasts, no_face = d["ids"], d["persons"], d["y"], d["part"], d["X"], d["last_frame"], d["no_face"]
    tr, te = np.where(part == "train")[0], np.where(part == "test")[0]

    log(f"5 s slice ridge: {len(ids)} videos, {len(set(persons))} people; {data.SLICES} slices x {len(data.SIGNALS)} signals = {X.shape[1]} numbers "
        f"per video ({time.time() - t0:.0f}s)")
    log(f"video lengths {data.seconds_of(lasts.min()):.1f}-{data.seconds_of(lasts.max()):.1f} s; frames without a face (filled with the last value "
        f"before): {no_face.sum()} of {len(ids) * data.FRAMES} in {(no_face > 0).sum()} videos (check_cache_nans.py lists them)")
    log(f"split {os.path.basename(split_file)}: training {len(set(persons[tr]))} people ({len(tr)} videos, {int(y[tr].sum())} lies), "
        f"test {len(set(persons[te]))} people ({len(te)} videos, {int(y[te].sum())} lies)")

    # ---- mean, spread, penalty and weights from the training videos
    m = fit(X, y, persons, tr)
    table = strength_table(X, y, tr, m)
    log(f"\npenalty strength: each of the {len(STRENGTHS)} strengths gets an inner person-grouped {INNER_FOLDS}-fold CV over the {len(set(persons[tr]))} "
        f"training people (fit on 4/5 of them, score the other 1/5, {INNER_FOLDS} times:")
    log(f"a held-out score for every training video); the best held-out AUC is chosen. Then each strength is fitted on all {len(tr)} training videos:")
    log(f"loss = error + lambda / {len(tr)} x sum of weights^2, error = mean (score - target)^2 (target lie +1, truth -1), "
        f"lambda = strength x {table[0]['lambda'] / STRENGTHS[0]:.0f}")
    log(f"(the average squared length of a video's {X.shape[1]} z-scored numbers). Test videos are not used here.")
    log(f"  {'':19s}|{'held-out (inner CV)':^22s}|{'fit on all ' + str(len(tr)) + ' training videos':^30s}|")
    log(f"  {'strength':>8s} {'lambda':>9s} |{'AUC':>7s}{'acc':>7s}{'error':>8s}|{'error':>9s}{'sum w^2':>11s}{'loss':>10s}|")
    for r in table:
        log(f"  {r['penalty']:>8g} {r['lambda']:>9.1f} |{r['inner_cv_auc']:>7.3f}{r['inner_cv_acc']:>7.3f}{r['inner_cv_error']:>8.3f}|"
            f"{r['train_error']:>9.4f}{r['weights_sum_sq']:>11.5f}{r['train_loss']:>10.4f}|" + ("  <- chosen" if r["chosen"] else ""))

    # ---- test once
    s_tr, s_te = score(X[tr], m), score(X[te], m)
    res_te, res_tr = all_metrics(y[te], s_te, persons[te]), all_metrics(y[tr], s_tr, persons[tr])
    log(f"\nresults (lie = positive class; 95% intervals resample people, {BOOTSTRAP} times):")
    log(metrics_line(f"test, {len(set(persons[te]))} people", res_te))
    log(metrics_line(f"training, {len(set(persons[tr]))} people", res_tr) + "   (videos the weights were fitted on)")

    s_all = np.zeros(len(ids))
    s_all[tr], s_all[te] = s_tr, s_te
    log("\nscores of the test videos (score = sum of the 2,256 z-scored numbers x their weights; > 0 = lie):")
    for p in sorted(set(persons[te])):
        st, sl = (s_all[(persons == p) & (part == "test") & (y == c)][0] for c in (0, 1))
        log(f"  {p}   truth video {st:+.3f} {'right' if st <= 0 else 'WRONG'}   lie video {sl:+.3f} {'right' if sl > 0 else 'WRONG'}"
            f"   lie scores higher: {'yes' if sl > st else 'NO'}")

    names = data.number_names()
    top = np.argsort(-np.abs(m["weight"]))[:10]
    log("\nlargest weights (on z-scored numbers; + pushes towards lie):")
    for k in top:
        sl, sec, sig = names[k]
        log(f"  {m['weight'][k]:+.5f}  slice {sl:2d} ({sec} s)  {sig}")

    # ---- save
    save_model(out_dir, m, persons, tr, te)
    data.write_csv(os.path.join(out_dir, "penalty_choice.csv"), table, list(table[0]), fmt="{:.6g}")
    data.write_csv(os.path.join(out_dir, "predictions_all.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "split": part[i], "score": float(s_all[i]),
                     "pred": int(s_all[i] > 0), "correct": int((s_all[i] > 0) == y[i])} for i in range(len(ids))],
                   ["video_id", "person", "label", "split", "score", "pred", "correct"])
    data.write_csv(os.path.join(out_dir, "predictions_test.csv"),
                   [{"video_id": ids[i], "person": persons[i], "label": int(y[i]), "score": float(s_te[k]), "pred": int(s_te[k] > 0),
                     "correct": int((s_te[k] > 0) == y[i])} for k, i in enumerate(te)],
                   ["video_id", "person", "label", "score", "pred", "correct"])
    fields = ["set", "people", "videos", "penalty"] + FIELDS
    data.write_csv(os.path.join(out_dir, "summary.csv"),
                   [{"set": "test", "people": len(set(persons[te])), "videos": len(te), "penalty": m["penalty"], **res_te},
                    {"set": "training (in-sample)", "people": len(set(persons[tr])), "videos": len(tr), "penalty": m["penalty"], **res_tr}],
                   fields)
    log(f"\nresults in {out_dir}: weights.npz, weights.csv, penalty_choice.csv, predictions_test.csv, predictions_all.csv, summary.csv, log.txt "
        f"({time.time() - t0:.0f}s)")


# =====================================================================================
# 4. Score new videos with the saved weights
# =====================================================================================
def predict(paths, out_dir, out_path):
    """Score LibreFace CSVs (same columns as D:/librefaceresults, frame numbers of the 35 fps video) of 240 s videos of the same task."""
    wpath = os.path.join(out_dir, "weights.npz")
    if not os.path.exists(wpath):
        raise SystemExit(f"[error] no {wpath}; train first")
    W = np.load(wpath, allow_pickle=False)
    rows = []
    for path in paths:
        v = data.video_series(path)
        sec = data.seconds_of(v["last_frame"])
        if not 0.95 * data.FRAMES / data.RATE <= sec <= 1.05 * data.FRAMES / data.RATE:
            raise SystemExit(f"[error] {path}: {sec:.0f} s of video; the model only fits 240 s videos of the same task")
        numbers = data.slice_levels(v["series"], float(W["seconds"]))[None].astype(np.float32)   # the model's slice length (5 s here)
        s = float((data.zscore(numbers, W["mean"], W["spread"]) @ W["weight"])[0])
        no_face = int((~v["has_row"]).sum())
        rows.append({"file": path, "score": s, "pred": "lie" if s > 0 else "truth", "frames_without_face": no_face})
        print(f"{os.path.basename(path):24s} score {s:+.3f} -> {rows[-1]['pred']:5s}  ({no_face} of {data.FRAMES} frames without a face, filled)",
              flush=True)
    if out_path:
        data.write_csv(out_path, rows, ["file", "score", "pred", "frames_without_face"])
        print(f"written to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Ridge lie detector on 5 s slices of LibreFace results: train and test, or score new videos "
                                            "(the split is made by ridge_5s_data.py --make_split)")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    p.add_argument("--predict", nargs="+", metavar="CSV", help="LibreFace CSVs of new 240 s videos to score with the saved weights")
    p.add_argument("--predictions_out", help="also write the scores of --predict to this CSV")
    args = p.parse_args()
    if args.predict:
        predict(args.predict, args.out_dir, args.predictions_out)
    else:
        train(args.split, args.out_dir, max(1, args.workers))


if __name__ == "__main__":
    sys.exit(main())
