#!/usr/bin/env python
"""
Ridge lie detector on the FULL 240 s of every video, in 20 s slices, tested by 5-fold person-grouped cross-validation.

Everything is the same as ridge_10s_cv5.py (whose code is used, unchanged), only with 20 s slices instead of 10 s:
    numbers    ridge_5s_data.slice_levels with seconds=20: 12 slices of 350 frames (20 s x 17.5; 12 x 350 = 4,200 frames = the whole 240 s,
               no frame left over); per slice and signal (slice average - video average) / video spread, average and spread over the whole
               video -> 12 x 48 = 576 numbers per video
    folds      the 121 people in 5 folds of 24 or 25 people (48-50 videos), both videos of a person in the same fold (sklearn GroupKFold over
               the video order of split_90_31.csv): the same folds as ridge_5s_cv5.py, ridge_10s_cv5.py, ridge_15s_cv5.py and the minute scripts
    model      per fold a completely new model from the other 4 folds only (about 97 people): mean and spread -> z-scores, cut at +-4 ->
               penalty strength by an inner person-grouped 5-fold CV over its training people -> ridge weights -> tested once on its fold

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_20s_cv5.py

Results in results_ridge_20s_cv5/ (the same files as ridge_10s_cv5.py): summary.csv, per_fold.csv, predictions.csv, folds.csv,
fold_01 ... fold_05/ (weights.npz with seconds = 20, penalty_choice.csv), weights_all_folds.csv, log.txt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_10s_cv5 as base                    # noqa: E402

SECONDS = 20                                    # the only difference to ridge_10s_cv5.py
FOLDS = 5
OUT_DIR = os.path.join(HERE, "results_ridge_20s_cv5")


def main():
    p = argparse.ArgumentParser(description="Ridge on the full 240 s of every video, 20 s slices, 5-fold person-grouped CV")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    base.run(args.split, args.out_dir, max(1, args.workers), SECONDS, FOLDS)


if __name__ == "__main__":
    sys.exit(main())
