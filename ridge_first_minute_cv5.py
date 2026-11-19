#!/usr/bin/env python
"""
Ridge lie detector on the FIRST MINUTE of every video only, in 5 s slices, tested by 5-fold person-grouped cross-validation.

Everything is the same as ridge_first_minute_cv10.py (whose code is used, unchanged), only with 5 folds instead of 10:
    numbers    exactly the first 60 s = 1,050 frames, 12 slices of 87-88 frames; per slice and signal (slice average - first-minute average) /
               first-minute spread -> 12 x 48 = 576 numbers per video
    folds      the 121 people in 5 folds of 24 or 25 people (48-50 videos), both videos of a person in the same fold (sklearn GroupKFold over
               the video order of split_90_31.csv); every person is tested exactly once
    model      per fold a completely new model from the other 4 folds only (about 97 people): mean and spread -> z-scores, cut at +-4 ->
               penalty strength by an inner person-grouped 5-fold CV over its training people -> ridge weights -> tested once on its fold

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_first_minute_cv5.py

Results in results_ridge_first_minute_cv5/ (the same files as ridge_first_minute_cv10.py): summary.csv, per_fold.csv, predictions.csv,
folds.csv, fold_01 ... fold_05/ (weights.npz, penalty_choice.csv), weights_all_folds.csv, log.txt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_first_minute_cv10 as fm            # noqa: E402

FOLDS = 5
OUT_DIR = os.path.join(HERE, "results_ridge_first_minute_cv5")


def main():
    p = argparse.ArgumentParser(description="Ridge on the first minute of every video, 5 s slices, 5-fold person-grouped CV")
    p.add_argument("--first_seconds", type=int, default=fm.FIRST_SECONDS,
                   help=f"use only this start of every video, a multiple of 5 s (default {fm.FIRST_SECONDS})")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    fm.FOLDS = FOLDS                            # the only difference to ridge_first_minute_cv10.py
    fm.run(args.split, args.out_dir, max(1, args.workers), args.first_seconds)


if __name__ == "__main__":
    sys.exit(main())
