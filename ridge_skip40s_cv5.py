#!/usr/bin/env python
"""
Ridge lie detector on every video WITHOUT ITS FIRST 40 s (40-240 s), in 5 s slices, tested by 5-fold person-grouped cross-validation.

Everything is the same as ridge_second_minute_cv5.py (whose code is used, unchanged), only with the window 40-240 s instead of 60-120 s:
    numbers    exactly frames 700-4,200 (40 s x 17.5 to 240 s x 17.5, the end of the video) = 3,500 frames = 200 s, 40 slices of 87-88 frames
               (the same 5 s grid as the minute scripts: the boundary k x 5 s is frame ceil(k x 87.5)); per slice and signal
               (slice average - average of 40-240 s) / spread of 40-240 s -> 40 x 48 = 1,920 numbers per video.
               No frame of the first 40 s is used anywhere, not even for the average and spread.
    folds      the 121 people in 5 folds of 24 or 25 people (48-50 videos), both videos of a person in the same fold (sklearn GroupKFold over
               the video order of split_90_31.csv): the same folds as ridge_5s_cv5.py, the minute scripts and the slice-length scripts
    model      per fold a completely new model from the other 4 folds only (about 97 people): mean and spread -> z-scores, cut at +-4 ->
               penalty strength by an inner person-grouped 5-fold CV over its training people -> ridge weights -> tested once on its fold

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_skip40s_cv5.py
    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_skip40s_cv5.py --skip_seconds 60 --out_dir results_ridge_skip60s_cv5
                                                                       # skip another start: a multiple of 5 s

Results in results_ridge_skip40s_cv5/ (the same files as ridge_second_minute_cv5.py): summary.csv, per_fold.csv, predictions.csv,
folds.csv, fold_01 ... fold_05/ (weights.npz with start_frame, end_frame, slice_bounds; penalty_choice.csv), weights_all_folds.csv, log.txt
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_second_minute_cv5 as sm            # noqa: E402

SKIP_SECONDS = 40                               # the start of every video that is thrown away
END_SECONDS = int(data.FRAMES / data.RATE)      # 240 s: up to the end of the video
FOLDS = 5
OUT_DIR = os.path.join(HERE, "results_ridge_skip40s_cv5")


def main():
    p = argparse.ArgumentParser(description="Ridge on every video without its first 40 s (40-240 s), 5 s slices, 5-fold person-grouped CV")
    p.add_argument("--skip_seconds", type=int, default=SKIP_SECONDS, help=f"start of the video to leave out, a multiple of 5 s (default {SKIP_SECONDS})")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    sm.run(args.split, args.out_dir, max(1, args.workers), args.skip_seconds, END_SECONDS, FOLDS)


if __name__ == "__main__":
    sys.exit(main())
