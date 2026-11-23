#!/usr/bin/env python
"""
Control for ridge_skip40s_cv5.py: the same 40 slices of 40-240 s, but every signal normalised with the average and spread of the WHOLE video
(0-240 s), in 5 s slices, tested by 5-fold person-grouped cross-validation. Not a true "skip the first 40 s" model: the first 40 s are used for
the average and spread of every signal (nothing else).

Why: ridge_skip40s_cv5.py differs from the full video (0-240 s, same code: ridge_second_minute_cv5.py --start_seconds 0 --end_seconds 240) in
two ways at once: (1) the 8 slices of 0-40 s are gone and (2) average and spread come from 40-240 s instead of 0-240 s. This run keeps (2) as in
the full video and changes only (1); so
    full video (0-240 s)  vs  this run            = the effect of dropping the 8 slices of 0-40 s alone
    this run              vs  ridge_skip40s_cv5   = the effect of the normalisation window alone (0-240 s vs 40-240 s)

Everything is the same as ridge_second_minute_cv5.py (whose code is used, with norm="video"):
    numbers    frames 700-4,200 in 40 slices of 87-88 frames (the boundary k x 5 s is frame ceil(k x 87.5)); per slice and signal
               (slice average - average of the whole video) / spread of the whole video -> 40 x 48 = 1,920 numbers per video
    folds      the 121 people in 5 folds of 24 or 25 people (48-50 videos), both videos of a person in the same fold (sklearn GroupKFold over
               the video order of split_90_31.csv): the same folds as ridge_5s_cv5.py, ridge_skip40s_cv5.py and the other 5-fold scripts
    model      per fold a completely new model from the other 4 folds only (about 97 people): mean and spread -> z-scores, cut at +-4 ->
               penalty strength by an inner person-grouped 5-fold CV over its training people -> ridge weights -> tested once on its fold

    C:\\Users\\hiwi\\miniforge3\\envs\\envlie\\python.exe -I -X utf8 ridge_skip40s_videonorm_cv5.py

Results in results_ridge_skip40s_videonorm_cv5/ (the same files as ridge_second_minute_cv5.py; weights.npz also holds norm = "video")
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # python -I leaves this folder off the import path

import ridge_5s_data as data                    # noqa: E402
import ridge_second_minute_cv5 as sm            # noqa: E402

SKIP_SECONDS = 40                               # the slices of the first 40 s are left out ...
END_SECONDS = int(data.FRAMES / data.RATE)      # ... up to the end of the video (240 s)
NORM = "video"                                  # ... but average and spread come from the whole video
FOLDS = 5
OUT_DIR = os.path.join(HERE, "results_ridge_skip40s_videonorm_cv5")


def main():
    p = argparse.ArgumentParser(description="Ridge on the 40 slices of 40-240 s, normalised over the whole video, 5-fold person-grouped CV")
    p.add_argument("--skip_seconds", type=int, default=SKIP_SECONDS, help=f"start whose slices are left out, a multiple of 5 s (default {SKIP_SECONDS})")
    p.add_argument("--split", default=data.SPLIT_FILE, help=f"split file: its videos, people and labels (default {data.SPLIT_FILE})")
    p.add_argument("--out_dir", default=OUT_DIR, help=f"results folder (default {OUT_DIR})")
    p.add_argument("--workers", type=int, default=4, help="processes for reading raw CSVs into cache/ the first time (default 4)")
    args = p.parse_args()
    sm.run(args.split, args.out_dir, max(1, args.workers), args.skip_seconds, END_SECONDS, FOLDS, NORM)


if __name__ == "__main__":
    sys.exit(main())
