"""
DR Landscape Experiment
-----------------------
Run DR algorithms across datasets and compute projection quality metrics.

Usage:
  python experiment.py                          # run all
  python experiment.py --datasets iris wine     # specific datasets
  python experiment.py --algorithms pca         # specific algorithms
  python experiment.py --rerun                  # ignore cache
  python experiment.py --jobs 4                 # parallel jobs
  python experiment.py --summary                # print results table
"""
import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from dr_landscape.runner import run_experiment
from dr_landscape.storage import results_to_dataframe
from dr_landscape.datasets import DATASETS
from dr_landscape.algorithms import ALGORITHMS


def main():
    parser = argparse.ArgumentParser(description="DR landscape experiment")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to run (default: all). Available: {list(DATASETS.keys())}")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help=f"Algorithms to run (default: all). Available: {list(ALGORITHMS.keys())}")
    parser.add_argument("--results-dir", default="results",
                        help="Directory for storing results (default: ./results)")
    parser.add_argument("--rerun", action="store_true",
                        help="Recompute even if cached results exist")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel jobs (default: 1). Use -1 for all CPUs")
    parser.add_argument("--summary", action="store_true",
                        help="Print a pivot table of results and exit")
    args = parser.parse_args()

    if args.summary:
        df = results_to_dataframe(args.results_dir)
        if df.empty:
            print("No results found. Run the experiment first.")
            return
        pivot = df.pivot_table(index=["dataset", "algorithm"], columns="metric", values="value")
        print(pivot.to_string())
        return

    run_experiment(
        datasets=args.datasets,
        algorithms=args.algorithms,
        results_dir=args.results_dir,
        rerun=args.rerun,
        n_jobs=args.jobs,
    )

    print("\nDone. Load results with:")
    print("  from dr_landscape.storage import results_to_dataframe")
    print("  df = results_to_dataframe()")


if __name__ == "__main__":
    main()
