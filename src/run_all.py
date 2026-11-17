"""Run the four experiments, then draw the figures of the report.

Run:  python src/run_all.py [--fonts FOLDER]

Each script runs in its own Python process, exactly as when it is started by
hand, so the result files are the same as with separate runs. The figures of
the report come last because they are drawn from the result files of the
experiments. --fonts is passed on to report_figures.py. Most of the time goes
to E2b, which takes about 10 minutes.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent

# (name, script), in the order in which they run.
STEPS = [
    ("E1", "e1_compas.py"),
    ("E2a", "e2a_unemployment.py"),
    ("E2b", "e2b_electricity.py"),
    ("E3", "e3_rashomon.py"),
    ("Report figures", "report_figures.py"),
]


def duration(seconds):
    """A duration written in minutes and seconds."""
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds:02d} s"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fonts", help="folder that contains the Archivo font files, passed on to report_figures.py")
    args = parser.parse_args()

    durations = []
    for name, script in STEPS:
        command = [sys.executable, str(SRC / script)]
        if script == "report_figures.py" and args.fonts:
            command += ["--fonts", args.fonts]
        print(f"\n==== {name}: src/{script} ====", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command)
        if result.returncode != 0:
            sys.exit(f"{name} failed (exit code {result.returncode}); the steps after it were not run.")
        durations.append((name, time.perf_counter() - start))
        print(f"{name} took {duration(durations[-1][1])}", flush=True)

    print("\nDurations")
    for name, seconds in durations:
        print(f"  {name:<15} {duration(seconds)}")
    print(f"  {'Total':<15} {duration(sum(seconds for _, seconds in durations))}")


if __name__ == "__main__":
    main()
