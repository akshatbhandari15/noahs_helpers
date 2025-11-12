import random

from core.runner import ArkRunner
from core.parse_args import parse_args


def main():
    args = parse_args()
    random.seed(args.seed)

    runner = ArkRunner(args.player, args.num_helpers, args.animals, args.time, args.ark)

    if args.gui:
        score, times = runner.run_gui()
    else:
        score, times = runner.run()

    print("RESULTS")
    print(f"{'#' * 20}")
    print(f"SCORE={score}")
    if len(times):
        print(f"TOTAL_TURN_TIME={sum(times):.4f}s")
        print(f"TURNS_PER_SECOND={1 / (sum(times) / len(times)):.0f}")
    else:
        print("TOTAL_TURN_TIME=-1")
        print("TURNS_PER_SECOND=-1")
    print(f"{'#' * 20}")


if __name__ == "__main__":
    main()
