from __future__ import annotations

import argparse

from digital_brain.engine import DigitalBrain, make_text_stimulus


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Digital Brain cognitive cycles.")
    parser.add_argument("--database", default=".brain/brain.sqlite3")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--stimulus", default="Observe the environment and improve future behavior.")
    args = parser.parse_args()

    brain = DigitalBrain(args.database)
    try:
        for _ in range(max(1, args.cycles)):
            cycle = brain.cycle(make_text_stimulus(args.stimulus))
            print(cycle.response)
    finally:
        brain.close()


if __name__ == "__main__":
    main()
