from __future__ import annotations

from . import engine


def run_high_level_flow() -> None:
    # High-level flow:
    # 1) Read CLI options from user
    # 2) Delegate detailed execution to engine
    # 3) Engine performs solving and writes run summary
    engine.main()


def main() -> None:
    run_high_level_flow()


if __name__ == "__main__":
    main()
