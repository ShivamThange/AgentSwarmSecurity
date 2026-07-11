from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin.config import get_settings
from twin.engine import Engine


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prune twin nodes/edges/checkpoints older than N days "
                    "(audit log is never pruned). Run from cron.")
    parser.add_argument("--days", type=int,
                        help="retention window; defaults to "
                             "TWIN_RETENTION_DAYS")
    args = parser.parse_args()

    settings = get_settings()
    days = args.days or settings.retention_days
    if not days or days < 1:
        print("no retention window configured (pass --days or set "
              "TWIN_RETENTION_DAYS)", file=sys.stderr)
        return 2

    engine = Engine(settings)
    try:
        result = engine.run_retention(days, actor="retention-cli")
        print(f"pruned nodes={result['nodes']} edges={result['edges']} "
              f"checkpoints={result['checkpoints']} (older than {days}d)")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
