from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from twin.config import get_settings
from twin.db import build_engine, build_session_factory, init_schema
from twin.security import ROLES, ApiKeyManager


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Manage Twin API keys (uses TWIN_DATABASE_URL)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create", help="create a new API key")
    p_create.add_argument("--name", required=True)
    p_create.add_argument("--role", required=True, choices=list(ROLES))

    sub.add_parser("list", help="list keys")

    p_disable = sub.add_parser("disable", help="disable a key")
    p_disable.add_argument("key_id")

    p_enable = sub.add_parser("enable", help="re-enable a key")
    p_enable.add_argument("key_id")

    args = parser.parse_args()

    settings = get_settings()
    engine = build_engine(settings)
    init_schema(engine)
    manager = ApiKeyManager(build_session_factory(engine))

    if args.cmd == "create":
        row, full_key = manager.create(name=args.name, role=args.role)
        print(f"key_id: {row.key_id}")
        print(f"role:   {row.role}")
        print(f"api key (store it now, it is not retrievable later):")
        print(full_key)
    elif args.cmd == "list":
        for k in manager.list_keys():
            state = "disabled" if k["disabled"] else "active"
            print(f"{k['key_id']}  {k['role']:<9} {state:<9} {k['name']}")
    elif args.cmd == "disable":
        ok = manager.set_disabled(args.key_id, True)
        print("disabled" if ok else "key not found")
        return 0 if ok else 1
    elif args.cmd == "enable":
        ok = manager.set_disabled(args.key_id, False)
        print("enabled" if ok else "key not found")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
