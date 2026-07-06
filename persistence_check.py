"""Cross-process persistence proof for the digital twin (Section 8.1).

The moat is durable cross-run causal state. This demonstrates it the only way that
counts: one process writes the graph, a SEPARATE process reads it back and answers
a causal query (blast radius from the root cause) WITHOUT re-ingesting anything.

Usage:
    python persistence_check.py               # orchestrate both phases + assert
    python persistence_check.py write <db>    # phase 1: seed + persist
    python persistence_check.py read  <db>    # phase 2: fresh process, query only
"""
from __future__ import annotations

import os
import subprocess
import sys

from twin import Engine

DEFAULT_DB = "persist_demo.db"


def phase_write(db: str) -> None:
    engine = Engine(db_path=db)
    engine.seed(background=30)
    br = sorted(engine.store.blast_radius("A2"))
    print(f"[write] pid={os.getpid()} nodes={len(engine.store.all_nodes())} "
          f"blast_radius(A2)={br} audit_entries={len(engine.audit.entries())}")


def phase_read(db: str) -> None:
    # brand-new Engine object on the same DB file: no seed(), no ingest().
    engine = Engine(db_path=db)
    engine.load_or_seed()   # must report "loaded", not "seeded"
    nodes = engine.store.all_nodes()
    br = sorted(engine.store.blast_radius("A2"))
    root = engine.incident_narrative.root_cause_node if engine.incident_narrative else None
    print(f"[read ] pid={os.getpid()} boot_mode={engine.boot_mode} "
          f"nodes={len(nodes)} blast_radius(A2)={br} root_cause={root} "
          f"audit_chain_valid={engine.audit.verify_chain()}")
    assert engine.boot_mode == "loaded", "graph did not survive the restart"
    assert set(br) == {"A3", "A5", "A7"}, f"blast radius lost across restart: {br}"
    assert root == "A2", f"attribution lost across restart: {root}"
    assert engine.audit.verify_chain(), "audit chain broke across restart"
    print("PERSISTENCE OK")


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "write":
        phase_write(args[1] if len(args) > 1 else DEFAULT_DB)
    elif args and args[0] == "read":
        phase_read(args[1] if len(args) > 1 else DEFAULT_DB)
    else:
        db = DEFAULT_DB
        for p in (db, db + "-wal", db + "-shm"):
            if os.path.exists(p):
                os.remove(p)
        subprocess.run([sys.executable, __file__, "write", db], check=True)
        # separate process reads it back — the actual restart
        subprocess.run([sys.executable, __file__, "read", db], check=True)
        for p in (db, db + "-wal", db + "-shm"):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
