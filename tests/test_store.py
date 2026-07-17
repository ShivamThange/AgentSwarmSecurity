from __future__ import annotations

from twin.engine import Engine
from twin.models import TwinEdge, TwinNode


def _mk(engine: Engine, node_id: str, ts: float = 0.0):
    engine.store.upsert_node(TwinNode(node_id=node_id, agent_id=node_id,
                                      trace_id="t", timestamp=ts or 1.0))


def test_graph_traversal_via_cte(engine: Engine):
    for n in ("a", "b", "c", "d", "e"):
        _mk(engine, n)
    for src, dst in (("a", "b"), ("b", "c"), ("b", "d"), ("e", "d")):
        engine.store.add_edge(TwinEdge(src=src, dst=dst))

    assert set(engine.store.blast_radius("a")) == {"b", "c", "d"}
    assert set(engine.store.upstream("d")) == {"a", "b", "e"}
    assert engine.store.propagation_path("a", "c") == ["a", "b", "c"]
    assert engine.store.propagation_path("a", "e") == []
    order = engine.store.topo_order({"a", "b", "c", "d", "e"})
    assert order.index("a") < order.index("b") < order.index("c")


def test_cycle_does_not_hang_traversal(engine: Engine):
    for n in ("x", "y", "z"):
        _mk(engine, n)
    engine.store.add_edge(TwinEdge(src="x", dst="y"))
    engine.store.add_edge(TwinEdge(src="y", dst="z"))
    engine.store.add_edge(TwinEdge(src="z", dst="x"))
    assert set(engine.store.blast_radius("x")) == {"y", "z"}
    order = engine.store.topo_order({"x", "y", "z"})
    assert set(order) == {"x", "y", "z"}


def test_counters_accumulate(engine: Engine):
    engine.store.incr_counters({"k1": 2})
    engine.store.incr_counters({"k1": 3, "k2": 1})
    got = engine.store.get_counters(["k1", "k2", "k3"])
    assert got == {"k1": 5.0, "k2": 1.0, "k3": 0.0}


def test_persistence_survives_engine_restart(tmp_path):
    from .conftest import make_settings
    from . import fixtures

    settings = make_settings(tmp_path)
    e1 = Engine(settings)
    for span in fixtures.incident_spans():
        e1.ingest(span)
    nodes_before = e1.store.node_count()
    audit_before = e1.audit.count()
    e1.close()

    e2 = Engine(settings)
    try:
        assert e2.store.node_count() == nodes_before
        assert e2.audit.count() == audit_before
        assert e2.audit.verify_chain()["valid"] is True
        assert e2.store.get_node("A2").drift.status.value == "flagged"
        assert set(e2.store.blast_radius("A2")) == {"A3", "A5", "A7"}
    finally:
        e2.close()
