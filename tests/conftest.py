from __future__ import annotations

import pytest

from twin.config import Settings
from twin.engine import Engine

from . import fixtures


def make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        database_url=f"sqlite:///{tmp_path}/twin-test.db",
        auth_enabled=False,
        embeddings_backend="hashing",
        llm_api_key=None,
        log_json=False,
        log_level="WARNING",
    )
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def engine(settings):
    e = Engine(settings)
    yield e
    e.close()


@pytest.fixture
def seeded(engine):
    for span in fixtures.background_spans(60):
        engine.ingest(span)
    for span in fixtures.incident_spans():
        engine.ingest(span)
    return engine
