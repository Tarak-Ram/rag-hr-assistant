import os

import pytest

os.environ.setdefault("HRA_PROVIDER", "offline")
os.environ.setdefault("HRA_QDRANT_LOCATION", ":memory:")


@pytest.fixture(scope="session")
def assistant():
    from app.assistant import Assistant

    a = Assistant()
    a.ingest()
    return a
