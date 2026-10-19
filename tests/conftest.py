from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CITRON_FIXTURES = FIXTURES_DIR / "citron"


@pytest.fixture
def citron_fixtures() -> Path:
    return CITRON_FIXTURES
