import pytest

from data.paper_case import DECISION_MATRICES
from src.madm import HiddenConfig, hidden_layer


@pytest.fixture(scope="session")
def diagnostic_hidden():
    return hidden_layer(DECISION_MATRICES, HiddenConfig())
