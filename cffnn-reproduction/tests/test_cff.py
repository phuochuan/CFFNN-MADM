from dataclasses import FrozenInstanceError
import pytest

from src.cff import CFFNumber
from data.paper_case import DECISION_MATRICES, TABLE1_RAW, TABLE2_RAW, CRITERION_WEIGHTS


def test_four_components_are_independent_and_immutable():
    x = CFFNumber(.1,.2,.3,.4)
    assert x.components == (.1,.2,.3,.4)
    with pytest.raises(FrozenInstanceError):
        x.mu_phase = .1


@pytest.mark.parametrize("components", [(-.1,0,0,0), (0,1.1,0,0),
    (float("nan"),0,0,0), (0,0,float("inf"),0), (.9,0,.9,0), (0,.9,0,.9)])
def test_invalid_components_rejected(components):
    with pytest.raises(ValueError):
        CFFNumber(*components)


@pytest.mark.parametrize("x", [(0,0,1,1), (1,1,0,0), (0,0,0,0)])
def test_boundaries(x):
    assert CFFNumber(*x).components == x


def test_transcription_preserves_unequal_phase_and_small_decimals():
    assert TABLE1_RAW[0][1] == (.05,.05,.1,.7)
    assert TABLE2_RAW[2][0] == (.024,.024,.028,.028)
    assert TABLE2_RAW[3][1] == (.0024,.0024,.0028,.0028)
    assert CRITERION_WEIGHTS == (.23,.24,.25,.28)
    assert len(DECISION_MATRICES) == 2
    assert all(len(t) == 4 and all(len(r) == 4 for r in t) for t in DECISION_MATRICES)
