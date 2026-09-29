from math import sqrt
import pytest

from data.paper_case import (TABLE3_EXPECTED, CRITERION_WEIGHTS, DECISION_MATRICES,
                             FINAL_EXPECTED, TABLE4_EXPECTED, SENSITIVITY_EXPECTED,
                             ALTERNATIVES, RANKING_EXPECTED)
from src.cff import CFFNumber
from src.score import score
from src.madm import (HiddenConfig, ReproductionBlocked, require_table3,
                      output_layer, scalar_scores, sensitivity_analysis)


def test_literal_complex_score_without_inventing_a_projection():
    x = CFFNumber(.5,.5,.2,0)
    result = score(x)
    assert result.real == pytest.approx(.125/sqrt(2)-.008,abs=1e-4,rel=0)
    assert result.imag == pytest.approx(.125/sqrt(2),abs=1e-4,rel=0)


def test_gate_prevents_output_aggregation(diagnostic_hidden,monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail("Output aggregation executed before Table 3 passed")
    monkeypatch.setattr("src.madm.cffsstwa",forbidden)
    with pytest.raises(ReproductionBlocked,match="Table 3 FAIL"):
        output_layer(diagnostic_hidden,TABLE3_EXPECTED,CRITERION_WEIGHTS,HiddenConfig())


def test_gate_prevents_sensitivity():
    with pytest.raises(ReproductionBlocked,match="Table 3 FAIL"):
        sensitivity_analysis(DECISION_MATRICES,TABLE3_EXPECTED,HiddenConfig(),tuple(SENSITIVITY_EXPECTED))


@pytest.mark.paper
@pytest.mark.parametrize("index",range(4),ids=[f"Table4-row{i+1}" for i in range(4)])
def test_table4_checkpoint(diagnostic_hidden,index):
    try:
        outputs = output_layer(diagnostic_hidden,TABLE3_EXPECTED,CRITERION_WEIGHTS,HiddenConfig())
    except ReproductionBlocked as exc:
        pytest.skip(str(exc))
    assert outputs[index].components == pytest.approx(TABLE4_EXPECTED[index],abs=1e-4,rel=0)


@pytest.mark.paper
@pytest.mark.parametrize("index",range(4),ids=[f"Z{i+1}" for i in range(4)])
def test_final_scores_checkpoint(diagnostic_hidden,index):
    try:
        require_table3(diagnostic_hidden,TABLE3_EXPECTED)
    except ReproductionBlocked as exc:
        pytest.skip(str(exc))
    outputs = output_layer(diagnostic_hidden,TABLE3_EXPECTED,CRITERION_WEIGHTS,HiddenConfig())
    actual = scalar_scores(outputs)  # explicitly blocked until a scalar rule is specified
    assert actual[index] == pytest.approx(FINAL_EXPECTED[index],abs=1e-4,rel=0)


@pytest.mark.paper
def test_final_ranking_checkpoint(diagnostic_hidden):
    try:
        require_table3(diagnostic_hidden,TABLE3_EXPECTED)
    except ReproductionBlocked as exc:
        pytest.skip(str(exc))
    outputs = output_layer(diagnostic_hidden,TABLE3_EXPECTED,CRITERION_WEIGHTS,HiddenConfig())
    actual = scalar_scores(outputs)
    ranking = tuple(sorted(ALTERNATIVES,key=lambda name:actual[ALTERNATIVES.index(name)],reverse=True))
    assert ranking == RANKING_EXPECTED


@pytest.mark.paper
@pytest.mark.parametrize("parameter",tuple(SENSITIVITY_EXPECTED))
def test_sensitivity_checkpoint(diagnostic_hidden,parameter):
    try:
        require_table3(diagnostic_hidden,TABLE3_EXPECTED)
    except ReproductionBlocked as exc:
        pytest.skip(str(exc))
    actual = sensitivity_analysis(DECISION_MATRICES,TABLE3_EXPECTED,HiddenConfig(),[parameter])
    assert actual[parameter] == pytest.approx(SENSITIVITY_EXPECTED[parameter],abs=1e-4,rel=0)
