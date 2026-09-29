from itertools import product
import pytest

from data.paper_case import DECISION_MATRICES, TABLE3_EXPECTED
from src.diagnostics import audit_table3
from src.madm import HiddenConfig, hidden_layer, checkpoint_errors


@pytest.mark.paper
@pytest.mark.parametrize("i,j,k",list(product(range(4),range(4),range(4))),
    ids=[f"HIC{i+1}-LIS{j+1}-{('mu_mag','mu_phase','nu_mag','nu_phase')[k]}"
         for i,j,k in product(range(4),range(4),range(4))])
def test_each_table3_component(diagnostic_hidden,i,j,k):
    # These are real failures, neither xfail nor replaced by expected values.
    assert diagnostic_hidden[i][j].components[k] == pytest.approx(TABLE3_EXPECTED[i][j][k],abs=1e-4,rel=0)


def test_input_membership_symmetry_makes_published_table_impossible(diagnostic_hidden):
    assert all(x.mu_magnitude == x.mu_phase for table in DECISION_MATRICES for row in table for x in row)
    assert all(x.mu_magnitude == x.mu_phase for row in diagnostic_hidden for x in row)
    audit = audit_table3(TABLE3_EXPECTED)
    assert audit["unequal_membership_pairs"] == 14
    assert audit["hic1_lis1_symmetry_error_lower_bound"] == pytest.approx(.18345,abs=1e-4,rel=0)


def test_table3_contains_violations_that_rounding_cannot_fix():
    audit = audit_table3(TABLE3_EXPECTED)
    assert len(audit["invalid_constraints"]) == 6
    assert len({(v["alternative"],v["criterion"]) for v in audit["invalid_constraints"]}) == 4
    assert all(v["rounding_lower_bound"] > 1 for v in audit["invalid_constraints"])


def test_repeated_expert_entries_must_stay_identical_under_idempotency(diagnostic_hidden):
    for i,j in [(1,3),(2,2),(3,2)]:
        assert DECISION_MATRICES[0][i][j] == DECISION_MATRICES[1][i][j]
        assert diagnostic_hidden[i][j].components == pytest.approx(
            DECISION_MATRICES[0][i][j].components,abs=1e-4,rel=0)


def test_literal_hidden_layer_reports_domain_failure():
    with pytest.raises(ValueError):
        hidden_layer(DECISION_MATRICES,HiddenConfig(interpretation="literal"))


def test_shape_validation(diagnostic_hidden):
    with pytest.raises(ValueError):
        hidden_layer([],HiddenConfig())
    with pytest.raises(ValueError):
        checkpoint_errors(diagnostic_hidden,[[[0,0,0,0]]])
