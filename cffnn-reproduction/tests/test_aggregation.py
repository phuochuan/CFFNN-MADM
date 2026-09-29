from math import cbrt
import pytest

from src.cff import CFFNumber
from src.aggregation import cffsstwa, cffsstowa, cffssthwa, printed_hwa_product

A = CFFNumber(.2,.1,.4,.3)
B = CFFNumber(.6,.2,.3,.2)


@pytest.mark.parametrize("mode", ["sign_corrected","fermatean"])
def test_wa_matches_independent_weighted_generator_mean(mode):
    result = cffsstwa([A,B],[.3,.7],-1,interpretation=mode)
    out = []
    for i, (a,b) in enumerate(zip(A.components,B.components)):
        if i < 2:
            out.append(cbrt(1-1/(.3/(1-a**3)+.7/(1-b**3))))
        elif mode == "fermatean":
            out.append(cbrt(1/(.3/a**3+.7/b**3)))
        else:
            out.append(1/(.3/a+.7/b))
    assert result.components == pytest.approx(out,abs=1e-4,rel=0)


@pytest.mark.parametrize("mode", ["sign_corrected","fermatean"])
@pytest.mark.parametrize("operator", [cffsstwa,cffsstowa,cffssthwa])
def test_idempotency_when_sign_is_corrected(operator,mode):
    x = operator([A,A],[.23,.77],-.1,interpretation=mode)
    assert x.components == pytest.approx(A.components,abs=1e-4,rel=0)


@pytest.mark.parametrize("weights", [[],[1],[.3,.3],[-.1,1.1],[float("nan"),.5]])
def test_weight_validation(weights):
    with pytest.raises(ValueError):
        cffsstwa([A,B],weights,-.1)


def test_printed_definitions_do_not_invent_an_ordering_rule():
    options = dict(interpretation="sign_corrected")
    expected = cffsstwa([A,B],[.3,.7],-.1,**options)
    assert cffsstowa([A,B],[.3,.7],-.1,**options) == expected
    assert cffssthwa([A,B],[.3,.7],-.1,**options) == expected
    ordered = cffsstowa([A,B],[.3,.7],-.1,key=lambda x:x.mu_magnitude,**options)
    assert ordered.components == pytest.approx(cffsstwa([B,A],[.3,.7],-.1,**options).components,abs=1e-4,rel=0)


def test_explicit_hybrid_reduces_to_owa_for_uniform_input_weights():
    key = lambda x:x.mu_magnitude
    result = cffssthwa([A,B],[.5,.5],-.1,order_weights=[.2,.8],key=key,interpretation="sign_corrected")
    expected = cffsstowa([A,B],[.2,.8],-.1,key=key,interpretation="sign_corrected")
    assert result.components == pytest.approx(expected.components,abs=1e-4,rel=0)
    with pytest.raises(ValueError):
        cffssthwa([A,B],[.5,.5],-.1,order_weights=[.5,.5])


def test_printed_product_at_rho_one_repeated_index():
    # At rho=1 repeated L1/M1 produces mu=cbrt(1-(1-mu1^3)^n), nu=nu1^n.
    result = printed_hwa_product([A,B],rho=1,theta=.1)
    expected = (cbrt(1-(1-.2**3)**2),cbrt(1-(1-.1**3)**2),.4**2,.3**2)
    assert result.components == pytest.approx(expected,abs=1e-4,rel=0)
    assert abs(result.mu_magnitude-A.mu_magnitude) > 1e-4  # not idempotent


def test_printed_product_index_repair_is_explicit():
    result = printed_hwa_product([A,B],rho=1,theta=.1,indexing="indexed")
    expected = (cbrt(1-(1-.2**3)*(1-.6**3)),cbrt(1-(1-.1**3)*(1-.2**3)),.4*.3,.3*.2)
    assert result.components == pytest.approx(expected,abs=1e-4,rel=0)
