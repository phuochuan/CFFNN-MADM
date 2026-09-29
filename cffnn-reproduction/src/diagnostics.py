"""Audits of printed observations and explicitly named hidden-layer hypotheses."""

from dataclasses import asdict
from itertools import product
import numpy as np

from .aggregation import printed_hwa_product, cffssthwa
from .madm import HiddenConfig, hidden_layer, components


def audit_table3(raw, tolerance=1e-4):
    """Check domain and symmetry without treating targets as model inputs."""
    invalid = []
    for i, row in enumerate(raw):
        for j, (mm, mp, nm, np_) in enumerate(row):
            for label, a, b in (("magnitude", mm, nm), ("phase", mp, np_)):
                cube_sum = a**3+b**3
                if cube_sum > 1+1e-12:
                    invalid.append(dict(alternative=f"HIC{i+1}", criterion=f"LIS{j+1}",
                                        branch=label, cube_sum=cube_sum,
                                        rounding_lower_bound=max(0,a-tolerance)**3+max(0,b-tolerance)**3))
    arr = np.asarray(raw)
    return dict(invalid_constraints=invalid,
                unequal_membership_pairs=int(np.count_nonzero(np.abs(arr[:,:,0]-arr[:,:,1]) > 2*tolerance)),
                hic1_lis1_symmetry_error_lower_bound=float(max(abs(arr[0,0,0]-arr[0,0,1]),
                                                               abs(arr[0,0,2]-arr[0,0,3]))/2))


def _summary(label, compute, target):
    result = dict(hypothesis=label)
    try:
        arr = components(compute())
        error = np.abs(arr-np.asarray(target))
        result.update(status="PASS" if np.all(error <= 1e-4) else "MISMATCH",
                      max_absolute_error=float(error.max()),
                      mean_absolute_error=float(error.mean()),
                      matched_components=int(np.count_nonzero(error <= 1e-4)),
                      hic1_lis1=arr[0,0].tolist())
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        result.update(status="DOMAIN_ERROR", reason=str(exc))
    return result


def interpretation_sweep(matrices, target):
    """Predeclared hypotheses, no optimization/fitting against the table.

    q grid is used ONLY to probe sign ambiguity at the hidden checkpoint;
    these are not claims of reproducing Section 5.7 sensitivity scores.
    """
    rows = []
    parameters = (-1., -.6, -.5, -.4, -.3, -.2, -.1, 0., .1, .2, .3, .4, .5, .6, 1.)
    for interpretation, parameter, first_weight in product(
            ("literal", "sign_corrected", "fermatean"), parameters, (.25,.5,.75)):
        extensions = ("literal", "truncate") if parameter > 0 else ("reject",)
        for positive in extensions:
            config = HiddenConfig(parameter, (first_weight,1-first_weight), interpretation, positive)
            rows.append(_summary(asdict(config), lambda c=config: hidden_layer(matrices,c), target))
    # Thm.4.3.2 is algebraically different from the definition. No rho was supplied.
    for rho, theta, indexing in product((.5,1.,2.), (-.6,-.1,.1,.6,1.), ("repeated_first","indexed")):
        def compute(rho=rho, theta=theta, indexing=indexing):
            return tuple(tuple(printed_hwa_product([m[i][j] for m in matrices],
                                                    rho=rho,theta=theta,indexing=indexing)
                               for j in range(4)) for i in range(4))
        rows.append(_summary(dict(expression="Thm.4.3.2",rho=rho,theta=theta,indexing=indexing),compute,target))
    # Conventional hybrid is not the printed HWA. Explicit sorting hypotheses
    # use magnitude-only or component-average scores, never claim to be Def.2.3.
    for mode, first_weight, ordered_weight, key_name in product(
            ("sign_corrected","fermatean"), (.25,.5,.75), (.25,.5,.75),
            ("magnitude_difference", "four_component_difference")):
        key = ((lambda x: x.mu_magnitude**3-x.nu_magnitude**3)
               if key_name == "magnitude_difference" else
               (lambda x: (x.mu_magnitude**3+x.mu_phase**3-x.nu_magnitude**3-x.nu_phase**3)/2))
        def compute(mode=mode, first_weight=first_weight, ordered_weight=ordered_weight, key=key):
            return tuple(tuple(cffssthwa([m[i][j] for m in matrices],
                                         (first_weight,1-first_weight), -.1,
                                         interpretation=mode, key=key,
                                         order_weights=(ordered_weight,1-ordered_weight))
                               for j in range(4)) for i in range(4))
        rows.append(_summary(dict(expression="conventional_hybrid_hypothesis", parameter=-.1,
                                  interpretation=mode, expert_weight=first_weight,
                                  ordered_weight=ordered_weight, ordering=key_name), compute, target))
    return rows
