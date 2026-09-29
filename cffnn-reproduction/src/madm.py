"""Hidden-layer hypotheses and hard checkpoint gates (Section 5, pp.6--9)."""

from dataclasses import dataclass
import numpy as np

from .aggregation import cffssthwa, cffsstwa, validate_weights


class ReproductionBlocked(RuntimeError):
    """A required published checkpoint or specification is missing."""


@dataclass(frozen=True)
class HiddenConfig:
    """Explicit diagnostic assumptions, NOT paper-specified baseline settings."""
    parameter: float = -0.1
    expert_weights: tuple = (0.5, 0.5)
    interpretation: str = "sign_corrected"
    positive: str = "reject"


def hidden_layer(matrices, config):
    """Cellwise fusion of Tables 1 and 2, one plausible 4x4 hidden topology.

    Pseudocode on p.7 instead collapses each alternative's criteria to one H_i;
    the paper does not specify how that produces its 4x4 Table 3.
    """
    shape = {(len(table), tuple(len(row) for row in table)) for table in matrices}
    if len(shape) != 1 or not matrices or not matrices[0] or not matrices[0][0]:
        raise ValueError("Decision matrices must have the same non-empty rectangular shape")
    n, widths = next(iter(shape))
    if len(set(widths)) != 1:
        raise ValueError("Decision matrices must be rectangular")
    validate_weights(matrices, config.expert_weights)
    return tuple(tuple(cffssthwa([table[i][j] for table in matrices], config.expert_weights,
                                config.parameter, interpretation=config.interpretation,
                                positive=config.positive)
                       for j in range(widths[0])) for i in range(n))


def components(matrix):
    return np.array([[x.components for x in row] for row in matrix], dtype=float)


def checkpoint_errors(hidden, expected):
    actual, expected = components(hidden), np.asarray(expected, dtype=float)
    if actual.shape != expected.shape:
        raise ValueError(f"Checkpoint shape mismatch: {actual.shape} vs {expected.shape}")
    if not np.isfinite(expected).all():
        raise ValueError("Checkpoint values must be finite")
    return np.abs(actual-expected)


def require_table3(hidden, expected, tolerance=1e-4):
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Tolerance must be finite and non-negative")
    errors = checkpoint_errors(hidden, expected)
    if np.any(errors > tolerance):
        raise ReproductionBlocked(
            f"Table 3 FAIL: {np.count_nonzero(errors > tolerance)}/{errors.size} components; "
            f"max absolute error={errors.max():.9f}. Output, score and sensitivity are blocked.")


def output_layer(hidden, expected_table3, weights, config, tolerance=1e-4):
    """p.9 CFFSSTWA, callable only after all Table 3 entries agree."""
    require_table3(hidden, expected_table3, tolerance)
    return tuple(cffsstwa(row, weights, config.parameter,
                         interpretation=config.interpretation, positive=config.positive)
                 for row in hidden)


def scalar_scores(outputs):
    """No real-valued score conversion is specified in the paper."""
    raise ReproductionBlocked("Definition 2.3 gives a complex score; a real ranking rule is unspecified")


def sensitivity_analysis(matrices, expected_table3, baseline_config, parameters):
    """Do not guess what q denotes or run sensitivity before baseline passes."""
    require_table3(hidden_layer(matrices, baseline_config), expected_table3)
    raise ReproductionBlocked("Sensitivity q is not mapped to Gamma, rho, or theta in Section 5.7")
