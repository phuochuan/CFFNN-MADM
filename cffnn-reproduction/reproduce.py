"""Run the intermediate audit. A Table 3 mismatch exits nonzero by design."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

from data.paper_case import (ALTERNATIVES, CRITERIA, DECISION_MATRICES, TABLE3_EXPECTED,
                             CRITERION_WEIGHTS, FINAL_EXPECTED, SENSITIVITY_EXPECTED)
from src.diagnostics import audit_table3, interpretation_sweep
from src.madm import (HiddenConfig, ReproductionBlocked, hidden_layer, components,
                      output_layer, scalar_scores)


def run(report_dir):
    config = HiddenConfig()
    hidden = hidden_layer(DECISION_MATRICES, config)
    actual = components(hidden)
    sweep = interpretation_sweep(DECISION_MATRICES, TABLE3_EXPECTED)
    audit = audit_table3(TABLE3_EXPECTED)
    try:
        outputs = output_layer(hidden, TABLE3_EXPECTED, CRITERION_WEIGHTS, config)
        scalar_scores(outputs)
    except ReproductionBlocked as exc:
        reason = str(exc)
    else:
        raise AssertionError("A scalar score specification is still required")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(status="BLOCKED_AT_TABLE3", diagnostic_config=asdict(config), reason=reason,
                  table3_actual=actual.tolist(), table3_expected=TABLE3_EXPECTED,
                  audit=audit, interpretation_sweep=sweep,
                  final_actual=None, final_expected=FINAL_EXPECTED,
                  ranking_actual=None, sensitivity_actual=None,
                  sensitivity_expected=SENSITIVITY_EXPECTED)
    (report_dir/"reproduction.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n",encoding="utf-8")
    lines = ["# Numerical reproduction report", "", "**Table 3 FAIL; downstream BLOCKED.**", "",
             "The actual values below use a diagnostic assumption, not a claimed paper baseline:",
             f"`{asdict(config)}`", "", reason, "",
             "| Checkpoint | Paper | Code | Absolute difference | Status |",
             "|---|---:|---:|---:|---|"]
    names = ("mu magnitude","mu phase","nu magnitude","nu phase")
    for i, alt in enumerate(ALTERNATIVES):
        for j, criterion in enumerate(CRITERIA):
            for k, name in enumerate(names):
                expected = TABLE3_EXPECTED[i][j][k]
                delta = abs(actual[i,j,k]-expected)
                lines.append(f"| Table 3 {alt}/{criterion} {name} | {expected:.4f} | {actual[i,j,k]:.9f} | {delta:.9f} | {'PASS' if delta <= 1e-4 else 'FAIL'} |")
    for i, expected in enumerate(FINAL_EXPECTED):
        lines.append(f"| Z{i+1} | {expected:.4f} | not computed | N/A | BLOCKED |")
    lines += ["", "Table 4, scalar scores, ranking, and Section 5.7 sensitivity were not evaluated.",
              "", f"Interpretation sweep: {len(sweep)} hypotheses; {dict(Counter(r['status'] for r in sweep))}.",
              "Full assumptions and domain failures are in `reproduction.json`.", "",
              "Invalid Table 3 componentwise Fermatean constraints:", "",
              "| Cell | Branch | mu^3 + nu^3 | Lower bound allowing 1e-4 on both components |",
              "|---|---|---:|---:|"]
    for item in audit["invalid_constraints"]:
        lines.append(f"| {item['alternative']}/{item['criterion']} | {item['branch']} | {item['cube_sum']:.9f} | {item['rounding_lower_bound']:.9f} |")
    (report_dir/"reproduction.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("Diagnostic hidden-layer configuration:", asdict(config))
    print("HIC1/LIS1 actual:", ", ".join(f"{x:.9f}" for x in actual[0,0]))
    print("HIC1/LIS1 expected:", TABLE3_EXPECTED[0][0])
    print(reason)
    print("Interpretations:", dict(Counter(r["status"] for r in sweep)))
    print("Ranking and sensitivity: BLOCKED (no final scores calculated)")
    print("Reports:", report_dir.resolve())
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parent/"reports")
    args = parser.parse_args()
    raise SystemExit(run(args.report_dir))
