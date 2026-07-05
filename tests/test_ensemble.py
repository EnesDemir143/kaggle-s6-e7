import numpy as np
import pandas as pd
import pytest
import yaml

from kaggle_s6_e7.candidate_experiments import run_candidate_suite
from kaggle_s6_e7.ensemble import (
    apply_multipliers,
    blend_probabilities,
    consensus_correction,
    disagreement_rate,
    eligibility_reasons,
    search_multiplier_scales,
)


def test_blend_probabilities_normalizes_weights_and_rows():
    first = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    second = np.array([[0.6, 0.3, 0.1], [0.4, 0.4, 0.2]])

    blended = blend_probabilities([first, second], [3.0, 1.0])

    assert blended == pytest.approx(np.array([[0.75, 0.15, 0.10], [0.25, 0.625, 0.125]]))
    assert blended.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_blend_probabilities_rejects_incompatible_sources():
    with pytest.raises(ValueError, match="same shape"):
        blend_probabilities([np.ones((2, 3)), np.ones((3, 3))], [0.5, 0.5])


def test_multiplier_search_uses_only_supplied_oof_labels():
    proba = np.array(
        [
            [0.60, 0.35, 0.05],
            [0.40, 0.55, 0.05],
            [0.55, 0.05, 0.40],
        ]
    )
    labels = np.array(["at-risk", "fit", "unhealthy"])
    base = np.array([1.0, 1.0, 1.0])

    multipliers, metrics = search_multiplier_scales(labels, proba, base, [1.0, 1.5])

    assert multipliers == pytest.approx([1.0, 1.0, 1.5])
    assert metrics["balanced_accuracy"] == pytest.approx(1.0)
    assert apply_multipliers(proba, multipliers).tolist() == ["at-risk", "fit", "unhealthy"]


def test_multiplier_search_can_limit_candidates_to_explicit_scale_pairs():
    proba = np.array([[0.51, 0.49, 0.0], [0.51, 0.0, 0.49], [0.9, 0.05, 0.05]])
    labels = np.array(["fit", "unhealthy", "at-risk"])
    base = np.ones(3)

    multipliers, _ = search_multiplier_scales(
        labels,
        proba,
        base,
        [],
        scale_pairs=[[1.0, 1.0], [1.05, 1.0]],
    )

    assert multipliers == pytest.approx([1.0, 1.05, 1.0])


def test_consensus_correction_changes_only_joint_disagreements():
    base = np.array(["at-risk", "fit", "unhealthy", "at-risk"])
    left = np.array(["fit", "fit", "at-risk", "unhealthy"])
    right = np.array(["fit", "unhealthy", "at-risk", "fit"])

    corrected = consensus_correction(base, left, right)

    assert corrected.tolist() == ["fit", "fit", "at-risk", "at-risk"]
    assert disagreement_rate(base, corrected) == pytest.approx(0.5)


def test_eligibility_reports_all_failed_bounds():
    reasons = eligibility_reasons(
        distribution={"at-risk": 0.82, "fit": 0.07, "unhealthy": 0.11},
        distribution_bounds={"at-risk": [0.815, 0.819], "fit": [0.071, 0.073]},
        disagreement=0.0005,
        disagreement_bounds=[0.001, 0.015],
        oof_score=0.9480,
        min_oof_score=0.9482,
    )

    assert len(reasons) == 4
    assert any("at-risk" in reason for reason in reasons)
    assert any("fit" in reason for reason in reasons)
    assert any("disagreement" in reason for reason in reasons)
    assert any("OOF" in reason for reason in reasons)


def _write_source(root, name, oof, test, labels):
    source = root / name
    source.mkdir(parents=True)
    np.save(source / "oof_proba.npy", oof)
    np.save(source / "test_proba.npy", test)
    pd.DataFrame({"id": range(len(labels)), "y_true": labels}).to_csv(
        source / "oof_pred.csv", index=False
    )
    pd.DataFrame({"id": range(100, 100 + len(test)), "class": "at-risk"}).to_csv(
        source / "submission_argmax.csv", index=False
    )
    (source / "label_mapping.json").write_text(
        '{"at-risk": 0, "fit": 1, "unhealthy": 2}\n'
    )
    (source / "best_multipliers.json").write_text(
        '{"at-risk": 1.0, "fit": 1.0, "unhealthy": 1.0}\n'
    )


def test_candidate_suite_writes_submission_manifest_and_report(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    test = proba[:2]
    _write_source(source_root, "E002", proba, test, labels)
    _write_source(source_root, "E004", proba, test, labels)
    (source_root / "E004" / "best_multipliers.json").unlink()
    config = {
        "base_experiment": "E002",
        "experiments": {
            "E009": {
                "kind": "blend_tuned",
                "output_dir": "E009_example",
                "sources": {"E002": 0.75, "E004": 0.25},
                "multiplier_source": "E002",
                "scales": [1.0],
                "submission_name": "submission_tuned.csv",
                "eligibility": {},
            }
        },
    }
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output_root = tmp_path / "generated"

    report = run_candidate_suite(config_path, source_root, output_root)

    candidate = output_root / "E009_example"
    assert (candidate / "submission_tuned.csv").is_file()
    assert (candidate / "config.json").is_file()
    assert (candidate / "metrics.json").is_file()
    assert (candidate / "eligibility.json").is_file()
    assert (output_root / "eligibility_report.csv").is_file()
    assert report.loc[0, "experiment"] == "E009"
    assert bool(report.loc[0, "eligible"])


def test_candidate_suite_supports_a_dedicated_report_name(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.eye(3)
    _write_source(source_root, "E002", proba, proba[:2], labels)
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment": "E002",
                "report_stem": "e014_e015_eligibility_report",
                "experiments": {
                    "E014": {
                        "kind": "blend_tuned",
                        "output_dir": "E014_example",
                        "sources": {"E002": 1.0},
                        "multiplier_source": "E002",
                        "scale_pairs": [[1.0, 1.0]],
                        "submission_name": "submission.csv",
                    }
                },
            }
        )
    )
    output_root = tmp_path / "generated"

    run_candidate_suite(config_path, source_root, output_root)

    assert (output_root / "e014_e015_eligibility_report.csv").is_file()
    assert not (output_root / "eligibility_report.csv").exists()


def test_filtered_source_search_writes_no_submission_when_all_candidates_fail(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.eye(3)
    _write_source(source_root, "E002", proba, proba[:2], labels)
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment": "E002",
                "experiments": {
                    "E016": {
                        "kind": "source_tuned_filtered",
                        "output_dir": "E016_example",
                        "source": "E002",
                        "multiplier_source": "E002",
                        "candidates": [["A", 1.0025, 1.0025]],
                        "selection_filters": {
                            "min_oof_score": 1.1,
                            "disagreement_bounds": [0.1, 0.2],
                        },
                        "submission_name": "submission.csv",
                    }
                },
            }
        )
    )
    output_root = tmp_path / "generated"

    report = run_candidate_suite(config_path, source_root, output_root)

    candidate = output_root / "E016_example"
    assert not (candidate / "submission.csv").exists()
    assert (candidate / "metrics.json").is_file()
    assert not bool(report.loc[0, "eligible"])


def test_selective_margin_correction_selects_an_eligible_direction_rule(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["unhealthy", "at-risk", "fit"])
    base_oof = np.array([[0.51, 0.01, 0.48], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    alt_oof = np.array([[0.45, 0.01, 0.54], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    base_test = np.array([[0.51, 0.01, 0.48], [0.8, 0.1, 0.1]])
    alt_test = np.array([[0.45, 0.01, 0.54], [0.8, 0.1, 0.1]])
    _write_source(source_root, "E002", base_oof, base_test, labels)
    _write_source(source_root, "E004", alt_oof, alt_test, labels)
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment": "E002",
                "experiments": {
                    "E017": {
                        "kind": "selective_margin_correction",
                        "output_dir": "E017_example",
                        "base": "E002",
                        "alternative_sources": {
                            "E004": {"source": "E004", "multiplier_source": "E004"}
                        },
                        "directions": [["at-risk", "unhealthy"]],
                        "base_margin_quantiles": [1.0],
                        "alt_gain_quantiles": [0.0],
                        "min_alt_margins": [0.0],
                        "selection_filters": {
                            "min_oof_score": 0.0,
                            "disagreement_bounds": [0.0, 1.0],
                            "changed_test_rows": [1, 2],
                            "max_test_count_difference": {
                                "at-risk": 2,
                                "fit": 2,
                                "unhealthy": 2,
                            },
                        },
                        "submission_name": "submission.csv",
                    }
                },
            }
        )
    )
    output_root = tmp_path / "generated"

    report = run_candidate_suite(config_path, source_root, output_root)
    submission = pd.read_csv(output_root / "E017_example" / "submission.csv")

    assert bool(report.loc[0, "eligible"])
    assert submission["health_condition"].tolist() == ["unhealthy", "at-risk"]


def test_candidate_suite_rejects_misaligned_oof_ids(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.eye(3)
    _write_source(source_root, "E002", proba, proba[:2], labels)
    _write_source(source_root, "E004", proba, proba[:2], labels)
    oof = pd.read_csv(source_root / "E004" / "oof_pred.csv")
    oof["id"] = [3, 4, 5]
    oof.to_csv(source_root / "E004" / "oof_pred.csv", index=False)
    config_path = tmp_path / "candidates.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_experiment": "E002",
                "experiments": {
                    "E009": {
                        "kind": "blend_tuned",
                        "output_dir": "E009",
                        "sources": {"E002": 0.75, "E004": 0.25},
                        "multiplier_source": "E002",
                        "scales": [1.0],
                        "submission_name": "submission_tuned.csv",
                        "eligibility": {},
                    }
                },
            }
        )
    )

    with pytest.raises(ValueError, match="OOF IDs"):
        run_candidate_suite(config_path, source_root, tmp_path / "out")


def test_blend_scales_are_applied_on_top_of_e002_multipliers(tmp_path):
    source_root = tmp_path / "sources"
    labels = np.array(["at-risk", "fit", "unhealthy"])
    proba = np.eye(3)
    _write_source(source_root, "E002", proba, proba[:2], labels)
    _write_source(source_root, "E019", proba, proba[:2], labels)
    base = {"at-risk": 0.18923, "fit": 1.44445, "unhealthy": 1.36632}
    (source_root / "E002" / "best_multipliers.json").write_text(
        __import__("json").dumps(base)
    )
    config = {
        "base_experiment": "E002",
        "experiments": {
            "E021": {
                "kind": "blend_tuned",
                "output_dir": "E021",
                "sources": {"E002": 0.95, "E019": 0.05},
                "multiplier_source": "E002",
                "scale_pairs": [[1.005, 1.010]],
                "submission_name": "submission.csv",
            }
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    run_candidate_suite(path, source_root, tmp_path / "out")
    metrics = __import__("json").loads(
        (tmp_path / "out" / "E021" / "metrics.json").read_text()
    )
    assert metrics["multipliers"][0] == pytest.approx(base["at-risk"])
    assert metrics["multipliers"][1] != pytest.approx(1.005)
    assert (tmp_path / "out" / "E021" / "risk_summary.json").is_file()
