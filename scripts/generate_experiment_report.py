"""Generate the final experiment decision report and its supporting figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "kaggle-s6e7-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = ROOT / "outputs" / "experiments"
FIG_ROOT = ROOT / "reports" / "figures" / "final_experiments"
REPORT_PATH = ROOT / "docs" / "final-experiment-report.md"
CLASSES = ["at-risk", "fit", "unhealthy"]
MAIN_IDS = ["E001", "E002", "E003", "E004", "E005", "E006", "E008"]
TUNED_IDS = ["E002", "E004", "E006", "E008"]

EXPERIMENT_INFO = {
    "E001": ("V1 baseline", "Median imputasyon + missing flag", "Referans"),
    "E002": ("V2-Core", "Missing count, ratio, interaction, outlier flag", "E001 karşılaştırması"),
    "E003": ("Gender/activity", "E002 + gender_activity interaction", "Interaction ablation"),
    "E004": ("Rule flags", "E002 + sekiz eşik flag'i", "Rule ablation"),
    "E005": ("Clipping", "E002 + %0,1/%99,9 clipping", "Outlier ablation"),
    "E006": ("Log ratios", "E002 + altı log1p ratio", "Dönüşüm ablation"),
    "E008": ("Sqrt balanced", "E002 + sqrt-balanced sample weight", "Sınıf dengesi"),
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def pct(value: float) -> str:
    return f"%{100 * value:.2f}".replace(".", ",")


def f6(value: float) -> str:
    return f"{value:.6f}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns, *frame.astype(str).values.tolist()]
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]

    def render(row: list[str]) -> str:
        cells = [value.replace("|", "\\|").ljust(widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    separator = "| " + " | ".join("---" for _ in columns) + " |"
    return "\n".join([render(columns), separator, *(render(row) for row in rows[1:])])


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIG_ROOT / name, dpi=180, bbox_inches="tight")
    plt.close()


def load_main_results() -> pd.DataFrame:
    rows = []
    for exp_id in MAIN_IDS:
        metrics = load_json(EXP_ROOT / exp_id / "metrics.json")
        fold = metrics["fold_summary"]
        rows.append(
            {
                "experiment": exp_id,
                "name": EXPERIMENT_INFO[exp_id][0],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "std": fold["balanced_accuracy"]["std"],
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["f1_macro"],
                "mcc": metrics["mcc"],
                "log_loss": metrics["log_loss"],
                "roc_auc": metrics["roc_auc_ovr_macro"],
                **{f"recall_{c}": metrics["class_recall"][c] for c in CLASSES},
                **{
                    f"pred_{c}": metrics["prediction_distribution"][c]["rate"]
                    for c in CLASSES
                },
                "best_iteration": fold["best_iteration"]["mean"],
                "features": int(fold["features_count"]["mean"]),
            }
        )
    return pd.DataFrame(rows)


def load_tuned_results(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for exp_id in TUNED_IDS:
        tuned = load_json(EXP_ROOT / exp_id / "metrics_tuned.json")
        multipliers = load_json(EXP_ROOT / exp_id / "best_multipliers.json")
        base = main.loc[main.experiment == exp_id].iloc[0]
        rows.append(
            {
                "experiment": exp_id,
                "argmax": base.balanced_accuracy,
                "tuned": tuned["balanced_accuracy"],
                "delta": tuned["balanced_accuracy"] - base.balanced_accuracy,
                "macro_f1": tuned["macro_f1"],
                **{f"recall_{c}": tuned["class_recall"][c] for c in CLASSES},
                **{f"pred_{c}": tuned["prediction_distribution"][c] for c in CLASSES},
                **{f"mult_{c}": multipliers[c] for c in CLASSES},
            }
        )
    return pd.DataFrame(rows)


def load_sweeps() -> pd.DataFrame:
    rows = []
    for path in sorted(EXP_ROOT.glob("SWEEP_*/metrics.json")):
        exp_dir = path.parent
        metrics = load_json(path)
        config = load_json(exp_dir / "config.json")
        overrides = config["model_overrides"]
        rows.append(
            {
                "experiment": exp_dir.name,
                "balanced_accuracy": metrics["balanced_accuracy"],
                "std": metrics["fold_summary"]["balanced_accuracy"]["std"],
                "macro_f1": metrics["f1_macro"],
                "learning_rate": overrides["learning_rate"],
                "num_leaves": overrides["num_leaves"],
                "min_child_samples": overrides["min_child_samples"],
                "reg_alpha": overrides["reg_alpha"],
                "reg_lambda": overrides["reg_lambda"],
                "subsample": overrides["subsample"],
                "colsample_bytree": overrides["colsample_bytree"],
                "best_iteration": metrics["fold_summary"]["best_iteration"]["mean"],
            }
        )
    return pd.DataFrame(rows)


def candidate_submissions(best_sweep: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = {
        "E001_argmax": EXP_ROOT / "E001" / "submission_argmax.csv",
        "E002_argmax": EXP_ROOT / "E002" / "submission_E002_argmax.csv",
        "E002_tuned": EXP_ROOT / "E002" / "submission_E002_tuned.csv",
        "E004_tuned": EXP_ROOT / "E004" / "submission_E004_tuned.csv",
        "E006_tuned": EXP_ROOT / "E006" / "submission_E006_tuned.csv",
        "E008_argmax": EXP_ROOT / "E008" / "submission_E008_argmax.csv",
        "E008_tuned": EXP_ROOT / "E008" / "submission_E008_tuned.csv",
        f"{best_sweep}_argmax": EXP_ROOT / best_sweep / "submission_argmax.csv",
    }
    predictions: dict[str, pd.Series] = {}
    rows = []
    for name, path in paths.items():
        frame = pd.read_csv(path)
        pred = frame["health_condition"]
        predictions[name] = pred
        counts = pred.value_counts(normalize=True)
        rows.append(
            {
                "candidate": name,
                "file": str(path.relative_to(ROOT)),
                "rows": len(frame),
                **{c: counts.get(c, 0.0) for c in CLASSES},
                "labels_valid": set(pred.unique()) <= set(CLASSES),
                "id_unique": frame["id"].is_unique,
            }
        )
    names = list(predictions)
    disagreement = pd.DataFrame(index=names, columns=names, dtype=float)
    for left in names:
        for right in names:
            disagreement.loc[left, right] = np.mean(
                predictions[left].to_numpy() != predictions[right].to_numpy()
            )
    return pd.DataFrame(rows), disagreement


def generate_figures(
    main: pd.DataFrame,
    tuned: pd.DataFrame,
    sweeps: pd.DataFrame,
    submissions: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    ordered = main.sort_values("balanced_accuracy")
    plt.figure(figsize=(9, 5))
    colors = ["#2ca02c" if x == "E008" else "#4c78a8" for x in ordered.experiment]
    plt.barh(ordered.experiment, ordered.balanced_accuracy, xerr=ordered["std"], color=colors)
    plt.xlim(0.87, 0.92)
    plt.xlabel("OOF balanced accuracy (hata çubuğu: fold std)")
    plt.title("Ana deneylerin balanced accuracy karşılaştırması")
    savefig("01_main_balanced_accuracy.png")

    recalls = main.set_index("experiment")[[f"recall_{c}" for c in CLASSES]]
    recalls.columns = CLASSES
    recalls.plot(kind="bar", figsize=(10, 5), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.ylim(0.78, 1.0)
    plt.ylabel("Recall")
    plt.xticks(rotation=0)
    plt.title("Ana deneylerde sınıf bazlı recall")
    savefig("02_main_class_recall.png")

    true_labels = pd.read_csv(EXP_ROOT / "E001" / "oof_pred.csv")["y_true"]
    true_rates = true_labels.value_counts(normalize=True).reindex(CLASSES)
    dist = main.set_index("experiment")[[f"pred_{c}" for c in CLASSES]]
    dist.columns = CLASSES
    dist.loc["Gerçek"] = true_rates
    dist.plot(kind="bar", stacked=True, figsize=(10, 5), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.ylabel("Oran")
    plt.xticks(rotation=0)
    plt.title("Gerçek hedef ve OOF tahmin dağılımları")
    savefig("03_oof_prediction_distribution.png")

    tune_plot = tuned.set_index("experiment")[["argmax", "tuned"]]
    tune_plot.plot(kind="bar", figsize=(9, 5), color=["#9ecae9", "#2ca02c"])
    plt.ylim(0.86, 0.96)
    plt.ylabel("OOF balanced accuracy")
    plt.xticks(rotation=0)
    plt.title("Class multiplier tuning öncesi ve sonrası")
    savefig("04_multiplier_gain.png")

    multipliers = tuned.set_index("experiment")[[f"mult_{c}" for c in CLASSES]]
    multipliers.columns = CLASSES
    multipliers.plot(kind="bar", figsize=(9, 5), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.ylabel("Normalize multiplier")
    plt.xticks(rotation=0)
    plt.title("Bulunan sınıf multiplier değerleri")
    savefig("05_multiplier_values.png")

    sweep_order = sweeps.sort_values("experiment")
    plt.figure(figsize=(12, 5))
    colors = ["#2ca02c" if x == sweeps.loc[sweeps.balanced_accuracy.idxmax(), "experiment"] else "#4c78a8" for x in sweep_order.experiment]
    plt.bar(sweep_order.experiment, sweep_order.balanced_accuracy, color=colors)
    plt.axhline(main.loc[main.experiment == "E002", "balanced_accuracy"].iloc[0], color="#e45756", linestyle="--", label="E002")
    plt.axhline(main.loc[main.experiment == "E001", "balanced_accuracy"].iloc[0], color="black", linestyle=":", label="E001")
    plt.ylim(0.8755, 0.8785)
    plt.xticks(rotation=70)
    plt.ylabel("OOF balanced accuracy")
    plt.title("20 LightGBM sweep trial sonucu")
    plt.legend()
    savefig("06_sweep_scores.png")

    folds = []
    for exp_id in MAIN_IDS:
        frame = pd.read_csv(EXP_ROOT / exp_id / "fold_metrics.csv")
        folds.append(frame.assign(experiment=exp_id))
    fold_frame = pd.concat(folds)
    pivot = fold_frame.pivot(index="experiment", columns="fold", values="balanced_accuracy")
    pivot.plot(kind="bar", figsize=(10, 5), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.ylim(0.87, 0.92)
    plt.ylabel("Balanced accuracy")
    plt.xticks(rotation=0)
    plt.title("Ana deneylerde fold bazlı skorlar")
    savefig("07_fold_stability.png")

    confusion = pd.read_csv(EXP_ROOT / "E008" / "confusion_matrix.csv", index_col=0)
    normalized = confusion.div(confusion.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{confusion.iloc[i,j]:,}\n({normalized.iloc[i,j]:.1%})", ha="center", va="center", color="white" if normalized.iloc[i,j] > 0.55 else "black")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("E008 OOF confusion matrix")
    fig.colorbar(image, ax=ax, label="Satır-normalize oran")
    savefig("08_e008_confusion_matrix.png")

    importance = pd.read_csv(EXP_ROOT / "E008" / "feature_importance.csv")
    importance = importance[importance["fold"].astype(str) == "mean"].nlargest(20, "importance_gain")
    plt.figure(figsize=(9, 7))
    plt.barh(importance.feature[::-1], importance.importance_gain[::-1], color="#4c78a8")
    plt.xlabel("Ortalama gain importance")
    plt.title("E008 — en yüksek 20 feature importance")
    savefig("09_e008_feature_importance.png")

    sub_plot = submissions.set_index("candidate")[CLASSES]
    sub_plot.plot(kind="bar", stacked=True, figsize=(11, 5), color=["#4c78a8", "#f2cf5b", "#e45756"])
    plt.ylabel("Test tahmin oranı")
    plt.xticks(rotation=35, ha="right")
    plt.title("Submission adaylarının test sınıf dağılımı")
    savefig("10_submission_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(disagreement.to_numpy(), cmap="YlOrRd", vmin=0, vmax=disagreement.to_numpy().max())
    for i in range(len(disagreement)):
        for j in range(len(disagreement)):
            ax.text(j, i, f"{disagreement.iloc[i,j]:.1%}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(disagreement)), disagreement.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(disagreement)), disagreement.index)
    ax.set_title("Submission adayları pairwise disagreement")
    fig.colorbar(image, ax=ax, label="Farklı tahmin edilen satır oranı")
    savefig("11_submission_disagreement.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, exp_id in zip(axes, ["E001", "E002", "E008"], strict=True):
        ax.imshow(plt.imread(EXP_ROOT / exp_id / "roc_curves.png"))
        ax.axis("off")
        ax.set_title(exp_id)
    fig.suptitle("Seçili ana deneylerin OOF ROC eğrileri")
    savefig("12_selected_roc_curves.png")

    history = load_json(EXP_ROOT / "E008" / "training_history.json")
    plt.figure(figsize=(9, 5))
    for fold in history:
        values = fold["valid_loss"]
        plt.plot(range(1, len(values) + 1), values, label=f"Fold {fold['fold']}")
        plt.axvline(fold["best_iteration"], color="gray", alpha=0.25, linewidth=1)
    plt.xlabel("Boosting iteration")
    plt.ylabel("Validation multi_logloss")
    plt.title("E008 validation loss ve best iteration")
    plt.legend()
    savefig("13_e008_training_history.png")

    parameters = [
        "learning_rate",
        "num_leaves",
        "min_child_samples",
        "reg_alpha",
        "reg_lambda",
        "subsample",
        "colsample_bytree",
    ]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, parameter in zip(axes.flat, parameters, strict=False):
        grouped = sweeps.groupby(parameter).balanced_accuracy.agg(["mean", "min", "max"])
        x = np.arange(len(grouped))
        ax.errorbar(
            x,
            grouped["mean"],
            yerr=[grouped["mean"] - grouped["min"], grouped["max"] - grouped["mean"]],
            fmt="o-",
            capsize=3,
        )
        ax.set_xticks(x, [str(value) for value in grouped.index], rotation=35)
        ax.set_title(parameter)
        ax.set_ylabel("Mean balanced accuracy")
    axes.flat[-1].axis("off")
    fig.suptitle("Sweep parametre seviyeleri — gözlemsel ortalama ve aralık")
    savefig("14_sweep_parameter_effects.png")


def build_report(
    main: pd.DataFrame,
    tuned: pd.DataFrame,
    sweeps: pd.DataFrame,
    submissions: pd.DataFrame,
    disagreement: pd.DataFrame,
) -> str:
    best_main = main.loc[main.balanced_accuracy.idxmax()]
    best_tuned = tuned.loc[tuned.tuned.idxmax()]
    best_sweep = sweeps.loc[sweeps.balanced_accuracy.idxmax()]
    e001 = main.loc[main.experiment == "E001"].iloc[0]
    e002 = main.loc[main.experiment == "E002"].iloc[0]
    target = pd.read_csv(EXP_ROOT / "E001" / "oof_pred.csv")["y_true"].value_counts()
    manifest = load_json(EXP_ROOT / "E008" / "run_manifest.json")

    main_table = main.copy()
    main_table["değişiklik"] = main_table.experiment.map(lambda x: EXPERIMENT_INFO[x][1])
    main_table["Δ E001"] = main_table.balanced_accuracy - e001.balanced_accuracy
    main_table = main_table[["experiment", "değişiklik", "features", "balanced_accuracy", "std", "macro_f1", "recall_at-risk", "recall_fit", "recall_unhealthy", "best_iteration", "Δ E001"]]
    for col in ["balanced_accuracy", "std", "macro_f1", "recall_at-risk", "recall_fit", "recall_unhealthy", "Δ E001"]:
        main_table[col] = main_table[col].map(f6)
    main_table["best_iteration"] = main_table.best_iteration.map(lambda x: f"{x:.1f}")

    tuned_table = tuned.copy()
    tuned_table = tuned_table[["experiment", "argmax", "tuned", "delta", "macro_f1", "recall_at-risk", "recall_fit", "recall_unhealthy", "mult_at-risk", "mult_fit", "mult_unhealthy"]]
    for col in tuned_table.columns[1:]:
        tuned_table[col] = tuned_table[col].map(lambda x: f"{x:.6f}")

    sweep_table = sweeps.sort_values("balanced_accuracy", ascending=False).copy()
    sweep_table["rank"] = range(1, len(sweep_table) + 1)
    sweep_table = sweep_table[["rank", "experiment", "balanced_accuracy", "std", "learning_rate", "num_leaves", "min_child_samples", "reg_alpha", "reg_lambda", "subsample", "colsample_bytree", "best_iteration"]]
    sweep_table["balanced_accuracy"] = sweep_table.balanced_accuracy.map(f6)
    sweep_table["std"] = sweep_table["std"].map(f6)
    sweep_table["best_iteration"] = sweep_table.best_iteration.map(lambda x: f"{x:.1f}")

    sub_table = submissions.copy()
    for col in CLASSES:
        sub_table[col] = sub_table[col].map(pct)
    sub_table = sub_table[["candidate", "file", "rows", *CLASSES, "labels_valid", "id_unique"]]

    importance = pd.read_csv(EXP_ROOT / "E008" / "feature_importance.csv")
    importance = importance[importance["fold"].astype(str) == "mean"].nlargest(20, "importance_gain")
    importance_table = importance[["feature", "importance_gain", "importance_split"]].copy()
    importance_table["importance_gain"] = importance_table.importance_gain.map(lambda x: f"{x:,.0f}")
    importance_table["importance_split"] = importance_table.importance_split.map(lambda x: f"{x:,.0f}")

    fold_rows = []
    for exp_id in MAIN_IDS:
        fold = pd.read_csv(EXP_ROOT / exp_id / "fold_metrics.csv")
        fold_rows.append({"Deney": exp_id, **{f"Fold {int(r.fold)}": f"{r.balanced_accuracy:.6f}" for _, r in fold.iterrows()}})
    fold_table = pd.DataFrame(fold_rows)

    confusion = pd.read_csv(EXP_ROOT / "E008" / "confusion_matrix.csv", index_col=0)
    confusion.insert(0, "Gerçek sınıf", confusion.index)

    sections = [f'''---
title: "Kaggle S6E7 — Nihai Deney Sonuçları ve Submission Karar Raporu"
author: "Deney pipeline artefaktlarından otomatik üretilmiştir"
date: "2026-07-04"
lang: tr-TR
geometry: margin=1.7cm
papersize: a4
documentclass: extarticle
fontsize: 9pt
header-includes:
  - \\usepackage{{longtable}}
  - \\usepackage{{booktabs}}
  - \\usepackage{{float}}
  - \\usepackage{{graphicx}}
  - \\floatplacement{{figure}}{{H}}
  - \\setkeys{{Gin}}{{width=0.92\\linewidth,keepaspectratio}}
---

# Yönetici özeti

Bu rapor, tamamlanan **7 ana model deneyi**, **4 OOF class-multiplier optimizasyonu** ve
**20 LightGBM sweep trial'ını** submission seçimi için tek karar paketinde toplar. Bütün
skorlar train etiketlerinden üretilen out-of-fold (OOF) tahminlere dayanır; test etiketi
ve public leaderboard bilgisi optimizasyonda kullanılmamıştır.

En önemli sonuçlar:

1. **En iyi doğrudan argmax model {best_main.experiment}:** balanced accuracy
   `{best_main.balanced_accuracy:.6f} ± {best_main['std']:.6f}`. Sqrt-balanced sample
   weight, E002'ye göre `{best_main.balanced_accuracy - e002.balanced_accuracy:+.6f}`
   kazandırmıştır.
2. **En iyi tuned OOF sonucu {best_tuned.experiment}:** `{best_tuned.tuned:.6f}`.
   Class multiplier tuning dört kaynak modelin tamamını yaklaşık 0,948 seviyesine taşımıştır.
3. **En iyi sweep {best_sweep.experiment}:** `{best_sweep.balanced_accuracy:.6f}`; E002'ye
   göre yalnız `{best_sweep.balanced_accuracy - e002.balanced_accuracy:+.6f}` ve E001'e
   göre `{best_sweep.balanced_accuracy - e001.balanced_accuracy:+.6f}` fark üretmiştir.
   Sweep, sınıf ağırlığı kadar güçlü bir kazanım sağlamamıştır.
4. **Submission seçimindeki ana belirsizlik multiplier overfit riskidir.** Aynı OOF üzerinde
   multiplier seçilip skorlandığı için tuned skorlar bağımsız doğrulama skoru değildir.
5. **Karar için güvenli kısa liste:** E008 argmax, E002 tuned, E006 tuned, E008 tuned ve
   çeşitlilik kontrolü için {best_sweep.experiment} argmax. İlk gönderimlerde argmax/tuned
   çifti, postprocess'in gerçek leaderboard katkısını ölçmek için birlikte değerlendirilebilir.

# 1. Kapsam, veri ve tekrar üretilebilirlik

| Alan | Değer |
|---|---|
| Train satırı | {manifest['train_rows']:,} |
| Test satırı | {manifest['test_rows']:,} |
| Ham feature | 13 (7 numeric + 6 categorical) |
| Target | `health_condition` |
| Sınıflar | `at-risk`, `fit`, `unhealthy` |
| CV | StratifiedKFold, 3 fold, shuffle, seed 42 |
| Model | LightGBM 4.6.0, CPU/OpenMP, `n_jobs=6` |
| Python | {manifest['python']} |
| Train SHA-256 | `{manifest['data_fingerprint']['train']['sha256']}` |
| Test SHA-256 | `{manifest['data_fingerprint']['test']['sha256']}` |

Gerçek train sınıf dağılımı:

| Sınıf | Adet | Oran |
|---|---:|---:|
| at-risk | {target['at-risk']:,} | {pct(target['at-risk']/target.sum())} |
| fit | {target['fit']:,} | {pct(target['fit']/target.sum())} |
| unhealthy | {target['unhealthy']:,} | {pct(target['unhealthy']/target.sum())} |

Bu dengesizlik nedeniyle normal accuracy ana seçim metriği değildir. Balanced accuracy,
üç sınıf recall değerinin ortalamasıdır ve azınlık sınıflarını eşit ağırlıkla değerlendirir.

# 2. Pipeline nasıl çalıştı?

```text
CSV yükleme ve schema doğrulama
  -> stratified 3-fold ayırma
  -> her fold'da yalnız fold-train ile preprocessing fit
  -> LightGBM + early stopping
  -> fold-valid OOF probabilities
  -> fold-test probabilities ve üç fold ortalaması
  -> metrikler, modeller, importance, confusion matrix, ROC, manifest
  -> seçili deneylerde OOF-only class multiplier tuning
  -> argmax ve tuned submission üretimi
  -> E002 feature pipeline üzerinde 20 trial parameter sweep
```

## 2.1 Fold-safe preprocessing

Median, kategori seviyeleri, outlier quantile ve clipping sınırları yalnız fold-train'de
öğrenilir. Aynı dönüşüm fold-valid ve test'e uygulanır. Bu tasarım validation satırlarının
kendi preprocessing istatistiklerini etkilemesini, yani veri sızıntısını önler.

## 2.2 LightGBM ve early stopping

Başlangıç modeli `learning_rate=0.035`, `num_leaves=96`, `min_child_samples=200`,
`subsample=0.85`, `colsample_bytree=0.90`, `reg_alpha=0.1`, `reg_lambda=2.0` kullanır.
`n_estimators=12000` yalnız üst sınırdır. Validation multi-logloss 300 tur iyileşmezse
eğitim durur ve her fold'un en iyi iterasyonu kullanılır.

## 2.3 OOF ve test olasılıkları

Her train satırının OOF olasılığı, o satırı eğitimde görmeyen modelden gelir. Model seçimi,
multiplier tuning ve rapordaki lokal metrikler bu olasılıklar üzerinden hesaplanır. Test
olasılıkları üç fold modelinin ortalamasıdır.

# 3. Deney tasarımı: ne değiştirildi?

| Deney | Pipeline | Değişiklik | Ölçülen hipotez |
|---|---|---|---|
| E001 | V1 baseline | Median + missing category/flag | Güvenli referans |
| E002 | V2-Core | Missing count, 6 ratio, 3 interaction, outlier flag/count | EDA feature set'i ek sinyal taşıyor mu? |
| E003 | E002 + interaction | `gender_activity` | Gender/activity birlikte yararlı mı? |
| E004 | E002 + rules | 8 threshold flag | Açık eşikler rare-class recall artırıyor mu? |
| E005 | E002 + clipping | Fold-train %0,1/%99,9 clipping | Uç değer sıkıştırma genellemeyi artırıyor mu? |
| E006 | E002 + log | 6 `log1p(ratio)` | Uzun kuyrukları sıkıştırmak yararlı mı? |
| E007 | Postprocess | OOF class multiplier | Argmax karar sınırı dengelenebilir mi? |
| E008 | E002 + weight | Sqrt-balanced sample weight | Rare class eğitim ağırlığı faydalı mı? |

Feature yöntemlerinin formülleri ve sayısal örnekleri ayrıca
[`experiments-detailed-explanation.md`](experiments-detailed-explanation.md) belgesinde
ayrıntılı olarak açıklanmıştır.

## 3.1 Feature üretim yöntemleri ve tam eşikler

### Missing yönetimi

- 7 numeric kolon fold-train medianı ile doldurulur.
- 6 categorical kolon için eksik değer açık `missing` kategorisidir.
- Her ham kolon için `<kolon>_is_missing` ikili flag'i üretilir.
- E002 ve türevlerinde 13 ham kolondaki eksiklerin toplamı `missing_count` olur.
- Fold-train'de görülmeyen kategoriler `__UNKNOWN__` seviyesine taşınır.

### Ratio feature'ları

Bütün oranlarda güvenli bölme `pay / (payda + 1)` kullanılır. `+1`, paydanın sıfır olduğu
satırlarda sonsuz değer oluşmasını önler.

| Feature | Formül | Amaç |
|---|---|---|
| `calorie_per_step` | calorie_expenditure / (step_count + 1) | Adım başına enerji |
| `calorie_per_exercise_min` | calorie_expenditure / (exercise_duration + 1) | Egzersiz süresine göre enerji |
| `step_per_exercise_min` | step_count / (exercise_duration + 1) | Egzersiz süresine göre hareket |
| `water_per_bmi` | water_intake / (bmi + 1) | BMI'a göre su tüketimi |
| `exercise_per_bmi` | exercise_duration / (bmi + 1) | BMI'a göre egzersiz |
| `steps_per_sleep_hour` | step_count / (sleep_duration + 1) | Uyku süresine göre hareket |

### Categorical interaction'lar

`stress_level__sleep_quality`, `physical_activity_level__diet_type` ve
`smoking_alcohol__physical_activity_level` birleşimleri yeni kategori olarak üretilir.
E003 ayrıca `gender__physical_activity_level` ekler.

### Outlier ve clipping eşikleri

- E002 outlier flag sınırları fold-train `%0,5` ve `%99,5` quantile'larıdır.
- Ham 7 numeric ve 6 ratio için `_outlier_low`/`_outlier_high` flag'leri üretilir.
- Aktif flag sayısı `outlier_count` olarak tutulur; satır silinmez.
- E005 clipping sınırları fold-train `%0,1` ve `%99,9` quantile'larıdır.
- Outlier flag clipping'den **önce** hesaplanır; uçta olma bilgisi korunur.

### Rule flag eşikleri

| Flag | Aktif olma koşulu |
|---|---|
| `low_sleep_flag` | sleep_duration < 6 |
| `high_sleep_flag` | sleep_duration > 9 |
| `high_bmi_flag` | bmi >= 30 |
| `low_bmi_flag` | bmi < 18,5 |
| `high_heart_rate_flag` | heart_rate > 100 |
| `low_heart_rate_flag` | heart_rate < 60 |
| `low_steps_flag` | step_count < 3.000 |
| `high_steps_flag` | step_count > 12.000 |

### Log dönüşümü ve sample weight

E006, her ratio için `log(1 + max(ratio, 0))` varyantını orijinal feature'ı silmeden ekler.
E008 sınıf ağırlığını fold-train sınıf sayısından hesaplar:

```text
balanced_weight(class) = N / (3 * class_count)
sqrt_balanced_weight(class) = sqrt(balanced_weight(class))
```

Bu yöntem rare-class hatalarını eğitim loss'unda daha pahalı yapar fakat tam balanced
weight'e göre çoğunluk sınıfını daha az baskılar.

# 4. Ana deney sonuçları

{markdown_table(main_table)}

![Ana deney balanced accuracy](../reports/figures/final_experiments/01_main_balanced_accuracy.png)

![Ana deney sınıf recall](../reports/figures/final_experiments/02_main_class_recall.png)

![OOF tahmin dağılımları](../reports/figures/final_experiments/03_oof_prediction_distribution.png)

## 4.1 Fold kararlılığı

{markdown_table(fold_table)}

![Fold kararlılığı](../reports/figures/final_experiments/07_fold_stability.png)

E008 yalnız ortalamada en iyi değildir; fold standard deviation değeri de ana deneylerin
en düşüğüdür. `unhealthy` recall fold'lar arasında diğer sınıflardan daha fazla oynasa da
üç fold'un tamamında E001/E002 seviyesinin belirgin üzerindedir.

## 4.2 Deney bazında kararlar

- **E001:** Güçlü baseline. 26 feature ile bütün ağırlıksız V2 varyantlarını geçmiştir.
- **E002:** EDA feature paketinin toplu eklenmesi `-0.001167` kayıp üretmiştir. Tek tek
  feature'ların kötü olduğunu kanıtlamaz; paket olarak faydalı değildir.
- **E003:** Gender/activity interaction E002'yi geçmemiştir; gender shift riski karşılığında
  kazanım yoktur.
- **E004:** E002'ye `+0.000529` ile ağırlıksız V2 varyantlarının en iyisidir; fark gürültü
  eşiğinin altındadır ve E001'i geçmez.
- **E005:** Clipping skoru düşürmüştür. Uç değerlerin hedef sinyalini taşıdığı EDA bulgusuyla
  uyumludur.
- **E006:** Log ratio varyantları pratik olarak nötrdür; 6 ek feature karşılığında kazanım yoktur.
- **E008:** Açık kazanan argmax modeldir. Rare-class recall artışı ortalama skoru yaklaşık
  0,037 yükseltmiştir.

# 5. E008 ayrıntılı teşhis

E008 overall accuracy `{best_main.accuracy:.6f}`, balanced accuracy
`{best_main.balanced_accuracy:.6f}`, macro F1 `{best_main.macro_f1:.6f}`, MCC
`{best_main.mcc:.6f}`, log-loss `{best_main.log_loss:.6f}` ve macro OvR ROC-AUC
`{best_main.roc_auc:.6f}` üretmiştir.

## 5.1 Confusion matrix

{markdown_table(confusion.reset_index(drop=True))}

![E008 confusion matrix](../reports/figures/final_experiments/08_e008_confusion_matrix.png)

`fit` örneklerinin büyük bölümü doğru bulunurken 4.351 tanesi `at-risk` tahmin edilmiştir.
`unhealthy` için ana hata yönü yine `at-risk` sınıfıdır (6.877 satır). Bu davranış sonraki
postprocess'in neden azınlık sınıfı multiplier'larını büyüttüğünü açıklar.

## 5.2 Feature importance

{markdown_table(importance_table)}

![E008 feature importance](../reports/figures/final_experiments/09_e008_feature_importance.png)

Gain importance nedensellik değildir ve korelasyonlu feature'lar önemi paylaşabilir.
Bununla birlikte `sleep_duration`, `stress_level` ve `physical_activity_level` ana sinyal
kaynaklarıdır. Missing flag ve engineered feature katkıları ham feature'ların gölgesindedir.

## 5.3 ROC ve eğitim geçmişi

![Seçili ROC eğrileri](../reports/figures/final_experiments/12_selected_roc_curves.png)

![E008 training history](../reports/figures/final_experiments/13_e008_training_history.png)

ROC-AUC sınıfların olasılık sıralamasını, balanced accuracy ise seçilen kararın recall
dengesini ölçer. E008 macro OvR ROC-AUC değeri yüksek olduğu için model olasılıklarının
sıralama gücü vardır; multiplier tuning bu sıralamayı farklı karar sınırına çevirir.
Training history'deki dikey çizgiler fold best-iteration noktalarını gösterir.

# 6. E007 — Class multiplier / karar sınırı optimizasyonu

Varsayılan karar `argmax(probability)` iken tuned karar şöyledir:

```text
prediction = argmax(probability * class_multiplier)
```

Arama, `at-risk=1` referansıyla `fit` ve `unhealthy` için 0,80–1,50 coarse grid (841 aday),
ardından en iyi komşulukta seeded 2.000 random trial uygular. Son vektör ortalaması 1 olacak
şekilde normalize edilir.

{markdown_table(tuned_table)}

![Multiplier tuning kazancı](../reports/figures/final_experiments/04_multiplier_gain.png)

![Multiplier değerleri](../reports/figures/final_experiments/05_multiplier_values.png)

## 6.1 Kritik metodolojik uyarı

Multiplier aynı OOF tahminleri üzerinde hem **seçilmiş** hem **raporlanmıştır**. 2.841 aday
arasından en iyiyi seçmek, OOF skoruna seçim yanlılığı ekleyebilir. Bu nedenle `0.948584`
bağımsız bir holdout skoru gibi kabul edilmemelidir. Güvenilirlik sırası:

1. Aynı multiplier'ı farklı seed OOF tahminlerinde doğrulamak.
2. Nested CV ile her dış fold için multiplier'ı yalnız diğer fold'larda seçmek.
3. Public LB'de bir argmax/tuned çiftini kontrollü karşılaştırmak.

Tuned adaylar arasında fark yalnız yaklaşık 0,0008'dir; tek başına bu sıralama kesin model
üstünlüğü sayılmaz.

# 7. LightGBM sweep sonuçları

Sweep yalnız E002 feature pipeline üzerinde çalışmıştır; E008'in sample weighting yaklaşımı
sweep'e dahil değildir. Aşağıdaki tablo 20 trial'ın tamamını OOF balanced accuracy'ye göre
sıralar.

{markdown_table(sweep_table)}

![Sweep skorları](../reports/figures/final_experiments/06_sweep_scores.png)

![Sweep parametre etkileri](../reports/figures/final_experiments/14_sweep_parameter_effects.png)

Parametre grafiği kontrollü tek-değişken deneyi değildir: her trial aynı anda birden fazla
parametreyi değiştirdiği için noktalar nedensel etki olarak okunmamalıdır. Yalnız arama
uzayında hangi seviyelerin daha iyi trial'larla birlikte görüldüğünü özetler.

## 7.1 Sweep yorumu

- En iyi trial **{best_sweep.experiment}**, parametreleri:
  `learning_rate={best_sweep.learning_rate}`, `num_leaves={int(best_sweep.num_leaves)}`,
  `min_child_samples={int(best_sweep.min_child_samples)}`, `reg_alpha={best_sweep.reg_alpha}`,
  `reg_lambda={best_sweep.reg_lambda}`, `subsample={best_sweep.subsample}`,
  `colsample_bytree={best_sweep.colsample_bytree}`.
- En iyi ile en kötü sweep arasındaki aralık yalnız
  `{sweeps.balanced_accuracy.max() - sweeps.balanced_accuracy.min():.6f}`'dır.
- Hiçbir sweep E008'e yaklaşmamıştır.
- Aynı 3 fold üzerinde 20 aday seçildiği için en iyi trial skoru da hafif seçim yanlılığı
  taşıyabilir.
- Sonraki HPO, yapılacaksa E008 veya E001+sqrt-balanced tabanı üzerinde ve ayrı seed ile
  doğrulanmalıdır.

# 8. Submission adaylarının doğrulanması

{markdown_table(sub_table)}

Bütün kısa liste dosyaları 295.753 satır, benzersiz ID, doğru iki kolon ve izin verilen üç
label kontrolünü geçmiştir.

![Submission dağılımları](../reports/figures/final_experiments/10_submission_distribution.png)

![Submission disagreement](../reports/figures/final_experiments/11_submission_disagreement.png)

## 8.1 Disagreement nasıl okunur?

Pairwise disagreement, iki submission'ın farklı label verdiği test satırı oranıdır. Düşük
oran iki dosyanın günlük submit hakkı açısından birbirini tekrar ettiğini; daha yüksek oran
ise farklı karar sınırı veya model davranışı taşıdığını gösterir. Fakat çeşitlilik tek başına
kalite kanıtı değildir.

En yüksek kısa-liste disagreement:
`{disagreement.where(~np.eye(len(disagreement), dtype=bool)).stack().idxmax()[0]}` ile
`{disagreement.where(~np.eye(len(disagreement), dtype=bool)).stack().idxmax()[1]}` arasında
`{disagreement.where(~np.eye(len(disagreement), dtype=bool)).stack().max():.2%}`.

# 9. Submission seçimi için karar matrisi

| Aday | Güçlü yön | Ana risk | Önerilen rol |
|---|---|---|---|
| E008 argmax | En iyi bağımsız ana deney skoru, düşük fold std | Tuned modellere göre düşük OOF skor | Güvenli ilk referans |
| E002 tuned | En iyi tuned OOF skoru | Multiplier seçim yanlılığı | Ana tuned aday |
| E006 tuned | E002 tuned'a çok yakın, farklı feature set | Log feature katkısı kanıtlanmadı | İkinci tuned aday |
| E008 tuned | En yüksek unhealthy recall | E002 tuned'dan düşük, postprocess overfit | Recall-ağırlıklı aday |
| {best_sweep.experiment} argmax | En iyi HPO trial | E001'den iyi değil | HPO kontrol adayı |
| E001 argmax | En güçlü sade ağırlıksız baseline | Rare-class recall düşük | Baseline kontrolü |

## 9.1 Önerilen kontrollü submit sırası

Bu sıra lokal kanıta dayanır; yarışma leaderboard sonucu bilinmeden kesin kazanan iddiası
değildir.

1. `outputs/experiments/E008/submission_E008_argmax.csv`
2. `outputs/experiments/E002/submission_E002_tuned.csv`
3. `outputs/experiments/E008/submission_E008_tuned.csv`
4. `outputs/experiments/E006/submission_E006_tuned.csv`
5. Gerekirse `{best_sweep.experiment}/submission_argmax.csv`

İlk iki dosya, eğitimde sınıf ağırlığı ile postprocess karar sınırının leaderboard'daki
gerçek katkısını ayırmak için en bilgi verici çifttir. Birbirine çok yakın tuned varyantların
tamamını aynı gün göndermek yerine sonuç geldikçe sonraki aday seçilmelidir.

# 10. Bilinen riskler ve eksik doğrulamalar

1. **Tek CV seed:** Bütün ana deneyler seed 42 ile aynı üç fold'u kullanır.
2. **Multiplier overfit:** Class multiplier aynı OOF üzerinde seçilip değerlendirilmiştir.
3. **Sweep selection bias:** 20 aday aynı fold'larda karşılaştırılmıştır.
4. **Public/private LB farkı:** Test target dağılımı ve hidden split bilinmez.
5. **Feature importance nedensel değildir:** Gain değerleri yalnız model içi kullanım ölçüsüdür.
6. **Probability calibration ölçülmedi:** Balanced accuracy iyi olsa bile olasılıkların
   kalibrasyonu ayrıca doğrulanmamıştır.
7. **E001 + sqrt-balanced eksik:** Ağırlık kazancının V2 feature set'e bağlı olup olmadığını
   ayıracak en önemli yeni ablation henüz çalıştırılmamıştır.

# 11. Sonraki en değerli deneyler

1. **E009 = E001 + sqrt-balanced:** Baseline feature'larla weighting etkisini izole et.
2. **Multi-seed doğrulama:** E001, E008, E002-tuned için en az 3 ek seed.
3. **Nested multiplier CV:** Postprocess iyimserliğini ölç.
4. **E008 tabanlı küçük sweep:** HPO'yu sınıf ağırlıklı kazanan pipeline'a uygula.
5. **OOF probability blend:** E008 ve E002/E006 olasılıklarını yalnız OOF ile ağırlıklandır.

# 12. Artefakt ve kaynak dizini

- Ana sonuçlar: `outputs/experiments/E001` … `E008`
- Sweep sonuçları: `outputs/experiments/SWEEP_000` … `SWEEP_019`
- Lokal leaderboard: `outputs/leaderboard_local.csv`
- Pipeline logu: `outputs/logs/pipeline_20260704_143235.log`
- Deney config'i: `configs/experiments.yaml`
- Model config'i: `configs/lgbm_base.yaml`
- Sweep uzayı: `configs/sweeps.yaml`
- Eğitim kodu: `src/kaggle_s6_e7/training.py`
- Postprocess kodu: `src/kaggle_s6_e7/postprocess.py`
- Bu raporun üreticisi: `scripts/generate_experiment_report.py`

Rapor yalnız repository içindeki gerçekleşmiş artefaktlardan üretilmiştir. Public veya
private leaderboard skoru içermez.
''']
    return "\n".join(sections)


def main() -> None:
    main_results = load_main_results()
    tuned_results = load_tuned_results(main_results)
    sweeps = load_sweeps()
    if len(sweeps) != 20:
        raise RuntimeError(f"Expected 20 completed sweeps, found {len(sweeps)}")
    best_sweep = sweeps.loc[sweeps.balanced_accuracy.idxmax(), "experiment"]
    submissions, disagreement = candidate_submissions(best_sweep)
    generate_figures(main_results, tuned_results, sweeps, submissions, disagreement)
    REPORT_PATH.write_text(
        build_report(main_results, tuned_results, sweeps, submissions, disagreement)
    )
    print(REPORT_PATH.relative_to(ROOT))
    print(f"figures={len(list(FIG_ROOT.glob('*.png')))} sweeps={len(sweeps)}")


if __name__ == "__main__":
    main()
