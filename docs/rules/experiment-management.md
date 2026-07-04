# Experiment Management Rules

1. Deney kimlikleri `E001` biçimindedir ve yeniden kullanılmaz.
2. Her deney başlamadan hipotez, baseline, tek ana değişiklik ve başarı ölçütünü kaydeder.
3. Config, metrik, not ve artefaktlar `outputs/experiments/E###/` altında tutulur.
4. Ana metric balanced accuracy; MCC, macro/weighted F1, OvR ROC-AUC, confusion
   matrix, OOF tahmin dağılımı ve sınıf bazlı recall zorunludur.
5. Submission deney klasöründe `submission_argmax.csv` veya `submission_tuned.csv` olur.
6. Preprocessing değerleri yalnız train veya fold-train üzerinden öğrenilir.
7. Sonucu etkileyen config değişikliği yeni deney kimliği alır.
8. Notebook keşif içindir; tekrar üretilebilir deneyler `scripts/` ile çalıştırılır.
9. EDA kararları kesinleşmeden model deneylerine başlanmaz.
