# Experiments

Deneyler kalıcı, üç basamaklı kimlik alır: `E001`, `E002`, …, `E011`.

```text
experiments/E001/
├── README.md
├── config.yaml
├── metrics.json
└── artifacts/
```

- Kimlikler yeniden kullanılmaz veya numaralandırılmaz.
- Her deney hipotezini, baseline kimliğini ve tek ana değişkenini kaydeder.
- Ana metric balanced accuracy; sınıf bazlı recall zorunludur.
- Submission varsa `submissions/E###_submission.csv` kullanılır.
