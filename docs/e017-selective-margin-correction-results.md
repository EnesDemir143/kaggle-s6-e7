# E017 — E002 Selective Margin Correction Pipeline ve Sonuç Raporu

## 1. Amaç

E017, E011/E014/E009/E004 alternatiflerini komple blend olarak kullanmak yerine yalnız
E002'nin düşük margin'li olduğu ve alternatif modelin yeterli skor kazancı gösterdiği
seçilmiş satırlarda düzeltme yapar.

Base model **E002 tuned** olarak sabittir. Alternatifler öncelik sırasıyla E011, E014,
E009 ve E004'tür. Her kaynak ve label direction bağımsız taranır; submission yalnız
önceden belirlenen OOF, disagreement, satır sayısı ve dağılım filtrelerini geçerse üretilir.

## 2. Yeniden kurulan alternatifler

Postprocess deneyleri ayrı probability dosyası saklamadığından probability'ler kaynak
artefaktlardan deterministik biçimde yeniden kuruldu:

| Kaynak | Probability | Multiplier |
|---|---|---|
| E011 | `0.60 E002 + 0.30 E004 + 0.10 E006` | E002 × `[1, 1.01, 1.01]` |
| E014 | `0.60 E002 + 0.30 E004 + 0.10 E006` | E002 × `[1, 1.005, 1.005]` |
| E009 | `0.75 E002 + 0.25 E004` | E002 × `[1, 1.01, 1.01]` |
| E004 | E004 probability | E004 tuned multiplier |

## 3. Satır sinyalleri

Her OOF/test satırı için adjusted score `probability × multiplier` olarak hesaplandı.
Arama sinyalleri:

```text
base_margin = E002 adjusted top1 - E002 adjusted top2
alt_margin  = alternative adjusted top1 - alternative adjusted top2
alt_gain    = alternative score(alt label) - E002 score(base label)
entropy     = -Σ E002_probability × log(E002_probability)
direction   = E002 label -> alternative label
```

Quantile eşikleri her alternatif kaynağın tüm `alt_pred != E002_pred` OOF havuzundan
öğrenildi. Daha sonra direction maskesi uygulandı. Böylece aynı kaynaktaki direction'lar
karşılaştırılabilir ortak margin/gain ölçeği kullanır.

Entropy seçim filtresi değildir; seçilen OOF satırlarının belirsizlik tanısı olarak
raporlanır.

## 4. Arama uzayı

```text
source               = {E011, E014, E009, E004}
direction            = 6 ordered label transition
base_margin_quantile = {0.10, 0.20, 0.30, 0.40, 0.50}
alt_gain_quantile    = {0.50, 0.60, 0.70, 0.80}
min_alt_margin       = {0.000, 0.005, 0.010}
```

Toplam:

```text
4 × 6 × 5 × 4 × 3 = 1,440 kural
```

Her kural tek source ve tek direction uygular. Bu tasarım, farklı kaynaklardan öğrenilen
kuralları aynı submission içinde karıştırarak search uzayını kontrolsüz büyütmez.

## 5. Eligibility filtreleri

| Filtre | Eşik |
|---|---:|
| Minimum OOF balanced accuracy | 0.94860 |
| Minimum E002 disagreement | %0.025 |
| Maksimum E002 disagreement | %0.12 |
| Değişen test satırı | 75–350 |
| Maksimum at-risk adet farkı | 150 |
| Maksimum fit adet farkı | 100 |
| Maksimum unhealthy adet farkı | 150 |

1.440 kuralın yalnız **5** tanesi bütün filtreleri geçti. Geçenler arasında en yüksek OOF
balanced accuracy otomatik seçildi.

## 6. Seçilen kural

```text
source                 = E004 tuned
direction              = at-risk -> unhealthy
base_margin_quantile   = 0.50
alt_gain_quantile      = 0.60
min_alt_margin         = 0.000
base_margin_threshold  = 0.0154675979
alt_gain_threshold     = 0.0092075398
```

Kuralın açık hali:

```text
E004_pred == unhealthy
E002_pred == at-risk
E002_base_margin <= 0.0154675979
E004_alt_gain >= 0.0092075398
E004_alt_margin >= 0.000
```

Bu koşulların tümü sağlanırsa E002 etiketi `unhealthy` olarak değiştirilir; diğer bütün
satırlar E002 tuned olarak kalır.

E011 ve E014 önce tarandı ancak tanımlı eligibility filtrelerini birlikte sağlayan bir
kural üretmedi. E004'ün seçilmesi öncelik sırasının ihlali değildir; öncelik arama sırasını
belirler, nihai seçim ise filtreleri geçen adaylar arasında OOF skoruna göre yapılır.

## 7. OOF ve test sonuçları

| Metrik | E017 |
|---|---:|
| OOF balanced accuracy | **0.948658173849** |
| E002 tuned OOF | 0.948584170185 |
| Δ E002 OOF | **+0.000074003664** |
| Değişen OOF satırı | 275 |
| Seçilen OOF satırlarında ortalama E002 entropy | 0.401217222214 |
| Değişen test satırı | 90 |
| E002 disagreement | **%0.03043** |
| Eligibility | **PASS** |

Testte değişen 90 satırın tamamı:

```text
at-risk -> unhealthy
```

Sınıf adetleri:

| Sınıf | E002 | E017 | Fark |
|---|---:|---:|---:|
| at-risk | 241,579 | 241,489 | -90 |
| fit | 21,297 | 21,297 | 0 |
| unhealthy | 32,877 | 32,967 | +90 |

Test oranları:

| Sınıf | E017 oranı |
|---|---:|
| at-risk | %81.65226 |
| fit | %7.20094 |
| unhealthy | %11.14680 |

Bütün dağılım farkları tanımlanan limitlerin içindedir.

## 8. Filtreyi geçen beş kural

| Sıra | Source | Direction | Base q | Gain q | Min alt margin | OOF | Test değişim |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | **E004** | at-risk→unhealthy | 0.50 | 0.60 | 0.000 | **0.948658174** | 90 |
| 2 | E004 | at-risk→unhealthy | 0.50 | 0.50 | 0.000 | 0.948655924 | 90 |
| 3 | E004 | at-risk→unhealthy | 0.50 | 0.70 | 0.000 | 0.948642802 | 75 |
| 4 | E004 | at-risk→unhealthy | 0.40 | 0.60 | 0.000 | 0.948635041 | 80 |
| 5 | E004 | at-risk→unhealthy | 0.40 | 0.50 | 0.000 | 0.948633916 | 80 |

Sonucun aynı source ve direction çevresindeki beş yakın kuralla desteklenmesi, seçimin tek
bir izole threshold tesadüfü olmadığını gösterir. Bununla birlikte farklar küçük olduğu
için public artış garantisi değildir.

## 9. Çalıştırma

```bash
bash scripts/run_e017_experiment.sh
```

Yeniden üretim:

```bash
FORCE=1 bash scripts/run_e017_experiment.sh
```

Config ve runner:

```text
configs/e017_experiment.yaml
scripts/run_e017_experiment.sh
```

## 10. Üretilen artefaktlar

```text
outputs/experiments/E017_E002_selective_margin_correction/submission.csv
outputs/experiments/E017_E002_selective_margin_correction/config.json
outputs/experiments/E017_E002_selective_margin_correction/metrics.json
outputs/experiments/E017_E002_selective_margin_correction/eligibility.json
outputs/experiments/e017_eligibility_report.csv
outputs/experiments/e017_eligibility_report.json
```

Submit dosyası:

```text
outputs/experiments/E017_E002_selective_margin_correction/submission.csv
```

## 11. Sonuç ve risk yorumu

E017 bütün eligibility filtrelerini geçti ve tek submission üretti. E002'nin 295.753 test
satırından yalnız 90'ını değiştirdiği için base karar sınırını büyük ölçüde korur. OOF'ta
E002'ye göre `+0.000074` artış sağlar ve değişimler tek, yorumlanabilir direction üzerindedir.

Bu aday submit edilmeye teknik olarak uygundur. Ancak OOF üzerinde 1.440 kural tarandığı
için threshold-selection overfit riski vardır. Public leaderboard sonucu gelene kadar
**E002 tuned güvenli final**, E017 ise kontrollü son-deney adayı olarak değerlendirilmelidir.
