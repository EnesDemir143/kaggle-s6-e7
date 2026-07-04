# E009–E013 Postprocess Pipeline ve Sonuç Raporu

## 1. Amaç ve kapsam

Bu rapor, public leaderboard'da en iyi sonucu veren **E002 tuned** modelinin etrafında
oluşturulan beş kontrollü postprocess deneyini açıklar. E009–E013 sırasında yeni feature
üretilmedi ve LightGBM modeli yeniden eğitilmedi. Mevcut modellerin OOF ve test
olasılıkları kullanılarak probability blend, dar multiplier araması, sabit boundary
perturbation ve consensus correction uygulandı.

Referans sonuçlar:

| Submission | Public skor |
|---|---:|
| **E002 tuned** | **0.94960** |
| E004 tuned | 0.94941 |
| E006 tuned | 0.94905 |
| E008 tuned | 0.94894 |
| E008 argmax | 0.91517 |

E002 tuned OOF balanced accuracy değeri `0.948584` ve multiplier vektörü şöyledir:

```text
at-risk   = 0.1892298422
fit       = 1.4444532324
unhealthy = 1.3663169255
```

## 2. Pipeline nasıl çalışır?

### 2.1 Girdiler

Pipeline kaynak deneylerin aşağıdaki artefaktlarını okur:

| Artefakt | Kullanım amacı |
|---|---|
| `oof_proba.npy` | Blend ve multiplier seçimini train OOF tahminlerinde yapmak |
| `test_proba.npy` | Seçilen kararı test verisine uygulamak |
| `oof_pred.csv` | OOF gerçek etiketleri ve ID sırasını doğrulamak |
| `submission_argmax.csv` | Test ID sırasını doğrulamak |
| `label_mapping.json` | Olasılık kolonlarının sınıf sırasını doğrulamak |
| `best_multipliers.json` | Tuned kaynakların multiplier vektörünü okumak |

Ana kaynaklar `E002`, `E004`, `E006` ve `SWEEP_002` dizinleridir. Pipeline, işlemden önce
tüm kaynaklarda OOF ID'lerinin, gerçek etiketlerin, test ID'lerinin, probability shape'ının
ve class mapping'in aynı olduğunu doğrular. Uyuşmazlık varsa submission üretimi durur.

### 2.2 Ortak hesap sırası

Her aday için şu sıra izlenir:

1. İlgili OOF ve test olasılıklarını yükle.
2. Deney tarifindeki blend veya correction işlemini OOF üzerinde uygula.
3. Gerekliyse yalnız OOF balanced accuracy kullanarak multiplier/weight seç.
4. Seçilen işlemi değişiklik yapmadan test olasılıklarına uygula.
5. E002 tuned ile test-label disagreement oranını hesapla.
6. Test tahminlerinin sınıf dağılımını hesapla.
7. Eligibility eşiklerini değerlendir ve PASS/FAIL nedenlerini kaydet.
8. Submission, config, metrics ve eligibility artefaktlarını yaz.

Test veya leaderboard etiketleri seçim sırasında kullanılmaz. Public skorlar yalnız deney
yönünü belirlemek için referanstır.

### 2.3 Dry-run ve production

Önce izole doğrulama çalıştırılır:

```bash
bash scripts/dry_run_postprocess_experiments.sh
```

Dry-run gerçek probability artefaktlarını salt okunur kaynak olarak kullanır ve yalnız
`outputs/dry_runs/postprocess_experiments/` altına yazar. Çalışma öncesi ve sonrasında
`outputs/cache` ile `outputs/experiments` envanterlerini karşılaştırır. Böylece dry-run'dan
kalan feature cache'in production çalışmasını bozması engellenir.

Production çalışması:

```bash
bash scripts/run_postprocess_experiments.sh
```

Tamamlanmış adaylar varsayılan olarak yeniden hesaplanmaz. Bilinçli yeniden üretim için:

```bash
FORCE=1 bash scripts/run_postprocess_experiments.sh
```

Deney tarifleri `configs/postprocess_experiments.yaml`, ortak hesap motoru ise
`src/kaggle_s6_e7/candidate_experiments.py` içindedir.

## 3. Toplu production sonuçları

| Deney | Eligibility | OOF bal. acc. | Δ E002 OOF | E002 disagreement | Test at-risk | Test fit | Test unhealthy |
|---|---|---:|---:|---:|---:|---:|---:|
| E009 | FAIL | 0.948639 | +0.000055 | %0.0426 | %81.6678 | %7.2050 | %11.1272 |
| E010 | **PASS** | 0.948392 | -0.000192 | %0.1802 | %81.6658 | %7.1996 | %11.1346 |
| E011 | **PASS** | **0.948654** | **+0.000069** | %0.0575 | %81.6709 | %7.2036 | %11.1255 |
| E012 | FAIL | 0.948540 | -0.000044 | %0.0179 | %81.6922 | %7.2060 | %11.1018 |
| E013 | FAIL | 0.948418 | -0.000166 | %0.0859 | %81.7277 | %7.1776 | %11.0947 |

`Eligibility=FAIL`, dosyanın üretilemediği anlamına gelmez. Beş deney de geçerli submission
CSV'si üretmiştir. FAIL yalnız önceden tanımlanan kontrollü varyasyon eşiğinin aşılmadığını
veya dağılım hedefinin kaçırıldığını belirtir.

## 4. Deneylerin ayrıntıları

### 4.1 E009 — E002/E004 75/25 blend tuned

Formül:

```text
p_blend = 0.75 * p_E002 + 0.25 * p_E004
prediction = argmax(p_blend * multiplier)
```

E002 multiplier çevresinde `fit` ve `unhealthy` için
`{0.990, 1.000, 1.010}` micro grid tarandı. Seçilen multiplier:

```text
[0.1892298422, 1.4588977647, 1.3799800947]
```

OOF balanced accuracy `0.948639` ile E002 tuned'dan yaklaşık `+0.000055` yüksektir.
Test dağılımı belirlenen bantların içindedir. Ancak E002 ile disagreement yalnız
`0.000426`, yani `%0.0426` oldu. Minimum `%0.1` eşiği aşılmadığı için **FAIL** verildi.
Blend, E002'den yeterince farklı bir submission üretmedi.

Submission:

```text
outputs/experiments/E009_blend_E002_E004_75_25/submission_tuned.csv
```

### 4.2 E010 — SWEEP_002 tuned

SWEEP_002 olasılıklarına E002 multiplier tabanı uygulandı. `fit` ve `unhealthy` için
`{0.970, 0.985, 1.000, 1.015, 1.030}` grid tarandı. Seçilen multiplier:

```text
[0.1892298422, 1.4877868293, 1.4073064332]
```

OOF balanced accuracy `0.948392`, E002 tuned'dan `-0.000192` düşüktür. Buna karşılık
E002 disagreement `%0.1802` ile beş aday arasındaki en yüksek farklılıktır. E010 için ek
dağılım veya disagreement hard gate tanımlanmadığından şema ve artefakt kontrollerini
geçerek **PASS** oldu. Bu adayın değeri, E002'ye göre çeşitlilik sağlamasıdır.

Submission:

```text
outputs/experiments/E010_SWEEP_002_tuned/submission_tuned.csv
```

### 4.3 E011 — E002/E004/E006 blend tuned

OOF üzerinde üç ağırlık seçeneği karşılaştırıldı:

```text
A) 0.60 E002 + 0.30 E004 + 0.10 E006
B) 0.65 E002 + 0.25 E004 + 0.10 E006
C) 0.70 E002 + 0.20 E004 + 0.10 E006
```

En iyi seçenek **A** oldu. Seçilen multiplier E009 ile aynıdır:

```text
[0.1892298422, 1.4588977647, 1.3799800947]
```

OOF balanced accuracy `0.948654`, E002 tuned'dan `+0.000069` yüksek ve bu serinin en iyi
OOF sonucudur. Test sınıf oranlarının üçü de eligibility bantlarında kaldığı için E011
**PASS** oldu. Disagreement `%0.0575` ile küçük olsa da E011 için disagreement hard gate
tanımlanmamıştır.

Submission:

```text
outputs/experiments/E011_blend_E002_E004_E006/submission_tuned.csv
```

### 4.4 E012 — E002 fit-up / unhealthy-down

E002 olasılıkları korunup multiplier sabit olarak değiştirildi:

```text
m_at-risk   = E002_at-risk
m_fit       = E002_fit * 1.012
m_unhealthy = E002_unhealthy * 0.992
```

Son multiplier:

```text
[0.1892298422, 1.4617866712, 1.3553863901]
```

OOF balanced accuracy `0.948540`, E002 tuned'dan `-0.000044` düşüktür. Fit oranı
`0.072060` ile beklenen `[0.0725, 0.0735]` aralığının altında kaldı. Değişiklik E002
etiketlerinin yalnız `%0.0179` kadarını etkiledi. Bu nedenle deney **FAIL** oldu.

Submission:

```text
outputs/experiments/E012_E002_fit_up_unhealthy_down/submission.csv
```

### 4.5 E013 — E002 consensus correction

Base tahmin E002 tuned'dır. E004 tuned ve E006 tuned aynı etikette birleşip E002'den
ayrıldığında E002 etiketi değiştirilir:

```text
if pred_E004 == pred_E006 and pred_E004 != pred_E002:
    final_pred = pred_E004
else:
    final_pred = pred_E002
```

OOF balanced accuracy `0.948418` ile minimum `0.9482` eşiğini geçti. Ancak test
disagreement `0.000859`, yani `%0.0859` oldu ve gerekli `%0.2–1.0` bandının altında
kaldı. Consensus kuralı E002'ye yeterince müdahale etmediği için deney **FAIL** oldu.

Submission:

```text
outputs/experiments/E013_E002_consensus_correction/submission.csv
```

## 5. Üretilen beş submission

Beş dosyanın tamamı production çalışmasında oluşturulmuştur:

```text
outputs/experiments/E009_blend_E002_E004_75_25/submission_tuned.csv
outputs/experiments/E010_SWEEP_002_tuned/submission_tuned.csv
outputs/experiments/E011_blend_E002_E004_E006/submission_tuned.csv
outputs/experiments/E012_E002_fit_up_unhealthy_down/submission.csv
outputs/experiments/E013_E002_consensus_correction/submission.csv
```

Her deney dizininde ayrıca şu dosyalar bulunur:

| Dosya | İçerik |
|---|---|
| `config.json` | Gerçek çalışmada kullanılan deney tarifi |
| `metrics.json` | OOF skor, multiplier/weight, dağılım ve disagreement |
| `eligibility.json` | PASS/FAIL kararı ve gerekçeleri |

Toplu makine-okunur raporlar:

```text
outputs/experiments/eligibility_report.csv
outputs/experiments/eligibility_report.json
```

## 6. Sonuç ve submit yorumu

- **E011**, serinin en yüksek OOF skoruna sahiptir ve dağılım kontrollerini geçmiştir.
- **E010**, OOF'ta E002'den düşük olsa da en yüksek disagreement ile en farklı adaydır.
- **E009**, OOF'ta küçük artış sağlamasına rağmen E002'ye aşırı benzer kalmıştır.
- **E012**, hedeflenen fit artışını üretmemiştir.
- **E013**, OOF tabanını korumuş fakat yeterli sayıda consensus düzeltmesi yapmamıştır.

Önceden tanımlanan eligibility politikasına göre öncelikli submit sırası **E010 ve E011**'dir.
E009, E012 ve E013 dosyaları teknik olarak geçerlidir; ancak submit edilmeleri kontrollü
deney planındaki eleme kararını bilinçli biçimde geçersiz kılmak anlamına gelir.
