---
goal: E002 merkezli XGBoost ve CatBoost diversity deneylerini minimum konfigürasyonla çalıştırmak
version: 1.0
date_created: 2026-07-05
last_updated: 2026-07-05
owner: repository maintainer
status: 'In progress'
tags: [feature, modeling, xgboost, catboost, ensemble, cache]
---

# Introduction

![Status: In progress](https://img.shields.io/badge/status-In%20progress-yellow)

Bu plan E019-E023 deneylerini geniş hiperparametre taraması yapmadan uygular. Amaç,
E002'nin public LB `0.94960` omurgasını korurken XGBoost ve CatBoost ile farklı hata
paternleri üretmek ve bu paternleri küçük oranlı probability blend'lerinde kullanmaktır.
Plan, önce kısa ve ucuz sözleşme koşuları, sonra yalnız iki tam model eğitimi, son olarak
eğitimsiz ensemble değerlendirmeleri yapar.

Mevcut yerel ölçüm tabanı: 690,088 train, 295,753 test, 63 V2-Core feature, 3-fold E002
tam koşusu yaklaşık 4 dakika ve cache boyutu yaklaşık 1.1 GiB'dir. Aşağıdaki XGBoost ve
CatBoost süreleri henüz ölçülmemiş tahminlerdir; ilk kısa koşunun gerçek süresiyle otomatik
yeniden hesaplanmalıdır.

## 1. Requirements & Constraints

- **REQ-001**: E019 ve E020, E002 ile aynı V2-Core feature sözleşmesini, `seed=42` ve aynı 3 stratified fold atamalarını kullanmalıdır.
- **REQ-002**: Model ailesi başına yalnız bir ana konfigürasyon tam veri üzerinde çalıştırılmalıdır; otomatik grid/random sweep yapılmamalıdır.
- **REQ-003**: E019-E023 raporları OOF balanced accuracy, fold-wise balanced accuracy, macro F1, üç sınıf recall, test class distribution, E002'ye göre changed rows, transition matrix ve disagreement confidence analizini içermelidir.
- **REQ-004**: E021, E022 ve E023 yalnız saklanmış OOF/test probability artefaktlarını kullanmalı; model yeniden eğitmemelidir.
- **REQ-005**: Standalone model için tuned OOF, ensemble için E002 tuned OOF referans alınmalıdır.
- **REQ-006**: Tüm probability matrisleri `CLASS_NAMES = [at-risk, fit, unhealthy]` sırasını kullanmalı ve satır toplamları `1 ± 1e-6` olmalıdır.
- **REQ-007**: Her tam model fold sonunda crash-recovery artefaktı yazmalıdır; tamamlanmış fold tekrar çalıştırılmamalıdır.
- **REQ-008**: E021-E023 multiplier scale değerleri ham multiplier değildir; `final_at_risk = E002_at_risk`, `final_fit = E002_fit × fit_scale`, `final_unhealthy = E002_unhealthy × unhealthy_scale` formülü zorunludur.
- **REQ-009**: Ana model diversity üretmezse model ailesi başına en fazla bir fallback config koşulabilir; fallback varsayılan pipeline içinde çalışmamalıdır.
- **REQ-010**: CatBoost adapter, preprocess edilmiş frame'de categorical kolon bulunup bulunmadığını çalışma anında belirlemeli; varsa native `cat_features`, yoksa numeric-only fit kullanmalıdır.
- **REQ-011**: Her ensemble raporu E018 benzeri tek yönlü minority push riskini ayrı bir `risk_summary` bölümüyle göstermelidir.
- **CON-001**: XGBoost ve CatBoost mevcut bağımlılıklarda yoktur; sürümleri `pyproject.toml` içinde alt/üst sınırla sabitlenmelidir.
- **CON-002**: Preprocessing fold train verisinde fit edilmeli; validation/test bilgisi feature fit veya multiplier seçiminde kullanılmamalıdır.
- **CON-003**: Dry-run metrikleri model seçmek için kullanılamaz; yalnız pipeline, schema, cache ve süre doğrulaması içindir.
- **CON-004**: Cache silinmemeli veya yeniden yazılmamalı; cache anahtarı fold indeksleri, data fingerprint, feature config ve encoding türünü içermelidir.
- **CON-005**: Aynı makinede XGBoost ve CatBoost tam koşuları paralel başlatılmamalıdır; CPU/RAM çekişmesi süre ve sonucu bozabilir.
- **GUD-001**: Önce test yaz, sonra en küçük model-adapter değişikliğini yap, hedefli testleri geçir, kısa koşuyu çalıştır, yalnız başarılıysa tam koşuya geç.
- **GUD-002**: E019/E020 standalone skorları düşük olsa bile diversity ölçümleri iyi ise ensemble kaynağı olarak korunmalıdır.
- **GATE-001**: Standalone submission yalnız tuned OOF `>= E002 tuned OOF + 0.00010`, foldların en az 2/3'ünde artış ve güvenli test dağılımı sağlarsa üretilmelidir.
- **GATE-002**: Ensemble submission adayı yalnız tuned OOF E002'yi geçer, foldların en az 2/3'ünde gerilemez, toplam changed rows 30-180 aralığında kalır, `at-risk -> {fit, unhealthy}` toplamı 150'yi aşmaz, `{fit, unhealthy} -> at-risk` toplamı 30'u aşmaz ve test class distribution E002'ye göre her sınıfta en fazla 0.10 yüzde puan saparsa uygun sayılmalıdır.
- **GATE-003**: Bir ana model için fallback yalnız şu koşullardan biri gerçekleşirse tetiklenebilir: tuned OOF `E002 tuned OOF - 0.00150` altındadır; E002 tuned test label disagreement oranı `%0.10` altındadır; veya ilgili micro blend dokuz multiplier pair'inin hiçbirinde GATE-002'yi geçmez. Fallback kararı `fallback_decision.json` ile kanıtlanmalıdır.

### Seçilmiş sabit model konfigürasyonları

#### E019 — XGBoost V2-Core

```yaml
model: xgboost
objective: multi:softprob
eval_metric: mlogloss
tree_method: hist
max_depth: 4
learning_rate: 0.03
n_estimators: 6000
min_child_weight: 100
subsample: 0.90
colsample_bytree: 0.90
reg_lambda: 10.0
reg_alpha: 0.1
max_bin: 256
class_weight_mode: sqrt_balanced
early_stopping_rounds: 250
n_jobs: 6
random_state: 42
```

Gerekçe: depth 4 büyük veri üzerinde variance'ı sınırlar; `min_child_weight=100` minority
split'lerini tamamen kapatmadan küçük leaf gürültüsünü azaltır; `lr=0.03`, güçlü L2 ve
sqrt-balanced ağırlık E002'den farklı ama kontrollü bir boundary hedefler.

#### E020 — CatBoost V2-Core

```yaml
model: catboost
loss_function: MultiClass
eval_metric: MultiClass
depth: 6
learning_rate: 0.035
iterations: 5000
l2_leaf_reg: 8.0
random_strength: 1.0
bootstrap_type: Bayesian
bagging_temperature: 0.5
class_weight_mode: sqrt_balanced
od_type: Iter
od_wait: 250
thread_count: 6
random_seed: 42
allow_writing_files: false
verbose: false
```

Gerekçe: depth 6 CatBoost'un categorical interaction kapasitesini korur; orta regularization
ve düşük bagging temperature, 690k satırda aşırı rastlantısallık yaratmadan LightGBM'den
farklı boundary üretir.

#### Koşullu fallback konfigürasyonları

Fallback'ler config dosyalarında tanımlanır fakat normal `train-xgb` ve `train-cat`
stage'lerinde çalıştırılmaz. Yalnız GATE-003 sağlanırsa ilgili `--stage fallback-xgb` veya
`--stage fallback-cat` açıkça ve bir kez çalışır.

```yaml
E019_XGB_alt:
  model: xgboost
  objective: multi:softprob
  eval_metric: mlogloss
  tree_method: hist
  max_depth: 5
  learning_rate: 0.025
  n_estimators: 7000
  min_child_weight: 50
  subsample: 0.90
  colsample_bytree: 0.85
  reg_lambda: 15.0
  reg_alpha: 0.1
  max_bin: 256
  class_weight_mode: sqrt_balanced
  early_stopping_rounds: 250
  n_jobs: 6
  random_state: 42

E020_CAT_alt:
  model: catboost
  loss_function: MultiClass
  eval_metric: MultiClass
  depth: 5
  learning_rate: 0.03
  iterations: 6000
  l2_leaf_reg: 12.0
  random_strength: 1.5
  bootstrap_type: Bayesian
  bagging_temperature: 0.7
  class_weight_mode: sqrt_balanced
  od_type: Iter
  od_wait: 250
  thread_count: 6
  random_seed: 42
  allow_writing_files: false
  verbose: false
```

### Seçilmiş ensemble konfigürasyonları

| Deney | Probability ağırlıkları | Multiplier scale pairs | Changed-row kapısı |
|---|---|---|---|
| E021 | E002 `0.95`, E019 `0.05` | fit/unhealthy Cartesian `[1.000, 1.005, 1.010]` | 30-150 |
| E022 | E002 `0.97`, E020 `0.03` | fit/unhealthy Cartesian `[1.000, 1.005, 1.010]` | 30-150 |
| E023 | E002 `0.94`, E019 `0.03`, E020 `0.03` | fit/unhealthy Cartesian `[1.000, 1.005, 1.010]` | 50-180 |

Dokuz scale pair ham multiplier değildir ve mevcut E002 tuned multiplier vektörü
`[0.1892298422, 1.4444532324, 1.3663169255]` üzerine uygulanır:

```text
candidate = [
  E002_at_risk_multiplier,
  E002_fit_multiplier * fit_scale,
  E002_unhealthy_multiplier * unhealthy_scale,
]
```

At-risk multiplier her pair içinde sabit tutulur. Seçim önce OOF balanced accuracy,
eşitlikte daha az changed row, sonra E002'ye daha yakın test class distribution sırasıyla
yapılır. Bu dar aralık E018'deki agresif `+4%` hareketini tekrar etmez.

## 2. Implementation Steps

### Implementation Phase 1 — Sözleşmeleri testlerle kilitle

- GOAL-001: Model ailesinden bağımsız CV, cache ve artefakt sözleşmesini regression testleriyle tanımla.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-001 | `tests/test_model_training.py` içinde model registry'nin `lightgbm`, `xgboost`, `catboost` değerlerini kabul ettiğini; bilinmeyen modeli reddettiğini test et. | | |
| TASK-002 | `tests/test_model_training.py` içinde her adapter için 150 satırlık sentetik 1-fold fit/predict testi yaz; çıktı shape, class order, normalization ve deterministic seed kontrolü ekle. | | |
| TASK-003 | `tests/test_experiment_framework.py` içinde fold assignment'ın E002/E019/E020 arasında byte-identical olduğunu test et. | | |
| TASK-004 | `tests/test_cache.py` içinde native-categorical ve one-hot encoding cache anahtarlarının çakışmadığını, aynı E002 fold'unun ikinci okumada hit verdiğini test et. | | |
| TASK-005 | `tests/test_candidate_experiments.py` içinde E021-E023 weight sum, ID alignment, E002 multiplier üzerine uygulanan 3x3 scale grid, transition matrix ve eligibility gate testlerini ekle; raw `1.005` multiplier kullanımının fail ettiğini doğrula. | | |
| TASK-037 | `tests/test_candidate_experiments.py` içinde fallback'in varsayılan kapalı olduğunu ve yalnız GATE-003 kanıtıyla tek kez açıldığını test et. | | |

Tamamlanma kriteri: yeni testler önce mevcut implementasyonda beklenen nedenle fail eder; Phase 2 sonrası tümü geçer.

### Implementation Phase 2 — Ortak CV runner ve model adapter'ları

- GOAL-002: Mevcut LightGBM davranışını bozmadan XGBoost ve CatBoost desteği ekle.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-006 | `pyproject.toml` dependencies listesine uyumlu sabit aralıklarla `xgboost` ve `catboost` ekle; lock dosyasını `uv sync --dev` ile güncelle. | | |
| TASK-007 | `src/kaggle_s6_e7/model_adapters.py` oluştur; `fit`, `predict_proba`, `best_iteration`, `save_model`, `feature_importance` ortak arayüzünü tanımla. | | |
| TASK-008 | `src/kaggle_s6_e7/training.py` içindeki LightGBM doğrudan çağrısını adapter registry ile değiştir; mevcut E002 artefakt isimlerini ve sonuçlarını koru. | | |
| TASK-009 | XGBoost için `build_one_hot_encoder()` dönüşümünü yalnız fold train'de fit et; sparse train/valid/test matrislerini encoding-aware cache'e yaz. | | |
| TASK-010 | CatBoost için preprocess edilmiş frame'de `category`, `object` veya `string` dtype kolonlarını çalışma anında bul; liste boş değilse native `cat_features` olarak geçir, boşsa `cat_features` argümanı vermeden numeric-only fit et. Missing/unknown kategorileri mevcut `FoldPreprocessor` sözleşmesiyle koru. | | |
| TASK-011 | `scripts/run_experiment.py` model-config seçimini experiment modeline göre çöz; dry-run limitlerini adapter bazında uygula. | | |
| TASK-012 | `configs/xgb_v2_core.yaml`, `configs/catboost_v2_core.yaml` ve `configs/e019_e020_experiments.yaml` dosyalarını ana ve koşullu fallback değerleriyle oluştur; fallback tanımlarını `enabled_by_default: false` yap. | | |
| TASK-038 | CatBoost adapter manifest'ine tespit edilen `categorical_columns`, `categorical_feature_count` ve seçilen `input_mode: native_categorical|numeric_only` değerlerini yaz. | | |
| TASK-013 | Run manifest'e model library/version, encoding cache key, wall-clock fold süresi, peak RSS bulunabiliyorsa peak RSS ve stop reason yaz. | | |

Tamamlanma kriteri: eski E002 hedefli regression testleri ile tüm adapter unit testleri geçer; cache miss/hit loglarda görünür.

### Implementation Phase 3 — Feature cache hazırlığı ve kısa koşular

- GOAL-003: Tam eğitime başlamadan veri yolu, cache reuse, bellek ve süreyi ölç.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-014 | E002'nin 3 fold assignment dosyasını doğrula; yoksa seed 42 ile bir kez üret ve E019/E020'de aynı dosyayı zorunlu input yap. | | |
| TASK-015 | E019 cache warm-up/dry-run çalıştır: 12,000 stratified train, 4,000 test, 1 fold, 100 estimator, early stop 20. | | |
| TASK-016 | Aynı E019 dry-run'ı ikinci kez çalıştır; preprocessing/encoding cache hit doğrula ve ikinci sürenin ilk süreden kısa olduğunu kaydet. | | |
| TASK-017 | E020 dry-run çalıştır: 12,000 stratified train, 4,000 test, 1 fold, 150 iteration, `od_wait=20`. | | |
| TASK-018 | Her dry-run için `metrics.json`, probability artefaktları, model dosyası, manifest ve süre kaydı bulunduğunu doğrula. NaN, class-order veya cache hatasında tam koşuyu bloke et. | | |
| TASK-019 | `scripts/estimate_runtime.py` ile `full_estimate = dry_seconds × 3 × full_rows / dry_rows × 0.65` başlangıç tahminini üret; tahmine `%35` güven aralığı ekle ve gerçek ilk full fold sonrası kalan süreyi fold medyanıyla güncelle. | | |

Beklenen yerel süreler:

| İş | İlk koşu | Cache-hit tekrar | Not |
|---|---:|---:|---|
| Unit/contract testleri | 10-30 sn | aynı | Model paket kurulum süresi hariç |
| E019 kısa koşu | 30-90 sn | 20-60 sn | Sparse one-hot cache ilk seferde yazılır |
| E020 kısa koşu | 45-120 sn | 35-100 sn | Native categorical cache mevcut fold frame'lerini kullanır |
| E019 tam 3-fold | 12-25 dk | N/A | İlk fold sonrası yeniden tahmin edilir |
| E020 tam 3-fold | 18-40 dk | N/A | CPU CatBoost tahmini |
| E021-E023 toplam | 20-90 sn | 10-45 sn | Model eğitimi yok, yalnız NumPy/metric işlemleri |
| Tüm pipeline | 32-68 dk | — | Paket indirme ve başarısız retry hariç |
| E019 fallback, koşullu | +15-30 dk | N/A | Yalnız GATE-003 XGB için sağlanırsa |
| E020 fallback, koşullu | +20-45 dk | N/A | Yalnız GATE-003 CatBoost için sağlanırsa |

Normal toplam süre 32-68 dakikadır. İki fallback'in de tetiklendiği en kötü koşullu senaryo
yaklaşık 67-143 dakikadır; fallback süreleri normal ETA'ya peşinen dahil edilmez.

Süre stop koşulu: dry-run tek model için 3 dakikayı veya tahmini tam koşu 60 dakikayı aşarsa
tam koşuya geçme; thread oversubscription, sparse densification ve cache miss sebebini düzelt.

### Implementation Phase 4 — E019 ve E020 tam OOF üretimi

- GOAL-004: Tek konfigürasyonlu iki diversity kaynağını aynı fold'larla üret.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-020 | `scripts/run_e019_e023_pipeline.sh --stage train-xgb` ile E019 tam 3-fold koşusunu başlat; fold başına probability ve checkpoint yaz. | | |
| TASK-021 | E019 tamamlanınca tuned standalone multipliers'ı geniş 2000-trial arama yerine yalnız `[0.98, 0.99, 1.00, 1.01, 1.02]` fit/unhealthy 25-pair grid ile seç. | | |
| TASK-022 | E019 OOF, fold stability, distribution ve E002 disagreement raporunu üret; başarısız standalone gate ensemble üretimini durdurmamalıdır. | | |
| TASK-023 | `scripts/run_e019_e023_pipeline.sh --stage train-cat` ile E020 tam 3-fold koşusunu E019 bittikten sonra çalıştır. | | |
| TASK-024 | E020 için aynı 25-pair standalone tuning ve diversity raporunu üret. | | |
| TASK-039 | E019/E020 ana sonuçlarından GATE-003 değerlerini hesapla; gerekirse yalnız ilgili fallback stage'ini çalıştır, ana ve fallback arasından daha yüksek diversity-güvenlik sıralamasına sahip probability kaynağını E021-E023'e bağla. | | |

Tamamlanma kriteri: `outputs/experiments/E019/` ve `E020/` altında OOF/test/fold probability,
fold metrics, tuned metrics, model files, label mapping, manifest ve diversity raporu eksiksizdir.

### Implementation Phase 5 — E021-E023 ensemble ve seçim

- GOAL-005: Üç sabit blend'i dar multiplier grid ile değerlendir ve en fazla üç güvenli submission adayı üret.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-025 | `configs/e021_e023_ensembles.yaml` içine yalnız E021 `95/5`, E022 `97/3`, E023 `94/3/3` tanımlarını ve E002 best multiplier vektörü üzerine uygulanacak dokuz scale pair'i yaz. | | |
| TASK-026 | `src/kaggle_s6_e7/candidate_experiments.py` raporuna fold-wise ensemble metrics, changed-row transition matrix, disagreement confidence bucket'ları ve `risk_summary` ekle. | | |
| TASK-027 | Confidence analizinde E002 disagreement satırlarını E002 margin quartile'larına ayır; her bucket için satır sayısı, E002/model accuracy, mean max probability ve mean entropy raporla. | | |
| TASK-028 | E021, E022 ve E023'ü tek probability pass ile üret; scale pair'leri E002 tuned multiplier üzerine çarp ve her candidate için eligibility gate'i uygula. | | |
| TASK-029 | Uygun adayları OOF gain, fold win count, changed-row güvenliği ve distribution drift sırasıyla sırala; öncelik eşitlikte E023, sonra E021, sonra E022 olsun. | | |
| TASK-030 | Yalnız gate geçen adaylar için `submission_<EXP>_tuned.csv` yaz; gate geçmeyenler için submission yerine `rejection_reasons.json` yaz. | | |
| TASK-040 | Her candidate için `risk_summary.json` yaz: toplam changed rows, altı yönlü transition sayıları, toplam at-risk-to-minority, toplam minority-to-at-risk, E018-benzeri tek yönlü push boolean'ı ve E002'ye göre sınıf bazında test distribution yüzde-puan sapması. | | |

Tamamlanma kriteri: Her deney için tek satırlık summary ve ayrıntılı JSON/CSV rapor vardır; hiçbir candidate gizli test label veya LB skoruyla seçilmez.

### Implementation Phase 6 — Orkestrasyon, dokümantasyon ve doğrulama

- GOAL-006: Pipeline'ı idempotent, kısa-run destekli ve yeniden başlatılabilir yap.

| Task | Description | Completed | Date |
|---|---|---|---|
| TASK-031 | `scripts/run_e019_e023_pipeline.sh` oluştur; sıralama `preflight -> dry-run -> E019 -> E020 -> E021/E022/E023 -> validate -> report` olsun. | | |
| TASK-032 | Runner'a `--stage`, `--resume`, `--dry-run`, `--force` ve `MIN_FREE_GB=8` kontrolleri ekle; `fallback-xgb`/`fallback-cat` stage'lerini GATE-003 kanıtı yoksa reddet ve mevcut başarılı artefaktı varsayılan olarak atla. | | |
| TASK-033 | `docs/e019-e023-diversity-pipeline.md` oluştur; komutlar, cache dizinleri, artefakt ağacı, süre tahmin yöntemi, failure recovery ve submission gate'lerini anlat. | | |
| TASK-034 | `scripts/validate_experiments.py` içine E019-E023 config schema, source existence, class mapping, ID alignment ve probability normalization kontrolleri ekle. | | |
| TASK-035 | Hedefli pytest, tüm pytest, Ruff, mypy ve compileall çalıştır; ardından iki kısa koşu ve ensemble smoke run çalıştır. | | |
| TASK-036 | `docs/e019-e023-diversity-results.md` dosyasını gerçek süreler, cache hit oranı ve bütün zorunlu metriklerle otomatik üret. | | |

### Yürütme komutları

```bash
# 1. Bağımlılıklar ve kalite kapıları
uv sync --dev
uv run pytest tests/test_model_training.py tests/test_cache.py tests/test_candidate_experiments.py -q
uv run python scripts/check.py

# 2. Kısa sözleşme koşuları ve gerçek süre tahmini
bash scripts/run_e019_e023_pipeline.sh --stage dry-run

# 3. Tam modeller; aynı makinede sıralı
bash scripts/run_e019_e023_pipeline.sh --stage train-xgb --resume
bash scripts/run_e019_e023_pipeline.sh --stage train-cat --resume

# 4. Yalnız fallback_decision.json gerekli diyorsa ilgili satırı çalıştır
bash scripts/run_e019_e023_pipeline.sh --stage fallback-xgb --resume
bash scripts/run_e019_e023_pipeline.sh --stage fallback-cat --resume

# 5. Eğitimsiz blend ve raporlama
bash scripts/run_e019_e023_pipeline.sh --stage ensemble --resume
bash scripts/run_e019_e023_pipeline.sh --stage validate
```

## 3. Alternatives

- **ALT-001**: 8-12 XGBoost ve CatBoost config sweep'i reddedildi; hesap maliyetini büyütür ve küçük OOF farklarında selection noise üretir.
- **ALT-002**: E021/E022/E023 için dört-beş blend oranı reddedildi; kullanıcı hedefi minimum ama güçlü adaylardır.
- **ALT-003**: Balanced class weights ana seçim olarak reddedildi; mevcut E002 boundary'sinden fazla minority shift riski taşır. Sqrt-balanced iki yeni model için kontrollü başlangıçtır.
- **ALT-004**: XGBoost'a ham pandas category geçmek reddedildi; mevcut explicit unknown handling kullanan fold-safe one-hot yolu daha denetlenebilirdir.
- **ALT-005**: XGBoost ve CatBoost tam koşularını paralel çalıştırmak reddedildi; 6 thread kullanan iki model aynı makinede ölçüm ve bellek istikrarını bozar.
- **ALT-006**: E002'nin 2000-trial multiplier aramasını her blend için tekrarlamak reddedildi; 9-pair dar grid daha hızlıdır ve E018 benzeri aşırı shift'i sınırlar.
- **ALT-007**: Fallback config'leri varsayılan çalıştırmak reddedildi; yalnız ölçülmüş diversity başarısızlığı ek hesap maliyetini haklı çıkarır.
- **ALT-008**: CatBoost'a koşulsuz `cat_features` vermek reddedildi; gerçek transformed dtype kontrolü native-categorical ve numeric-only pipeline'ları güvenli ayırır.

## 4. Dependencies

- **DEP-001**: Mevcut `FoldPreprocessor`, `FoldFeatureCache`, metric ve candidate experiment altyapısı.
- **DEP-002**: `xgboost` CPU histogram implementation.
- **DEP-003**: `catboost` CPU multiclass implementation.
- **DEP-004**: NumPy, pandas, scikit-learn, PyArrow ve YAML mevcut bağımlılıkları.
- **DEP-005**: E002'nin `oof_proba.npy`, `test_proba.npy`, `fold_assignments.csv`, `best_multipliers.json` ve ID artefaktları.

## 5. Files

- **FILE-001**: `pyproject.toml` — XGBoost ve CatBoost bağımlılıkları.
- **FILE-002**: `src/kaggle_s6_e7/model_adapters.py` — ortak model adapter registry.
- **FILE-003**: `src/kaggle_s6_e7/training.py` — modelden bağımsız CV ve checkpoint.
- **FILE-004**: `src/kaggle_s6_e7/cache.py` — encoding-aware cache metadata.
- **FILE-005**: `src/kaggle_s6_e7/candidate_experiments.py` — E021-E023 diagnostics ve gate'ler.
- **FILE-006**: `scripts/run_experiment.py` — model-aware config ve kısa koşu.
- **FILE-007**: `scripts/run_e019_e023_pipeline.sh` — idempotent orchestration.
- **FILE-008**: `scripts/estimate_runtime.py` — ölçüme dayalı süre tahmini.
- **FILE-009**: `configs/e019_e020_experiments.yaml` — iki standalone deney.
- **FILE-010**: `configs/xgb_v2_core.yaml` ve `configs/catboost_v2_core.yaml` — seçilmiş sabit parametreler.
- **FILE-011**: `configs/e021_e023_ensembles.yaml` — üç sabit micro blend.
- **FILE-012**: `docs/e019-e023-diversity-pipeline.md` — çalışma ve recovery rehberi.
- **FILE-013**: `docs/e019-e023-diversity-results.md` — otomatik sonuç raporu.
- **FILE-014**: `tests/test_model_training.py`, `tests/test_cache.py`, `tests/test_candidate_experiments.py` — regression ve contract testleri.

## 6. Testing

- **TEST-001**: Model adapter'ları aynı input üzerinde deterministic ve normalize probability üretir.
- **TEST-002**: E002 mevcut LightGBM config/artefakt davranışı regression testini geçer.
- **TEST-003**: E002/E019/E020 fold assignment'ları aynıdır.
- **TEST-004**: Feature/encoding cache aynı anahtarda hit, farklı encoding'de miss üretir.
- **TEST-005**: OOF/test ID ve label mapping uyuşmazlığı blend'i hard fail eder.
- **TEST-006**: Blend weight toplamı 1 değilse config validation fail eder.
- **TEST-007**: Multiplier grid yalnız 9 tanımlı pair'i değerlendirir ve deterministic tie-break uygular.
- **TEST-008**: Transition matrix changed-row toplamına eşittir.
- **TEST-009**: Eligibility sınırı dışındaki aday submission üretmez.
- **TEST-010**: Dry-run E019/E020 tüm zorunlu artefaktları üretir ve ikinci E019 koşusu cache hit verir.
- **TEST-011**: `uv run pytest`, Ruff, mypy ve compileall sıfır hata ile biter.
- **TEST-012**: Scale `[1.000, 1.005, 1.010]` doğrudan multiplier olarak değil E002 best multiplier vektörü üzerinde uygulanır; at-risk değeri tüm candidate'larda aynıdır.
- **TEST-013**: CatBoost categorical kolonlu frame'de native mode, categorical kolonsuz frame'de numeric-only mode seçer.
- **TEST-014**: Fallback varsayılan koşuda çalışmaz; yalnız GATE-003 ve `fallback_decision.json` mevcutsa bir kez çalışır.
- **TEST-015**: `risk_summary` transition toplamlarıyla tutarlıdır ve 150 at-risk-to-minority, 30 minority-to-at-risk, 0.10 yüzde-puan distribution sınırlarını doğru uygular.

## 7. Risks & Assumptions

- **RISK-001**: XGBoost one-hot matrisi cache boyutunu ve RAM kullanımını artırabilir. Sparse format zorunludur; densification hard fail olmalıdır.
- **RISK-002**: CatBoost native categorical training tahminden uzun sürebilir. İlk full fold sonrası ETA 60 dakikayı aşarsa pipeline güvenli biçimde durmalıdır.
- **RISK-003**: Sqrt-balanced weighting standalone tuned OOF'u yükseltirken test distribution'ı fazla minority'ye itebilir. Distribution ve transition gate bunu engeller.
- **RISK-004**: Aynı OOF üzerinde multiplier ve blend seçimi küçük overfit yaratabilir. Grid bilinçli olarak dar ve aday sayısı üç ile sınırlıdır.
- **RISK-005**: 3-fold küçük farklarda gürültülüdür. Submission için fold win/stability kapısı OOF tek skorundan daha önceliklidir.
- **RISK-006**: İki fallback de tetiklenirse toplam süre normal tahmini aşabilir; bu koşullu üst sınır 67-143 dakika olarak ayrıca raporlanmalıdır.
- **ASSUMPTION-001**: E002 artefaktları değişmeden ve doğru ID sırasıyla mevcuttur.
- **ASSUMPTION-002**: Yerel makine en az 8 GiB boş disk ve 6 CPU thread sağlayabilir.
- **ASSUMPTION-003**: E002 mevcut tam koşu süresi yaklaşık 4 dakikadır; yeni model tahminleri bu ölçüme ve 690k/296k veri boyutuna dayanır.
- **ASSUMPTION-004**: E019 veya E020 standalone gate'i geçmese bile geçerli probability ürettiyse E021-E023 devam eder.

## 8. Related Specifications / Further Reading

- `docs/experiment-runbook.md`
- `docs/e018-e002-heavy-micro-blend-results.md`
- `docs/rules/experiment-management.md`
- `docs/rules/quality-gates.md`
- `plan/data-fold-feature-cache-1.md`
- `plan/feature-experiment-pipelines-1.md`
