# E014–E015 E002-Merkezli Mikro Blend Pipeline ve Sonuç Raporu

## 1. Amaç ve karar bağlamı

E010'un public leaderboard skorunun `0.94901` gelmesi, E002'den daha yüksek model
çeşitliliğinin tek başına fayda sağlamadığını gösterdi. Bu nedenle E014 ve E015 yeni model
veya geniş sweep yerine, public lider **E002 tuned** karar sınırına yakın iki kontrollü
probability blend olarak tasarlandı.

Public referansları:

| Submission | Public skor |
|---|---:|
| **E002 tuned** | **0.94960** |
| E011 — 60/30/10 blend | 0.94957 |
| E009 — 75/25 blend | 0.94948 |
| E004 tuned | 0.94941 |
| E006 tuned | 0.94905 |
| E010 — SWEEP_002 tuned | 0.94901 |

E002 multiplier tabanı:

```text
at-risk   = 0.1892298422
fit       = 1.4444532324
unhealthy = 1.3663169255
```

E014, public'te E002'nin yalnız `0.00003` gerisinde kalan E011'in multiplier sınırını
E002'ye yaklaştırır. E015 ise aynı üç modeli kullanırken E002 probability ağırlığını
`0.60`tan `0.70`e çıkarır. Her iki deneyde seçim yalnız train OOF balanced accuracy ile
yapıldı; test etiketleri ve leaderboard sonuçları tuning sırasında kullanılmadı.

## 2. Pipeline'ın kapsamı ve mimarisi

### 2.1 Pipeline ne yapar, ne yapmaz?

E014–E015 bir training pipeline değildir. Yeni LightGBM modeli eğitmez, feature üretmez
veya fold'ları yeniden kurmaz. Daha önce aynı fold düzeniyle üretilmiş E002, E004 ve E006
OOF/test olasılıklarını yeniden kullanır.

İşlem sırası:

1. Kaynak deneylerin OOF ve test probability matrislerini yükle.
2. Kaynakların aynı satır ve sınıf düzeninde olduğunu doğrula.
3. Deney tarifindeki ağırlıklarla probability-level blend oluştur.
4. Yalnız OOF gerçek etiketleriyle izin verilen multiplier adaylarını karşılaştır.
5. En yüksek OOF balanced accuracy veren kombinasyonu seç.
6. Aynı blend ve multiplier'ı değiştirmeden test probability'lerine uygula.
7. E002 tuned ile disagreement ve test sınıf dağılımını hesapla.
8. Submission, config, metrics, eligibility ve toplu rapor artefaktlarını yaz.

Public skorlar deney hipotezini oluşturmak için kullanılmıştır; ağırlık veya multiplier
seçimini doğrudan optimize etmek için kullanılmamıştır.

### 2.2 Girdi artefaktları

Her kaynak deneyden aşağıdaki dosyalar okunur:

| Artefakt | Pipeline'daki görevi |
|---|---|
| `oof_proba.npy` | Blend ve multiplier seçimini train OOF üzerinde yapmak |
| `test_proba.npy` | Seçilen karar kuralını test setine uygulamak |
| `oof_pred.csv` | OOF ID sırasını ve gerçek etiketleri almak |
| `submission_argmax.csv` | Test ID sırasını almak |
| `label_mapping.json` | Probability kolonlarının sınıf sırasını doğrulamak |
| `best_multipliers.json` | E002 tuned multiplier vektörünü taban almak |

Kaynak dizinler:

```text
outputs/experiments/E002/
outputs/experiments/E004/
outputs/experiments/E006/
```

### 2.3 Hizalama ve güvenlik kontrolleri

Submission üretiminden önce pipeline şu invariant'ları doğrular:

- OOF ID dizileri kaynakların tamamında birebir aynıdır.
- OOF gerçek etiketleri kaynakların tamamında birebir aynıdır.
- Test ID dizileri ve sıraları birebir aynıdır.
- OOF ve test probability shape'ları aynıdır.
- Class mapping `at-risk=0`, `fit=1`, `unhealthy=2` düzenindedir.
- Multiplier değerleri sonlu ve pozitiftir.
- Probability matrisleri sonlu, pozitif toplamlı ve üç sınıflıdır.

Bir kontrol başarısız olursa pipeline submission yazmadan hata verir. Böylece farklı satır
sıralarındaki modellerin sessizce blend edilmesi engellenir.

### 2.4 Probability blend ve karar kuralı

Kaynak probability'leri satır bazında normalize edilir. Ağırlıklar da toplamları `1.0`
olacak şekilde normalize edildikten sonra:

```text
p_blend(i, c) = Σ weight_model * p_model(i, c)
```

hesaplanır. Nihai sınıf:

```text
prediction(i) = argmax_c(p_blend(i, c) * multiplier(c))
```

ile seçilir. `at-risk` multiplier sabit tutulur; yalnız `fit` ve `unhealthy` scale
değerleri taranır. Böylece E002 karar sınırından uzaklaşma kontrollü kalır.

### 2.5 Kod bileşenleri

| Dosya | Sorumluluk |
|---|---|
| `configs/e014_e015_experiments.yaml` | Blend ağırlıkları, multiplier adayları ve çıktı adları |
| `scripts/run_e014_e015_experiments.sh` | Tek komutluk production entrypoint |
| `scripts/generate_postprocess_experiments.py` | CLI argümanları ve suite çağrısı |
| `src/kaggle_s6_e7/candidate_experiments.py` | Kaynak yükleme, hizalama, seçim ve artefakt üretimi |
| `src/kaggle_s6_e7/ensemble.py` | Normalization, blend, multiplier search ve disagreement |

E014 için motora explicit `scale_pairs` desteği eklendi. Böylece yalnız M0, M1 ve M2
karşılaştırılır. Genel kartezyen grid kullanılsaydı planda olmayan
`fit=1.000 / unhealthy=1.005` adayı da yanlışlıkla aramaya girecekti.

## 3. Çalıştırma ve tekrar üretilebilirlik

İki aday tek komutla üretilir:

```bash
bash scripts/run_e014_e015_experiments.sh
```

Mevcut çıktıları bilinçli olarak yeniden üretmek için:

```bash
FORCE=1 bash scripts/run_e014_e015_experiments.sh
```

Varsayılan çalışma tamamlanmış adayları tekrar hesaplamaz. `FORCE=1`, E014/E015
artefaktlarını aynı config ve kaynaklarla yeniden yazar. Yeni toplu rapor eski E009–E013
raporunu ezmemek için ayrı bir stem kullanır:

```text
e014_e015_eligibility_report
```

## 4. Referans OOF sonuçları

E002 tuned:

| Metrik | Değer |
|---|---:|
| Balanced accuracy | 0.948584170185 |
| Macro F1 | 0.874625177251 |
| At-risk recall | 0.942891955427 |
| Fit recall | 0.945355877698 |
| Unhealthy recall | 0.957504677431 |

E002 test tahmin dağılımı:

| Sınıf | Adet | Oran |
|---|---:|---:|
| at-risk | 241,579 | %81.68269 |
| fit | 21,297 | %7.20095 |
| unhealthy | 32,877 | %11.11636 |

E011, `60/30/10` blend ve E002 tabanına göre `+1%/+1%` multiplier scale ile
`0.948653658184` OOF balanced accuracy üretmişti. E014 bu multiplier sınırını yumuşatır;
E015 ise E002 probability ağırlığını artırır.

## 5. E014 — E011 blend, daha az agresif multiplier

Probability formülü E011 ile aynıdır:

```text
p = 0.60 * E002 + 0.30 * E004 + 0.10 * E006
```

Yalnız üç önceden belirlenmiş multiplier adayı OOF üzerinde karşılaştırıldı:

| Aday | Fit scale | Unhealthy scale | OOF balanced accuracy |
|---|---:|---:|---:|
| M0 | 1.000 | 1.000 | 0.948580668262 |
| **M1** | **1.005** | **1.005** | **0.948654398698** |
| M2 | 1.005 | 1.000 | 0.948638101877 |

OOF seçimi **M1** oldu. Nihai multiplier:

```text
at-risk   = 0.1892298422
fit       = 1.4516754985
unhealthy = 1.3731485101
```

Bunlar E011'in `+1%/+1%` multiplier'ından daha az agresiftir. E014, E002 tuned'a göre
`+0.000070228513`; E011'e göre yalnız `+0.000000740515` OOF artışı üretmiştir. E011 farkı
pratik olarak eşit kabul edilmelidir.

### 5.1 E014 test etkisi

E014, 295,753 test satırının yalnız **173** tanesinde E002'den farklı sınıf üretmiştir:

```text
disagreement = 173 / 295753 = %0.05849
```

| Sınıf | E002 | E014 | Net fark |
|---|---:|---:|---:|
| at-risk | 241,579 | 241,571 | -8 |
| fit | 21,297 | 21,302 | +5 |
| unhealthy | 32,877 | 32,880 | +3 |

Net kolon farklarının mutlak toplamı 173 değildir; değişimlerin bir kısmı azınlık
sınıfları arasında karşılıklı geçiştir.

## 6. E015 — E002 ağırlığı yüksek 70/20/10 blend

Probability formülü:

```text
p = 0.70 * E002 + 0.20 * E004 + 0.10 * E006
```

Fit ve unhealthy için E002 multiplier çevresindeki `{1.000, 1.005, 1.010}` kartezyen
mikro-grid tarandı:

| Fit scale | Unhealthy scale | OOF balanced accuracy |
|---:|---:|---:|
| 1.000 | 1.000 | 0.948549970469 |
| 1.000 | 1.005 | 0.948583677041 |
| 1.000 | 1.010 | 0.948591173929 |
| 1.005 | 1.000 | 0.948570031433 |
| 1.005 | 1.005 | 0.948597963400 |
| 1.005 | 1.010 | 0.948611234894 |
| 1.010 | 1.000 | 0.948578905170 |
| 1.010 | 1.005 | 0.948606837136 |
| **1.010** | **1.010** | **0.948614334024** |

OOF, grid içindeki `+1%/+1%` adayını seçti. Nihai multiplier:

```text
at-risk   = 0.1892298422
fit       = 1.4588977647
unhealthy = 1.3799800947
```

E015, E002 tuned'a göre `+0.000030163839` OOF artışı üretmiştir; ancak E011 ve E014'ten
düşüktür. Avantajı E011'e göre daha yüksek E002 probability ağırlığıdır.

### 6.1 E015 test etkisi

E015, 295,753 test satırının **131** tanesinde E002'den farklıdır:

```text
disagreement = 131 / 295753 = %0.04429
```

| Sınıf | E002 | E015 | Net fark |
|---|---:|---:|---:|
| at-risk | 241,579 | 241,530 | -49 |
| fit | 21,297 | 21,310 | +13 |
| unhealthy | 32,877 | 32,913 | +36 |

E015 probability olarak E002'ye daha fazla ağırlık verse de seçilen `+1%/+1%` multiplier
azınlık sınıflarını artırır. Ölçülen label disagreement yine de E014'ten düşüktür.

## 7. Production sonuç özeti

| Deney | OOF bal. acc. | Δ E002 OOF | E002 disagreement | Değişen satır | Test at-risk | Test fit | Test unhealthy |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E014** | **0.948654398698** | **+0.000070228513** | %0.05849 | 173 | %81.67998 | %7.20263 | %11.11739 |
| E015 | 0.948614334024 | +0.000030163839 | %0.04429 | 131 | %81.66612 | %7.20534 | %11.12854 |

İki aday da teknik olarak geçerli üretildi. Config'te ek bir hard eligibility bandı
olmadığından ikisi de `eligible=true` durumundadır. Bu değer public skor garantisi değil,
artefakt ve tanımlı kontrollerin geçtiği anlamına gelir.

### 7.1 OOF yorumu

- E014 iki yeni aday arasındaki en yüksek OOF balanced accuracy değerine sahiptir.
- E014 ile E011 arasındaki fark yalnız `0.00000074`; anlamlı üstünlük sayılmamalıdır.
- E015 E002'den yüksek, fakat E011/E014'ten düşük OOF üretmiştir.
- Disagreement oranları `%0.06` altında olduğundan bunlar bağımsız modeller değil, E002
  karar sınırının mikro varyantlarıdır.
- OOF artışı leaderboard artışını garanti etmez; E009–E010 sonuçları bunu göstermiştir.

### 7.2 Public leaderboard yorumu

E014 ve E015 için bu rapor hazırlanırken public skor bulunmamaktadır. Bu nedenle:

- E014, daha yüksek OOF ve E011'e göre yumuşatılmış sınırla ilk submit adayıdır.
- E015, E002 ağırlığı yüksek ikinci mikro adaydır.
- E002 tuned, yeni public sonuç gelene kadar güvenli final referansıdır.

## 8. Üretilen dosyalar ve artefakt sözlüğü

Submit dosyaları:

```text
outputs/experiments/E014_E011_less_aggressive_multiplier/submission.csv
outputs/experiments/E015_blend_E002_E004_E006_70_20_10/submission.csv
```

Her deney dizininde:

| Dosya | İçerik |
|---|---|
| `submission.csv` | Kaggle'a yüklenebilir `id,health_condition` dosyası |
| `config.json` | Gerçek çalışmada kullanılan blend ve search tarifi |
| `metrics.json` | OOF skor, multiplier, weights, recall, dağılım ve disagreement |
| `eligibility.json` | PASS/FAIL durumu, nedenler ve toplu rapor satırı |

Toplu raporlar:

```text
outputs/experiments/e014_e015_eligibility_report.csv
outputs/experiments/e014_e015_eligibility_report.json
```

Her submission 295,753 veri satırı ve header içerir. Kolon şeması ve ID sırası E002 tuned
submission ile birebir doğrulanmıştır.

## 9. Doğrulama kanıtı

| Kontrol | Sonuç |
|---|---|
| Unit test (`tests/test_ensemble.py`) | 9 test geçti |
| Ruff | Hata yok |
| mypy | Hata yok |
| Shell syntax (`bash -n`) | Geçti |
| Submission satır sayısı | İki dosyada da 295,753 |
| Submission kolonları | E002 ile aynı |
| Submission ID sırası | E002 ile birebir aynı |
| Null target kontrolü | Null yok |

Testler ayrıca E014 explicit scale pair aramasının yalnız izin verilen adayları kullandığını
ve yeni toplu raporun eski E009–E013 eligibility raporunu ezmediğini doğrular.

## 10. Submit sırası ve final karar çerçevesi

Önerilen sıra:

1. **E014** — E011'in public'te `0.00003` eksik kalan sınırını daha az agresif multiplier
   ile düzeltmeyi hedefliyor ve iki yeni adayın en yüksek OOF skoruna sahip.
2. **E015** — E002'ye daha yüksek probability ağırlığı veren muhafazakâr alternatif.

Beklenen kazanç büyük değildir; hedef yaklaşık `0.00003–0.0002` ölçeğinde mikro
iyileştirmedir. Public sonuçlar geldikten sonra karar kuralı:

1. E014 veya E015 `0.94960` üstüne çıkarsa yeni public lider olur.
2. İkisi de E002 altında kalırsa final için E002 tuned korunur.
3. Private seçimde ikinci dosya hakkı varsa public'e çok yakın ve ensemble niteliği taşıyan
   E011 alternatif olarak tutulabilir.
4. OOF farkları çok küçük olduğundan yalnız local skora bakarak kesin üstünlük iddiası
   yapılmaz.
