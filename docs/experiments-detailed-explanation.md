# Kaggle S6E7 Deneylerinin Ayrıntılı Açıklaması

Bu belge projede tanımlanan ve çalıştırılan deneyleri yalnızca **ne yapıldı** düzeyinde
değil, **neden yapıldı, yöntem nasıl çalışıyor, hangi aşamada uygulanıyor ve sonuç nasıl
yorumlanmalı** sorularıyla açıklar. Hedef kitle; projeyi yeni devralan, makine öğrenmesine
temel düzeyde aşina bir geliştirici veya veri bilimcidir.

> Sonuç anlık görüntüsü: `outputs/experiments/` ve `outputs/leaderboard_local.csv`,
> 4 Temmuz 2026. Sonraki koşular skorları değiştirebilir.

## 1. Büyük resim: Deney zinciri neyi ölçüyor?

Bu bir üç sınıflı sınıflandırma problemidir. Model her satırı `at-risk`, `fit` veya
`unhealthy` sınıflarından birine atar. Deneylerin ana soruları şunlardır:

1. **E001:** Ham değişkenlerle güvenilir bir referans kurulabiliyor mu?
2. **E002–E006:** Feature engineering fikirlerinden hangisi gerçekten ek bilgi taşıyor?
3. **E007:** Modeli yeniden eğitmeden karar sınırları iyileştirilebilir mi?
4. **E008:** Sınıf dengesizliği eğitim sırasında daha iyi ele alınabilir mi?
5. **SWEEP deneyleri:** Feature set sabitken LightGBM parametreleri iyileştirilebilir mi?

Bu ayrım önemlidir. Aynı anda hem feature’ları hem model parametrelerini değiştirirsek skor
değişiminin nedenini bilemeyiz. Bu nedenle önce sabit LightGBM ayarlarıyla feature ablation,
sonra kazanan pipeline üzerinde model parametre araması yapılır.

## 2. Ortak veri ve değerlendirme akışı

Her eğitim deneyinde aynı akış uygulanır:

```text
train.csv
  -> hedefi sınıf oranını koruyarak 3 fold'a ayır
  -> her fold için preprocessing istatistiklerini yalnız fold-train'de öğren
  -> fold-train ile LightGBM eğit
  -> fold-valid için olasılık üret (OOF)
  -> test için olasılık üret
  -> üç fold'un test olasılıklarını ortala
  -> OOF metriklerini ve bütün artefaktları kaydet
```

### 2.1 Stratified 3-fold çapraz doğrulama

`StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` kullanılır. Stratification,
her fold içinde üç sınıfın oranını ana veriye yakın tutar.

Örnek olarak veride kabaca `%85 at-risk`, `%6 fit`, `%9 unhealthy` varsa rastgele ama
stratified bir bölme her valid fold’da bu oranları korumaya çalışır. Normal rastgele
bölmede azınlık sınıfı bir fold’a daha az düşebilir; bu da recall karşılaştırmasını
gereksiz yere oynatır.

### 2.2 Fold-safe preprocessing ve veri sızıntısı

Median, quantile sınırı ve kategori seviyeleri gibi veriyle öğrenilen bütün değerler yalnız
o fold’un eğitim kısmında hesaplanır. Valid ve test parçalarına aynı değerler uygulanır.

```text
Yanlış: tüm train medianı -> fold validation'a uygula
Doğru: fold-train medianı -> aynı fold'un validation ve test parçasına uygula
```

Yanlış yöntemde validation satırları kendi preprocessing değerlerini dolaylı biçimde
etkiler. Buna veri sızıntısı denir ve lokal skoru olduğundan iyimser gösterebilir.

### 2.3 OOF olasılığı nedir?

OOF, “out-of-fold” demektir. Her train satırının tahmini, o satırı eğitimde görmeyen model
tarafından üretilir. Üç fold sonunda her satır için üç sınıf olasılığı bulunur:

```text
satır 42 -> [at-risk=0.71, fit=0.18, unhealthy=0.11]
```

Bu olasılıklar bir araya getirilerek `oof_proba.npy` oluşturulur. Model seçimi ve E007
multiplier tuning bu dosya üzerinde yapılır. Test etiketleri bilinmediği için test
olasılıklarıyla ayar yapmak yasaktır.

### 2.4 Neden balanced accuracy?

Normal accuracy çoğunluk sınıfına ağırlık verir. Balanced accuracy ise her sınıf recall
değerinin aritmetik ortalamasıdır:

```text
balanced_accuracy = (recall_at-risk + recall_fit + recall_unhealthy) / 3
```

Örneğin recall değerleri `0.99`, `0.60`, `0.60` ise balanced accuracy `0.73` olur. Veri
çoğunlukla `at-risk` olsa bile ilk sınıftaki yüksek başarı diğer iki sınıftaki zayıflığı
gizleyemez.

Ek olarak macro F1, fold standard deviation, confusion matrix ve tahmin dağılımı izlenir.
Fold std düşükse skor farklı veri parçalarında daha kararlıdır.

### 2.5 LightGBM bu projede nasıl çalışıyor?

LightGBM ardışık karar ağaçları kuran gradient boosting yöntemidir. İlk ağaçların yaptığı
hatalar sonraki ağaçlar tarafından düzeltilmeye çalışılır. `learning_rate=0.035`, her yeni
ağacın katkısını küçültür; `n_estimators=12000` üst sınırdır. Model çoğunlukla bu sınıra
ulaşmaz çünkü validation `multi_logloss` 300 tur iyileşmezse early stopping devreye girer.

Basit bir ağaç dalı şu fikre benzeyebilir:

```text
bmi >= 30?
  evet -> step_count < 3000?
           evet -> unhealthy olasılığını artır
  hayır -> sleep_quality kategorisine bak
```

Gerçek model bu tür çok sayıda ağacın katkısını toplar. Kategorik kolonlar LightGBM’e
native categorical dtype ile verilir; bilinmeyen kategori `__UNKNOWN__` seviyesine alınır.

## 3. Ortak feature yöntemleri

### 3.1 Median imputasyon

Eksik numeric değer, ilgili kolonun **fold-train medianı** ile doldurulur. Median uç
değerlerden ortalamaya göre daha az etkilenir.

```text
fold-train BMI: [21, 23, 24, 31, 50]
median: 24
valid satırındaki eksik BMI -> 24
```

Categorical eksikler `missing`, fold-train’de görülmeyen kategoriler `__UNKNOWN__` olur.

### 3.2 Missing flag ve missing count

İmputasyon eksikliği giderir ama “bu değer aslında eksikti” bilgisini silebilir. Her ham
kolon için `<kolon>_is_missing` üretilir. E002’den itibaren ayrıca satırdaki toplam eksik
sayısı `missing_count` tutulur.

```text
sleep_duration = NaN, bmi = NaN, diğerleri dolu
sleep_duration_is_missing = 1
bmi_is_missing = 1
missing_count = 2
```

Eksiklik rastgele değilse bu işaretler hedef hakkında sinyal taşıyabilir.

### 3.3 Güvenli ratio feature’ları

Oranlar iki ham değeri bağlamsallaştırır. Kodda bölme `pay / (payda + 1)` biçimindedir;
`+1`, sıfıra bölmeyi önler.

| Feature | Formül | Sezgisel anlam |
|---|---|---|
| `calorie_per_step` | calorie / (step + 1) | Adım başına enerji harcaması |
| `calorie_per_exercise_min` | calorie / (exercise + 1) | Egzersiz dakikası başına enerji |
| `step_per_exercise_min` | step / (exercise + 1) | Egzersiz süresine göre hareket yoğunluğu |
| `water_per_bmi` | water / (BMI + 1) | Vücut ölçüsüne göre su tüketimi |
| `exercise_per_bmi` | exercise / (BMI + 1) | BMI’a göre egzersiz süresi |
| `steps_per_sleep_hour` | step / (sleep + 1) | Uyku süresine göre günlük hareket |

Örnek: `step_count=8000`, `exercise_duration=39` ise
`step_per_exercise_min = 8000 / 40 = 200` olur. Tek başına 8000 adım yerine egzersiz
süresiyle birlikte yorumlanabilen yeni bir sinyal elde edilir.

### 3.4 Categorical interaction

İki kategori string birleştirme ile yeni bir kategoriye dönüştürülür:

```text
stress_level=high + sleep_quality=poor
-> stress_sleep_quality=high__poor
```

E002’de `stress_sleep_quality`, `activity_diet` ve `smoking_activity` vardır. Böylece model
“yüksek stres” ve “kötü uyku” etkilerini yalnız ayrı ayrı değil, birlikte de öğrenebilir.

### 3.5 Outlier flag

E002’nin her fold’unda ham numeric ve ratio kolonlarının `%0.5` ve `%99.5` quantile
sınırları fold-train’den öğrenilir. Değer değiştirilmez; yalnız düşük/yüksek uçta olduğunu
gösteren ikili kolonlar ve toplam `outlier_count` eklenir.

```text
fold-train heart_rate %99.5 sınırı = 112
valid heart_rate = 118
heart_rate_outlier_high = 1
```

Bu yöntem uç değeri silmez. LightGBM hem gerçek `118` değerini hem de “uç değer” işaretini
görür.

### 3.6 Clipping

Clipping, değeri öğrenilen alt ve üst sınıra sıkıştırır:

```text
alt sınır=15, üst sınır=45, BMI=61 -> clipped BMI=45
```

E005’te sınırlar fold-train `%0.1/%99.9` quantile’larıdır. Outlier flag clipping’den önce
üretilir; böylece değer sıkıştırılsa bile uçta olduğu bilgisi korunur.

### 3.7 Log1p dönüşümü

E006’da her ratio’nun negatif olmayan hali için `log(1 + x)` eklenir. Dönüşüm büyük
değerler arasındaki mesafeyi sıkıştırır:

```text
x=9    -> log1p(x)=2.30
x=999  -> log1p(x)=6.91
```

Orijinal ratio silinmez. Model gerektiğinde ham veya log ölçeğini seçebilir.

### 3.8 Rule flag

E004 sekiz eşik tabanlı feature ekler:

| Flag | Koşul |
|---|---|
| `low_sleep_flag` | sleep < 6 |
| `high_sleep_flag` | sleep > 9 |
| `high_bmi_flag` | BMI >= 30 |
| `low_bmi_flag` | BMI < 18.5 |
| `high_heart_rate_flag` | heart rate > 100 |
| `low_heart_rate_flag` | heart rate < 60 |
| `low_steps_flag` | step < 3000 |
| `high_steps_flag` | step > 12000 |

Bunlar tıbbi tanı değildir; yalnız modelin eşik çevresindeki örüntüyü daha kolay bulması
için oluşturulan aday sinyallerdir.

## 4. Deney bazında ayrıntılı açıklama

### E001 — V1 baseline

**Amaç:** Daha karmaşık feature engineering’in karşılaştırılacağı sade referansı kurmak.

**Uygulananlar:** Ham 7 numeric ve 6 categorical kolon, numeric median imputasyon,
categorical `missing`/`__UNKNOWN__` yönetimi ve 13 missing flag. Toplam 26 feature.

**Uygulanmayanlar:** Ratio, interaction, outlier, clipping, log dönüşümü ve class weight.

**Sonuç:** Balanced accuracy `0.878103 ± 0.001471`; recall değerleri sırasıyla
`at-risk=0.991208`, `fit=0.832400`, `unhealthy=0.810703`.

**Yorum:** Baseline, sonraki ağırlıksız feature deneylerinin hepsinden daha iyi çıktı.
Bu, ek feature’ların otomatik olarak faydalı olmadığını ve baseline’ın güçlü olduğunu
gösterir. Özellikle `unhealthy` recall geliştirmeye açık kalmıştır.

### E002 — V2-Core

**Amaç:** Eksiklik yoğunluğu, altı ratio, üç interaction ve outlier bilgisinin birlikte
baseline’ı geçip geçmediğini sınamak.

**E001’e eklenenler:** `missing_count`, 6 ratio, 3 interaction, ham+ratio outlier flag’leri
ve `outlier_count`. Toplam 63 feature.

**Sonuç:** Balanced accuracy `0.876937 ± 0.001883`; E001’den yaklaşık `-0.001167` düşük.
Recall: `0.991228 / 0.829988 / 0.809594`.

**Yorum:** V2-Core topluca fayda sağlamadı. Bu sonuç tek tek bütün feature’ların kötü
olduğunu kanıtlamaz; bazı yararlı feature’lar gürültülü olanlarla birlikte maskelenmiş
olabilir. E002 yine de E003–E006’nın kontrol grubu ve E007’nin ilk kaynağıdır.

### E003 — V2-Core + gender_activity

**Amaç:** Cinsiyet ile fiziksel aktivite seviyesinin birleşiminin ek sinyal taşıyıp
taşımadığını ölçmek.

**Örnek:** `gender=female`, `physical_activity_level=high` girdisi
`female__high` kategorisine dönüşür.

**Risk:** Train ve test arasında gender dağılımı değişiyorsa model bu interaction’a aşırı
bağlanabilir ve genelleme zayıflayabilir.

**Sonuç:** `0.876819 ± 0.001484`; E002’ye göre `-0.000117`. Feature sayısı 64.

**Karar:** Ölçülebilir iyileşme yoktur; interaction final feature set için desteklenmez.

### E004 — V2-Core + rule flags

**Amaç:** Ağaçların öğrenebileceği ama doğrudan verilince daha kolay kullanabileceği sekiz
eşik bilgisini sınamak.

**Sonuç:** `0.877466 ± 0.001418`; E002’ye göre `+0.000529`, E001’e göre `-0.000638`.
Feature sayısı 71.

**Yorum:** V2-Core içindeki en iyi ekleme budur ancak kazanç `<0.001` olduğu için gürültü
bölgesindedir. Rare-class recall değerlerinde de belirgin sıçrama yoktur. Ek seed/CV ile
tekrarlanmadan güçlü bir kazanım sayılmaz.

### E005 — V2-Core + clipping

**Amaç:** Çok uç numeric ve ratio değerlerinin ağacın bölmelerini gereksiz yere
etkilemesini önlemek.

**Sonuç:** `0.876661 ± 0.001735`; E002’ye göre `-0.000275`. Feature sayısı 63.

**Yorum:** Clipping fayda sağlamadı. Olası açıklama, extreme değerlerin gürültü değil hedef
sinyali taşımasıdır. Bu nedenle uç değerleri koruyan E002 yaklaşımı tercih edilir.

### E006 — V2-Core + log ratio varyantları

**Amaç:** Uzun kuyruklu ratio’ları sıkıştırılmış ölçekte de modele sunmak.

**Sonuç:** `0.876852 ± 0.001698`; E002’ye göre `-0.000085`. Altı log feature ile toplam
69 feature.

**Yorum:** Sonuç pratik olarak E002 ile aynıdır ve pozitif sinyal yoktur. Orijinal ve log
feature’ların birlikte bulunması model karmaşıklığını artırmış, ölçülebilir fayda vermemiştir.

### E007 — OOF class multiplier tuning

**Amaç:** Modeli yeniden eğitmeden, argmax kararının sınıf dengesizliği nedeniyle
çoğunluk sınıfına kaymasını düzeltmek.

Normal karar:

```text
prediction = argmax([p_at-risk, p_fit, p_unhealthy])
```

Tuned karar:

```text
prediction = argmax(probabilities * class_multipliers)
```

Örnek:

```text
olasılık     = [0.55, 0.30, 0.15]
multiplier   = [0.20, 1.40, 1.40]
çarpım       = [0.11, 0.42, 0.21]
argmax önce  = at-risk
argmax sonra = fit
```

Arama önce `at-risk=1.0` referansıyla `fit` ve `unhealthy` için `0.80–1.50` grid tarar,
sonra en iyi noktanın çevresinde seeded 2.000 random deneme yapar. Son vektör ortalaması
1 olacak şekilde normalize edilir; ortak ölçek argmax sonucunu değiştirmez.

| Kaynak | Argmax bal. acc. | Tuned bal. acc. | Tuned multiplier (`at-risk/fit/unhealthy`) |
|---|---:|---:|---|
| E002 | 0.876937 | **0.948584** | 0.1892 / 1.4445 / 1.3663 |
| E004 | 0.877466 | **0.948188** | 0.2019 / 1.3834 / 1.4147 |
| E006 | 0.876852 | **0.948308** | 0.1965 / 1.4136 / 1.3900 |
| E008 | 0.914002 | **0.947802** | 0.2265 / 1.3503 / 1.4232 |

**Yorum:** Büyük artış modelin olasılık sıralamasının güçlü, varsayılan argmax karar
sınırlarının ise balanced accuracy için uygun olmadığını gösterir. En yüksek tuned OOF
E002’dedir. Ancak multiplier araması aynı OOF üzerinde seçilip raporlandığı için skor bir
miktar iyimser olabilir; sağlam karar için farklı seed veya nested doğrulama önerilir.

### E008 — V2-Core + sqrt balanced sample weights

**Amaç:** Azınlık sınıflarına eğitim sırasında daha fazla önem vermek ama tam balanced
weight kadar agresif olmamak.

Önce klasik balanced ağırlık hesaplanır:

```text
w_class = N / (K * class_count)
```

Burada `N` fold-train satır sayısı, `K=3` sınıf sayısıdır. E008 bunun karekökünü kullanır:

```text
w_sqrt_balanced = sqrt(w_class)
```

Örnek olarak balanced ağırlıklar `[0.39, 5.5, 3.7]` ise karekök yaklaşımı yaklaşık
`[0.62, 2.35, 1.92]` üretir. Böylece rare class hataları daha pahalı olur ama aşırı
düzeltme riski azalır.

**Sonuç:** `0.914002 ± 0.001098`; E002’den `+0.037065`, E001’den `+0.035898` yüksek.
Recall `at-risk=0.975356`, `fit=0.887395`, `unhealthy=0.879253`.

**Yorum:** Ana deneyler içindeki açık ara en güçlü ve en kararlı argmax sonucu E008’dir.
Çoğunluk recall’ından kontrollü ödün vererek iki azınlık sınıfını ciddi biçimde iyileştirir.
E008 tuned sonucu ise E002 tuned’dan biraz düşüktür; çünkü multiplier zaten karar sınırını
ayarladığında eğitim ağırlığının ek avantajı küçülür.

### Public leaderboard geri bildirimi ve yeni yön

İlk beş submission, lokal argmax sıralamasından daha güçlü bir karar verdi:

| Submission | Public balanced accuracy |
|---|---:|
| **E002 tuned** | **0.94960** |
| E004 tuned | 0.94941 |
| E006 tuned | 0.94905 |
| E008 tuned | 0.94894 |
| E008 argmax | 0.91517 |

Public sonuçlar model varyantları arasındaki farktan çok class multiplier ile değiştirilen
karar sınırının kazandırdığını gösterir. Yeni seri bu nedenle yeni feature veya geniş HPO
yerine E002 olasılıklarını küçük blend, multiplier perturbation ve consensus düzeltmeleriyle
değiştirir. OOF üzerinde seçilen multiplier'ların overfit riski sürdüğü için her adayın
dağılımı ve E002 ile disagreement oranı ayrıca kaydedilir.

### E009 — E002/E004 75/25 probability blend tuned

**Amaç:** E002'nin güçlü karar yüzeyini koruyup E004 rule flag modelinin farklı yakaladığı
satırlardan küçük kazanım almak.

```text
p = 0.75 * p_E002 + 0.25 * p_E004
prediction = argmax(p * multiplier)
```

E002 multiplier'ı çevresinde `fit` ve `unhealthy` için bağımsız
`{0.990, 1.000, 1.010}` ölçekleri OOF balanced accuracy ile taranır. Test dağılımı
`at-risk=%81.6–81.9`, `fit=%7.1–7.3`, `unhealthy=%11.0–11.2`; E002 disagreement ise
`%0.1–1.5` dışında kalırsa aday üretilir fakat submit için FAIL işaretlenir.

### E010 — SWEEP_002 tuned

**Amaç:** Argmax OOF sonucu E002'den az miktarda yüksek olan en iyi sweep modelini E002
çevresinde kontrollü karar sınırıyla değerlendirmek.

SWEEP_002 OOF/test olasılıklarına E002 multiplier tabanı uygulanır. `fit` ve `unhealthy`
ölçekleri bağımsız olarak `{0.970, 0.985, 1.000, 1.015, 1.030}` taranır. Bu deney yeni
HPO yapmaz; mevcut en iyi sweep artefaktını tuned karara dönüştürür.

### E011 — E002/E004/E006 probability blend tuned

**Amaç:** E006 log-ratio modeline yalnız `%10` ağırlık vererek üç feature görünümünün
kararsız satırlardaki olasılıklarını yumuşatmak.

OOF üzerinde `60/30/10`, `65/25/10` ve `70/20/10` ağırlıkları karşılaştırılır. Her ağırlık
için E009 micro multiplier grid'i taranır; en yüksek OOF balanced accuracy seçilir. Test
dağılımı `at-risk=%81.3–82.1`, `fit=%7.0–7.4`, `unhealthy=%10.8–11.4` dışında kalırsa
submission FAIL işaretlenir.

### E012 — E002 fit-up / unhealthy-down boundary

**Amaç:** Borderline `at-risk/fit` satırlarını hafifçe fit'e çekerken unhealthy oranını
şişirmemek.

```text
m_at-risk   = E002_at-risk
m_fit       = E002_fit * 1.012
m_unhealthy = E002_unhealthy * 0.992
```

Beklenen test bantları `at-risk=%81.55–81.75`, `fit=%7.25–7.35` ve
`unhealthy=%10.95–11.15` olarak hard eligibility kontrolüne dönüştürülür.

### E013 — E002 consensus correction

**Amaç:** E002 tuned etiketini yalnız E004 tuned ve E006 tuned aynı farklı etikette
birleştiğinde değiştirmek.

```text
if pred_E004 == pred_E006 and pred_E004 != pred_E002:
    final = pred_E004
else:
    final = pred_E002
```

Aynı kural OOF üzerinde uygulanır. Submit uygunluğu için OOF balanced accuracy en az
`0.9482`, test disagreement ise `%0.2–1.0` olmalıdır.

Yeni adayların önerilen submit sırası E009, E010, E011, E012 ve E013'tür. Eski argmax
submission'lar bu serinin parçası değildir.

## 5. LightGBM parameter sweep deneyleri

Sweep, E002 feature pipeline’ını sabit tutup şu parametreleri değiştirir: learning rate,
leaf sayısı, minimum leaf örneği, L1/L2 regularization, row sampling ve column sampling.

- **`learning_rate`:** Her ağacın katkısı. Küçük değer daha yavaş ama kontrollü öğrenir.
- **`num_leaves`:** Bir ağacın karmaşıklığı. Artınca detaylı örüntü ve overfit riski artar.
- **`min_child_samples`:** Bir leaf için gereken minimum örnek. Büyük değer ağacı düzenler.
- **`reg_alpha`:** L1 regularization; gereksiz katkıları sıfıra itebilir.
- **`reg_lambda`:** L2 regularization; büyük leaf ağırlıklarını yumuşatır.
- **`subsample`:** Her iterasyonda kullanılan satır oranı.
- **`colsample_bytree`:** Her ağaçta kullanılan feature oranı.

Tamamlanan 20 trial içinde en iyi sonuç `SWEEP_002 = 0.877851 ± 0.001796` olmuştur.
Parametreleri: `learning_rate=0.04`, `num_leaves=127`, `min_child_samples=200`,
`reg_alpha=0`, `reg_lambda=10`, `subsample=0.85`, `colsample_bytree=0.95`.

Bu sonuç E002’den yalnız `+0.000915`, E001’den `-0.000252` farklıdır. Yani E002 üzerinde
HPO güçlü bir sıçrama üretmemiştir. Asıl kazanım model parametresinden değil, E008’deki
sınıf ağırlığı ve E007’deki karar sınırı düzenlemesinden gelmiştir.

`SWEEP_000`–`SWEEP_019` trial'larının tamamı metric ve model artefaktlarını üretmiştir.
Tüm parametreler, skorlar ve sıralama `final-experiment-report.md` içinde verilmiştir.

## 6. Toplu sonuç ve çıkarımlar

### 6.1 Argmax sıralaması

| Deney | Balanced accuracy | Fold std | Ana değişiklik |
|---|---:|---:|---|
| **E008** | **0.914002** | **0.001098** | sqrt balanced weights |
| E001 | 0.878103 | 0.001471 | baseline |
| E004 | 0.877466 | 0.001418 | rule flags |
| E002 | 0.876937 | 0.001883 | V2-Core |
| E006 | 0.876852 | 0.001698 | log ratios |
| E003 | 0.876819 | 0.001484 | gender/activity |
| E005 | 0.876661 | 0.001735 | clipping |

### 6.2 Deneylerin öğrettiği temel dersler

1. **Daha fazla feature her zaman daha iyi değildir.** E001, bütün ağırlıksız V2
   varyantlarını geçti.
2. **Sınıf dengesizliği ana problemdir.** E008’in kazancı bütün feature ablation
   farklarından çok büyüktür.
3. **Olasılıklar karardan daha iyi olabilir.** E007’de multiplier sonrası büyük artış,
   model ranking’inin argmax kararından daha güçlü olduğunu gösterir.
4. **Clipping burada faydalı değildir.** Extreme değerler olası hedef sinyalidir.
5. **Küçük farklar kesin sonuç değildir.** `<0.001` farklar ek seed/CV olmadan gürültü
   kabul edilmelidir.
6. **Sweep feature sorununu çözmez.** E002 tabanı üzerinde parametre araması baseline’ı
   anlamlı biçimde geçemedi.

## 7. Sonraki deney için önerilen doğrulama

1. E001, E008 ve E002-tuned için 3–5 farklı seed çalıştır.
2. Multiplier seçimini nested CV veya ayrı holdout ile doğrula.
3. E001 feature set’i üzerinde `sqrt_balanced` deneyerek feature ve weighting etkisini
   birbirinden ayır.
4. E008 için probability calibration ve daha dar multiplier araması dene.
5. Blend yapılacaksa yalnız OOF olasılıklarıyla ağırlık seç; test/LB dağılımına göre ayar
   yapma.

## 8. Artefaktları okuma rehberi

| Dosya | Ne anlatır? |
|---|---|
| `metrics.json` | Deneyin overall OOF metrikleri ve fold özetleri |
| `fold_metrics.csv` | Fold bazında skor, recall, iterasyon ve feature sayısı |
| `oof_proba.npy` | Train için sızıntısız OOF olasılıkları |
| `test_proba.npy` | Fold ortalamalı test olasılıkları |
| `confusion_matrix.csv` | Hangi sınıfların birbiriyle karıştığı |
| `feature_importance.csv` | Feature’ların gain/split katkısı |
| `training_history.json` | Fold valid loss seyri ve overfit incelemesi |
| `best_multipliers.json` | E007’de bulunan sınıf çarpanları |
| `metrics_tuned.json` | Multiplier sonrası OOF sonuçları |
| `submission_argmax.csv` | Varsayılan karar ile test tahmini |
| `submission_tuned.csv` | Multiplier sonrası test tahmini |

## 9. Kaynak dosyalar

- Deney tanımları: `configs/experiments.yaml`
- Ortak model ayarları: `configs/lgbm_base.yaml`
- Sweep uzayı: `configs/sweeps.yaml`
- Feature formülleri: `src/kaggle_s6_e7/features.py`
- Fold-safe preprocessing: `src/kaggle_s6_e7/preprocessing.py`
- CV ve eğitim: `src/kaggle_s6_e7/training.py`
- Multiplier araması: `src/kaggle_s6_e7/postprocess.py`
- E009–E013 tarifleri: `configs/postprocess_experiments.yaml`
- Ensemble ve eligibility motoru: `src/kaggle_s6_e7/candidate_experiments.py`
- Gerçekleşmiş sonuçlar: `outputs/experiments/<EXP_ID>/`
- Lokal sıralama: `outputs/leaderboard_local.csv`
