# E018 Serisi — E002-Ağırlıklı Mikro Blend: Tam Analiz Raporu

**Tarih:** 2026-07-05 | **Pipeline:** `scripts/run_e018_experiments.sh`

---

## 1. Deney Tasarımı

| Deney | E002 | E004 | E006 | OOF Bal Acc | Δ E014 |
|:------|-----:|-----:|-----:|------------:|-------:|
| **E018C** | 85% | 12% | 3% | **0.948809** | +0.000155 |
| E018B | 88% | 10% | 2% | 0.948805 | +0.000151 |
| **E018A** | 90% | 8% | 2% | 0.948800 | +0.000146 |
| E018D | 92% | 6% | 2% | 0.948791 | +0.000137 |
| E018E | 95% | 5% | — | 0.948781 | +0.000127 |

Her adayda 7×7 = 49 multiplier scale grid tarandı:
`{0.96, 0.98, 0.99, 1.00, 1.01, 1.02, 1.04}` × aynı set.

**Kritik bulgu:** 5 adayın 5'i de **aynı multiplier'ı** seçti.

---

## 2. Multiplier Karşılaştırması

| Deney | at-risk | fit | unhealthy | E002'ye göre |
|:------|--------:|----:|----------:|:-------------|
| E002 | 0.18923 | 1.44445 | 1.36632 | — (taban) |
| E011 | 0.18923 | 1.45890 | 1.37998 | +%1.0 / +%1.0 |
| E014 | 0.18923 | 1.45168 | 1.37315 | +%0.5 / +%0.5 |
| **E018 (tümü)** | 0.18923 | **1.50223** | **1.42097** | **+%4.0 / +%4.0** |

E018, E011'den 4 kat daha agresif multiplier kullanıyor. Sebep: E002-heavy
blend'in olasılık dağılımı saf E002'den farklı; azınlık sınıf olasılıkları
biraz daha düşük, bu yüzden daha büyük düzeltme gerekiyor.

---

## 3. Submission Karşılaştırması: Tam Matris

### 3.1 Test Dağılımları

| Deney | at-risk | fit | unhealthy |
|:------|--------:|----:|----------:|
| E002 | 241,579 (%81.68) | 21,297 (%7.20) | 32,877 (%11.12) |
| E011 | 241,544 (%81.67) | 21,305 (%7.20) | 32,904 (%11.13) |
| E014 | 241,571 (%81.68) | 21,302 (%7.20) | 32,880 (%11.12) |
| E015 | 241,530 (%81.67) | 21,310 (%7.21) | 32,913 (%11.13) |
| **E018A** | 241,404 (%81.62) | 21,333 (%7.21) | 33,016 (%11.16) |
| **E018B** | 241,408 (%81.62) | 21,332 (%7.21) | 33,013 (%11.16) |
| **E018C** | 241,406 (%81.62) | 21,332 (%7.21) | 33,015 (%11.16) |
| **E018D** | 241,403 (%81.62) | 21,334 (%7.21) | 33,016 (%11.16) |
| **E018E** | 241,402 (%81.62) | 21,333 (%7.21) | 33,018 (%11.16) |

> Tüm E018 varyantları **aynı dağılımı** üretiyor (fark ≤16 satır).
> E002'ye göre: at-risk ~175 satır azalmış, azınlık sınıfları ~140 satır artmış.

### 3.2 Pairwise Disagreement (değişen satır sayısı)

```
              E002    E011    E014    E015   E018A   E018B   E018C   E018D   E018E
E002           -      170     173     131     177     174     178     178     179
E011          170      -       27      41     165     154     152     172     179
E014          173     27       -       54     190     179     177     197     202
E015          131     41      54       -      142     131     131     147     152
E018A         177    165     190     142       -       15      25       7      18
E018B         174    154     179     131      15       -       10      22      33
E018C         178    152     177     131      25      10       -       32      43
E018D         178    172     197     147       7      22      32       -      11
E018E         179    179     202     152      18      33      43      11       -
```

**En yakın çiftler:** E018A ↔ E018D (7 satır), E018A ↔ E018B (15 satır)  
**En uzak çiftler:** E014 ↔ E018E (202 satır), E014 ↔ E018D (197 satır)

### 3.3 E018 Aile İçi Tutarlılık

5 E018 varyantı **295,710 satırda** (%99.985) aynı tahmini yapıyor.  
Sadece **43 satırda** (0.0145%) herhangi bir anlaşmazlık var — ve bu 43 satırın
hiçbirinde 2'den fazla farklı etiket üretilmiyor.

Bu 43 kararsız satırda E002'nin tahmini:
- 41 satır `at-risk` (%95.3)
- 1 satır `fit`, 1 satır `unhealthy`

Yani E018'lerin kendi aralarında anlaşamadığı satırlar, E002'nin `at-risk`
dediği borderline satırlar. Blend oranındaki küçük farklar bu satırların
kaderini değiştiriyor.

---

## 4. E018A vs E002: Ne Değişti?

**177 satır** (%0.060) farklı:

```
              E018A →
E002 ↓        fit   unhealthy   TOPLAM
at-risk        34       141       175
unhealthy       2         0         2
TOPLAM         36       141       177
```

- **175 satır** E002'de `at-risk` → E018A'da 34'ü `fit`, 141'i `unhealthy`
- **2 satır** E002'de `unhealthy` → E018A'da `fit`

E018A'nın E002'den tek farkı: agresif multiplier (×1.04) borderline at-risk
satırlarını azınlık sınıflara itiyor. Hiçbir fit→at-risk veya unhealthy→at-risk
geçişi yok.

### E018C vs E002:

**178 satır** (%0.060) farklı:

```
              E018C →
E002 ↓        at-risk  fit  unhealthy  TOPLAM
at-risk            0   34       140      174
fit                0    0         1        1
unhealthy          1    2         0        3
TOPLAM             1   36       141      178
```

E018A ile neredeyse aynı pattern. E018C'de ek olarak 1 satır
at-risk→unhealthy yerine unhealthy→at-risk yapmış (net fark).

---

## 5. E018 vs Diğer Postprocess Deneyleri

### E018A vs E011 (60/30/10 blend)

**165 satır farklı.** E011, E002'ye daha yakın bir karar sınırı kullanıyor
(×1.01 vs ×1.04). Farkın çoğu, E018A'nın daha fazla at-risk satırını
azınlık sınıflara itmesinden kaynaklanıyor.

### E018A vs E014 (E011 + soft multiplier)

**190 satır farklı.** E014 en yumuşak multiplier'a sahip (×1.005).
E018A'nın agresifliği en çok E014'ten uzaklaştırıyor.

### E018A vs E015 (70/20/10 blend)

**142 satır farklı.** E015, E011 ile aynı multiplier'ı kullanıyor ama
E002 ağırlığı daha yüksek (%70). Bu, E018'e en yakın eski deney.

---

## 6. OOF Sıralaması (Tüm Postprocess Deneyleri)

| Sıra | Deney | OOF Bal Acc | Blend | Multiplier (fit/uh) |
|-----:|:------|------------:|:------|:--------------------|
| 1 | **E018C** | **0.948809** | 85/12/3 | +%4.0 / +%4.0 |
| 2 | E018B | 0.948805 | 88/10/2 | +%4.0 / +%4.0 |
| 3 | E018A | 0.948800 | 90/8/2 | +%4.0 / +%4.0 |
| 4 | E018D | 0.948791 | 92/6/2 | +%4.0 / +%4.0 |
| 5 | E018E | 0.948781 | 95/5 | +%4.0 / +%4.0 |
| 6 | E014 | 0.948654 | 60/30/10 | +%0.5 / +%0.5 |
| 7 | E011 | 0.948654 | 60/30/10 | +%1.0 / +%1.0 |
| 8 | E015 | 0.948614 | 70/20/10 | +%1.0 / +%1.0 |
| 9 | E009 | 0.948639 | 75/25 | +%1.0 / +%1.0 |
| 10 | E010 | 0.948392 | SWEEP_002 | +%1.5 / +%3.0 |

---

## 7. OOF Metrik Detayı

| Metrik | E018A | E018B | E018C | E018D | E018E |
|:-------|------:|------:|------:|------:|------:|
| Bal Acc | 0.948800 | 0.948805 | **0.948809** | 0.948791 | 0.948781 |
| Macro F1 | 0.873813 | 0.873824 | 0.873803 | 0.873799 | 0.873829 |
| Recall at-risk | 0.942197 | 0.942203 | 0.942190 | 0.942188 | 0.942219 |
| Recall fit | 0.945883 | 0.945858 | 0.945883 | 0.945883 | 0.945858 |
| Recall unhealthy | 0.958319 | 0.958354 | 0.958354 | 0.958302 | 0.958267 |

E002 tuned referans:
- Bal acc: 0.948584
- Macro F1: 0.874625
- Recall: at-risk=0.942892, fit=0.945356, unhealthy=0.957505

---

## 8. LB Beklentisi ve Risk Değerlendirmesi

### OOF → LB Korelasyonu (geçmiş veri)

| Deney | OOF | LB | OOF→LB farkı |
|:------|----:|---:|:-------------|
| E002 tuned | — | **0.94960** | — |
| E011 | 0.948654 | 0.94957 | +0.00092 |
| E009 | 0.948639 | 0.94948 | +0.00084 |
| E004 tuned | — | 0.94941 | — |
| E010 | 0.948392 | 0.94901 | +0.00062 |

OOF artışı her zaman LB artışı anlamına gelmiyor. E011, E014'ten OOF'da
düşük olmasına rağmen LB'de daha iyi.

### E018 için LB projeksiyonu

E018C OOF: 0.948809 (+0.000155 vs E014 DD)

Geçmiş OOF→LB spread'i 0.0006–0.0009 aralığında. Bu spread'le:
- İyimser: ~**0.94975**
- Nötr: ~0.94965
- Kötümser: ~0.94955

**E002'yi (0.94960) geçme olasılığı mevcut, garanti değil.**

### Riskler

1. **×1.04 multiplier agresif** — test dağılımı train'den farklıysa overfit riski
2. **5 E018 neredeyse aynı** — hepsi aynı anda submit edilirse günlük hak israfı
3. **OOF farkları minik** (max 0.000028) — sıralama gürültü seviyesinde

---

## 9. Önerilen Submit Stratejisi

```
1. E018C (en yüksek OOF)
2. E018A (kullanıcının en umutlu adayı)
3. Sonuçlara göre: E018B veya E018D (yedek)
```

İlk iki submit'ten sonra LB sonuçlarına bak:
- İkisi de E002'nin altında kalırsa → E018 serisini bırak, E002 final
- Biri geçerse → kazananı ikinci kez submit et (güvenlik)
- E014/E015 henüz submit edilmediyse onları da dene

---

## 10. Üretilen Dosyalar

### Submission'lar
```
outputs/experiments/E018A_blend_90_08_02/submission_tuned.csv
outputs/experiments/E018B_blend_88_10_02/submission_tuned.csv
outputs/experiments/E018C_blend_85_12_03/submission_tuned.csv
outputs/experiments/E018D_blend_92_06_02/submission_tuned.csv
outputs/experiments/E018E_blend_95_05/submission_tuned.csv
```

### Config & Script
```
configs/e018_experiments.yaml
scripts/run_e018_experiments.sh
```

### Rapor ve artefaktlar
```
outputs/experiments/e018_eligibility_report.csv
outputs/experiments/e018_eligibility_report.json
docs/e018-e002-heavy-micro-blend-results.md
```

### Tekrar çalıştırma
```bash
bash scripts/run_e018_experiments.sh       # mevcut çıktıları korur
FORCE=1 bash scripts/run_e018_experiments.sh  # yeniden üretir
```
