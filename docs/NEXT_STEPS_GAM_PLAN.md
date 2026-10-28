# 작업 계획과 진행 기록: ASCA 순열검정 이후

작성일: 2026-10-09
기준 결과: `outputs/asca_permutation/` (script 25)

## 0. Script 25에서 확인된 것과 그 의미

| 결과 | 의미 | 다음 단계에 주는 영향 |
|---|---|---|
| 모든 Pad/Contrast 항이 p = 0.0001 (최소값) | 이미지 단위 잡음이 매우 작아 거의 모든 항이 유의 | 유의성보다 효과 크기와 형태를 보고해야 함 |
| Background pH: strip 단위 p = 0.57, 이미지 단위 p = 0.0001 | 이미지를 독립으로 보면 결론이 뒤집힘 (pseudoreplication) | 논문의 통계적 핵심 사례로 사용. GAM에도 strip 랜덤효과 필수 |
| pH x phone은 strip 수준 df = 0 | ASCA로는 "폰 효과 vs strip 차이"를 검정할 수 없음 | **GAM을 쓰는 통계적 이유**가 됨 (아래 1절) |
| Contrast pH PC1과 pH x phone PC1의 각도 57.6° (CI 56.3–59.0) | 두 효과는 서로 다른 색 방향 | 초안 3.4 주장 유지 가능 |
| 상호작용 PC1의 주 채널이 척도에 따라 바뀜 (표준화 L*, raw b*) | "주로 L*" 같은 표현은 척도 의존적 | 척도와 무관한 특징(a* 부호가 pH와 반대)으로 서술 |

## 1. ASCA → GAM으로 넘어가는 논리 (Methods 2.6 서두에 쓸 내용)

ASCA는 pH를 13개 범주로 다루므로, phone x pH 셀마다 strip이 하나뿐인 이 설계에서는
pH x phone 효과와 strip 간 차이를 구분할 오차항이 없다(df = 0).
GAM은 pH를 연속 변수로 보고 "pH에 대해 매끄러운 phone별 곡선"을 가정한다.
이 가정 아래에서 pH x phone 편차는

- pH를 따라 매끄럽게 변하는 phone별 반응 곡선(smooth), 그리고
- 매끄럽지 않은 strip별 편차(strip 랜덤효과)

로 나뉘며, strip 수준의 오차항이 생긴다.
즉 GAM은 ASCA가 남긴 식별 문제를 "매끄러움 가정"으로 해결하는 다음 단계이다.

## 2. GAM 모형 사전 명세 (결과 보기 전에 고정)

- 소프트웨어: R 4.5.2, `mgcv` 1.9.3
- 자료: `outputs/factorial_analysis/analysis_factorial_table.csv` (195행)
- 반응변수: Lab 채널별 단변량 적합
  - 주 분석: Contrast L*, a*, b*
  - 민감도 분석: Pad, Background
- 공통 고정효과: `light`, `phone:light` (ASCA에서 유의, 크기는 작음)
- 공통 랜덤효과: `s(strip_id, bs = "re")`
- 추정: `method = "REML"`

| 모형 | 식 (mgcv) |
|---|---|
| M0 | `y ~ light + phone:light + s(ph, k = 10) + s(strip_id, bs = "re")` |
| M1 | M0 + `phone` |
| M2 | M1 + `s(ph, by = phone_ord, k = 10)` (phone_ord: 순서형 factor, 기준 폰 대비 차이 곡선) |

- 주 k = 10 (서로 다른 pH 값 13개). 민감도 분석: k = 6, 8, 12. basis는 `tp`이고, `cr`은 민감도 분석용.
- 모형 비교: ML로 다시 적합해 AIC를 비교하고, 차이 smooth의 근사 검정(`summary.gam`)을 함께 보고
- 핵심 진단 지표: M1에서 M2로 갈 때 strip 랜덤효과 분산이 얼마나 줄어드는지
  (pH x phone 편차 중 매끄러운 부분의 비율)
- 진단: `gam.check`, `concurvity`, 잔차 대 적합값, k-index

## 3. 작업 순서

| 순서 | 작업 | 산출물 | 비고 |
|---|---|---|---|
| 1 | **26_gam_phone_specific_curves.R**: M0–M2 적합, phone별 곡선 및 95% CI, 쌍별 차이 곡선, CI가 0에서 벗어나는 pH 구간, edf, 랜덤효과 분산 분해 | `outputs/gam_curves/` | 초안 3.5, Figure 5–6 |
| 2 | k와 basis 민감도 분석 (같은 스크립트) | `outputs/gam_curves/` (spec별 결과) | RCS의 knot 민감도와 대비 |
| 3 | **27_gam_leave_one_strip_out.R**: 39-fold, 랜덤효과 제외 예측, 채널별 예측을 합친 ΔE, pH 2–12 / pH 1·13 / 전체, strip 단위 paired bootstrap | `outputs/gam_strip_holdout/` | 초안 3.9, script 19와 같은 부분집합 |
| 4 | GAM–RCS 일관성 표: 내부 pH 개선량, 경계 악화량, knot/k 민감도를 한 표로 정리 | `gam_rcs_consistency.csv` | 초안 Table 1 확장 |
| 5 | 그림 정리 (matplotlib): 효과 구성, ASCA loading CI, GAM 곡선, LOSO 결과 | `outputs/paper_figures/` | `scripts/33_make_paper_figures.py` |
| 6 | 초안 수정 (아래 4절) | — | |

## 4. 초안 수정 사항

1. 2.5 / 3.4: 순열검정 방식(restricted, Freedman–Lane)과 결과 추가. 각도와 신뢰구간 보고.
   "상호작용은 L* 중심" 대신 "a* 방향이 pH 효과와 반대"로 서술.
2. 3.1: Background pH가 strip 단위 검정에서 유의하지 않다는 점(p = 0.57)을 근거로 추가.
3. 새 소절 (Methods 또는 Discussion): 실험 단위와 pseudoreplication.
   이미지 단위 검정(3.1 대비)과 검증(3.6 대비) 두 곳에서 같은 문제가 나타남을 한 번에 정리.
4. 3.8: RCS 결과를 knot 3/4/5 모두 보고.
5. 2.9 / 4.6: 증강 실험은 동기 설명 수준으로 축소하고, "target-characterized"라는 이름은 수정.
6. 투고처를 통계 학술지로 정했으므로 서론은 "다요인 설계 색 자료에서 상호작용의 방향과 형태를 분석하는 절차"를 중심으로 재구성.

## 5. 남은 위험 요소

- GAM에서 phone별 곡선과 strip 랜덤효과가 서로 분산을 나눠 가지면서 추정이 불안정할 수 있음 → concurvity 진단과 REML 분산 성분 보고
- pH 1/13은 LOSO에서 외삽 → 경계 결과는 "외삽 성능"으로 따로 보고
- GAM에서도 내부 pH 개선이 k에 따라 흔들리면, "재현된다"가 아니라 "방향은 일관되나 크기는 불확실"로 결론을 낮춰야 함

## 6. 진행 기록

### 2026-10-09: 작업 1–2 완료 (script 26)

- 결과: `outputs/gam_curves/`, 설명: `docs/README_26_GAM_PHONE_CURVES.md`
- Contrast에서 폰별 곡선이 strip 랜덤효과 분산의 L* 39%, a* 66%, b* 39%를 설명함
- 차이 곡선의 모양은 k = 6–12와 tp/cr basis에 거의 영향을 받지 않음 (RCS knot 민감도와 대비)
- AIC는 구조상 모형 간 차이가 거의 없으므로 판단 기준에서 제외
- 그림은 matplotlib로 작성 (`scripts/26_plot_gam_curves.py`)

### 추가로 생긴 작업

- 차이 곡선의 "shape" 성분은 상수만큼 식별되지 않으므로, 기울기 차이(1차 도함수) 곡선을 추가하는 것을 검토
- Background에서 iPhone 16 Pro의 pH 의존 차이가 유의함 → 배경 ROI에 패드 색이 번졌는지 이미지로 확인
- 다음 단계: 작업 3 (27번, GAM leave-one-strip-out)

### 2026-10-09: 배경 ROI 확인, 작업 3–4 완료 (script 27)

- 배경 ROI: 패드와 같은 크기, 패드 폭의 2.5배만큼 왼쪽, 흰 종이 위 (`docs/BACKGROUND_ROI_CHECK.md`).
  직접적인 번짐 가능성은 낮고, 카메라 자동 보정 가능성이 더 큼
- GAM LOSO (`docs/README_27_GAM_STRIP_HOLDOUT.md`):
  - Contrast pH 2–12: M2 − M1 = −1.16 [−1.92, −0.33], 5개 설정 모두 신뢰구간 < 0
  - RCS는 knot 4에서만 유의 → 초안 3.8의 knot 의존성 문제를 GAM이 해소
  - pH 1·13: GAM은 유의한 악화 없음 (+0.64 [−2.93, +4.21]); iPhone 16 Pro pH 13 한 strip이 평균을 끌어올림
- 다음: 작업 5 (논문용 그림 정리), 작업 6 (초안 수정)

### 2026-10-09: 역추정 pH 분석 완료 (script 28)

- 폰별 곡선(M2)은 색 예측은 개선(ΔE −1.1)하지만 pH 판독은 유의하게 개선하지 못함:
  pH 2–12 |pH 오차| M2 − M1 = −0.03 [−0.10, +0.05], 5개 설정 모두 유의하지 않음
- 실용적 이득의 대부분은 폰별 상수 보정(M1)에서 나옴: M1 − M0 = −0.20 pH, 정확 일치율 +24%p
- 설명: pH 축 방향 색 변화가 커서(중앙값 ΔE 9.7/pH) ΔE 1.1은 최대 약 0.1 pH에 해당하고,
  상호작용 방향이 pH 방향과 다름(ASCA 각도 58°)
- 산/염기 이진 분류는 폰 정보를 쓰는 모든 모형에서 100% → 증강 실험의 이진 과제로는 차이를 볼 수 없었던 이유
- 상세: docs/README_28_INVERSE_PH.md

### 2026-10-09: 모의실험 S1–S3 완료 (scripts 29–32)

- S1: 이미지 단위 순열검정은 strip 편차가 조금만 있어도 1종 오류 75–100%; strip 단위 검정은 4.6–5.8% 유지.
  폰별 곡선이 없어도 ASCA pH×phone 검정은 실제 Contrast 조건에서 100% 유의 → GAM 단계의 근거
- S2: strip 랜덤효과 없으면 곡선 검정 1종 오류 ≈ 100%. 랜덤효과 포함 시 약 10%(명목 5%의 2배, mgcv 근사 p값의 한계).
  곡선이 없을 때 strip 분산 감소율 6–8% (실제 39–66%). 작고 구불구불한 곡선은 동시 신뢰구간 포함률이 낮음(0.35–0.61)
- S2-LOSO: 곡선이 없으면 GAM의 거짓 개선 0%. 실제 개선(−1.16)은 A = 1 기대값(−1.03)과 일치
- S3: RCS 3 knot의 경계 실패는 방법 자체의 특성. 다만 실제 자료의 경계 결과는 매끄러운 참값의 기대보다 나빠 실제 데이터에도 경계 특이성이 있음
- 상세: docs/README_29_32_SIMULATION.md

### 2026-10-09: 보정 단계 분석 완료 (script 28 `--light-mode`, scripts 34–36)

- 조명을 몰라도 pH 판독은 거의 변하지 않음 (Contrast M0 0.583 → 0.585): 이 실험의 조명 범위는 판독에 영향이 작음
- 외란 성분별 pH 판독 영향 (pH 2–12): 폰 상수 0.12–0.63 pH (가장 큼), 곡선 모양 0.13–0.22, 조명 ≤ 0.16, strip 잡음 SD 0.30
- 폰 상수로 예측한 pH 치우침(0.63 / 0.53 / 0.11)과 새 폰에서 관찰된 치우침(−0.57 / +0.64 / −0.12)이 일치
- 새 폰: 기준 strip 0 / 1 / 2 / 3개 → pH 오차 0.73 / 0.56 / 0.48 / 0.45 (전체 보정 0.41). 2–3개로 이득의 77–86%
- 상세: docs/README_34_36_CALIBRATION_MECHANISM.md

### 2026-10-09: 배경 기반 보정 분석 완료 (scripts 37–38)

- 배경만으로 하는 보정 4가지(Lab 빼기, 선형 RGB 화이트밸런스, Bradford 색순응, RGB 비율) 비교
- 보정 없는 새 폰 오차는 모두 0.71–0.73으로 비슷하고(패드만 0.85), 폰 치우침은 남음 (iPhone 13 Pro +0.64)
- 흰 배경으로는 진한 시험지 색의 표현 차이를 알 수 없음 → 기준 시험지를 대체할 수 없음
- iPhone 16 Pro는 어둡게 촬영되어 65장 중 39장에서 패드 파란 채널이 포화(0)
- 상세: docs/README_37_38_BACKGROUND_CORRECTION.md

### 2026-10-09: 논문 방향 정리

- 논문 방향과 그림 구성: docs/PAPER_OUTLINE.md, `outputs/paper_figures/`
