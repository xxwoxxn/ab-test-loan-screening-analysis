# A/B 테스트 기반 대출 심사 AI 모델 성과 분석

> **Warwick Business School** Advanced Data Analysis (IB98D0)  
> Group 6 원본 보고서(2025.02)를 기반으로, 방법론적 한계를 보완하고  
> 행동 분석·비즈니스 임팩트 레이어를 추가한 분석 프로젝트.

---

## 분석 개요

10일간의 A/B 테스트 데이터를 활용해 AI 보조 대출 심사 모델이 심사관의 의사결정 정확도를 유의미하게 개선하는지 검증한다.

- **Control:** 기존 방식 심사관 10명
- **Treatment:** AI 보조 모델 활용 심사관 28명
- **Primary OEC:** F1 Score (Type I / Type II 오류 균형 반영)

---

## 핵심 결과

| 지표 | Control | Treatment | 향상폭 | Cohen's d |
|------|---------|-----------|--------|-----------|
| **F1** (Primary) | 43.69% | 61.52% | **+17.83pp** | 3.73 *(Huge)* |
| PrecisionAp | 74.49% | 86.46% | +11.97pp | 4.56 *(Huge)* |
| PrecisionRej | 34.26% | 55.01% | +20.75pp | 3.70 *(Huge)* |

Bonferroni 보정 후(α = 0.017) 세 지표 모두 p ≈ 0으로 통계적으로 유의미.

**→ 결론: 실험 중단 후 신규 모델 전면 도입 권고**

**비즈니스 임팩트 (심사관 1인당 10일 기준):** $107K ~ $187K 손실 절감  
(대출금 $20K / $27.5K / $35K 세 가지 시나리오 모두 AI 모델 유리)

---

## 원본 대비 개선 사항

| 항목 | Warwick 원본 | 본 분석 |
|------|------------|--------|
| 정규성 검정 | Skewness/Kurtosis만 | + Shapiro-Wilk 추가 |
| 다중 비교 보정 | 없음 | Bonferroni 보정 적용 |
| Cohen's d 기준 | 잘못된 레퍼런스 인용 | Sawilowsky(2009) 정정 |
| 신뢰구간 활용 | 표에만 기재 | 실무 해석 문장 작성 |
| 실험 무결성 검증 | 없음 | 1일차 Sanity Check 추가 |
| 행동 분석 | 없음 | 확신도·Automation Bias·숙련도별 분석 |
| 비즈니스 임팩트 | 없음 | 달러 환산 + 민감도 분석 |

---

## 행동 분석 주요 발견

- **확신도 변화:** Treatment +6.74 vs Control +4.73 (p = 0.525, 유의미하지 않음)
- **Automation Bias:** Treatment 심사관 중 25%(7명)가 AI 추천 과의존 의심
- **숙련도별 효과:** 숙련 심사관 +20.56pp > 초보 심사관 +15.31pp → 숙련자 우선 배포 권장

---

## 분석 구조

```
1단계 — 데이터 클리닝 및 Sanity Check
2단계 — OEC 산출 (F1, PrecisionAp, PrecisionRej)
3단계 — 정규성 & 등분산 검정 (Shapiro-Wilk, Levene's)
4단계 — 가설 검정 (t-test, Bonferroni, Cohen's d, 95% CI)
5단계 — 행동 분석 (확신도, Automation Bias, 숙련도별 분석)
6단계 — 비즈니스 임팩트 (달러 환산 + 민감도 분석)
```

---

## 기술 스택

`Python` · `pandas` · `numpy` · `scipy` · `statsmodels` · `seaborn` · `matplotlib`

---

## 실행 방법

**Google Colab 버전 (`AB_test_loan_analysis.ipynb`)**
1. `ADAproject_2025_data.xlsx`를 Google Drive에 업로드
2. Colab에서 파일 열기
3. `FILE_PATH`를 본인 Drive 경로로 수정
4. Runtime → Run all

**로컬 버전 (`AB_test_loan_analysis_local.ipynb`)**
1. `ADAproject_2025_data.xlsx`를 로컬에 저장
2. Spyder 또는 Jupyter에서 파일 열기
3. `FILE_PATH`를 본인 로컬 경로로 수정
4. 전체 실행

> 데이터 파일은 과제 데이터로 본 레포에 포함되지 않습니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `AB_test_loan_analysis.ipynb` | Colab 버전 분석 노트북 (69셀) |
| `AB_test_loan_analysis_local.ipynb` | 로컬 버전 분석 노트북 (67셀, Spyder/Jupyter용) |
| `REPORT.md` | 최종 보고서 (Executive Summary + 권고안) |
