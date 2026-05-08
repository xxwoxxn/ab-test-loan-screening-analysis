# -*- coding: utf-8 -*-
"""
A/B 테스트 기반 대출 심사 AI 모델 성과 분석

Spyder에서 실행하기 위해 변환된 스크립트
"""

# # A/B 테스트 기반 대출 심사 AI 모델 성과 분석
# 
# **원본:** Warwick Business School — Advanced Data Analysis (IB98D0), Group 6 (2025.02)
# **목적:** 원본 보고서의 방법론적 한계를 보완하고, 원본에 없는 행동 분석·비즈니스 임팩트 레이어를 추가한 실무 수준의 A/B 테스트 분석
# 
# ---
# 
# ## 원본 대비 개선 사항
# 
# | 항목 | Warwick 원본 | 본 분석 |
# |------|------------|--------|
# | 정규성 검정 | Skewness/Kurtosis만 | + Shapiro-Wilk 추가 |
# | 다중 비교 보정 | 없음 | Bonferroni 보정 적용 |
# | Cohen's d 기준 | 잘못된 레퍼런스 인용 | Sawilowsky(2009) 정정 |
# | 실험 무결성 검증 | 없음 | 1일차 Sanity Check 추가 |
# | 행동 분석 | 없음 | 확신도·Automation Bias·숙련도별 분석 |
# | 비즈니스 임팩트 | 없음 | 달러 환산 + 민감도 분석 |

# ## 0. 라이브러리 설치 및 임포트

# pip install pingouin  # 최초 1회만 실행

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120
print('라이브러리 설치 완료')

# ---
# 
# ## 1단계 — 데이터 클리닝 및 Sanity Check
# 
# **왜 하는가:**
# - 분석에 쓸 수 없는 미완료 데이터를 제거하고, 심사관 단위로 집계한다.
# - **Sanity Check**(원본에 없는 추가): 실험 시작 전 두 그룹의 초기 실력이 동질했는지 검증한다.
#   두 그룹이 처음부터 실력 차이가 있었다면, 이후 성과 차이를 모두 AI 모델 덕분으로 볼 수 없다.
# 
# **오류 정의:**
# - **Type I Error:** 우량 대출을 잘못 거절 → 이자 수익 기회 손실
# - **Type II Error:** 불량 대출을 잘못 승인 → 원금 손실

# ### 1-1. 데이터 로드

FILE_PATH = r'C:/temp/ADAproject_2025_data.xlsx'  # 데이터 파일 경로를 본인 환경에 맞게 수정

df = pd.read_excel(FILE_PATH)
print(f'데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열')

# ### 1-2. EDA — 기본 구조 확인

print('=== Shape ===')
print(df.shape)

print('\n=== Dtypes ===')
print(df.dtypes)

print('\n=== 결측치 현황 ===')
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else '결측치 없음 (0개)')

print('\n=== 상위 5행 ===')
df.head()

# 정제 전 그룹 비율 확인
vc = df['Variant'].value_counts()
total = len(df)
print('=== 그룹 비율 (정제 전) ===')
for v, cnt in vc.items():
    print(f'  {v}: {cnt}개 ({cnt/total*100:.1f}%)')

print('\n=== fully_complt 분포 (0=미완료, 10=완료) ===')
print(df['fully_complt'].value_counts().sort_index())

# ### 1-3. 필터링 — 미완료 행 제거 + 카테고리형 변환

# fully_complt = 0: 심사를 끝까지 완료하지 않은 행 → 제거
df_clean = df[df['fully_complt'] != 0].copy()
print(f'정제 전: {len(df)}행  →  정제 후: {len(df_clean)}행  ({len(df)-len(df_clean)}행 제거)')

df_clean['Variant'] = df_clean['Variant'].astype('category')
print(f'Variant dtype: {df_clean["Variant"].dtype}')
print(f'카테고리: {df_clean["Variant"].cat.categories.tolist()}')

# ### 1-4. OEC 계산 + 심사관 단위 10일 평균 집계

# **OEC (Overall Evaluation Criterion) 정의**
# 
# ```
# TP = badloans_num - typeII_fin    # 불량 대출 중 올바르게 거절한 건수
# TN = goodloans_num - typeI_fin    # 우량 대출 중 올바르게 승인한 건수
# 
# PrecisionRej = TP / (TP + FP) × 100   # 불량 대출 거절 정확도 (Supporting OEC)
# PrecisionAp  = TN / (TN + FN) × 100   # 우량 대출 승인 정확도 (Supporting OEC)
# F1 = 2 × (PrecisionAp × PrecisionRej) / (PrecisionAp + PrecisionRej)  # Primary OEC
# ```
# 
# F1을 Primary OEC로 쓰는 이유: 하나의 지표만 최적화하면 다른 쪽을 희생하는 트레이드오프가 발생한다.
# F1은 두 정확도를 균형 있게 반영하므로 단일 지표로 전반적 성과를 평가하기에 적합하다.

# 행별 OEC 계산
TN = df_clean['goodloans_num'] - df_clean['typeI_fin']
FN = df_clean['typeII_fin']
TP = df_clean['badloans_num'] - df_clean['typeII_fin']
FP = df_clean['typeI_fin']

df_clean['PrecisionAp']  = np.where((TN + FN) == 0, np.nan, TN / (TN + FN) * 100)
df_clean['PrecisionRej'] = np.where((TP + FP) == 0, np.nan, TP / (TP + FP) * 100)
df_clean['F1'] = (
    2 * df_clean['PrecisionAp'] * df_clean['PrecisionRej'] /
    (df_clean['PrecisionAp'] + df_clean['PrecisionRej'])
)

# 심사관(loanofficer_id) 단위로 10일 평균 집계
agg = (
    df_clean
    .groupby(['loanofficer_id', 'Variant'], observed=True)[['PrecisionAp', 'PrecisionRej', 'F1']]
    .mean().reset_index()
    .rename(columns={'PrecisionAp': 'Avg_PrecisionAp', 'PrecisionRej': 'Avg_PrecisionRej', 'F1': 'Avg_F1'})
)

# PrecisionAp = NA 심사관 제거 (우량 대출 승인이 0건 → 지표 계산 불가)
before = len(agg)
agg = agg.dropna(subset=['Avg_PrecisionAp']).reset_index(drop=True)
print(f'NA 제거: {before}명 → {len(agg)}명  (제거: {before - len(agg)}명)')

vc2 = agg['Variant'].value_counts()
total2 = len(agg)
print('\n=== 정제 후 그룹 비율 ===')
for v, cnt in vc2.items():
    print(f'  {v}: {cnt}명 ({cnt/total2*100:.1f}%)')

print(f'\n=== 심사관별 평균 OEC (상위 5명) ===')
print(agg.head().round(2).to_string(index=False))

# ### 1-5. Sanity Check — 실험 시작 전 그룹 동질성 검증

# **왜 Sanity Check가 필요한가:**
# 무작위 배정이 제대로 됐다면 실험 시작 전 두 그룹의 실력이 비슷해야 한다.
# `day = 1` 데이터(AI를 아직 보지 않은 시점)에서 초기 오류율을 비교해 이를 검증한다.
# 
# - `typeI_init`: AI 추천 전 Type I 오류 수
# - `typeII_init`: AI 추천 전 Type II 오류 수
# 
# p > 0.05이면 두 그룹이 통계적으로 동질하다고 볼 수 있다.

# day=1 데이터만 추출 (AI 추천 전 순수 심사관 실력 비교)
day1   = df[df['day'] == 1]
ctrl_d1 = day1[day1['Variant'] == 'Control']
trt_d1  = day1[day1['Variant'] == 'Treatment']

print('=== Sanity Check: day=1 기준 그룹 간 초기 실력 동질성 검증 ===')
print(f'Control n={len(ctrl_d1)}, Treatment n={len(trt_d1)}\n')

for metric in ['typeI_init', 'typeII_init']:
    c = ctrl_d1[metric]
    t = trt_d1[metric]
    t_stat, p_val = stats.ttest_ind(c, t)
    verdict = '유의미하지 않음 (p>0.05) ✅' if p_val > 0.05 else '유의미한 차이 있음 (p<0.05) ⚠️'
    print(f'{metric}:')
    print(f'  Control mean = {c.mean():.2f},  Treatment mean = {t.mean():.2f}')
    print(f'  t = {t_stat:.3f},  p = {p_val:.4f}  →  {verdict}')
    print()

# **Sanity Check 결과 해석:**
# 
# | 지표 | Control | Treatment | p-value | 판정 |
# |------|---------|-----------|---------|------|
# | typeI_init | 2.11 | 1.50 | 0.0816 | ✅ 동질 |
# | typeII_init | 0.79 | 0.43 | 0.0477 | ⚠️ 차이 있음 |
# 
# **⚠️ 한계:** `typeII_init`에서 Treatment 그룹이 실험 전부터 Type II 오류율이 낮았다.
# 이는 F1 및 PrecisionRej 개선의 일부가 AI 모델 효과가 아닌 초기 실력 차이에서 기인했을 가능성을 배제할 수 없음을 의미한다. 보고서 한계점 섹션에 명시한다.

# ---
# 
# ## 2단계 — OEC 산출
# 
# 1단계 집계에서 `PrecisionAp` / `PrecisionRej` / `F1` 컬럼과 심사관별 평균(`agg`)이 이미 산출됐다.
# 여기서는 `TP` / `TN` / `FP` / `FN`을 명시적 컬럼으로 추가하고, 그룹별 최종 비교표를 정리한다.

# TP, TN, FP, FN 명시적 컬럼 추가
df_clean['TN'] = df_clean['goodloans_num'] - df_clean['typeI_fin']
df_clean['FN'] = df_clean['typeII_fin']
df_clean['TP'] = df_clean['badloans_num'] - df_clean['typeII_fin']
df_clean['FP'] = df_clean['typeI_fin']

print('=== TP / TN / FP / FN 검증 (상위 5행) ===')
df_clean[[
    'loanofficer_id', 'Variant', 'day',
    'badloans_num', 'goodloans_num', 'typeI_fin', 'typeII_fin',
    'TP', 'TN', 'FP', 'FN', 'PrecisionAp', 'PrecisionRej', 'F1'
]].head()

# 그룹별 평균 vs 원본 기준값 비교
summary = (agg
           .groupby('Variant', observed=True)[['Avg_F1', 'Avg_PrecisionAp', 'Avg_PrecisionRej']]
           .mean().round(2))

ref = pd.DataFrame({
    'Variant':      ['Control',  'Treatment'],
    'F1_ref':       [43.69,       61.52],
    'PrecAp_ref':   [74.49,       86.46],
    'PrecRej_ref':  [34.42,       55.01],
})

print('=== 그룹별 OEC 평균 (산출값) ===')
print(summary)
print()
print('=== 원본 기준값 비교 ===')
print(ref.to_string(index=False))
print()
print('=== 그룹 간 차이 ===')
for metric, label in [('Avg_F1','F1'), ('Avg_PrecisionAp','PrecisionAp'), ('Avg_PrecisionRej','PrecisionRej')]:
    ctrl = summary.loc['Control', metric]
    trt  = summary.loc['Treatment', metric]
    diff = trt - ctrl
    pct  = diff / ctrl * 100
    print(f'  {label:15s}: Control {ctrl:6.2f} → Treatment {trt:6.2f}  (+{diff:.2f}pp, +{pct:.1f}%)')

# **2단계 결과 해석:**
# 
# | 지표 | Control | Treatment | 차이 | 원본 일치 |
# |------|---------|-----------|------|----------|
# | F1 (Primary OEC) | 43.69 | 61.52 | +17.83pp | ✅ |
# | PrecisionAp | 74.49 | 86.46 | +11.97pp | ✅ |
# | PrecisionRej | 34.26 | 55.01 | +20.75pp | ⚠️ Δ0.16pp |
# 
# PrecisionRej 미세 차이(34.26 vs 34.42)는 `PrecisionAp = NA` 제거 후 집계 분모 변화에 의한 것으로 해석에 영향 없다.
# 세 지표 모두 Treatment > Control. 단, 이것이 통계적으로 유의미한지는 4단계에서 검정한다.

# ---
# 
# ## 3단계 — 정규성 & 등분산 검정
# 
# **왜 하는가:**
# t-test는 두 가지 가정을 전제로 한다: ① 데이터가 정규분포를 따를 것, ② 두 그룹의 분산이 같을 것.
# 4단계 t-test를 돌리기 전에 이 가정들이 성립하는지 검증해야 한다.
# 
# **원본 대비 개선:**
# 원본 보고서는 Skewness/Kurtosis와 Levene's Test만 사용했다.
# Skewness/Kurtosis는 '분포 모양 서술'일 뿐 정규성 **검정**이 아니다.
# 심사관 단위 집계 후 n=10~28 수준 소표본에서는 **Shapiro-Wilk**가 필수다.

# ### 3-1. Violin Plot — 분포 시각화

metrics = ['Avg_F1', 'Avg_PrecisionAp', 'Avg_PrecisionRej']
labels  = ['F1 (Primary OEC)', 'PrecisionAp (Supporting)', 'PrecisionRej (Supporting)']
palette = {'Control': '#4C72B0', 'Treatment': '#DD8452'}

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ax, metric, label in zip(axes, metrics, labels):
    sns.violinplot(data=agg, x='Variant', y=metric, ax=ax,
                   palette=palette, inner='box', order=['Control', 'Treatment'])
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Score (%)')

plt.suptitle('OEC 분포 — Control vs Treatment', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('violin_oec.png', dpi=120, bbox_inches='tight')
plt.show()
print('violin_oec.png 저장 완료')

# ### 3-2. Shapiro-Wilk 정규성 검정

# 6개 조합 (OEC 3개 × 그룹 2개) 전부 검정
# p > 0.05: 정규성 가정 성립 → t-test 사용 가능
groups = ['Control', 'Treatment']
sw_rows = []

for metric in metrics:
    for group in groups:
        data = agg.loc[agg['Variant'] == group, metric].dropna()
        stat, p = stats.shapiro(data)
        sw_rows.append({
            'OEC':       metric.replace('Avg_', ''),
            'Group':     group,
            'n':         len(data),
            'Shapiro-W': round(stat, 4),
            'p-value':   round(p, 4),
            '정규성':    '✅ 성립 (p>0.05)' if p > 0.05 else '❌ 불성립 (p≤0.05)'
        })

sw_df = pd.DataFrame(sw_rows)
print('=== Shapiro-Wilk 정규성 검정 ===')
print(sw_df.to_string(index=False))

# ### 3-3. Skewness / Kurtosis (원본 보고서 비교용)

sk_rows = []
for metric in metrics:
    for group in groups:
        data = agg.loc[agg['Variant'] == group, metric].dropna()
        skew = round(data.skew(), 3)
        kurt = round(data.kurtosis(), 3)  # Fisher's excess kurtosis
        sk_rows.append({
            'OEC':          metric.replace('Avg_', ''),
            'Group':        group,
            'Skewness':     skew,
            'Kurtosis':     kurt,
            'Skew OK':      '✅' if abs(skew) < 1 else '⚠️',
            'Kurt OK':      '✅' if abs(kurt) < 4 else '⚠️',
        })

sk_df = pd.DataFrame(sk_rows)
print('=== Skewness / Kurtosis ===')
print('기준: |Skewness| < 1, |Kurtosis(excess)| < 4 → 정규분포 근사')
print()
print(sk_df.to_string(index=False))

# ### 3-4. 이상치 확인 (IQR 기준)

out_rows = []
for metric in metrics:
    for group in groups:
        data = agg.loc[agg['Variant'] == group, metric].dropna()
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers = data[(data < lower) | (data > upper)]
        out_rows.append({
            'OEC':       metric.replace('Avg_', ''),
            'Group':     group,
            'n':         len(data),
            '이상치 수': len(outliers),
            '이상치 값': outliers.round(2).tolist() if len(outliers) > 0 else '-'
        })

out_df = pd.DataFrame(out_rows)
print('=== IQR 기반 이상치 탐지 ===')
print(out_df.to_string(index=False))
print()
print('판단: 이상치가 있어도 심사관 실력의 자연스러운 분산으로 해석 → 제거하지 않음')
print('      (소표본 n=10~28에서 제거 시 정보 손실 > 이상치 제거 효과)')

# ### 3-5. Levene's Test — 등분산 검정

lev_rows = []
for metric in metrics:
    ctrl = agg.loc[agg['Variant'] == 'Control',   metric].dropna()
    trt  = agg.loc[agg['Variant'] == 'Treatment', metric].dropna()
    stat, p = stats.levene(ctrl, trt)
    lev_rows.append({
        'OEC':      metric.replace('Avg_', ''),
        'Levene-W': round(stat, 4),
        'p-value':  round(p, 4),
        '등분산':   '✅ 성립 (p>0.05)' if p > 0.05 else '❌ 불성립 (p≤0.05)'
    })

lev_df = pd.DataFrame(lev_rows)
print('=== Levene 등분산 검정 ===')
print(lev_df.to_string(index=False))
print()
print('→ p > 0.05이면 등분산 가정 성립: 4단계 t-test에서 equal_var=True 사용 가능')

# ### 3-6. 정규성·등분산 결과 요약

print('=' * 60)
print('3단계 요약 — 정규성 & 등분산 검정')
print('=' * 60)
print()
summary_sw = sw_df[['OEC', 'Group', 'n', 'p-value', '정규성']].copy()
summary_sw.columns = ['OEC', 'Group', 'n', 'Shapiro p', '정규성']
print(summary_sw.to_string(index=False))
print()
print('--- Levene 등분산 ---')
print(lev_df.to_string(index=False))
print()
print('--- 4단계 t-test 설정 결론 ---')
for row in lev_rows:
    ev = 'equal_var=True ' if row['p-value'] > 0.05 else 'equal_var=False'
    print(f'  {row["OEC"]:15s}: {ev}  (Levene p={row["p-value"]})')

# **3단계 결과 해석:**
# 
# - **F1 Treatment 정규성 불성립** (p=0.0172): Treatment n=28이라 CLT가 어느 정도 보호하지만, 엄밀한 가정 위반으로 보고서 한계점에 기록한다.
# - **나머지 5개 조합**: 모두 정규성 성립
# - **Levene's Test**: 3개 OEC 전부 등분산 성립 → **4단계 전체 `equal_var=True` 적용**

# ---
# 
# ## 4단계 — 가설 검정
# 
# **왜 하는가:**
# 2단계에서 Treatment가 더 높아 보이지만, 이것이 통계적으로 유의미한지(우연이 아닌지) 검증해야 한다.
# 
# **원본 대비 개선:**
# - **Bonferroni 보정 추가**: 원본은 3개 지표에 t-test를 3번 반복하면서 다중 비교 보정을 하지 않았다.
#   α=0.05로 3번 검정하면 실제 1종 오류율이 최대 14.3%까지 올라간다.
#   Bonferroni 보정으로 α를 0.05/3 ≈ 0.017로 낮춘다.
# - **Cohen's d 기준 정정**: 원본이 잘못 인용한 기준 대신 Sawilowsky(2009) 사용
# - **95% CI 해석문 추가**: 원본은 표에만 기재, 본 분석은 실무 언어로 해석

# ### 4-1. 초기 오류율 사전 체크 (Sanity Check 참조)

# 1단계 Sanity Check 결과 요약 (재실행 불필요)
print('=== 초기 오류율 사전 체크 (1단계 Sanity Check 결과) ===')
print()
print('  typeI_init  : Control 2.11 vs Treatment 1.50 | p=0.0816 → 유의미 차이 없음 ✅')
print('  typeII_init : Control 0.79 vs Treatment 0.43 | p=0.0477 → 유의미 차이 있음 ⚠️')
print()
print('⚠️  typeII_init 차이 → Treatment 그룹이 실험 전부터 Type II 오류율이 낮았음.')
print('     PrecisionRej 개선의 일부가 모델 효과가 아닌 초기 실력 차이일 가능성 → 한계점 기록')

# ### 4-2. 독립표본 t-test + Bonferroni 보정

metric_names = ['F1', 'PrecisionAp', 'PrecisionRej']
rows     = []
p_values = []

for metric, name in zip(metrics, metric_names):
    ctrl = agg.loc[agg['Variant'] == 'Control',   metric].dropna()
    trt  = agg.loc[agg['Variant'] == 'Treatment', metric].dropna()
    t_stat, p_val = stats.ttest_ind(ctrl, trt, equal_var=True)
    p_values.append(p_val)
    rows.append({
        'OEC':         name,
        'Control μ':   round(ctrl.mean(), 2),
        'Treatment μ': round(trt.mean(), 2),
        '평균차':      round(trt.mean() - ctrl.mean(), 2),
        't-stat':      round(t_stat, 3),
        'df':          len(ctrl) + len(trt) - 2,
        'p-value':     round(p_val, 4),
    })

# Bonferroni 보정
_, p_corr, _, _ = multipletests(p_values, method='bonferroni')
alpha_bonf = 0.05 / len(metrics)

for i, row in enumerate(rows):
    row['Bonferroni p'] = round(p_corr[i], 4)
    row['유의미']       = '✅' if p_corr[i] < alpha_bonf else '❌'

ttest_df = pd.DataFrame(rows)
print(f'=== 독립표본 t-test + Bonferroni 보정 (보정 α = {alpha_bonf:.4f}) ===')
print(ttest_df.to_string(index=False))

# ### 4-3. Cohen's d — 효과 크기

# Cohen's d 계산 (Sawilowsky 2009 기준)
# 기준: 0.2(Small) / 0.5(Medium) / 0.8(Large) / 1.2(Very Large) / 2.0+(Huge)
def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*x.std(ddof=1)**2 + (ny-1)*y.std(ddof=1)**2) / (nx+ny-2))
    return (y.mean() - x.mean()) / pooled_std  # Treatment - Control

def sawilowsky(d):
    d = abs(d)
    if   d < 0.2:  return 'Tiny'
    elif d < 0.5:  return 'Small'
    elif d < 0.8:  return 'Medium'
    elif d < 1.2:  return 'Large'
    elif d < 2.0:  return 'Very Large'
    else:          return 'Huge'

cohen_rows = []
for metric, name in zip(metrics, metric_names):
    ctrl = agg.loc[agg['Variant'] == 'Control',   metric].dropna()
    trt  = agg.loc[agg['Variant'] == 'Treatment', metric].dropna()
    d = cohens_d(ctrl, trt)
    cohen_rows.append({'OEC': name, "Cohen's d": round(d, 3), '해석': sawilowsky(d)})

cohen_df = pd.DataFrame(cohen_rows)
print("=== Cohen's d 효과 크기 (Sawilowsky 2009 기준) ===")
print('기준: 0.2 Small / 0.5 Medium / 0.8 Large / 1.2 Very Large / 2.0+ Huge')
print()
print(cohen_df.to_string(index=False))

# ### 4-4. 95% 신뢰구간

ci_rows = []
for metric, name in zip(metrics, metric_names):
    ctrl = agg.loc[agg['Variant'] == 'Control',   metric].dropna()
    trt  = agg.loc[agg['Variant'] == 'Treatment', metric].dropna()
    nc, nt = len(ctrl), len(trt)
    diff = trt.mean() - ctrl.mean()
    sp   = np.sqrt(((nc-1)*ctrl.std(ddof=1)**2 + (nt-1)*trt.std(ddof=1)**2) / (nc+nt-2))
    se   = sp * np.sqrt(1/nc + 1/nt)
    df_  = nc + nt - 2
    t_crit = stats.t.ppf(0.975, df_)
    lo, hi = diff - t_crit*se, diff + t_crit*se
    ci_rows.append({'OEC': name, '평균차': round(diff,2), 'CI 하한': round(lo,2), 'CI 상한': round(hi,2)})

ci_df = pd.DataFrame(ci_rows)
print('=== 95% 신뢰구간 ===')
print(ci_df.to_string(index=False))
print()
print('--- 실무 해석 문장 ---')
for row in ci_rows:
    print(f"  {row['OEC']:13s}: Treatment가 Control보다 최소 {row['CI 하한']:.1f}pp ~ 최대 {row['CI 상한']:.1f}pp 높음이 95% 확실")

# ### 4-5. 최종 요약 테이블

final = ttest_df.merge(cohen_df, on='OEC').merge(ci_df, on='OEC')
cols = ['OEC', 'Control μ', 'Treatment μ', '평균차',
        't-stat', 'df', 'p-value', 'Bonferroni p', '유의미',
        "Cohen's d", '해석', 'CI 하한', 'CI 상한']
print('=' * 90)
print('4단계 최종 요약 테이블')
print('=' * 90)
print(final[cols].to_string(index=False))

# **4단계 결과 해석:**
# 
# 세 지표 모두 Bonferroni 보정 후에도 p ≈ 0 (사실상 0에 가까운 극소값)으로 통계적으로 유의미하다.
# 
# | OEC | Cohen's d | 의미 |
# |-----|-----------|------|
# | F1 | 3.73 | 두 그룹 분포가 거의 겹치지 않음 |
# | PrecisionAp | 4.56 | 사회과학에서 극히 드문 효과 크기 |
# | PrecisionRej | 3.70 | AI 유무가 집단을 완전히 분리 |
# 
# **95% CI 결론:** F1 기준, 가장 보수적으로 봐도 최소 +14.3pp 개선이 95% 확실하다.

# ---
# 
# ## 5단계 — 행동 분석 (Behavioral Analysis)
# 
# **왜 하는가 (원본에 없는 레이어):**
# "AI 모델 성능이 좋다"를 넘어 "심사관이 AI를 어떻게 활용하고 있는가"를 파악해야
# 실제 도입 전략에 의미 있는 인사이트를 줄 수 있다.
# 
# - **확신도 변화**: AI 추천을 본 후 심사관의 자기 결정 확신이 높아졌는가
# - **Automation Bias**: AI 추천을 너무 맹목적으로 따르는 심사관이 있는가
# - **숙련도별 효과**: 초보 vs 숙련 심사관 중 누가 AI를 더 잘 활용하는가

# ### 5-1. 행동 변수 심사관 단위 집계

behav = (
    df_clean
    .groupby(['loanofficer_id', 'Variant'], observed=True)
    .agg(
        conf_init_sum = ('confidence_init_total', 'sum'),
        conf_fin_sum  = ('confidence_fin_total',  'sum'),
        cases_init    = ('complt_init', 'sum'),
        cases_fin     = ('complt_fin',  'sum'),
        rev_ai_sum    = ('revised_per_ai', 'sum'),
        typeI_init    = ('typeI_init',  'mean'),
        typeII_init   = ('typeII_init', 'mean'),
    ).reset_index()
)

# 파생 변수
behav['conf_init_per'] = behav['conf_init_sum'] / behav['cases_init'].replace(0, np.nan)
behav['conf_fin_per']  = behav['conf_fin_sum']  / behav['cases_fin'].replace(0, np.nan)
behav['conf_change']   = behav['conf_fin_per']  - behav['conf_init_per']   # 확신도 변화량
behav['rev_ai_rate']   = behav['rev_ai_sum']    / behav['cases_fin'].replace(0, np.nan) * 100  # AI 수정 비율(%)
behav['init_err']      = behav['typeI_init']    + behav['typeII_init']     # 초기 총 오류율

behav = behav.merge(agg[['loanofficer_id', 'Avg_F1', 'Avg_PrecisionAp', 'Avg_PrecisionRej']], on='loanofficer_id')
print(f'행동 분석 데이터: {len(behav)}명  (Control {(behav["Variant"]=="Control").sum()} / Treatment {(behav["Variant"]=="Treatment").sum()})')

# ### 5-2. 확신도 변화 분석

ctrl_c = behav.loc[behav['Variant'] == 'Control',   'conf_change'].dropna()
trt_c  = behav.loc[behav['Variant'] == 'Treatment', 'conf_change'].dropna()
t, p   = stats.ttest_ind(ctrl_c, trt_c, equal_var=True)

print('=== 확신도 변화 (confidence per case: fin - init) ===')
print(f'  Control   n={len(ctrl_c)}  mean={ctrl_c.mean():.2f}')
print(f'  Treatment n={len(trt_c)}  mean={trt_c.mean():.2f}')
print(f'  차이: {trt_c.mean()-ctrl_c.mean():.2f}  t={t:.3f}  p={p:.4f}  → {"유의미 ✅" if p<0.05 else "유의미하지 않음"}')

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=behav, x='Variant', y='conf_change', ax=ax,
            order=['Control', 'Treatment'],
            palette={'Control': '#4C72B0', 'Treatment': '#DD8452'})
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
ax.set_title('확신도 변화량 (AI 추천 전후)', fontsize=13)
ax.set_ylabel('Δ confidence per case')
plt.tight_layout()
plt.savefig('conf_change.png', dpi=120, bbox_inches='tight')
plt.show()

# ### 5-3. Automation Bias 탐지

# **Automation Bias란?**
# 심사관이 AI 추천을 비판적으로 검토하지 않고 맹목적으로 따르는 현상.
# `revised_per_ai / complt_fin`이 지나치게 높은 심사관을 식별한다.
# 오류가 줄었어도 그 이유가 "AI가 잘해서"가 아니라 "그냥 AI를 따라가서"일 수 있기 때문이다.

trt = behav[behav['Variant'] == 'Treatment'].copy()

print('=== Automation Bias 탐지 (Treatment 그룹) ===')
print(f'  rev_ai_rate: min={trt["rev_ai_rate"].min():.1f}%  '
      f'median={trt["rev_ai_rate"].median():.1f}%  '
      f'max={trt["rev_ai_rate"].max():.1f}%')

threshold = trt['rev_ai_rate'].quantile(0.75)
trt['bias_flag'] = trt['rev_ai_rate'] > threshold
print(f'  임계값 (75th percentile): {threshold:.1f}%')
print(f'  과의존 의심 심사관: {trt["bias_flag"].sum()}명 / {len(trt)}명')

if trt['bias_flag'].sum() > 0:
    print()
    print(trt[trt['bias_flag']][['loanofficer_id', 'rev_ai_rate', 'Avg_F1']].round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
colors = ['#DD8452' if f else '#AEC6E8' for f in trt['bias_flag']]
ax.barh(trt['loanofficer_id'], trt['rev_ai_rate'], color=colors)
ax.axvline(threshold, color='red', linestyle='--', label=f'임계값 {threshold:.1f}%')
ax.set_xlabel('AI 추천 수정 비율 (%)')
ax.set_title('심사관별 AI 추천 수정 비율 (Treatment)')
ax.legend()
plt.tight_layout()
plt.savefig('automation_bias.png', dpi=120, bbox_inches='tight')
plt.show()

# ### 5-4. 숙련도별 효과 분석

median_err = behav['init_err'].median()
behav['skill'] = np.where(behav['init_err'] <= median_err, '숙련(오류낮음)', '초보(오류높음)')

print(f'=== 숙련도 분류 (초기 오류율 중앙값: {median_err:.2f}) ===')
print(behav.groupby(['Variant', 'skill'], observed=True).size().to_string())
print()

skill_grp = behav.groupby(['Variant', 'skill'], observed=True)['Avg_F1'].agg(['mean', 'count']).round(2)
print('=== 숙련도 × 그룹별 평균 F1 ===')
print(skill_grp.to_string())
print()

ctrl_s = behav[behav['Variant'] == 'Control'].groupby('skill')['Avg_F1'].mean()
trt_s  = behav[behav['Variant'] == 'Treatment'].groupby('skill')['Avg_F1'].mean()
print('=== 향상폭 (Treatment - Control) ===')
for s in sorted(ctrl_s.index):
    if s in trt_s.index:
        print(f'  {s}: +{trt_s[s]-ctrl_s[s]:.2f}pp')

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=behav, x='skill', y='Avg_F1', hue='Variant',
            palette={'Control': '#4C72B0', 'Treatment': '#DD8452'}, ax=ax)
ax.set_title('숙련도별 F1 — Control vs Treatment', fontsize=13)
ax.set_ylabel('Avg F1 (%)')
ax.set_xlabel('')
plt.tight_layout()
plt.savefig('skill_f1.png', dpi=120, bbox_inches='tight')
plt.show()

# **5단계 결과 해석:**
# 
# | 분석 | 결과 | 해석 |
# |------|------|------|
# | 확신도 변화 | p=0.5252 (유의미하지 않음) | AI가 확신도를 높이는 효과는 제한적 |
# | Automation Bias | 7명/28명(25%) 과의존 의심 | 대다수는 적절히 활용 중, 소수 교육 필요 |
# | 숙련도별 향상폭 | 숙련 +20.56pp > 초보 +15.31pp | 숙련자가 AI를 더 효과적으로 활용 |
# 
# **권고안:** 숙련 심사관 우선 파일럿 배포 → AI 리터러시 교육 후 전체 확대

# ---
# 
# ## 6단계 — 비즈니스 임팩트 환산 (Financial Impact)
# 
# **왜 하는가 (원본에 없는 레이어):**
# 임원진은 p-value나 Cohen's d를 모른다.
# "Type II 오류가 20.59% 줄었다"보다 "연간 약 $X의 손실을 줄일 수 있다"가 의사결정에 직접적이다.
# 
# **가정 설정 (과제 지침 기반):**
# - 대출금 범위: $20,000 ~ $35,000 (중앙값 $27,500)
# - Type II Error 비용: 원금 손실 ≈ 대출금의 100%
# - Type I Error 비용: 이자 수익 기회비용 ≈ 대출금의 10%

# 심사관별 10일 오류 합산
err = (
    df_clean
    .groupby(['loanofficer_id', 'Variant'])
    .agg(typeI_sum=('typeI_fin', 'sum'), typeII_sum=('typeII_fin', 'sum'))
    .reset_index()
)

ctrl_g = err[err['Variant'] == 'Control']
trt_g  = err[err['Variant'] == 'Treatment']

ctrl_typeI  = ctrl_g['typeI_sum'].mean()
ctrl_typeII = ctrl_g['typeII_sum'].mean()
trt_typeI   = trt_g['typeI_sum'].mean()
trt_typeII  = trt_g['typeII_sum'].mean()

print('=== 심사관 1인당 10일 평균 오류 건수 ===')
print(f'               Type I (우량 거절)  Type II (불량 승인)')
print(f'  Control      {ctrl_typeI:>10.2f}건      {ctrl_typeII:>10.2f}건')
print(f'  Treatment    {trt_typeI:>10.2f}건      {trt_typeII:>10.2f}건')
print(f'  감소         {ctrl_typeI-trt_typeI:>+10.2f}건      {ctrl_typeII-trt_typeII:>+10.2f}건')

loan_scenarios = [20000, 27500, 35000]
results = []

print('=== 심사관 1인당 10일 손실 절감액 ===')
print(f'{"대출금":>10} | {"TypeI 절감":>12} | {"TypeII 절감":>13} | {"합계":>12}')
print('-' * 55)

for loan in loan_scenarios:
    cost_typeI  = loan * 0.10
    cost_typeII = loan * 1.00
    save_typeI  = (ctrl_typeI  - trt_typeI)  * cost_typeI
    save_typeII = (ctrl_typeII - trt_typeII) * cost_typeII
    total = save_typeI + save_typeII
    results.append({'loan': loan, 'typeI': save_typeI, 'typeII': save_typeII, 'total': total})
    print(f'${loan:>9,} | ${save_typeI:>11,.0f} | ${save_typeII:>12,.0f} | ${total:>11,.0f}')

# 전체 은행 스케일
n_officers = len(err)
mid = results[1]
print()
print(f'=== 전체 심사관 {n_officers}명 기준 10일 절감액 (대출금 $27,500 시나리오) ===')
print(f'  Type I  절감: ${mid["typeI"]*n_officers:>12,.0f}')
print(f'  Type II 절감: ${mid["typeII"]*n_officers:>12,.0f}')
print(f'  합계:         ${mid["total"]*n_officers:>12,.0f}')

print('=== 민감도 분석 — 가정이 달라도 방향 확인 ===')
print(f'{"대출금":>10} | {"1인당 10일 절감액":>16} | 방향')
print('-' * 40)
for r in results:
    direction = "AI 모델 유리 ✅" if r['total'] > 0 else "Control 유리"
    print(f'${r["loan"]:>9,} | ${r["total"]:>15,.0f} | {direction}')

lo = results[0]['total']
hi = results[2]['total']
print()
print('=== 임원진용 요약 ===')
print(f'신규 AI 심사 모델 도입 시 심사관 1인당 10일 기준')
print(f'약 ${lo:,.0f} ~ ${hi:,.0f}의 손실 절감 효과 예상')
print(f'(3가지 대출금 시나리오 모두에서 AI 모델 유리 → 가정 변화에도 결론 동일)')

# **6단계 결과 해석:**
# 
# | 대출금 가정 | 1인당 10일 절감 |
# |------------|--------------|
# | $20,000 | $106,900 |
# | $27,500 | $146,988 |
# | $35,000 | $187,075 |
# 
# 세 시나리오 모두 AI 모델 유리 → 민감도 분석 통과.
# 전체 38명 기준 10일 절감액: **약 $5.6M** (중앙값 시나리오)

# ---
# 
# ## 시각화 파일 저장 및 다운로드

# Google Drive에 저장 (Drive가 마운트된 경우)
import os, shutil

save_dir = 'output'  # 저장 폴더 (자동 생성)
os.makedirs(save_dir, exist_ok=True)

fnames = ['violin_oec.png', 'conf_change.png', 'automation_bias.png', 'skill_f1.png']
for fname in fnames:
    if os.path.exists(fname):
        shutil.copy(fname, f'{save_dir}/{fname}')
        print(f'Drive 저장 완료: {fname}')
    else:
        print(f'파일 없음 (해당 셀 먼저 실행): {fname}')
