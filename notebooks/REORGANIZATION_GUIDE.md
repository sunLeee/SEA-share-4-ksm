# Playground 노트북 재구성 가이드

## 현재 문제점

섹션 9-11의 셀 순서가 역순 및 혼재되어 있어 순차 실행 시 오류 발생

## 올바른 셀 실행 순서

### 섹션 1-8 (정상)
- ✅ 환경 설정, 데이터 로드, 시각화, 정량적 분석

### 섹션 9: 고급 통계 분석 🔬

**cell-25** [마크다운]
```
---
# 고급 통계 분석 모듈 활용 🔬
## 9. 통계적 유의성 검증 및 효과 크기 분석
```

**cell-34** [코드] - 통계 분석기 초기화
```python
# 고급 통계 분석기 초기화
analyzer = StatisticsAnalyzer(alpha=0.01, confidence_level=0.99)
```

**cell-33** [코드] - 도시지역 고급 통계 분석
```python
# 도시지역 고급 통계 분석 - 통계적 유의성 및 효과 크기
print("=" * 80)
print("🏙️ 도시지역 셔클 도입 효과 - 고급 통계 분석")
...
```

**cell-32** [코드] - 농어촌지역 고급 통계 분석
```python
# 농어촌지역 고급 통계 분석 - 통계적 유의성 및 효과 크기
print("=" * 80)
print("🌾 농어촌지역 셔클 도입 효과 - 고급 통계 분석")
...
```

**cell-31** [코드] - 효과 크기 종합 비교
```python
# 효과 크기 종합 비교: 도시 vs 농어촌
print("=" * 80)
print("📊 효과 크기 종합 비교 - Cohen's d")
...
```

**cell-30** [마크다운] - 통계 분석 결과 요약
```
## 10. 통계 분석 결과 요약 및 결론 📋
```

### 섹션 11: 포레스트 플롯 시각화 🌲

**cell-29** [마크다운]
```
## 11. 포레스트 플롯 (Forest Plot) 시각화 🌲
```

**cell-28** [코드] - 도시지역 포레스트 플롯
```python
# 포레스트 플롯 함수 임포트
import importlib
...
# 도시지역 포레스트 플롯
```

**cell-27** [코드] - 농어촌지역 포레스트 플롯
```python
# 농어촌지역 포레스트 플롯
fig, ax = create_forest_plot(
    stats_results=rural_stats_results,
    ...
```

**cell-26** [마크다운] - 포레스트 플롯 해석 방법
```
### 포레스트 플롯 해석 방법 📖
```

## 재구성 방법

### 방법 1: Jupyter에서 수동 재배치

1. Jupyter 노트북 열기
2. 명령 모드(ESC)에서 셀 선택
3. 순서 조정:
   - `X`: 셀 잘라내기
   - 원하는 위치로 이동
   - `V`: 셀 붙여넣기

### 방법 2: Kernel Restart & Run All

현재 파일을 그대로 두고 다음 순서로 실행:
1. Kernel → Restart
2. 셀 1-24까지 순차 실행
3. 셀 25 (섹션 9 제목) 실행
4. 셀 34 (통계 분석기 초기화) 실행
5. 셀 33 (도시 통계) 실행
6. 셀 32 (농어촌 통계) 실행
7. 셀 31 (종합 비교) 실행
8. 셀 30 (요약) 실행
9. 셀 29 (섹션 11 제목) 실행
10. 셀 28 (도시 포레스트) 실행
11. 셀 27 (농어촌 포레스트) 실행
12. 셀 26 (해석 방법) 실행

### 방법 3: 개별 셀 실행 (권장)

섹션 1-8 실행 후:
```python
# 섹션 9
# 1. cell-34 실행 (analyzer 초기화)
# 2. cell-33 실행 (city_stats_results 생성)
# 3. cell-32 실행 (rural_stats_results 생성)
# 4. cell-31 실행 (종합 비교)

# 섹션 11
# 5. cell-28 실행 (create_forest_plot 임포트 및 도시 플롯)
# 6. cell-27 실행 (농어촌 플롯)
```

## 수정 완료 사항

- ✅ cell-34: StatisticsAnalyzer 초기화 파라미터 수정
  - `n_bootstrap=10000` → 제거 (내부적으로 `bootstrap_samples=10000` 사용)
  - `confidence_level=0.99` 추가

- ✅ cell-27: create_forest_plot 임포트 누락 수정
  - visualization 모듈 리로드 및 함수 임포트 추가

## 검증 체크리스트

- [ ] analyzer 초기화 성공
- [ ] city_stats_results 생성 확인
- [ ] rural_stats_results 생성 확인
- [ ] 도시지역 포레스트 플롯 표시
- [ ] 농어촌지역 포레스트 플롯 표시
- [ ] 모든 통계 분석 완료

## 주의사항

1. **순서가 중요합니다**: cell-34 → cell-33 → cell-32 순으로 반드시 실행
2. **변수 의존성**:
   - `analyzer` 없이는 cell-33, cell-32 실행 불가
   - `city_stats_results`, `rural_stats_results` 없이는 포레스트 플롯 생성 불가
3. **모듈 리로드**: visualization 모듈 변경 시 cell-28에서 자동 리로드

---
*Last Updated: 2025-09-30*
*Author: taeyang lee*