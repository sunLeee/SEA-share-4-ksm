# 셔클 효과 분석 패키지

**셔클(수요응답형 교통서비스) 도입 효과 시각화 및 통계 분석 도구**

## 개요

본 패키지는 셔클 도입 전후의 교통 데이터를 고품질 시각화와 통계적 유의성 검증을 통해 분석하는 전문 도구입니다. 데이터 분포를 직관적으로 시각화하고, 관찰된 개선 효과가 통계적으로 유의미함을 검증합니다.

## 주요 기능

### 1. 고급 시각화 (핵심 기능)

**다중 플롯 타입으로 데이터 분포 분석**
- **Boxplot**: 사분위수, 중앙값, 이상값을 명확하게 표시
- **Violin Plot**: 데이터 분포의 확률밀도를 시각적으로 표현
- **Boxen Plot**: 다층 분위수로 분포의 세밀한 구조 파악
- **Forest Plot**: 효과 크기와 99% 신뢰구간을 한눈에 비교

**시각화 옵션**
- 고품질 출력: PNG, PDF, SVG 형식 (300-600 DPI)
- 평균값 자동 표시 및 위치 최적화
- 이상값 표시 옵션
- 데이터 클리핑 (극단값 제거)
- 그래프 크기 및 스타일 커스터마이징

### 2. 통계적 유의성 검증 (시각화 검증)

시각적으로 관찰되는 개선 효과가 통계적으로도 유의미함을 검증합니다.

- **가설 검정**: Paired t-test, Wilcoxon signed-rank test, Bootstrap test
- **효과 크기**: Cohen's d 계산 및 해석 (Small/Medium/Large)
- **신뢰구간**: 99% 신뢰구간 추정
- **강건 통계**: 이상값에 민감하지 않은 절사평균, MAD 활용

### 3. 자동 데이터 처리

- 결측값 자동 제거
- 단위 변환 (초 → 분)
- 지역별 자동 분류 (도시/농어촌)
- 카테고리별 분석 (도보시간, 탑승시간, 대기시간, 총 이동시간)

## 설치 방법

### 필수 요구사항
- Python 3.8 이상
- Git
- 최소 4GB RAM 권장

### 1단계: 저장소 클론

터미널(또는 명령 프롬프트)을 열고 다음 명령어를 실행합니다:

```bash
# 저장소 클론
git clone https://github.com/sunLeee/SEA-share-4-ksm.git

# 클론된 디렉토리로 이동
cd SEA-share-4-ksm
```

### 2단계: 가상환경 생성 및 활성화

**Windows 사용자:**
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

**macOS/Linux 사용자:**
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

가상환경이 활성화되면 터미널 프롬프트 앞에 `(venv)`가 표시됩니다.

### 3단계: 패키지 설치

```bash
# 필수 패키지 설치
pip install -r requirements.txt
```

설치가 완료되면 다음과 같은 메시지가 표시됩니다:
```
Successfully installed pandas-2.0.0 numpy-1.24.0 matplotlib-3.6.0 ...
```

### 4단계: 설치 확인

```bash
# Python 실행 확인
python --version

# 패키지 설치 확인
python -c "import pandas; import matplotlib; print('설치 완료!')"
```

`설치 완료!` 메시지가 표시되면 정상적으로 설치된 것입니다.

## 사용 방법

### 5단계: 데이터 준비

분석할 데이터 파일을 `data/` 폴더에 배치합니다.

```bash
# data 폴더 확인
ls data/

# 예상 출력: shucle_analysis_dataset_20250929.csv
```

**데이터 형식 요구사항:**
- CSV 형식
- 필수 컬럼: `zone_type` (도시/농어촌 구분)
- 시간 데이터: 초 단위 (자동으로 분으로 변환됨)

### 6단계: 분석 실행 - 대화형 모드 (권장)

```bash
# 대화형 모드로 실행
python scripts/main.py --interactive
```

**전체 실행 과정:**

```bash
# 1. 프로그램 시작
(venv) $ python scripts/main.py --interactive

# 2. 환영 메시지 및 기본 정보
🔧 대화형 설정 모드
엔터를 누르면 기본값이 사용됩니다.

# 3. 데이터 파일 설정
데이터 파일 경로 [./data/shucle_analysis_dataset_20250929.csv]:
[엔터를 누르면 기본값 사용]

# 4. 출력 디렉토리 설정
출력 디렉토리 [./output]: ./results
[또는 엔터로 기본값 사용]

# 5. 플롯 저장 설정
플롯 저장 (Y/N) [Y]:
[Y 입력 또는 엔터]

# 6. 플롯 표시 설정 (화면에 그래프 표시 여부)
플롯 표시 (Y/N) [N]:
[N 입력 또는 엔터 - 파일로만 저장]

# 7. 시각화 세부 설정
📊 시각화 세부 설정:

# 플롯 타입 선택
플롯 타입 (쉼표로 구분) [boxplot, violin, boxen]: boxplot,violin,boxen
[모든 타입 생성하려면 엔터, 특정 타입만 원하면 쉼표로 구분하여 입력]

# 그래프 크기
그래프 크기 (가로x세로) [15x8]:
[기본 크기 사용하려면 엔터]

# 출력 형식
출력 형식 (png/pdf/svg) [png]: pdf
[고품질 벡터 그래픽이 필요하면 pdf 입력]

# 이미지 해상도
이미지 해상도 DPI [300]: 600
[고해상도가 필요하면 600 입력, 기본은 엔터]

# 8. 고급 시각화 옵션
🎨 고급 시각화 옵션:

이상값 표시 (Y/N) [N]:
[이상값을 보고 싶으면 Y, 아니면 엔터]

평균값 표시 (Y/N) [Y]:
[평균선을 표시하려면 엔터]

평균값 숫자 표시 (Y/N) [Y]:
[평균값을 숫자로 표시하려면 엔터]

데이터 클리핑 백분위수 (0.9-1.0) [0.95]:
[극단값 5%를 제거하려면 엔터, 10% 제거하려면 0.90 입력]

평균값 위치 전략 (adaptive/fixed_percentage/above_max) [adaptive]:
[자동 위치 조정이면 엔터]

# 9. 설정 완료 확인
✅ 설정 완료!
📊 플롯 타입: boxplot, violin, boxen
📐 그래프 크기: 15x8
🎯 해상도: 600 DPI
📁 출력: ./results

# 10. 분석 시작
🚀 셔클 효과 분석 시작...
📁 출력 디렉토리: results
✅ 데이터 로드 성공: 50,000개 행

# 11. 데이터 전처리
📊 데이터 전처리 중...
✅ 도시지역 처리 완료: 30,000개 행
✅ 농어촌지역 처리 완료: 20,000개 행

# 12. 시각화 생성
🎨 시각화 생성 중...
💾 도시지역 boxplot 저장: results/city_boxplot_analysis.pdf (DPI: 600)
💾 도시지역 violin 저장: results/city_violin_analysis.pdf (DPI: 600)
💾 도시지역 boxen 저장: results/city_boxen_analysis.pdf (DPI: 600)
💾 농어촌지역 boxplot 저장: results/rural_boxplot_analysis.pdf (DPI: 600)
💾 농어촌지역 violin 저장: results/rural_violin_analysis.pdf (DPI: 600)
💾 농어촌지역 boxen 저장: results/rural_boxen_analysis.pdf (DPI: 600)

# 13. 통계 분석
📈 통계 분석 수행 중...
🔬 고급 통계 분석 수행 중...
✅ 고급 통계 분석 완료

# 14. Forest Plot 생성
🌲 Forest Plot 생성 중...
💾 도시지역 Forest Plot 저장: results/city_forest_plot.pdf
💾 농어촌지역 Forest Plot 저장: results/rural_forest_plot.pdf
✅ Forest Plot 생성 완료

# 15. 분석 결과 요약
============================================================
🏆 분석 결과 요약
============================================================
  지역  카테고리  대중교통_평균  셔클_평균  평균_개선량  평균_개선율  중앙값_개선량  중앙값_개선율
  도시  도보시간       7.62    5.97      1.65      21.68         1.10        15.71
  도시  탑승시간      14.52   13.46      1.06       7.27         1.37        10.51
  ...

# 16. 완료
✅ 모든 분석 완료! 결과는 results에 저장되었습니다.

📋 저장된 파일 목록:
  • city_boxplot_analysis.pdf
  • city_violin_analysis.pdf
  • city_boxen_analysis.pdf
  • city_forest_plot.pdf
  • rural_boxplot_analysis.pdf
  • rural_violin_analysis.pdf
  • rural_boxen_analysis.pdf
  • rural_forest_plot.pdf
  • analysis_results.csv
  • walking_time_city.csv
  • walking_time_rural.csv
  • ...
```

### 기본 모드 실행 (설정 없이 빠른 실행)

```bash
# 기본 설정으로 빠르게 실행
python scripts/main.py

# 결과는 ./output 폴더에 PNG 형식으로 저장됨
```

## 시각화 결과 해석

### 생성되는 시각화

분석 실행 후 다음과 같은 시각화가 생성됩니다:

**1. Boxplot (박스플롯)**
- 중앙값(가운데 선), 사분위수(박스), 최소/최대값(수염) 표시
- 평균값이 숫자로 표시됨
- 도시/농어촌 지역별, 카테고리별 비교 가능
- **용도**: 데이터의 중심 경향성과 산포 파악

**2. Violin Plot (바이올린 플롯)**
- 데이터 분포의 확률밀도를 곡선으로 표시
- 분포가 넓은 곳은 데이터가 많이 집중됨
- 다봉 분포(여러 개의 봉우리) 확인 가능
- **용도**: 데이터 분포의 형태와 패턴 분석

**3. Boxen Plot (박센 플롯)**
- 다층 분위수를 여러 박스로 표시
- Boxplot보다 더 세밀한 분포 정보 제공
- 꼬리 분포(극단값 영역)를 더 잘 표현
- **용도**: 정밀한 분포 구조 파악

**4. Forest Plot (포레스트 플롯)**
- 평균 차이와 99% 신뢰구간을 점과 선으로 표시
- p-value와 Cohen's d가 함께 표시됨
- 0 기준선(효과 없음)과 비교하여 유의성 판단
- **용도**: 통계적 유의성과 효과 크기를 한눈에 확인

### Forest Plot 읽는 법

```
카테고리1  ●━━━━━━●  p<0.001, d=0.85 (큰 효과)
카테고리2      ●━━●   p=0.002, d=0.42 (중간 효과)
           |
           0 (효과 없음)
```

- **점(●)**: 평균 시간 단축량
- **선(━)**: 99% 신뢰구간 (더 짧을수록 정확한 추정)
- **0 기준선**: 신뢰구간이 0을 포함하지 않으면 통계적으로 유의함
- **p-value**: 0.01보다 작으면 통계적으로 유의함 (99% 신뢰수준)
- **Cohen's d**: 0.8 이상이면 큰 효과, 0.5~0.8이면 중간 효과

## 커스터마이제이션 옵션

### 시각화 옵션 상세

| 옵션            | 설명                    | 기본값                 | 권장 설정              |
| --------------- | ----------------------- | ---------------------- | ---------------------- |
| 플롯 타입       | 생성할 차트 종류        | boxplot, violin, boxen | 모두 생성 (엔터)       |
| 그래프 크기     | 차트 크기 (가로x세로)   | 15x8                   | 기본값 사용 (엔터)     |
| 출력 형식       | 이미지 파일 형식        | png                    | **pdf** (보고서용)     |
| 해상도 (DPI)    | 이미지 품질             | 300                    | **600** (고품질)       |
| 이상값 표시     | 차트에 이상값 표시 여부 | N                      | N (깔끔한 시각화)      |
| 평균값 표시     | 평균선 표시 여부        | Y                      | Y (비교 용이)          |
| 평균값 숫자     | 평균값 텍스트 표시      | Y                      | Y (정확한 값 확인)     |
| 클리핑 백분위수 | 극단값 제거 비율        | 0.95                   | 0.95 (상위 5% 제거)    |
| 평균 위치 전략  | 평균값 텍스트 위치      | adaptive               | adaptive (자동 최적화) |

**출력 형식별 특징:**
- **PNG**: 일반 이미지, 웹/문서 첨부용, 파일 크기 작음
- **PDF**: 벡터 그래픽, 확대해도 선명, **보고서/출판용 권장**
- **SVG**: 벡터 그래픽, 웹 게시용, 추가 편집 가능

**플롯 타입별 용도:**
- **Boxplot**: 기본 분포 비교, 빠른 파악
- **Violin**: 분포 형태 상세 분석
- **Boxen**: 정밀한 분위수 분석
- **Forest Plot**: 통계적 유의성 검증

## 분석 결과

### 출력 파일 구조

분석 완료 후 다음 파일들이 생성됩니다:

```
output/
├── 시각화 파일
│   ├── city_boxplot_analysis.png
│   ├── city_violin_analysis.png
│   ├── city_boxen_analysis.png
│   ├── rural_boxplot_analysis.png
│   ├── rural_violin_analysis.png
│   └── rural_boxen_analysis.png
│
├── 데이터 파일
│   ├── walking_time_city.csv
│   ├── walking_time_rural.csv
│   ├── onboard_time_city.csv
│   ├── onboard_time_rural.csv
│   ├── waiting_time_city.csv
│   ├── waiting_time_rural.csv
│   ├── total_time_city.csv
│   └── total_time_rural.csv
│
└── 분석 결과
    └── analysis_results.csv
```

### 분석 결과 해석

**`analysis_results.csv`** 파일에는 다음 정보가 포함됩니다:
- 지역별 (도시/농어촌) 구분
- 카테고리별 (도보/탑승/대기/총 이동시간) 구분
- 대중교통 평균값
- 셔클 평균값
- 평균 개선량 (분)
- 평균 개선율 (%)
- 중앙값 개선량 (분)
- 중앙값 개선율 (%)

## 프로젝트 구조

```
shucle-effect-analysis/
├── src/                      # 핵심 모듈
│   ├── visualization.py      # 시각화 기능
│   ├── statistics_analyzer.py # 통계 분석
│   ├── data_processor.py     # 데이터 처리
│   └── config_manager.py     # 설정 관리
│
├── scripts/
│   └── main.py               # 메인 실행 파일
│
├── notebooks/
│   └── playground.ipynb      # Jupyter 분석 노트북
│
├── data/                     # 입력 데이터
├── output/                   # 분석 결과
└── requirements.txt          # 패키지 의존성
```

## 고급 기능

### Jupyter 노트북 활용

고급 통계 분석 및 상세한 결과 확인을 위해 Jupyter 노트북을 제공합니다:

```bash
jupyter notebook notebooks/playground.ipynb
```

**노트북 주요 기능:**
- Cohen's d 효과 크기 분석
- Forest Plot 시각화
- 99% 신뢰구간 추정
- t-test, Wilcoxon, Bootstrap 검정
- 도시/농어촌 지역 비교 분석

## 문의사항

기술적 문의사항이나 개선 제안은 프로젝트 관리자에게 문의하시기 바랍니다.

**개발자:** taeyang lee

---

**주의:** 본 프로젝트는 내부 데이터를 사용하며 외부 공개가 제한됩니다.