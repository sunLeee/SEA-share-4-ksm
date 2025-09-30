# 셔클 효과 분석 패키지

**셔클(수요응답형 교통서비스) 도입 효과 분석 도구**

## 개요

본 패키지는 셔클 도입 전후의 교통 데이터를 분석하여 정량적 효과를 측정하는 전문 분석 도구입니다. 통계적 유의성 검증, 효과 크기 계산, 고급 시각화 기능을 제공합니다.

## 주요 기능

### 통계 분석
- **가설 검정**: Paired t-test, Wilcoxon signed-rank test, Bootstrap test
- **효과 크기 측정**: Cohen's d 계산 및 해석
- **신뢰구간 추정**: 99% 신뢰구간 기반 분석
- **강건 통계**: 절사평균, 중앙절대편차(MAD) 활용

### 시각화
- **다중 플롯 타입**: Boxplot, Violin plot, Boxen plot
- **Forest Plot**: 효과 크기와 신뢰구간 시각화
- **고품질 출력**: PNG, PDF, SVG 형식 지원 (300-600 DPI)
- **세밀한 제어**: 이상값 표시, 평균값 위치, 데이터 클리핑 옵션

### 데이터 처리
- **자동 전처리**: 결측값 제거, 단위 변환 (초→분)
- **지역별 분석**: 도시/농어촌 지역 구분 분석
- **카테고리별 분석**: 도보시간, 탑승시간, 대기시간, 총 이동시간

## 설치 방법

### 필수 요구사항
- Python 3.8 이상
- 최소 4GB RAM 권장

### 패키지 설치
```bash
# 프로젝트 디렉토리로 이동
cd shucle-effect-analysis

# 의존성 설치
pip install -r requirements.txt
```

## 사용 방법

### 기본 분석 실행
```bash
# 기본 설정으로 실행
python scripts/main.py

# 대화형 모드로 실행 (권장)
python scripts/main.py --interactive
```

### 대화형 모드 사용법

대화형 모드에서는 모든 분석 옵션을 직접 설정할 수 있습니다. 각 항목에서 엔터를 누르면 기본값이 사용됩니다.

```bash
python scripts/main.py --interactive
```

**설정 항목:**

1. **데이터 파일 경로**: 분석할 CSV 파일 위치
2. **출력 디렉토리**: 결과 파일 저장 위치
3. **플롯 저장 여부**: 그래프 이미지 파일로 저장 (Y/N)
4. **플롯 표시 여부**: 그래프를 화면에 표시 (Y/N)

**시각화 세부 설정:**
- 플롯 타입: boxplot, violin, boxen 중 선택 (쉼표로 구분)
- 그래프 크기: 가로x세로 픽셀 (예: 15x8)
- 출력 형식: png, pdf, svg 중 선택
- 이미지 해상도: DPI 설정 (기본 300, 고품질 600)

**고급 시각화 옵션:**
- 이상값 표시 여부
- 평균값 선 표시 여부
- 평균값 숫자 표시 여부
- 데이터 클리핑 백분위수 (0.9-1.0)
- 평균값 위치 전략 (adaptive/fixed_percentage/above_max)

**예시:**
```
데이터 파일 경로 [./data/shucle_analysis_dataset_20250929.csv]: [엔터]
출력 디렉토리 [./output]: ./results
플롯 저장 (Y/N) [Y]: [엔터]
플롯 표시 (Y/N) [N]: [엔터]

📊 시각화 세부 설정:
플롯 타입 (쉼표로 구분) [boxplot, violin, boxen]: boxplot,violin
그래프 크기 (가로x세로) [15x8]: [엔터]
출력 형식 (png/pdf/svg) [png]: pdf
이미지 해상도 DPI [300]: 600

🎨 고급 시각화 옵션:
이상값 표시 (Y/N) [N]: [엔터]
평균값 표시 (Y/N) [Y]: [엔터]
평균값 숫자 표시 (Y/N) [Y]: [엔터]
데이터 클리핑 백분위수 (0.9-1.0) [0.95]: [엔터]
평균값 위치 전략 (adaptive/fixed_percentage/above_max) [adaptive]: [엔터]
```

## 커스터마이제이션 옵션

### 시각화 옵션
| 옵션 | 설명 | 기본값 | 선택 가능한 값 |
|------|------|--------|---------------|
| **플롯 타입** | 생성할 차트 종류 | `boxplot, violin, boxen` | `boxplot`, `violin`, `boxen` (조합 가능) |
| **그래프 크기** | 차트 크기 (가로x세로) | `15x8` | 예: `12x6`, `20x10` |
| **출력 형식** | 이미지 파일 형식 | `png` | `png`, `pdf`, `svg` |
| **해상도 (DPI)** | 이미지 품질 | `300` | `150-600` (논문용 권장: 600) |
| **이상값 표시** | 차트에 이상값 표시 여부 | `False` | `True`, `False` |
| **평균값 표시** | 평균선 표시 여부 | `True` | `True`, `False` |
| **평균값 숫자** | 평균값 텍스트 표시 | `True` | `True`, `False` |
| **클리핑 백분위수** | 데이터 범위 제한 | `0.95` | `0.90-1.00` |
| **평균 위치 전략** | 평균값 텍스트 위치 | `adaptive` | `adaptive`, `fixed_percentage`, `above_max` |

### 출력 파일 형식별 특징
- **PNG**: 웹용, 빠른 로딩, 압축 효율적
- **PDF**: 논문용, 벡터 그래픽, 확대해도 선명
- **SVG**: 웹용 벡터, 브라우저에서 편집 가능

### 플롯 타입별 특징
- **Boxplot**: 사분위수, 중앙값, 이상값을 명확하게 표시
- **Violin**: 데이터 분포의 확률밀도를 시각화
- **Boxen**: 다층 분위수로 분포의 세밀한 구조 표시

## 📊 분석 결과 예시

### 출력 파일 구조
분석 완료 후 다음 파일들이 생성됩니다:

```
output/
├── 📈 시각화 파일 (6개)
│   ├── city_boxplot_analysis.png    # 도시지역 박스플롯
│   ├── city_violin_analysis.png     # 도시지역 바이올린 플롯
│   ├── city_boxen_analysis.png      # 도시지역 Boxen 플롯
│   ├── rural_boxplot_analysis.png   # 농어촌지역 박스플롯
│   ├── rural_violin_analysis.png    # 농어촌지역 바이올린 플롯
│   └── rural_boxen_analysis.png     # 농어촌지역 Boxen 플롯
├── 📋 데이터 파일 (8개)
│   ├── walking_time_city.csv        # 도시지역 도보시간 데이터
│   ├── walking_time_rural.csv       # 농어촌지역 도보시간 데이터
│   ├── onboard_time_city.csv        # 도시지역 탑승시간 데이터
│   ├── onboard_time_rural.csv       # 농어촌지역 탑승시간 데이터
│   ├── waiting_time_city.csv        # 도시지역 대기시간 데이터
│   ├── waiting_time_rural.csv       # 농어촌지역 대기시간 데이터
│   ├── total_time_city.csv          # 도시지역 총 이동시간 데이터
│   └── total_time_rural.csv         # 농어촌지역 총 이동시간 데이터
└── 📊 분석 결과
    └── analysis_results.csv         # 통계 분석 요약 결과
```

### 통계적 유의성 검증
```
🏙️ 도시지역 셔클 도입 효과
==========================================
📦 [도보시간] 통계적 검증
  📊 기술 통계:
    • 대중교통 평균: 7.62분
    • 셔클 평균: 5.97분
    • 평균 차이: +1.65분

  🔍 통계적 유의성:
    • t-test p-value: 1.23e-45
    • Cohen's d: 0.402
    • 효과 크기: 중간 효과

  ✅ 통계적으로 유의미한 개선 효과 (p < 0.01)
```

### 지역별 비교
| 카테고리 | 도시지역 | 농어촌지역 | 차이 |
|----------|----------|-----------|------|
| 도보시간 | 21.7% | 39.2% | +17.6%p |
| 탑승시간 | 7.3% | 34.4% | +27.1%p |
| 대기시간 | 20.6% | 72.1% | +51.6%p |
| 총 이동시간 | 16.6% | 62.5% | +45.9%p |

## 🧪 테스트

### 단위 테스트 실행
```bash
# 모든 테스트 실행
pytest tests/

# 커버리지 포함
pytest --cov=src tests/

# 특정 모듈 테스트
pytest tests/test_visualization.py -v
```

### 성능 테스트
```bash
# 성능 벤치마크 실행
python tests/test_performance.py
```

## 📁 프로젝트 구조

```
shucle-effect-analysis/
├── src/                      # 핵심 모듈
│   ├── __init__.py          # 패키지 초기화
│   ├── visualization.py     # 시각화 모듈
│   ├── data_processor.py    # 데이터 처리
│   ├── statistics_analyzer.py # 통계 분석
│   └── config_manager.py    # 설정 관리
├── scripts/                 # 실행 스크립트
│   ├── main.py             # 메인 실행 파일
│   ├── README.md           # 스크립트 사용법
│   └── config_example.yaml # 설정 예시
├── tests/                   # 테스트 코드
│   ├── __init__.py
│   ├── test_*.py
│   └── conftest.py
├── notebooks/               # Jupyter 노트북
│   └── playground.ipynb    # 분석 예시
├── data/                    # 데이터 파일
├── output/                  # 출력 결과
├── config/                  # 설정 파일
├── requirements.txt         # 의존성
├── setup.py                # 패키지 설정
└── README.md               # 이 파일
```

## 🤝 기여 방법

1. Fork 생성
2. 기능 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경사항 커밋 (`git commit -m 'Add amazing feature'`)
4. 브랜치에 Push (`git push origin feature/amazing-feature`)
5. Pull Request 생성

### 개발 환경 설정
```bash
# 개발용 의존성 설치
pip install -e ".[dev]"

# 코드 포맷팅
black src/ scripts/ tests/

# 타입 체크
mypy src/

# 린팅
flake8 src/ scripts/ tests/
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 👨‍💻 개발자

**taeyang lee**
- Email: taeyang.lee@example.com
- GitHub: [@your-username](https://github.com/your-username)

## 🙏 감사의 말

- [Pandas](https://pandas.pydata.org/) - 데이터 분석 프레임워크
- [Matplotlib](https://matplotlib.org/) - 시각화 라이브러리
- [NumPy](https://numpy.org/) - 수치 계산 라이브러리
- [SciPy](https://scipy.org/) - 과학 계산 라이브러리

## 📈 버전 히스토리

### v2.1.0 (2025-09-29) - Interactive Mode Release
- 🎛️ **대화형 모드**: 모든 시각화 옵션 실시간 커스터마이즈
- 📊 **고급 시각화**: 3가지 플롯 타입 선택적 조합 (boxplot, violin, boxen)
- 🎨 **완전한 제어**: DPI, 크기, 형식, 이상값 표시 등 모든 옵션 제어
- 📁 **스마트 파일 관리**: 일관된 파일명 규칙 및 자동 디렉토리 생성
- ⚡ **기본값 지원**: 엔터만 눌러도 최적화된 설정으로 빠른 실행

### v2.0.0 (2025-09-29)
- 🚀 **성능 최적화**: 3-5배 속도 향상
- 🔬 **고급 통계**: 가설검정, 효과 크기 계산
- ⚙️ **설정 관리**: YAML 기반 설정 시스템
- 📦 **모듈화**: 재사용 가능한 구조적 설계

### v1.0.0 (Initial Release)
- 기본 시각화 기능
- 데이터 전처리
- 통계 분석

---

⭐ 이 프로젝트가 도움이 되었다면 스타를 눌러주세요!