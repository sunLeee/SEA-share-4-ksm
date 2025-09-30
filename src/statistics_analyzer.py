"""
셔클 효과 분석 통계 분석 모듈

데이터사이언스 분석을 위한 고급 통계 분석 및 효과 측정 모듈
교통수단별 성능 비교, 개선 효과 정량화, 통계적 유의성 검증 기능 제공

Author: taeyang lee
Made Date: 2025-09-29 21:00
Modified Date: 2025-09-29 21:00

Logics:
    1. 기술통계량 계산 및 분포 분석
    2. 교통수단간 성능 차이 통계적 검증 (t-test, Wilcoxon)
    3. 효과 크기 계산 (Cohen's d, 상관분석)
    4. 신뢰구간 및 부트스트랩 분석
    5. 지역별, 시간대별 세분화 분석
    6. 벡터화 연산을 통한 대용량 데이터 처리

Example:
    >>> analyzer = StatisticsAnalyzer()
    >>> stats = analyzer.calculate_comprehensive_statistics(data)
    >>> effect = analyzer.analyze_improvement_effect(baseline, treatment)
    >>> significance = analyzer.test_statistical_significance(data1, data2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from scipy import stats
from scipy.stats import bootstrap
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class StatisticalResult:
    """
    통계 분석 결과 데이터 클래스

    통계 분석의 모든 결과를 구조화된 형태로 저장합니다.

    Attributes:
        statistic: 통계량 값
        p_value: p-값
        confidence_interval: 신뢰구간 (하한, 상한)
        effect_size: 효과 크기
        interpretation: 결과 해석
        method: 사용된 통계 방법
    """
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str
    method: str


@dataclass
class ImprovementAnalysis:
    """
    개선 효과 분석 결과

    교통수단 개선 효과에 대한 종합적인 분석 결과를 저장합니다.

    Attributes:
        absolute_improvement: 절대적 개선량 (분)
        relative_improvement: 상대적 개선율 (%)
        statistical_significance: 통계적 유의성 결과
        effect_size: 효과 크기 (Cohen's d)
        confidence_interval: 개선 효과 신뢰구간
        sample_size: 표본 크기
    """
    absolute_improvement: Dict[str, float]
    relative_improvement: Dict[str, float]
    statistical_significance: StatisticalResult
    effect_size: float
    confidence_interval: Tuple[float, float]
    sample_size: Dict[str, int]


class StatisticsAnalyzer:
    """
    통계 분석기 클래스

    교통 데이터의 종합적인 통계 분석을 수행하는 클래스입니다.
    기술통계, 추론통계, 효과 분석을 통합적으로 제공합니다.

    Attributes:
        alpha: 유의수준 (기본값: 0.05)
        confidence_level: 신뢰수준 (기본값: 0.95)
        bootstrap_samples: 부트스트랩 샘플 수

    Logics:
        1. 다양한 통계적 방법을 통한 종합적 분석
        2. 벡터화 연산으로 대용량 데이터 처리
        3. 병렬 처리를 통한 성능 최적화
        4. 강건한 통계 방법 및 비모수 검정 지원
        5. 실무적 해석과 통계적 엄밀성의 균형

    Example:
        >>> analyzer = StatisticsAnalyzer(alpha=0.05)
        >>> stats = analyzer.calculate_comprehensive_statistics(
        ...     data, group_col="transportation_mode"
        ... )
        >>> print(f"평균 차이: {stats['mean_difference']:.2f}분")
    """

    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        """
        통계 분석기 초기화

        Args:
            alpha: 유의수준 (Type I error rate)
            confidence_level: 신뢰수준
        """
        self.alpha = alpha
        self.confidence_level = confidence_level
        self.bootstrap_samples = 10000

    def calculate_comprehensive_statistics(
        self,
        data: pd.DataFrame,
        value_columns: List[str],
        group_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        종합적인 기술통계량 계산

        데이터의 중심경향성, 분산, 분포 특성을 종합적으로 분석합니다.

        Args:
            data: 분석할 데이터프레임
            value_columns: 분석할 값 컬럼 리스트
            group_column: 그룹 구분 컬럼 (None이면 전체 분석)

        Returns:
            Dict[str, Any]: 종합 통계 결과

        Logics:
            1. 기본 기술통계량 계산 (평균, 표준편차, 분위수)
            2. 분포 특성 분석 (왜도, 첨도, 정규성 검정)
            3. 그룹별 비교 통계량 계산
            4. 이상값 탐지 및 강건 통계량 계산

        Example:
            >>> stats = analyzer.calculate_comprehensive_statistics(
            ...     df, ["travel_time"], "mode"
            ... )
            >>> print(stats["descriptive"]["mean"])
        """
        results = {}

        if group_column and group_column in data.columns:
            # 그룹별 분석
            grouped_stats = {}
            for group in data[group_column].unique():
                group_data = data[data[group_column] == group]
                grouped_stats[group] = self._calculate_single_group_stats(
                    group_data, value_columns
                )
            results["by_group"] = grouped_stats

            # 그룹간 비교
            results["group_comparisons"] = self._compare_groups(
                data, value_columns, group_column
            )
        else:
            # 전체 분석
            results["overall"] = self._calculate_single_group_stats(
                data, value_columns
            )

        return results

    def analyze_improvement_effect(
        self,
        baseline_data: pd.Series,
        treatment_data: pd.Series,
        baseline_name: str = "기준",
        treatment_name: str = "개선"
    ) -> ImprovementAnalysis:
        """
        개선 효과 분석

        기준 데이터와 개선 데이터 간의 효과를 종합적으로 분석합니다.

        Args:
            baseline_data: 기준 데이터 (예: 대중교통)
            treatment_data: 개선 데이터 (예: 셔클)
            baseline_name: 기준 데이터 이름
            treatment_name: 개선 데이터 이름

        Returns:
            ImprovementAnalysis: 개선 효과 분석 결과

        Logics:
            1. 절대적/상대적 개선량 계산
            2. 통계적 유의성 검정 (t-test, Wilcoxon)
            3. 효과 크기 계산 (Cohen's d)
            4. 부트스트랩 신뢰구간 추정
            5. 실무적 의미 있는 차이 판단

        Example:
            >>> improvement = analyzer.analyze_improvement_effect(
            ...     public_transport_time, shuttle_time
            ... )
            >>> print(f"개선 효과: {improvement.absolute_improvement['mean']:.1f}분")
        """
        # 결측값 제거
        baseline_clean = baseline_data.dropna()
        treatment_clean = treatment_data.dropna()

        if baseline_clean.empty or treatment_clean.empty:
            raise ValueError("유효한 데이터가 없습니다")

        # 절대적 개선량 계산
        absolute_improvement = {
            "mean": baseline_clean.mean() - treatment_clean.mean(),
            "median": baseline_clean.median() - treatment_clean.median(),
            "trimmed_mean": self._trimmed_mean(baseline_clean) -
                          self._trimmed_mean(treatment_clean)
        }

        # 상대적 개선율 계산
        relative_improvement = {
            "mean": (baseline_clean.mean() - treatment_clean.mean()) /
                   baseline_clean.mean() * 100,
            "median": (baseline_clean.median() - treatment_clean.median()) /
                     baseline_clean.median() * 100
        }

        # 통계적 유의성 검정
        significance_result = self.test_statistical_significance(
            baseline_clean, treatment_clean
        )

        # 효과 크기 계산 (Cohen's d)
        effect_size = self._calculate_cohens_d(baseline_clean, treatment_clean)

        # 부트스트랩 신뢰구간
        confidence_interval = self._bootstrap_confidence_interval(
            baseline_clean, treatment_clean
        )

        # 표본 크기
        sample_size = {
            baseline_name: len(baseline_clean),
            treatment_name: len(treatment_clean)
        }

        return ImprovementAnalysis(
            absolute_improvement=absolute_improvement,
            relative_improvement=relative_improvement,
            statistical_significance=significance_result,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            sample_size=sample_size
        )

    def test_statistical_significance(
        self,
        group1: pd.Series,
        group2: pd.Series,
        method: str = "auto"
    ) -> StatisticalResult:
        """
        통계적 유의성 검정

        두 그룹 간의 차이에 대한 통계적 유의성을 검정합니다.

        Args:
            group1: 첫 번째 그룹 데이터
            group2: 두 번째 그룹 데이터
            method: 검정 방법 ("auto", "ttest", "wilcoxon", "bootstrap")

        Returns:
            StatisticalResult: 통계적 검정 결과

        Logics:
            1. 정규성 검정을 통한 적절한 방법 선택
            2. 등분산성 검정 (Levene test)
            3. 적절한 t-test 또는 비모수 검정 수행
            4. 효과 크기 및 신뢰구간 계산

        Example:
            >>> result = analyzer.test_statistical_significance(
            ...     public_data, shuttle_data
            ... )
            >>> print(f"p-value: {result.p_value:.4f}")
        """
        # 결측값 제거
        g1_clean = group1.dropna()
        g2_clean = group2.dropna()

        if len(g1_clean) < 3 or len(g2_clean) < 3:
            raise ValueError("통계 검정을 위한 충분한 데이터가 없습니다")

        # 방법 자동 선택
        if method == "auto":
            method = self._select_test_method(g1_clean, g2_clean)

        if method == "ttest":
            # Welch's t-test (등분산 가정하지 않음)
            statistic, p_value = stats.ttest_ind(
                g1_clean, g2_clean, equal_var=False
            )

            # 신뢰구간 계산
            diff_mean = g1_clean.mean() - g2_clean.mean()
            se_diff = np.sqrt(
                g1_clean.var(ddof=1)/len(g1_clean) +
                g2_clean.var(ddof=1)/len(g2_clean)
            )
            df = len(g1_clean) + len(g2_clean) - 2
            t_critical = stats.t.ppf(1 - self.alpha/2, df)
            ci = (
                diff_mean - t_critical * se_diff,
                diff_mean + t_critical * se_diff
            )

            effect_size = self._calculate_cohens_d(g1_clean, g2_clean)
            test_method = "Welch's t-test"

        elif method == "wilcoxon":
            # Mann-Whitney U 검정
            statistic, p_value = stats.mannwhitneyu(
                g1_clean, g2_clean, alternative='two-sided'
            )

            # 중위수 차이 신뢰구간 (Hodges-Lehmann 추정량)
            ci = self._hodges_lehmann_ci(g1_clean, g2_clean)
            effect_size = self._calculate_rank_biserial_correlation(
                g1_clean, g2_clean
            )
            test_method = "Mann-Whitney U test"

        elif method == "bootstrap":
            # 부트스트랩 검정
            statistic, p_value, ci = self._bootstrap_test(g1_clean, g2_clean)
            effect_size = self._calculate_cohens_d(g1_clean, g2_clean)
            test_method = "Bootstrap test"

        else:
            raise ValueError(f"지원하지 않는 검정 방법: {method}")

        # 결과 해석
        interpretation = self._interpret_test_result(
            p_value, effect_size, method
        )

        return StatisticalResult(
            statistic=statistic,
            p_value=p_value,
            confidence_interval=ci,
            effect_size=effect_size,
            interpretation=interpretation,
            method=test_method
        )

    def calculate_correlations(
        self,
        data: pd.DataFrame,
        columns: List[str],
        method: str = "pearson"
    ) -> Dict[str, Any]:
        """
        상관관계 분석

        지정된 컬럼들 간의 상관관계를 분석합니다.

        Args:
            data: 분석할 데이터프레임
            columns: 상관관계를 분석할 컬럼 리스트
            method: 상관계수 방법 ("pearson", "spearman", "kendall")

        Returns:
            Dict[str, Any]: 상관관계 분석 결과

        Example:
            >>> corr_results = analyzer.calculate_correlations(
            ...     df, ["wait_time", "travel_time"], method="spearman"
            ... )
        """
        # 수치형 데이터만 선택
        numeric_data = data[columns].select_dtypes(include=[np.number])

        if numeric_data.empty:
            raise ValueError("상관관계 분석할 수치형 데이터가 없습니다")

        # 상관계수 행렬 계산
        if method == "pearson":
            corr_matrix = numeric_data.corr(method='pearson')
        elif method == "spearman":
            corr_matrix = numeric_data.corr(method='spearman')
        elif method == "kendall":
            corr_matrix = numeric_data.corr(method='kendall')
        else:
            raise ValueError(f"지원하지 않는 상관계수 방법: {method}")

        # p-값 계산
        p_values = self._calculate_correlation_pvalues(
            numeric_data, method
        )

        # 강한 상관관계 탐지
        strong_correlations = self._find_strong_correlations(
            corr_matrix, threshold=0.7
        )

        return {
            "correlation_matrix": corr_matrix,
            "p_values": p_values,
            "strong_correlations": strong_correlations,
            "method": method,
            "sample_size": len(numeric_data)
        }

    def _calculate_single_group_stats(
        self,
        data: pd.DataFrame,
        value_columns: List[str]
    ) -> Dict[str, Any]:
        """단일 그룹 통계량 계산"""
        results = {}

        for col in value_columns:
            if col not in data.columns:
                continue

            col_data = data[col].dropna()
            if col_data.empty:
                continue

            # 기본 통계량
            basic_stats = {
                "count": len(col_data),
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "var": col_data.var(),
                "min": col_data.min(),
                "max": col_data.max(),
                "range": col_data.max() - col_data.min(),
                "iqr": col_data.quantile(0.75) - col_data.quantile(0.25)
            }

            # 분위수
            percentiles = {
                f"p{int(p*100)}": col_data.quantile(p)
                for p in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
            }

            # 분포 특성
            distribution_stats = {
                "skewness": stats.skew(col_data),
                "kurtosis": stats.kurtosis(col_data),
                "normality_test": stats.normaltest(col_data),
                "trimmed_mean": self._trimmed_mean(col_data, 0.1)
            }

            # 강건 통계량
            robust_stats = {
                "mad": stats.median_abs_deviation(col_data),  # Median Absolute Deviation
                "trimmed_std": self._trimmed_std(col_data, 0.1)
            }

            results[col] = {
                "basic": basic_stats,
                "percentiles": percentiles,
                "distribution": distribution_stats,
                "robust": robust_stats
            }

        return results

    def _compare_groups(
        self,
        data: pd.DataFrame,
        value_columns: List[str],
        group_column: str
    ) -> Dict[str, Any]:
        """그룹간 비교 분석"""
        results = {}
        groups = data[group_column].unique()

        if len(groups) < 2:
            return {"warning": "비교할 그룹이 충분하지 않습니다"}

        for col in value_columns:
            if col not in data.columns:
                continue

            col_results = {}

            # 두 그룹 비교 (첫 번째와 두 번째)
            if len(groups) >= 2:
                group1_data = data[data[group_column] == groups[0]][col].dropna()
                group2_data = data[data[group_column] == groups[1]][col].dropna()

                if not group1_data.empty and not group2_data.empty:
                    # 평균 차이
                    mean_diff = group1_data.mean() - group2_data.mean()

                    # 통계적 검정
                    try:
                        significance = self.test_statistical_significance(
                            group1_data, group2_data
                        )
                        col_results["significance_test"] = significance
                    except Exception as e:
                        col_results["significance_test"] = {"error": str(e)}

                    col_results["mean_difference"] = mean_diff
                    col_results["groups_compared"] = [groups[0], groups[1]]

            # 분산의 동질성 검정 (Levene test)
            if len(groups) >= 2:
                group_data = [
                    data[data[group_column] == group][col].dropna()
                    for group in groups
                ]
                group_data = [g for g in group_data if len(g) > 0]

                if len(group_data) >= 2:
                    try:
                        levene_stat, levene_p = stats.levene(*group_data)
                        col_results["levene_test"] = {
                            "statistic": levene_stat,
                            "p_value": levene_p
                        }
                    except Exception:
                        pass

            results[col] = col_results

        return results

    def _select_test_method(
        self,
        group1: pd.Series,
        group2: pd.Series
    ) -> str:
        """적절한 통계 검정 방법 선택"""
        # 표본 크기 확인
        if len(group1) < 30 or len(group2) < 30:
            # 정규성 검정
            _, p1 = stats.normaltest(group1)
            _, p2 = stats.normaltest(group2)

            if p1 > 0.05 and p2 > 0.05:
                return "ttest"
            else:
                return "wilcoxon"
        else:
            # 대표본: 중심극한정리에 의해 t-test 사용 가능
            return "ttest"

    def _calculate_cohens_d(
        self,
        group1: pd.Series,
        group2: pd.Series
    ) -> float:
        """Cohen's d 효과 크기 계산"""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(
            ((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) /
            (n1 + n2 - 2)
        )

        if pooled_std == 0:
            return 0.0

        return (group1.mean() - group2.mean()) / pooled_std

    def _calculate_rank_biserial_correlation(
        self,
        group1: pd.Series,
        group2: pd.Series
    ) -> float:
        """순위 이계열 상관계수 계산 (비모수 효과 크기)"""
        n1, n2 = len(group1), len(group2)
        U, _ = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return 1 - (2 * U) / (n1 * n2)

    def _bootstrap_confidence_interval(
        self,
        group1: pd.Series,
        group2: pd.Series,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float]:
        """부트스트랩 신뢰구간 계산"""
        def mean_difference(x, y):
            return np.mean(x) - np.mean(y)

        # 부트스트랩 샘플링
        differences = []
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=len(group1), replace=True)
            sample2 = np.random.choice(group2, size=len(group2), replace=True)
            differences.append(mean_difference(sample1, sample2))

        # 신뢰구간 계산
        alpha = 1 - self.confidence_level
        lower = np.percentile(differences, (alpha/2) * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)

        return (lower, upper)

    def _bootstrap_test(
        self,
        group1: pd.Series,
        group2: pd.Series,
        n_bootstrap: int = 10000
    ) -> Tuple[float, float, Tuple[float, float]]:
        """부트스트랩 검정"""
        observed_diff = group1.mean() - group2.mean()

        # 귀무가설 하에서 샘플링 (두 그룹 합쳐서)
        combined = pd.concat([group1, group2])
        n1, n2 = len(group1), len(group2)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(
                combined, size=len(combined), replace=True
            )
            sample1 = bootstrap_sample[:n1]
            sample2 = bootstrap_sample[n1:n1+n2]
            bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))

        # p-값 계산 (양측 검정)
        p_value = 2 * min(
            np.mean(np.array(bootstrap_diffs) >= observed_diff),
            np.mean(np.array(bootstrap_diffs) <= observed_diff)
        )

        # 신뢰구간
        ci = self._bootstrap_confidence_interval(group1, group2)

        return observed_diff, p_value, ci

    def _hodges_lehmann_ci(
        self,
        group1: pd.Series,
        group2: pd.Series
    ) -> Tuple[float, float]:
        """Hodges-Lehmann 추정량 신뢰구간"""
        # 모든 pairwise 차이 계산
        differences = []
        for x in group1:
            for y in group2:
                differences.append(x - y)

        differences = np.array(differences)
        alpha = 1 - self.confidence_level

        lower = np.percentile(differences, (alpha/2) * 100)
        upper = np.percentile(differences, (1 - alpha/2) * 100)

        return (lower, upper)

    def _interpret_test_result(
        self,
        p_value: float,
        effect_size: float,
        method: str
    ) -> str:
        """통계 검정 결과 해석"""
        # 통계적 유의성
        if p_value < 0.001:
            significance = "매우 강한 유의성"
        elif p_value < 0.01:
            significance = "강한 유의성"
        elif p_value < 0.05:
            significance = "유의함"
        else:
            significance = "유의하지 않음"

        # 효과 크기 해석 (Cohen's d 기준)
        if abs(effect_size) < 0.2:
            effect_interpretation = "작은 효과"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "중간 효과"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "큰 효과"
        else:
            effect_interpretation = "매우 큰 효과"

        return f"{significance} (p={p_value:.4f}), {effect_interpretation} (d={effect_size:.3f})"

    def _trimmed_mean(self, data: pd.Series, trim_ratio: float = 0.1) -> float:
        """절사평균 계산"""
        return stats.trim_mean(data, trim_ratio)

    def _trimmed_std(self, data: pd.Series, trim_ratio: float = 0.1) -> float:
        """절사표준편차 계산"""
        n = len(data)
        trim_count = int(n * trim_ratio / 2)
        sorted_data = np.sort(data)
        trimmed_data = sorted_data[trim_count:n-trim_count]
        return np.std(trimmed_data, ddof=1)

    def _calculate_correlation_pvalues(
        self,
        data: pd.DataFrame,
        method: str
    ) -> pd.DataFrame:
        """상관계수 p-값 계산"""
        n_vars = len(data.columns)
        p_values = np.zeros((n_vars, n_vars))

        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    col1, col2 = data.columns[i], data.columns[j]
                    clean_data = data[[col1, col2]].dropna()

                    if len(clean_data) > 3:
                        if method == "pearson":
                            _, p_val = stats.pearsonr(
                                clean_data[col1], clean_data[col2]
                            )
                        elif method == "spearman":
                            _, p_val = stats.spearmanr(
                                clean_data[col1], clean_data[col2]
                            )
                        else:  # kendall
                            _, p_val = stats.kendalltau(
                                clean_data[col1], clean_data[col2]
                            )
                        p_values[i, j] = p_val
                else:
                    p_values[i, j] = 0.0

        return pd.DataFrame(p_values, index=data.columns, columns=data.columns)

    def _find_strong_correlations(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[Tuple[str, str, float]]:
        """강한 상관관계 탐지"""
        strong_correlations = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_value
                    ))

        return strong_correlations