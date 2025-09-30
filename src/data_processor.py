"""
셔클 효과 분석 데이터 전처리 모듈

데이터사이언스 분석을 위한 교통 데이터 전처리 및 정제 모듈
지역별, 교통수단별 시간 데이터 전처리와 통계 분석 기능 제공

Author: taeyang lee
Made Date: 2025-09-29 21:00
Modified Date: 2025-09-29 21:00

Logics:
    1. CSV 데이터 로드 및 인코딩 처리
    2. 지역별 데이터 필터링 및 그룹화
    3. 시간 데이터 변환 (초→분) 및 정제
    4. 이상값 탐지 및 제거 (IQR, Z-score 방법)
    5. 벡터화 연산을 통한 성능 최적화

Example:
    >>> processor = ShuttleDataProcessor()
    >>> data = processor.load_data("data.csv")
    >>> filtered = processor.filter_by_zone_type(data, "도시")
    >>> cleaned = processor.remove_outliers(filtered, ["time_col"], method="iqr")
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp


class ShuttleDataProcessor:
    """
    셔클 데이터 전처리 클래스

    교통 데이터 분석을 위한 전처리 작업을 통합적으로 처리하는 클래스
    대용량 데이터 처리를 위한 병렬 처리 및 벡터화 연산 지원

    Attributes:
        column_mapping (Dict[str, str]): 컬럼명 매핑 정보
        processed_data (Dict): 전처리된 데이터 저장소

    Logics:
        1. 데이터 로드, 필터링, 정제 작업 수행
        2. 지역별, 교통수단별 데이터 분리
        3. 이상값 제거 및 통계적 정제
        4. 비동기 처리를 통한 성능 최적화

    Example:
        >>> processor = ShuttleDataProcessor()
        >>> processor.set_column_mapping({
        ...     "old_name": "new_name"
        ... })
        >>> data = processor.load_data("dataset.csv")
        >>> stats = processor.create_summary_statistics(data, ["time_col"])
    """

    def __init__(self):
        """
        데이터 프로세서 초기화

        데이터 전처리를 위한 기본 설정을 초기화합니다.

        Logics:
            1. 컬럼명 매핑 딕셔너리 초기화
            2. 전처리된 데이터 저장소 초기화
            3. 기본 설정값 적용

        Example:
            >>> processor = ShuttleDataProcessor()
            >>> print(type(processor.column_mapping))
            <class 'dict'>
        """
        self.column_mapping = {}
        self.processed_data = {}

    def load_data(
        self,
        filepath: str,
        encoding: str = 'utf-8'
    ) -> pd.DataFrame:
        """
        CSV 데이터 로드

        Parameters
        ----------
        filepath : str
            파일 경로
        encoding : str
            인코딩 방식

        Returns
        -------
        pd.DataFrame
            로드된 데이터
        """
        return pd.read_csv(filepath, encoding=encoding)

    def set_column_mapping(
        self,
        mapping: Dict[str, str]
    ):
        """
        컬럼명 매핑 설정

        Parameters
        ----------
        mapping : dict
            원본 컬럼명 -> 변경할 컬럼명 매핑
        """
        self.column_mapping = mapping

    def filter_by_zone_type(
        self,
        df: pd.DataFrame,
        zone_type: str,
        zone_column: str = 'zone_type'
    ) -> pd.DataFrame:
        """
        지역 유형별 필터링

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        zone_type : str
            지역 유형 (예: '도시', '농어촌')
        zone_column : str
            지역 유형 컬럼명

        Returns
        -------
        pd.DataFrame
            필터링된 데이터
        """
        return df[df[zone_column] == zone_type].copy()

    def rename_columns(
        self,
        df: pd.DataFrame,
        mapping: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        컬럼명 변경

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        mapping : dict, optional
            컬럼명 매핑 (None인 경우 기본 매핑 사용)

        Returns
        -------
        pd.DataFrame
            컬럼명이 변경된 데이터
        """
        if mapping is None:
            mapping = self.column_mapping

        return df.rename(columns=mapping)

    def extract_time_columns(
        self,
        df: pd.DataFrame,
        time_categories: Dict[str, List[str]]
    ) -> Dict[str, pd.DataFrame]:
        """
        시간 카테고리별 컬럼 추출

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        time_categories : dict
            카테고리별 컬럼 리스트

        Returns
        -------
        dict
            카테고리별 데이터프레임
        """
        result = {}

        for category, columns in time_categories.items():
            valid_columns = [col for col in columns if col in df.columns]
            if valid_columns:
                result[category] = df[valid_columns].copy()

        return result

    def convert_seconds_to_minutes(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        초를 분으로 변환

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        columns : list, optional
            변환할 컬럼 리스트 (None인 경우 모든 숫자 컬럼)

        Returns
        -------
        pd.DataFrame
            변환된 데이터
        """
        df_copy = df.copy()

        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col] / 60

        return df_copy

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        이상치 제거

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        columns : list
            이상치 제거할 컬럼
        method : str
            제거 방법 ('iqr', 'zscore')
        threshold : float
            임계값

        Returns
        -------
        pd.DataFrame
            이상치가 제거된 데이터
        """
        df_copy = df.copy()
        mask = pd.Series([True] * len(df_copy))

        for col in columns:
            if col not in df_copy.columns:
                continue

            if method == 'iqr':
                Q1 = df_copy[col].quantile(0.25)
                Q3 = df_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask &= (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)

            elif method == 'zscore':
                z_scores = np.abs((df_copy[col] - df_copy[col].mean()) / df_copy[col].std())
                mask &= z_scores < threshold

        return df_copy[mask]

    def aggregate_by_zone(
        self,
        df: pd.DataFrame,
        zone_column: str,
        agg_functions: Dict[str, Union[str, List[str]]]
    ) -> pd.DataFrame:
        """
        지역별 집계

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        zone_column : str
            지역 컬럼명
        agg_functions : dict
            집계 함수 설정

        Returns
        -------
        pd.DataFrame
            집계된 데이터
        """
        return df.groupby(zone_column).agg(agg_functions).reset_index()

    def calculate_time_differences(
        self,
        df: pd.DataFrame,
        baseline_columns: List[str],
        comparison_columns: List[str],
        suffix: str = '_diff'
    ) -> pd.DataFrame:
        """
        시간 차이 계산

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        baseline_columns : list
            기준 컬럼 리스트
        comparison_columns : list
            비교 컬럼 리스트
        suffix : str
            차이 컬럼 접미사

        Returns
        -------
        pd.DataFrame
            차이가 계산된 데이터
        """
        df_copy = df.copy()

        for base_col, comp_col in zip(baseline_columns, comparison_columns):
            if base_col in df.columns and comp_col in df.columns:
                diff_col_name = f"{base_col}_{comp_col}{suffix}"
                df_copy[diff_col_name] = df[base_col] - df[comp_col]

        return df_copy

    def create_summary_statistics(
        self,
        df: pd.DataFrame,
        columns: List[str],
        percentiles: List[float] = [0.25, 0.5, 0.75]
    ) -> pd.DataFrame:
        """
        요약 통계 생성

        Parameters
        ----------
        df : pd.DataFrame
            원본 데이터
        columns : list
            분석할 컬럼
        percentiles : list
            계산할 백분위수

        Returns
        -------
        pd.DataFrame
            요약 통계
        """
        stats = {}

        for col in columns:
            if col not in df.columns:
                continue

            col_stats = {
                'count': df[col].count(),
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }

            for p in percentiles:
                col_stats[f'{int(p*100)}%'] = df[col].quantile(p)

            stats[col] = col_stats

        return pd.DataFrame(stats).T