"""
셔클 효과 분석 시각화 모듈

데이터사이언스 분석을 위한 그룹화된 통계 차트 생성 모듈
교통수단별 시간 비교 분석을 위한 박스플롯, 바이올린플롯, Boxen플롯 제공

Author: taeyang lee
Made Date: 2025-09-29 21:00
Modified Date: 2025-09-29 21:00

Logics:
    1. 교통수단별 시간 데이터 전처리 및 그룹화
    2. 극단값 제거 및 통계적 클리핑 적용
    3. 평균값 위치 전략 (adaptive/fixed_top/fixed_percentage) 구현
    4. 사용자 정의 색상 및 축 범위 설정 지원
    5. 벡터화 연산을 통한 성능 최적화

Example:
    >>> data_dict = prepare_data(df, time_columns, convert_to_minutes=True)
    >>> fig, ax = create_grouped_boxplot(data_dict, title="분석 결과")
    >>> plt.show()
"""

import numpy as np
import pandas as pd

# matplotlib 백엔드 설정 (Jupyter 호환)
import matplotlib
import os

# Jupyter 환경 감지
try:
    get_ipython  # Jupyter/IPython 환경에서만 존재하는 함수
    # Jupyter 환경에서는 백엔드를 자동으로 설정하도록 둠
    import matplotlib.pyplot as plt
except NameError:
    # 일반 Python 스크립트 환경
    if os.getenv('DISPLAY') is None or os.getenv('MPLBACKEND') == 'Agg':
        matplotlib.use('Agg')  # 비-GUI 백엔드 사용
    else:
        # GUI 환경에서는 기본 백엔드 사용
        try:
            matplotlib.use('MacOSX')  # macOS 기본 백엔드
        except:
            matplotlib.use('Agg')  # 실패시 Agg 사용
    import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 글로벌 설정
DEFAULT_COLORS = {
    "대중교통": "#B0B0B0",
    "셔클": "#1F4EAA"
}

DEFAULT_MEAN_TEXT_COLORS = {
    "대중교통": "red",
    "셔클": "#1F4EAA"
}

DEFAULT_FONT_SIZES = {
    "title": 14,
    "label": 12,
    "tick": 11,
    "legend": 10
}

# 한글 폰트 설정
# 한글 폰트 설정 (안전한 처리)
try:
    plt.rcParams['font.family'] = 'AppleGothic'
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 폰트 설정 실패 시 기본 폰트 사용
    plt.rcParams['axes.unicode_minus'] = False


def prepare_data(
    df: pd.DataFrame,
    time_columns: Dict[str, List[str]],
    convert_to_minutes: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    데이터 전처리

    교통수단별 시간 데이터를 분석 가능한 형태로 전처리합니다.
    벡터화 연산을 통해 성능을 최적화하고 결측값을 제거합니다.

    Args:
        df: 원본 데이터프레임
        time_columns: 분석할 시간 컬럼 매핑 {카테고리: [컬럼리스트]}
        convert_to_minutes: 초를 분으로 변환 여부

    Returns:
        Dict[str, pd.DataFrame]: 전처리된 카테고리별 데이터

    Raises:
        ValueError: 유효한 컬럼이 없는 경우
        KeyError: 필수 컬럼이 누락된 경우

    Logics:
        1. 카테고리별 컬럼 유효성 검사
        2. 결측값 제거 및 벡터화 연산 적용
        3. 초를 분으로 변환 (convert_to_minutes=True시)
        4. 교통수단별 데이터 분리 및 DataFrame 생성

    Example:
        >>> time_cols = {
        ...     "총 이동시간": ["public_total_time", "drt_total_time"]
        ... }
        >>> processed = prepare_data(df, time_cols, True)
        >>> print(processed["총 이동시간"].columns)
        Index(['대중교통', '셔클'], dtype='object')
    """
    if df.empty:
        raise ValueError("입력 데이터프레임이 비어있습니다")

    processed_data = {}

    for category, columns in time_columns.items():
        category_data = {}
        valid_columns = [col for col in columns if col in df.columns]

        if not valid_columns:
            print(f"⚠️ '{category}' 카테고리에 유효한 컬럼이 없습니다: {columns}")
            continue

        # 벡터화 연산을 위한 데이터 추출
        for col in valid_columns:
            # 결측값 제거
            data = df[col].dropna()

            if data.empty:
                print(f"⚠️ '{col}' 컬럼에 유효한 데이터가 없습니다")
                continue

            # 벡터화 연산으로 초→분 변환
            if convert_to_minutes:
                data = data / 60

            # 교통수단 분류 (벡터화 접근)
            if "public" in col.lower() or "대중교통" in col:
                mode = "대중교통"
            elif "drt" in col.lower() or "셔클" in col:
                mode = "셔클"
            else:
                mode = col

            category_data[mode] = data

        # 유효한 데이터가 있는 경우에만 추가
        if category_data:
            processed_data[category] = pd.DataFrame(category_data)
        else:
            print(f"⚠️ '{category}' 카테고리에 처리할 데이터가 없습니다")

    if not processed_data:
        raise ValueError("처리된 데이터가 없습니다. 컬럼명과 데이터를 확인해주세요")

    return processed_data


def _prepare_grouped_data(
    data_dict: Dict[str, pd.DataFrame],
    clip_percentile: float = 1.0,
    spacing: float = 0.8,
    group_spacing: float = 2.0
) -> Tuple[List, List[float], List[str], List[str], List[float]]:
    """
    그룹화된 데이터 공통 준비 로직

    시각화를 위한 데이터 배치 및 색상 매핑을 준비합니다.
    극단값 제거와 그룹 간격 조정을 통해 시각적 품질을 향상시킵니다.

    Args:
        data_dict: 카테고리별 데이터프레임 딕셔너리
        clip_percentile: 극단값 제거 기준 (0.0-1.0)
        spacing: 같은 그룹 내 요소 간격
        group_spacing: 그룹 간 간격

    Returns:
        Tuple containing:
            - all_data: 클리핑된 데이터 리스트
            - all_positions: X축 위치 리스트
            - all_colors: 색상 리스트
            - group_labels: 그룹 라벨 리스트
            - group_positions: 그룹 중심 위치 리스트

    Logics:
        1. 데이터 순회하며 극단값 클리핑 적용
        2. 교통수단별 색상 매핑 (대중교통: 회색, 셔클: 파랑)
        3. X축 위치 계산 및 그룹별 중심점 계산
        4. 벡터화 연산을 통한 성능 최적화

    Example:
        >>> data, pos, colors, labels, group_pos = _prepare_grouped_data(
        ...     data_dict, clip_percentile=0.95
        ... )
    """
    all_data = []
    all_positions = []
    all_colors = []
    group_labels = []
    group_positions = []

    current_pos = 0

    for group_name, df in data_dict.items():
        if df.empty:
            print(f"⚠️ '{group_name}' 그룹에 데이터가 없습니다")
            continue

        group_start = current_pos
        group_has_data = False

        for col in df.columns:
            # 벡터화 연산으로 극단값 제거
            clipped_data = df[col].copy()
            if clip_percentile < 1.0:
                upper_limit = clipped_data.quantile(clip_percentile)
                clipped_data = clipped_data.clip(upper=upper_limit)

            # 결측값 제거
            clean_data = clipped_data.dropna()
            if clean_data.empty:
                print(f"⚠️ '{group_name}-{col}' 데이터가 비어있습니다")
                continue

            all_data.append(clean_data)
            all_positions.append(current_pos)

            # 효율적인 색상 매핑
            color_found = False
            for mode, color in DEFAULT_COLORS.items():
                if mode in col:
                    all_colors.append(color)
                    color_found = True
                    break

            if not color_found:
                all_colors.append('#808080')  # 기본 회색

            current_pos += spacing
            group_has_data = True

        # 그룹에 유효한 데이터가 있는 경우에만 추가
        if group_has_data:
            group_center = (group_start + current_pos - spacing) / 2
            group_positions.append(group_center)
            group_labels.append(group_name)
            current_pos += group_spacing - spacing
        else:
            # 데이터가 없는 그룹은 위치 조정 취소
            current_pos = group_start

    if not all_data:
        raise ValueError("시각화할 유효한 데이터가 없습니다")

    return all_data, all_positions, all_colors, group_labels, group_positions


def _calculate_mean_position(
    all_data: List[pd.Series],
    data: pd.Series,
    pos: float,
    mean_margin: float = 1.0,
    use_center_positioning: bool = True
) -> float:
    """
    평균값 텍스트 위치 계산

    데이터 분포에 따라 적응적으로 평균값 텍스트 위치를 결정합니다.
    데이터가 넓게 분포된 경우 중앙 배치, 그렇지 않으면 상단 배치를 적용합니다.

    Args:
        all_data: 전체 데이터 시리즈 리스트
        data: 현재 분석 중인 데이터 시리즈
        pos: X축 위치
        mean_margin: 평균값 텍스트 여백
        use_center_positioning: 중앙 배치 사용 여부

    Returns:
        float: 계산된 Y축 위치

    Logics:
        1. 전체 데이터 분포 분석 (IQR vs 95%-75% 범위)
        2. 분포가 넓은 경우 화면 60% 높이에 배치
        3. 분포가 좁은 경우 whisker 위에 배치
        4. 통계적 기법으로 이상값 영향 최소화

    Example:
        >>> y_pos = _calculate_mean_position(
        ...     all_data, current_data, x_pos, margin=1.5
        ... )
    """
    if use_center_positioning:
        # 전체 데이터가 흩어져 있는지 확인
        has_spread_data = False
        for d in all_data:
            q75 = d.quantile(0.75)
            q95 = d.quantile(0.95)
            iqr = d.quantile(0.75) - d.quantile(0.25)
            data_range = q95 - q75
            if data_range > iqr * 3:
                has_spread_data = True
                break

        if has_spread_data:
            # 전체 데이터의 y 범위 계산
            all_maxes = [d.quantile(0.9) for d in all_data]
            y_max = max(all_maxes)
            return y_max * 0.6  # 화면의 60% 높이에 배치

    # 기본 로직: whisker 또는 95% 분위수 위에 배치
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    upper_whisker = min(data.max(), q3 + 1.5 * iqr)
    return upper_whisker + mean_margin


def _add_mean_values(
    ax: plt.Axes,
    all_data: List[pd.Series],
    all_positions: List[float],
    all_colors: List[str],
    data_dict: Dict[str, pd.DataFrame],
    mean_format: str = "{:.0f}분",
    mean_margin: float = 1.0,
    mean_text_color: Optional[Dict[str, str]] = None,
    use_center_positioning: bool = True,
    mean_position_strategy: str = "adaptive",
    fixed_y_percentage: float = 0.8,
    y_axis_limits: Optional[Tuple[float, float]] = None
) -> None:
    """
    평균값 텍스트 추가

    다양한 위치 전략을 통해 평균값을 시각화에 표시합니다.
    사용자 정의 색상과 형식을 지원하여 가독성을 향상시킵니다.

    Args:
        ax: matplotlib Axes 객체
        all_data: 전체 데이터 시리즈 리스트
        all_positions: X축 위치 리스트
        all_colors: 색상 리스트
        data_dict: 카테고리별 데이터프레임 딕셔너리
        mean_format: 평균값 텍스트 형식
        mean_margin: 텍스트 여백
        mean_text_color: 교통수단별 텍스트 색상 매핑
        use_center_positioning: 중앙 배치 사용 여부
        mean_position_strategy: 위치 전략 (adaptive/fixed_top/fixed_percentage)
        fixed_y_percentage: fixed_percentage 전략 사용 시 비율
        y_axis_limits: Y축 범위 (min, max)

    Logics:
        1. Y축 범위 설정 (사용자 지정 또는 자동)
        2. 위치 전략에 따른 Y위치 계산
        3. 교통수단별 색상 매핑 및 적용
        4. 평균값 계산 및 텍스트 추가

    Example:
        >>> _add_mean_values(
        ...     ax, data_list, positions, colors, data_dict,
        ...     mean_format="{:.1f}분", mean_position_strategy="fixed_top"
        ... )
    """
    # y축 범위 설정
    if y_axis_limits:
        y_min, y_max = y_axis_limits
        ax.set_ylim(y_min, y_max)
    else:
        y_min, y_max = ax.get_ylim()

    for i, (data, pos) in enumerate(zip(all_data, all_positions)):
        # 새로운 위치 전략에 따른 y 위치 계산
        if mean_position_strategy == "fixed_top":
            # 상단 고정 위치 (90% 높이)
            y_position = y_max * 0.9
        elif mean_position_strategy == "fixed_percentage":
            # 사용자 지정 퍼센트 위치
            y_position = y_max * fixed_y_percentage
        else:  # adaptive (기존 로직)
            y_position = _calculate_mean_position(
                all_data, data, pos, mean_margin, use_center_positioning
            )

        mean_val = data.mean()

        # 색상 결정 (기본값을 DEFAULT_MEAN_TEXT_COLORS로 설정)
        if mean_text_color:
            text_color = mean_text_color
        else:
            text_color = DEFAULT_MEAN_TEXT_COLORS

        # 실제 색상 찾기
        final_color = all_colors[i]  # 기본값
        for mode, custom_color in text_color.items():
            group_name = list(data_dict.keys())[i // 2]
            col_name = list(data_dict[group_name].columns)[i % 2]
            if mode in col_name:
                final_color = custom_color
                break

        ax.text(pos, y_position, mean_format.format(mean_val),
               horizontalalignment='center',
               verticalalignment='bottom',
               fontsize=DEFAULT_FONT_SIZES['tick'],
               color=final_color,
               fontweight='bold')



def create_grouped_boxplot(
    data_dict: Dict[str, pd.DataFrame],
    title: str = "셔클 도입 효과 분석",
    ylabel: str = "시간(분)",
    figsize: Tuple[int, int] = (14, 8),
    show_outliers: bool = False,
    show_mean: bool = False,
    show_mean_value: bool = False,
    mean_format: str = "{:.0f}분",
    mean_margin: float = 1.0,
    clip_percentile: float = 1.0,
    mean_text_color: Optional[Dict[str, str]] = None,
    mean_position_strategy: str = "fixed_top",
    fixed_y_percentage: float = 0.8,
    y_axis_limits: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    그룹화된 박스플롯 생성

    교통수단별 시간 데이터를 박스플롯으로 시각화합니다.
    분위수, 중앙값, 이상값을 표시하며 평균값 텍스트를 추가할 수 있습니다.

    Args:
        data_dict: 카테고리별 데이터프레임 딕셔너리
        title: 그래프 제목
        ylabel: Y축 라벨
        figsize: 그래프 크기 (width, height)
        show_outliers: 이상값 표시 여부
        show_mean: 평균값 마커 표시 여부
        show_mean_value: 평균값 텍스트 표시 여부
        mean_format: 평균값 텍스트 형식
        mean_margin: 평균값 텍스트 여백
        clip_percentile: 극단값 제거 기준
        mean_text_color: 교통수단별 텍스트 색상
        mean_position_strategy: 평균값 위치 전략
        fixed_y_percentage: 고정 비율 위치 전략 사용 시 비율
        y_axis_limits: Y축 범위 (min, max)

    Returns:
        Tuple[plt.Figure, plt.Axes]: 생성된 그래프 객체

    Logics:
        1. 데이터 전처리 및 그룹화
        2. 박스플롯 생성 및 스타일 적용
        3. 교통수단별 색상 매핑
        4. 평균값 텍스트 추가 (옵션)
        5. 축 설정, 그리드, 범례 추가

    Example:
        >>> fig, ax = create_grouped_boxplot(
        ...     data_dict,
        ...     title="교통수단별 이동시간 비교",
        ...     show_mean_value=True,
        ...     mean_position_strategy="fixed_percentage",
        ...     fixed_y_percentage=0.85
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 데이터 준비
    all_data, all_positions, all_colors, group_labels, group_positions = _prepare_grouped_data(
        data_dict, clip_percentile
    )

    # 박스플롯 생성
    bp = ax.boxplot(
        all_data,
        positions=all_positions,
        widths=0.6,
        patch_artist=True,
        showfliers=show_outliers,
        showmeans=show_mean,
        meanprops=dict(marker='D', markerfacecolor='red', markeredgecolor='red', markersize=6),
        medianprops=dict(color='black', linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    # 색상 적용
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 평균값 텍스트 표시
    if show_mean_value:
        _add_mean_values(
            ax, all_data, all_positions, all_colors, data_dict,
            mean_format, mean_margin, mean_text_color, use_center_positioning=False,
            mean_position_strategy=mean_position_strategy,
            fixed_y_percentage=fixed_y_percentage,
            y_axis_limits=y_axis_limits
        )

    # 축 설정
    ax.set_ylabel(ylabel, fontsize=DEFAULT_FONT_SIZES['label'])
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZES['title'], fontweight='bold')
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, fontsize=DEFAULT_FONT_SIZES['tick'])

    # 그리드
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 범례
    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=mode)
        for mode, color in DEFAULT_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=DEFAULT_FONT_SIZES['legend'])

    plt.tight_layout()
    return fig, ax


def create_grouped_violin_plot(
    data_dict: Dict[str, pd.DataFrame],
    title: str = "셔클 도입 효과 분석 (바이올린 플롯)",
    ylabel: str = "시간(분)",
    figsize: Tuple[int, int] = (14, 8),
    show_outliers: bool = False,
    show_mean: bool = False,
    show_mean_value: bool = False,
    mean_format: str = "{:.0f}분",
    mean_margin: float = 1.0,
    clip_percentile: float = 1.0,
    mean_text_color: Optional[Dict[str, str]] = None,
    show_box: bool = False,
    mean_position_strategy: str = "fixed_top",
    fixed_y_percentage: float = 0.8,
    y_axis_limits: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    그룹화된 바이올린 플롯 생성

    교통수단별 시간 데이터의 확률밀도분포를 바이올린 형태로 시각화합니다.
    데이터 분포의 전체적인 모양과 밀도를 직관적으로 파악할 수 있습니다.

    Args:
        data_dict: 카테고리별 데이터프레임 딕셔너리
        title: 그래프 제목
        ylabel: Y축 라벨
        figsize: 그래프 크기 (width, height)
        show_outliers: 이상값 표시 여부
        show_mean: 평균값 마커 표시 여부
        show_mean_value: 평균값 텍스트 표시 여부
        mean_format: 평균값 텍스트 형식
        mean_margin: 평균값 텍스트 여백
        clip_percentile: 극단값 제거 기준
        mean_text_color: 교통수단별 텍스트 색상
        show_box: 내부 박스플롯 표시 여부
        mean_position_strategy: 평균값 위치 전략
        fixed_y_percentage: 고정 비율 위치 전략 사용 시 비율
        y_axis_limits: Y축 범위 (min, max)

    Returns:
        Tuple[plt.Figure, plt.Axes]: 생성된 그래프 객체

    Logics:
        1. 데이터 전처리 및 확률밀도 계산
        2. 바이올린 플롯 생성 및 색상 적용
        3. 옵션에 따른 내부 박스플롯 추가
        4. 적응적 Y축 범위 조정
        5. 평균값 텍스트 및 범례 추가

    Example:
        >>> fig, ax = create_grouped_violin_plot(
        ...     data_dict,
        ...     title="교통수단별 이동시간 분포",
        ...     show_box=True,
        ...     show_mean_value=True
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 데이터 준비
    all_data, all_positions, all_colors, group_labels, group_positions = _prepare_grouped_data(
        data_dict, clip_percentile
    )

    # 바이올린 플롯 생성
    parts = ax.violinplot(
        all_data,
        positions=all_positions,
        widths=0.6,
        showmeans=False,
        showmedians=True,
        showextrema=False
    )

    # 색상 적용
    for pc, color in zip(parts['bodies'], all_colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # 박스플롯 추가 (내부에)
    if show_box:
        bp = ax.boxplot(
            all_data,
            positions=all_positions,
            widths=0.2,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color='black', linewidth=1),
            boxprops=dict(facecolor='white', alpha=0.8, linewidth=1),
            whiskerprops=dict(linewidth=1),
            capprops=dict(linewidth=1)
        )

    # 평균값 텍스트 표시
    if show_mean_value:
        _add_mean_values(
            ax, all_data, all_positions, all_colors, data_dict,
            mean_format, mean_margin, mean_text_color, use_center_positioning=True,
            mean_position_strategy=mean_position_strategy,
            fixed_y_percentage=fixed_y_percentage,
            y_axis_limits=y_axis_limits
        )

        # y축 범위 조정 (중앙 배치를 위해)
        has_spread_data = any(
            (d.quantile(0.95) - d.quantile(0.75)) > (d.quantile(0.75) - d.quantile(0.25)) * 3
            for d in all_data
        )

        if has_spread_data:
            all_maxes = [d.quantile(0.9) for d in all_data]
            y_max = max(all_maxes)
            center_y = y_max * 0.6
            ax.set_ylim(0, max(center_y * 1.2, max([d.quantile(0.98) for d in all_data])))
        else:
            y_max = max([d.quantile(0.98) for d in all_data])
            ax.set_ylim(0, y_max * 1.1)

    # 축 설정
    ax.set_ylabel(ylabel, fontsize=DEFAULT_FONT_SIZES['label'])
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZES['title'], fontweight='bold')
    ax.set_xticks(group_positions)
    ax.set_xticklabels(group_labels, fontsize=DEFAULT_FONT_SIZES['tick'])

    # 그리드
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 범례
    legend_elements = [
        Patch(facecolor=color, alpha=0.7, label=mode)
        for mode, color in DEFAULT_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=DEFAULT_FONT_SIZES['legend'])

    plt.tight_layout()
    return fig, ax


def create_grouped_boxen_plot(
    data_dict: Dict[str, pd.DataFrame],
    title: str = "셔클 도입 효과 분석 (Boxen 플롯)",
    ylabel: str = "시간(분)",
    figsize: Tuple[int, int] = (14, 8),
    show_outliers: bool = False,
    show_mean: bool = False,
    show_mean_value: bool = False,
    mean_format: str = "{:.0f}분",
    mean_margin: float = 1.0,
    clip_percentile: float = 1.0,
    mean_text_color: Optional[Dict[str, str]] = None,
    mean_position_strategy: str = "fixed_top",
    fixed_y_percentage: float = 0.8,
    y_axis_limits: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    그룹화된 Boxen 플롯 생성

    교통수단별 시간 데이터를 다층 분위수 박스(Letter-value plot)로 시각화합니다.
    대용량 데이터에서 분포의 세부사항을 박스플롯보다 더 상세히 표현합니다.

    Args:
        data_dict: 카테고리별 데이터프레임 딕셔너리
        title: 그래프 제목
        ylabel: Y축 라벨
        figsize: 그래프 크기 (width, height)
        show_outliers: 이상값 표시 여부
        show_mean: 평균값 마커 표시 여부
        show_mean_value: 평균값 텍스트 표시 여부
        mean_format: 평균값 텍스트 형식
        mean_margin: 평균값 텍스트 여백
        clip_percentile: 극단값 제거 기준
        mean_text_color: 교통수단별 텍스트 색상
        mean_position_strategy: 평균값 위치 전략
        fixed_y_percentage: 고정 비율 위치 전략 사용 시 비율
        y_axis_limits: Y축 범위 (min, max)

    Returns:
        Tuple[plt.Figure, plt.Axes]: 생성된 그래프 객체

    Logics:
        1. 데이터를 seaborn 형식으로 변환
        2. Letter-value plot (boxenplot) 생성
        3. 교통수단별 색상 팔레트 적용
        4. 위치 전략에 따른 평균값 텍스트 추가
        5. 축 설정 및 범례 구성

    Example:
        >>> fig, ax = create_grouped_boxen_plot(
        ...     data_dict,
        ...     title="교통수단별 이동시간 상세 분포",
        ...     clip_percentile=0.95,
        ...     show_mean_value=True
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 데이터 준비
    plot_data = []
    for category, df in data_dict.items():
        if clip_percentile < 1.0:
            df = df.copy()
            for col in df.columns:
                upper_limit = df[col].quantile(clip_percentile)
                df = df[df[col] <= upper_limit]

        for mode in ["대중교통", "셔클"]:
            if mode in df.columns:
                values = df[mode].dropna()
                for val in values:
                    plot_data.append({
                        'category': category,
                        'mode': mode,
                        'value': val
                    })

    plot_df = pd.DataFrame(plot_data)

    # boxenplot 생성
    sns.boxenplot(
        data=plot_df,
        x='category',
        y='value',
        hue='mode',
        palette=DEFAULT_COLORS,
        ax=ax,
        showfliers=show_outliers
    )

    # 평균값 텍스트 표시
    if show_mean_value:
        categories = list(data_dict.keys())

        # y축 범위 설정
        if y_axis_limits:
            y_min, y_max = y_axis_limits
            ax.set_ylim(y_min, y_max)
        else:
            y_min, y_max = ax.get_ylim()

        for i, category in enumerate(categories):
            df = data_dict[category]
            if clip_percentile < 1.0:
                df = df.copy()
                for col in df.columns:
                    upper_limit = df[col].quantile(clip_percentile)
                    df = df[df[col] <= upper_limit]

            modes = ["대중교통", "셔클"]
            mode_positions = [i - 0.2, i + 0.2]

            for j, mode in enumerate(modes):
                if mode in df.columns:
                    mean_val = df[mode].mean()

                    # 새로운 위치 전략 적용
                    if mean_position_strategy == "fixed_top":
                        text_y = y_max * 0.9
                    elif mean_position_strategy == "fixed_percentage":
                        text_y = y_max * fixed_y_percentage
                    else:  # adaptive
                        q75 = df[mode].quantile(0.75)
                        spread = df[mode].quantile(0.95) - df[mode].quantile(0.75)
                        iqr = df[mode].quantile(0.75) - df[mode].quantile(0.25)
                        has_spread = spread > iqr * 3
                        if has_spread:
                            text_y = y_max * 0.6
                        else:
                            text_y = q75 + (y_max - q75) * 0.1

                    # 색상 설정 (기본값을 빨간색으로)
                    if mean_text_color:
                        color = mean_text_color.get(mode, DEFAULT_MEAN_TEXT_COLORS[mode])
                    else:
                        color = DEFAULT_MEAN_TEXT_COLORS[mode]

                    ax.text(
                        mode_positions[j], text_y,
                        mean_format.format(mean_val),
                        ha='center', va='bottom',
                        fontsize=DEFAULT_FONT_SIZES['tick'],
                        color=color,
                        fontweight='bold'
                    )

    # 축 설정
    ax.set_xlabel('')
    ax.set_ylabel(ylabel, fontsize=DEFAULT_FONT_SIZES['label'])
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZES['title'], fontweight='bold')

    # 그리드
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 범례를 두 개만 표시
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', fontsize=DEFAULT_FONT_SIZES['legend'])

    plt.tight_layout()
    return fig, ax


def create_swarm_plot(
    data: pd.DataFrame,
    title: str = "셔클 도입 효과 분석 (Swarm 플롯)",
    ylabel: str = "시간(분)",
    figsize: Tuple[int, int] = (14, 8),
    sample_size: Optional[int] = 1000,
    clip_percentile: float = 0.95
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Swarm 플롯 생성

    개별 데이터 포인트를 벌떼 형태로 시각화하여 데이터 분포와
    밀도를 직관적으로 표현합니다. 중간 규모 데이터셋에 적합합니다.

    Args:
        data: 교통수단별 시간 데이터프레임
        title: 그래프 제목
        ylabel: Y축 라벨
        figsize: 그래프 크기 (width, height)
        sample_size: 샘플링할 데이터 포인트 수 (성능 최적화용)
        clip_percentile: 극단값 제거 기준

    Returns:
        Tuple[plt.Figure, plt.Axes]: 생성된 그래프 객체

    Logics:
        1. 대용량 데이터 샘플링 (sample_size 지정시)
        2. 극단값 클리핑으로 시각적 품질 향상
        3. 데이터를 long format으로 변환
        4. 교통수단별 색상 매핑 적용
        5. seaborn swarmplot으로 벌떼 시각화 생성

    Example:
        >>> fig, ax = create_swarm_plot(
        ...     data,
        ...     title="교통수단별 개별 이동시간",
        ...     sample_size=500,
        ...     clip_percentile=0.98
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 데이터 전처리 및 극단값 제거
    plot_data = data.copy()

    if clip_percentile < 1.0:
        for col in plot_data.columns:
            upper_limit = plot_data[col].quantile(clip_percentile)
            plot_data[col] = plot_data[col].clip(upper=upper_limit)

    # 데이터 샘플링 (너무 많으면 느려짐)
    if sample_size and len(plot_data) > sample_size:
        plot_data = plot_data.sample(n=sample_size, random_state=42)

    # 데이터 준비
    melted = plot_data.melt(var_name='Mode', value_name='Value')

    # 팔레트 매핑
    palette_dict = {}
    for mode in melted['Mode'].unique():
        if '대중교통' in mode:
            palette_dict[mode] = DEFAULT_COLORS['대중교통']
        elif '셔클' in mode:
            palette_dict[mode] = DEFAULT_COLORS['셔클']
        else:
            palette_dict[mode] = '#808080'

    # Seaborn swarmplot
    sns.swarmplot(
        data=melted,
        x='Mode',
        y='Value',
        palette=palette_dict,
        size=3,
        alpha=0.6,
        ax=ax
    )

    ax.set_ylabel(ylabel, fontsize=DEFAULT_FONT_SIZES['label'])
    ax.set_title(title, fontsize=DEFAULT_FONT_SIZES['title'], fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig, ax


def calculate_statistics(
    data: pd.DataFrame,
    percentiles: List[float] = [0.25, 0.5, 0.75]
) -> pd.DataFrame:
    """
    통계 요약 계산

    교통수단별 시간 데이터의 기술통계량을 계산합니다.
    평균, 표준편차, 최솟값, 최댓값, 분위수를 포함합니다.

    Args:
        data: 교통수단별 시간 데이터프레임
        percentiles: 계산할 분위수 리스트

    Returns:
        pd.DataFrame: 통계량이 정리된 데이터프레임

    Logics:
        1. 각 컬럼(교통수단)별 기술통계량 계산
        2. 사용자 지정 분위수 계산
        3. 결과를 DataFrame 형태로 구조화
        4. 소수점 2자리로 반올림

    Example:
        >>> stats = calculate_statistics(
        ...     data, percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]
        ... )
        >>> print(stats)
                    대중교통    셔클
        평균         25.4    18.2
        표준편차      8.1     6.3
        Q25         19.5    14.1
    """
    stats = {}

    for col in data.columns:
        col_stats = {
            '평균': data[col].mean(),
            '표준편차': data[col].std(),
            '최소값': data[col].min(),
            '최대값': data[col].max()
        }

        for p in percentiles:
            col_stats[f'Q{int(p*100)}'] = data[col].quantile(p)

        stats[col] = col_stats

    return pd.DataFrame(stats).round(2)


def calculate_improvement(
    baseline_data: pd.Series,
    improved_data: pd.Series
) -> Dict[str, float]:
    """
    개선 효과 계산

    기준 교통수단(대중교통) 대비 개선된 교통수단(셔클)의
    시간 단축 효과를 절대량과 비율로 계산합니다.

    Args:
        baseline_data: 기준 교통수단 데이터 (대중교통)
        improved_data: 개선된 교통수단 데이터 (셔클)

    Returns:
        Dict[str, float]: 개선 효과 지표
            - 평균_개선량: 평균 시간 단축량 (분)
            - 평균_개선율: 평균 시간 단축율 (%)
            - 중앙값_개선량: 중앙값 시간 단축량 (분)
            - 중앙값_개선율: 중앙값 시간 단축율 (%)

    Logics:
        1. 평균값 기반 개선량 및 개선율 계산
        2. 중앙값 기반 개선량 및 개선율 계산
        3. 백분율 변환 (개선율 = (기준-개선)/기준 * 100)
        4. 음수는 시간 증가, 양수는 시간 단축을 의미

    Example:
        >>> improvement = calculate_improvement(
        ...     baseline_data=public_transport_time,
        ...     improved_data=shuttle_time
        ... )
        >>> print(f"평균 {improvement['평균_개선량']:.1f}분 단축")
        >>> print(f"개선율 {improvement['평균_개선율']:.1f}%")
    """
    return {
        '평균_개선량': baseline_data.mean() - improved_data.mean(),
        '평균_개선율': (baseline_data.mean() - improved_data.mean()) / baseline_data.mean() * 100,
        '중앙값_개선량': baseline_data.median() - improved_data.median(),
        '중앙값_개선율': (baseline_data.median() - improved_data.median()) / baseline_data.median() * 100
    }


def create_forest_plot(
    stats_results: Dict[str, any],
    region_name: str = "지역",
    title: str = "통계적 효과 크기 분석 (Forest Plot)",
    figsize: Tuple[int, int] = (12, 8),
    show_stats: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    포레스트 플롯 생성

    통계 분석 결과를 포레스트 플롯으로 시각화하여
    평균 차이와 신뢰구간을 직관적으로 표현합니다.

    Args:
        stats_results: 통계 분석 결과 딕셔너리
        region_name: 지역 이름 (도시/농어촌)
        title: 그래프 제목
        figsize: 그래프 크기 (width, height)
        show_stats: 통계량 표시 여부

    Returns:
        Tuple[plt.Figure, plt.Axes]: 생성된 그래프 객체

    Logics:
        1. 카테고리별 평균 차이와 신뢰구간 추출
        2. Y축에 카테고리 배치, X축에 평균 차이 표시
        3. 신뢰구간을 에러바로 표현
        4. p-value와 Cohen's d를 텍스트로 표시
        5. 0 기준선 추가 (개선 효과 없음)

    Example:
        >>> fig, ax = create_forest_plot(
        ...     city_stats_results,
        ...     region_name="도시지역",
        ...     title="도시지역 개선 효과 분석"
        ... )
        >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = list(stats_results.keys())
    n_categories = len(categories)

    # 데이터 추출
    mean_diffs = []
    ci_lowers = []
    ci_uppers = []
    p_values = []
    effect_sizes = []

    for category in categories:
        result = stats_results[category]

        # 평균 차이 계산
        mean_diff = (
            result.baseline_stats['mean'] -
            result.improved_stats['mean']
        )
        mean_diffs.append(mean_diff)

        # 신뢰구간
        ci_lower, ci_upper = result.confidence_interval
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)

        # 통계량
        p_values.append(result.statistical_significance.p_value)
        effect_sizes.append(result.effect_size)

    # Y축 위치
    y_positions = np.arange(n_categories)

    # 에러바 계산 (신뢰구간의 상하한)
    errors_lower = [
        mean_diffs[i] - ci_lowers[i]
        for i in range(n_categories)
    ]
    errors_upper = [
        ci_uppers[i] - mean_diffs[i]
        for i in range(n_categories)
    ]

    # 포레스트 플롯 그리기
    ax.errorbar(
        mean_diffs,
        y_positions,
        xerr=[errors_lower, errors_upper],
        fmt='o',
        markersize=8,
        capsize=5,
        capthick=2,
        color='#1F4EAA',
        ecolor='#1F4EAA',
        linewidth=2,
        label='평균 차이 (99% CI)'
    )

    # 0 기준선 (개선 효과 없음)
    ax.axvline(x=0, color='red', linestyle='--',
               linewidth=2, alpha=0.7, label='효과 없음')

    # 통계량 텍스트 추가
    if show_stats:
        for i, (category, mean_diff) in enumerate(
            zip(categories, mean_diffs)
        ):
            # p-value와 Cohen's d 텍스트
            p_val = p_values[i]
            d_val = effect_sizes[i]

            # p-value 표시 형식
            if p_val < 0.001:
                p_text = "p < 0.001***"
            elif p_val < 0.01:
                p_text = f"p = {p_val:.3f}**"
            elif p_val < 0.05:
                p_text = f"p = {p_val:.3f}*"
            else:
                p_text = f"p = {p_val:.3f}"

            # 텍스트 위치 조정
            text_x = max(ci_uppers) * 1.05

            ax.text(
                text_x, i,
                f"{p_text}\nd = {d_val:.2f}",
                verticalalignment='center',
                fontsize=9,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='wheat',
                    alpha=0.3
                )
            )

    # 축 설정
    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories, fontsize=DEFAULT_FONT_SIZES['tick'])
    ax.set_xlabel(
        '평균 시간 단축 (분)',
        fontsize=DEFAULT_FONT_SIZES['label']
    )
    ax.set_title(
        f"{title}\n{region_name}",
        fontsize=DEFAULT_FONT_SIZES['title'],
        fontweight='bold'
    )

    # 그리드
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # 범례
    ax.legend(loc='upper right', fontsize=DEFAULT_FONT_SIZES['legend'])

    # 레이아웃 조정
    plt.tight_layout()

    return fig, ax