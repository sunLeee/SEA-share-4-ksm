"""
셔클 효과 분석 패키지

데이터사이언스 분석을 위한 셔클(수요응답형 교통서비스) 도입 효과 분석 패키지
고성능 벡터화 연산, 병렬 처리, 고급 통계 분석을 지원하는 통합 분석 도구

Author: taeyang lee
Made Date: 2025-09-29 22:00
Modified Date: 2025-09-29 22:00

Modules:
    visualization: 시각화 모듈 (박스플롯, 바이올린, Boxen 플롯)
    data_processor: 데이터 처리 및 전처리 모듈
    statistics_analyzer: 고급 통계 분석 모듈 (가설검정, 효과 크기 계산)
    config_manager: YAML 기반 설정 관리 모듈

Example:
    >>> from src.visualization import create_grouped_boxplot
    >>> from src.statistics_analyzer import StatisticsAnalyzer
    >>> from src.config_manager import ConfigManager
    >>> from src.data_processor import ShuttleDataProcessor
"""

from .visualization import (
    prepare_data,
    create_grouped_boxplot,
    create_grouped_violin_plot,
    create_grouped_boxen_plot,
    create_swarm_plot,
    calculate_statistics,
    calculate_improvement
)

from .data_processor import ShuttleDataProcessor

from .statistics_analyzer import StatisticsAnalyzer

from .config_manager import ConfigManager

# 패키지 메타데이터
__version__ = "2.0.0"
__author__ = "taeyang lee"
__email__ = "taeyang.lee@example.com"
__description__ = "셔클 효과 분석을 위한 고성능 데이터사이언스 패키지"

# 성능 최적화 설정
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# 성능 최적화 설정
import pandas as pd
import numpy as np

pd.set_option('compute.use_bottleneck', True)
pd.set_option('compute.use_numexpr', True)
np.seterr(divide='ignore', invalid='ignore')

# 공개 API 정의
__all__ = [
    # 시각화 함수
    'prepare_data',
    'create_grouped_boxplot',
    'create_grouped_violin_plot',
    'create_grouped_boxen_plot',
    'create_swarm_plot',
    'calculate_statistics',
    'calculate_improvement',

    # 클래스들
    'ShuttleDataProcessor',
    'StatisticsAnalyzer',
    'ConfigManager',

    # 메타데이터
    '__version__',
    '__author__',
    '__description__'
]