"""
셔클 효과 분석 설정 관리 모듈

데이터사이언스 분석을 위한 YAML 설정 파일 로드, 검증, 관리 모듈
사용자 정의 설정과 기본값 병합, 설정 유효성 검사 기능 제공

Author: taeyang lee
Made Date: 2025-09-29 21:00
Modified Date: 2025-09-29 21:00

Logics:
    1. YAML 설정 파일 로드 및 파싱
    2. 설정 스키마 검증 및 유효성 검사
    3. 기본값과 사용자 설정 병합
    4. 환경변수 기반 설정 오버라이드
    5. 설정 템플릿 생성 및 관리

Example:
    >>> config_manager = ConfigManager()
    >>> config = config_manager.load_config("config.yaml")
    >>> config_manager.validate_config(config)
    >>> template = config_manager.create_config_template()
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import warnings


@dataclass
class VisualizationConfig:
    """
    시각화 설정 데이터 클래스

    시각화 관련 모든 설정을 구조화된 형태로 관리합니다.

    Attributes:
        types: 생성할 시각화 타입 리스트
        title_prefix: 그래프 제목 접두사
        ylabel: Y축 라벨
        figsize: 그래프 크기 [width, height]
        show_outliers: 이상값 표시 여부
        show_mean_value: 평균값 텍스트 표시 여부
        mean_format: 평균값 텍스트 형식
        mean_position_strategy: 평균값 위치 전략
        fixed_y_percentage: 고정 비율 위치 전략 사용 시 비율
        y_axis_min: Y축 최소값 (None이면 자동)
        y_axis_max: Y축 최대값 (None이면 자동)
        mean_text_colors: 교통수단별 텍스트 색상
    """
    types: List[str] = field(default_factory=lambda: ["boxplot", "violin", "boxen"])
    title_prefix: str = "셔클 도입 효과 분석"
    ylabel: str = "시간(분)"
    figsize: List[int] = field(default_factory=lambda: [15, 8])
    show_outliers: bool = False
    show_mean_value: bool = True
    mean_format: str = "{:.1f}분"
    mean_position_strategy: str = "fixed_percentage"
    fixed_y_percentage: float = 0.85
    y_axis_min: Optional[float] = None
    y_axis_max: Optional[float] = None
    mean_text_colors: Dict[str, str] = field(
        default_factory=lambda: {"대중교통": "red", "셔클": "#1F4EAA"}
    )


@dataclass
class DataConfig:
    """
    데이터 설정 데이터 클래스

    데이터 로드 및 처리 관련 설정을 관리합니다.

    Attributes:
        path: 데이터 파일 경로
        encoding: 파일 인코딩
        zone_types: 분석할 지역 유형 리스트
        time_columns: 시간 카테고리별 컬럼 매핑
    """
    path: str = "../data/shucle_analysis_dataset_20250929.csv"
    encoding: str = "utf-8"
    zone_types: List[str] = field(default_factory=lambda: ["도시", "농어촌"])
    time_columns: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class OutputConfig:
    """
    출력 설정 데이터 클래스

    결과 저장 및 출력 관련 설정을 관리합니다.

    Attributes:
        save_plots: 플롯 파일 저장 여부
        show_plots: 플롯 화면 표시 여부
        save_statistics: 통계 결과 저장 여부
        directory: 출력 디렉토리
        format: 플롯 파일 형식
        dpi: 이미지 해상도
    """
    save_plots: bool = True
    show_plots: bool = True
    save_statistics: bool = True
    directory: str = "../output"
    format: str = "png"
    dpi: int = 300


class ConfigManager:
    """
    설정 관리자 클래스

    YAML 설정 파일의 로드, 검증, 관리를 담당하는 클래스입니다.
    기본값과 사용자 설정의 병합, 환경변수 오버라이드 등을 지원합니다.

    Attributes:
        default_config: 기본 설정 딕셔너리
        schema: 설정 스키마 정의

    Logics:
        1. YAML 파일 로드 및 파싱
        2. 설정 스키마 기반 유효성 검사
        3. 기본값과 사용자 설정 재귀적 병합
        4. 환경변수를 통한 설정 오버라이드
        5. 설정 템플릿 및 예시 생성

    Example:
        >>> manager = ConfigManager()
        >>> config = manager.load_config("user_config.yaml")
        >>> if manager.validate_config(config):
        ...     print("설정이 유효합니다")
        >>> template = manager.create_config_template()
    """

    def __init__(self):
        """
        설정 관리자 초기화

        기본 설정과 스키마를 초기화합니다.

        Logics:
            1. 기본 설정 딕셔너리 생성
            2. 설정 스키마 정의
            3. 지원되는 설정 키 목록 설정
        """
        self.default_config = self._create_default_config()
        self.schema = self._define_schema()

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        설정 파일 로드

        YAML 설정 파일을 로드하고 기본값과 병합합니다.

        Args:
            config_path: 설정 파일 경로

        Returns:
            Dict[str, Any]: 병합된 설정 딕셔너리

        Raises:
            FileNotFoundError: 설정 파일이 없는 경우
            yaml.YAMLError: YAML 파싱 오류

        Logics:
            1. YAML 파일 존재 여부 확인
            2. YAML 파싱 및 설정 로드
            3. 기본값과 사용자 설정 병합
            4. 환경변수 오버라이드 적용

        Example:
            >>> config = manager.load_config("analysis_config.yaml")
            >>> print(config['data']['path'])
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML 파싱 오류: {str(e)}")

        # 기본값과 사용자 설정 병합
        merged_config = self._deep_merge(self.default_config, user_config)

        # 환경변수 오버라이드 적용
        merged_config = self._apply_env_overrides(merged_config)

        return merged_config

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        설정 유효성 검사

        설정 딕셔너리가 정의된 스키마에 맞는지 검사합니다.

        Args:
            config: 검사할 설정 딕셔너리

        Returns:
            bool: 유효성 검사 통과 여부

        Logics:
            1. 필수 섹션 존재 여부 확인
            2. 데이터 타입 검증
            3. 값 범위 및 허용값 검증
            4. 의존성 검증 (예: 전략과 관련 파라미터)

        Example:
            >>> is_valid = manager.validate_config(config)
            >>> if not is_valid:
            ...     print("설정에 문제가 있습니다")
        """
        try:
            # 필수 섹션 확인
            required_sections = ['data', 'visualization', 'output']
            for section in required_sections:
                if section not in config:
                    print(f"❌ 필수 섹션이 없습니다: {section}")
                    return False

            # 데이터 섹션 검증
            if not self._validate_data_config(config['data']):
                return False

            # 시각화 섹션 검증
            if not self._validate_visualization_config(config['visualization']):
                return False

            # 출력 섹션 검증
            if not self._validate_output_config(config['output']):
                return False

            print("✅ 설정 유효성 검사 통과")
            return True

        except Exception as e:
            print(f"❌ 설정 검증 중 오류 발생: {str(e)}")
            return False

    def create_config_template(self) -> str:
        """
        설정 템플릿 생성

        사용자가 참고할 수 있는 설정 파일 템플릿을 생성합니다.

        Returns:
            str: YAML 형식의 설정 템플릿

        Logics:
            1. 기본 설정을 YAML 형식으로 변환
            2. 주석과 설명 추가
            3. 예시 값과 선택 옵션 포함

        Example:
            >>> template = manager.create_config_template()
            >>> with open("new_config.yaml", "w") as f:
            ...     f.write(template)
        """
        template = """# 셔클 효과 분석 설정 파일
# 이 파일을 복사하여 원하는 설정으로 수정하세요.

# 데이터 설정
data:
  path: '../data/shucle_analysis_dataset_20250929.csv'  # 데이터 파일 경로
  encoding: 'utf-8'  # 파일 인코딩
  zone_types:  # 분석할 지역 유형
    - '도시'
    - '농어촌'
  time_columns:  # 시간 카테고리별 컬럼 매핑
    도보시간:
      - public_total_walking_time_seconds
      - drt_total_walking_time_seconds
    탑승시간:
      - public_onboard_time_seconds
      - drt_onboard_time_seconds
    대기시간:
      - public_waiting_time_seconds
      - drt_waiting_time_seconds
    총 이동시간:
      - public_total_time_seconds
      - drt_total_trip_time_seconds

# 시각화 설정
visualization:
  types:  # 생성할 시각화 타입 ['boxplot', 'violin', 'boxen']
    - boxplot
    - violin
    - boxen

  common_settings:  # 모든 시각화 공통 설정
    title_prefix: '셔클 도입 효과 분석'
    ylabel: '시간(분)'
    figsize: [15, 8]  # 그래프 크기 [width, height]
    show_outliers: false
    show_mean_value: true
    mean_format: '{:.1f}분'

    # 평균값 위치 전략 제어
    mean_position_strategy: 'fixed_percentage'  # 'adaptive', 'fixed_top', 'fixed_percentage'
    fixed_y_percentage: 0.85  # fixed_percentage 사용 시 Y축 비율 (0.0-1.0)

    # Y축 범위 직접 지정 (null이면 자동)
    y_axis_min: null
    y_axis_max: null

    # 평균값 텍스트 색상
    mean_text_colors:
      대중교통: 'red'
      셔클: '#1F4EAA'

# 통계 분석 설정
statistics:
  percentiles: [0.25, 0.5, 0.75, 0.9]

# 출력 설정
output:
  save_plots: true       # 플롯 파일 저장 여부
  show_plots: true       # 플롯 화면 표시 여부
  save_statistics: true  # 통계 결과 저장 여부
  directory: '../output' # 출력 디렉토리
  format: 'png'         # 파일 형식 ['png', 'pdf', 'svg']
  dpi: 300              # 이미지 해상도
"""
        return template

    def save_config(self, config: Dict[str, Any], filepath: str) -> None:
        """
        설정을 파일로 저장

        설정 딕셔너리를 YAML 파일로 저장합니다.

        Args:
            config: 저장할 설정 딕셔너리
            filepath: 저장할 파일 경로

        Logics:
            1. 디렉토리 존재 여부 확인 및 생성
            2. 설정을 YAML 형식으로 직렬화
            3. UTF-8 인코딩으로 파일 저장

        Example:
            >>> manager.save_config(config, "backup_config.yaml")
        """
        filepath = Path(filepath)

        # 디렉토리 자동 생성
        if not filepath.parent.exists():
            print(f"📁 설정 디렉토리 생성: {filepath.parent}")
            filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(
                config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
                sort_keys=False
            )

    def _create_default_config(self) -> Dict[str, Any]:
        """
        기본 설정 생성

        Returns:
            Dict[str, Any]: 기본 설정 딕셔너리
        """
        return {
            'data': {
                'path': '../data/shucle_analysis_dataset_20250929.csv',
                'encoding': 'utf-8',
                'zone_types': ['도시', '농어촌'],
                'time_columns': {
                    "도보시간": ["public_total_walking_time_seconds", "drt_total_walking_time_seconds"],
                    "탑승시간": ["public_onboard_time_seconds", "drt_onboard_time_seconds"],
                    "대기시간": ["public_waiting_time_seconds", "drt_waiting_time_seconds"],
                    "총 이동시간": ["public_total_time_seconds", "drt_total_trip_time_seconds"]
                }
            },
            'visualization': {
                'types': ['boxplot', 'violin', 'boxen'],
                'common_settings': {
                    'title_prefix': '셔클 도입 효과 분석',
                    'ylabel': '시간(분)',
                    'figsize': [15, 8],
                    'show_outliers': False,
                    'show_mean': False,
                    'show_mean_value': True,
                    'mean_format': '{:.1f}분',
                    'mean_margin': 1.0,
                    'clip_percentile': 0.99,
                    'mean_position_strategy': 'fixed_percentage',
                    'fixed_y_percentage': 0.85,
                    'y_axis_min': None,
                    'y_axis_max': None,
                    'mean_text_colors': {
                        '대중교통': 'red',
                        '셔클': '#1F4EAA'
                    }
                }
            },
            'statistics': {
                'percentiles': [0.25, 0.5, 0.75, 0.9]
            },
            'output': {
                'save_plots': True,
                'show_plots': True,
                'save_statistics': True,
                'directory': '../output',
                'format': 'png',
                'dpi': 300
            }
        }

    def _define_schema(self) -> Dict[str, Any]:
        """
        설정 스키마 정의

        Returns:
            Dict[str, Any]: 설정 스키마
        """
        return {
            'visualization': {
                'types': ['boxplot', 'violin', 'boxen'],
                'mean_position_strategies': ['adaptive', 'fixed_top', 'fixed_percentage'],
                'output_formats': ['png', 'pdf', 'svg']
            }
        }

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        딕셔너리 재귀적 병합

        Args:
            base: 기본 딕셔너리
            override: 오버라이드할 딕셔너리

        Returns:
            Dict: 병합된 딕셔너리
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        환경변수 오버라이드 적용

        Args:
            config: 원본 설정

        Returns:
            Dict[str, Any]: 환경변수가 적용된 설정
        """
        # 환경변수 기반 오버라이드 (예: SHUCLE_DATA_PATH)
        if os.getenv('SHUCLE_DATA_PATH'):
            config['data']['path'] = os.getenv('SHUCLE_DATA_PATH')

        if os.getenv('SHUCLE_OUTPUT_DIR'):
            config['output']['directory'] = os.getenv('SHUCLE_OUTPUT_DIR')

        if os.getenv('SHUCLE_OUTPUT_FORMAT'):
            config['output']['format'] = os.getenv('SHUCLE_OUTPUT_FORMAT')

        return config

    def _validate_data_config(self, data_config: Dict) -> bool:
        """데이터 설정 검증"""
        required_keys = ['path', 'zone_types', 'time_columns']
        for key in required_keys:
            if key not in data_config:
                print(f"❌ 데이터 설정에 필수 키가 없습니다: {key}")
                return False

        # 파일 경로 존재 확인 (상대 경로 고려)
        data_path = Path(data_config['path'])
        if not data_path.is_absolute():
            # 설정 파일 기준 상대 경로로 변환
            script_dir = Path(__file__).parent
            data_path = script_dir / data_config['path']

        if not data_path.exists():
            warnings.warn(f"⚠️ 데이터 파일이 존재하지 않습니다: {data_path}")

        return True

    def _validate_visualization_config(self, viz_config: Dict) -> bool:
        """시각화 설정 검증"""
        if 'types' not in viz_config:
            print("❌ 시각화 타입이 지정되지 않았습니다")
            return False

        valid_types = self.schema['visualization']['types']
        for viz_type in viz_config['types']:
            if viz_type not in valid_types:
                print(f"❌ 지원하지 않는 시각화 타입: {viz_type}")
                return False

        # 평균값 위치 전략 검증
        if 'common_settings' in viz_config:
            strategy = viz_config['common_settings'].get('mean_position_strategy')
            if strategy and strategy not in self.schema['visualization']['mean_position_strategies']:
                print(f"❌ 지원하지 않는 평균값 위치 전략: {strategy}")
                return False

            # fixed_percentage 전략 사용 시 비율 값 확인
            if strategy == 'fixed_percentage':
                percentage = viz_config['common_settings'].get('fixed_y_percentage')
                if percentage is None or not (0.0 <= percentage <= 1.0):
                    print("❌ fixed_y_percentage는 0.0과 1.0 사이의 값이어야 합니다")
                    return False

        return True

    def _validate_output_config(self, output_config: Dict) -> bool:
        """출력 설정 검증"""
        if 'format' in output_config:
            valid_formats = self.schema['visualization']['output_formats']
            if output_config['format'] not in valid_formats:
                print(f"❌ 지원하지 않는 출력 형식: {output_config['format']}")
                return False

        if 'dpi' in output_config:
            if not isinstance(output_config['dpi'], int) or output_config['dpi'] <= 0:
                print("❌ DPI는 양의 정수여야 합니다")
                return False

        return True