#!/usr/bin/env python3
"""
셔클 효과 분석 패키지 설정

데이터사이언스 분석을 위한 셔클(수요응답형 교통서비스) 도입 효과 분석 패키지
고성능 벡터화 연산, 병렬 처리, 고급 통계 분석을 지원하는 통합 분석 도구

Author: taeyang lee
Made Date: 2025-09-29 22:00
Modified Date: 2025-09-29 22:00

Usage:
    pip install -e .                    # 개발 모드 설치
    pip install .                       # 일반 설치
    python setup.py develop             # 개발 모드 (구버전)
    python setup.py install             # 일반 설치 (구버전)
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# 최소 Python 버전 체크
if sys.version_info < (3, 8):
    raise RuntimeError(
        "셔클 효과 분석 패키지는 Python 3.8 이상이 필요합니다. "
        f"현재 버전: {sys.version_info.major}.{sys.version_info.minor}"
    )

# 프로젝트 루트 디렉토리
project_root = Path(__file__).parent

# README 파일 읽기
readme_path = project_root / "scripts" / "README.md"
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "셔클 효과 분석을 위한 고성능 데이터사이언스 패키지"

# requirements.txt에서 의존성 읽기
requirements_path = project_root / "requirements.txt"
install_requires = []
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                install_requires.append(line)

# 개발용 의존성 정의
dev_requires = [
    'pytest>=7.0.0',
    'pytest-cov>=4.0.0',
    'black>=23.0.0',
    'flake8>=6.0.0',
    'mypy>=1.0.0',
    'pre-commit>=3.0.0',
    'sphinx>=5.0.0',
    'sphinx-rtd-theme>=1.0.0'
]

# 전체 의존성 (선택적 설치용)
all_requires = install_requires + dev_requires

# 패키지 메타데이터
setup(
    name="shucle-effect-analysis",
    version="2.0.0",
    author="taeyang lee",
    author_email="taeyang.lee@example.com",
    description="셔클 효과 분석을 위한 고성능 데이터사이언스 패키지",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/shucle-effect-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/shucle-effect-analysis/issues",
        "Documentation": "https://github.com/your-username/shucle-effect-analysis/wiki",
        "Source Code": "https://github.com/your-username/shucle-effect-analysis",
    },

    # 패키지 설정
    packages=find_packages(exclude=["tests*", "notebooks*", "docs*"]),
    package_dir={"": "."},
    include_package_data=True,

    # Python 버전 요구사항
    python_requires=">=3.8",

    # 의존성 설정
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
        'matplotlib>=3.6.0',
        'seaborn>=0.12.0',
        'pyyaml>=6.0.0',
        'fastjsonschema>=2.16.0',
        'scipy>=1.10.0',
        'statsmodels>=0.14.0',
        'psutil>=5.9.0',
        'numexpr>=2.8.0',
        'bottleneck>=1.3.0',
    ],

    # 선택적 의존성
    extras_require={
        "dev": dev_requires,
        "all": all_requires,
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=7.0.0",
            "ipykernel>=6.25.0"
        ],
        "excel": [
            "openpyxl>=3.1.0",
            "xlrd>=2.0.0"
        ]
    },

    # 진입점 (CLI 명령어)
    entry_points={
        "console_scripts": [
            "shucle-analysis=scripts.main:main",
            "shucle-config=scripts.main:create_config_command",
        ],
    },

    # 분류자
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],

    # 키워드
    keywords=[
        "data-science", "transportation", "shuttle", "analysis",
        "visualization", "statistics", "pandas", "matplotlib",
        "한국어", "교통분석", "셔클", "수요응답형교통"
    ],

    # 라이선스
    license="MIT",

    # 패키지 데이터 포함
    package_data={
        "scripts": ["*.yaml", "*.md"],
        "src": ["*.yaml", "*.json"],
    },

    # zip으로 압축하지 않음 (개발 편의성)
    zip_safe=False,
)