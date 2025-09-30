#!/usr/bin/env python3
"""
셔클 효과 분석 Main Script

데이터사이언스 분석을 위한 셔클(수요응답형 교통서비스) 도입 효과 분석 도구
사용자가 최대한 많은 설정을 조절할 수 있도록 구성된 메인 실행 파일

Author: taeyang lee
Made Date: 2025-09-29 21:00
Modified Date: 2025-09-29 22:00

Usage:
    python scripts/main.py --interactive  # 대화형 모드
    python scripts/main.py                # 기본 설정 실행
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

# 경고 무시
warnings.filterwarnings('ignore')

# matplotlib 백엔드 설정 (조건부 설정)
if os.getenv('DISPLAY') is None or os.getenv('MPLBACKEND') == 'Agg':
    matplotlib.use('Agg')  # 비-GUI 백엔드 사용
else:
    # GUI 환경에서는 기본 백엔드 사용
    try:
        import tkinter
        matplotlib.use('TkAgg')
    except ImportError:
        matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 프로젝트 경로 설정
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 최적화된 모듈들 import 시도
try:
    from src.visualization import (
        prepare_data,
        create_grouped_boxplot,
        create_grouped_violin_plot,
        create_grouped_boxen_plot,
        calculate_statistics,
        calculate_improvement
    )
    from src.data_processor import ShuttleDataProcessor
    from src.statistics_analyzer import StatisticsAnalyzer
    print("✅ 최적화된 모듈 import 성공")
    USE_OPTIMIZED = True
except ImportError as e:
    print(f"⚠️ 최적화된 모듈 import 실패: {e}")
    print("🔧 기본 함수들을 사용합니다...")
    USE_OPTIMIZED = False


def prepare_data_basic(df, time_columns, convert_to_minutes=True):
    """기본 데이터 준비 함수"""
    result = {}

    for category, columns in time_columns.items():
        if len(columns) >= 2 and all(col in df.columns for col in columns):
            category_data = pd.DataFrame()

            # 분 단위 변환
            if convert_to_minutes:
                category_data["대중교통"] = df[columns[0]] / 60
                category_data["셔클"] = df[columns[1]] / 60
            else:
                category_data["대중교통"] = df[columns[0]]
                category_data["셔클"] = df[columns[1]]

            # 결측값 제거
            category_data = category_data.dropna()
            result[category] = category_data

    return result


def calculate_statistics_basic(data, percentiles=[0.25, 0.5, 0.75, 0.9]):
    """기본 통계 계산 함수"""
    stats = pd.DataFrame()

    for col in data.columns:
        stats[col] = [
            data[col].mean(),
            data[col].std(),
            data[col].min(),
            data[col].max()
        ] + [data[col].quantile(p) for p in percentiles]

    stats.index = ['평균', '표준편차', '최소값', '최대값'] + [f'Q{int(p*100)}' for p in percentiles]
    return stats


def calculate_improvement_basic(baseline_data, improved_data):
    """기본 개선율 계산 함수"""
    baseline_mean = baseline_data.mean()
    improved_mean = improved_data.mean()
    baseline_median = baseline_data.median()
    improved_median = improved_data.median()

    return {
        '평균_개선량': baseline_mean - improved_mean,
        '평균_개선율': ((baseline_mean - improved_mean) / baseline_mean) * 100,
        '중앙값_개선량': baseline_median - improved_median,
        '중앙값_개선율': ((baseline_median - improved_median) / baseline_median) * 100
    }


def create_grouped_boxplot_basic(data_dict, title="", ylabel="값", figsize=(12, 8), **kwargs):
    """기본 박스플롯 함수"""
    fig, ax = plt.subplots(figsize=figsize)

    positions = []
    labels = []
    data_list = []

    pos = 1
    for category, df in data_dict.items():
        for col in df.columns:
            data_list.append(df[col].dropna())
            positions.append(pos)
            labels.append(f"{category}\n{col}")
            pos += 1
        pos += 0.5  # 카테고리 간 간격

    bp = ax.boxplot(data_list, positions=positions, patch_artist=True)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # 색상 설정
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors * len(data_dict)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.tight_layout()
    return fig, ax


# 함수 선택
if USE_OPTIMIZED:
    prepare_data_func = prepare_data
    calculate_statistics_func = calculate_statistics
    calculate_improvement_func = calculate_improvement
    create_grouped_boxplot_func = create_grouped_boxplot
else:
    prepare_data_func = prepare_data_basic
    calculate_statistics_func = calculate_statistics_basic
    calculate_improvement_func = calculate_improvement_basic
    create_grouped_boxplot_func = create_grouped_boxplot_basic


def get_default_config():
    """기본 설정 반환"""
    return {
        'data': {
            'path': './data/shucle_analysis_dataset_20250929.csv',
            'zone_types': ['도시', '농어촌']
        },
        'analysis': {
            'time_columns': {
                "도보시간": ["public_total_walking_time_seconds", "drt_total_walking_time_seconds"],
                "탑승시간": ["public_onboard_time_seconds", "drt_onboard_time_seconds"],
                "대기시간": ["public_waiting_time_seconds", "drt_waiting_time_seconds"],
                "총 이동시간": ["public_total_time_seconds", "drt_total_trip_time_seconds"]
            },
            'convert_to_minutes': True
        },
        'visualization': {
            'figsize': [15, 8],
            'save_plots': True,
            'show_plots': False,
            'plot_types': ['boxplot', 'violin', 'boxen'],
            'dpi': 300,
            'show_outliers': False,
            'show_mean': True,
            'show_mean_value': True,
            'clip_percentile': 0.95,
            'mean_position_strategy': 'adaptive'
        },
        'output': {
            'directory': './output',
            'format': 'png'
        }
    }


def interactive_config():
    """대화형 설정 생성"""
    print("🔧 대화형 설정 모드")
    print("엔터를 누르면 기본값이 사용됩니다.\n")

    config = get_default_config()

    try:
        # 데이터 파일 경로
        data_path = input(f"데이터 파일 경로 [{config['data']['path']}]: ").strip()
        if data_path:
            config['data']['path'] = data_path

        # 출력 디렉토리
        output_dir = input(f"출력 디렉토리 [{config['output']['directory']}]: ").strip()
        if output_dir:
            config['output']['directory'] = output_dir

        # 플롯 저장 여부
        save_plots = input(f"플롯 저장 (Y/N) [{'Y' if config['visualization']['save_plots'] else 'N'}]: ").strip().upper()
        if save_plots == 'N':
            config['visualization']['save_plots'] = False
        elif save_plots == 'Y':
            config['visualization']['save_plots'] = True

        # 플롯 표시 여부
        show_plots = input(f"플롯 표시 (Y/N) [{'Y' if config['visualization']['show_plots'] else 'N'}]: ").strip().upper()
        if show_plots == 'N':
            config['visualization']['show_plots'] = False
        elif show_plots == 'Y':
            config['visualization']['show_plots'] = True

        print("\n📊 시각화 세부 설정:")

        # 플롯 타입 선택
        available_types = ['boxplot', 'violin', 'boxen']
        current_types = ', '.join(available_types)
        plot_types_input = input(f"플롯 타입 (쉼표로 구분) [{current_types}]: ").strip()
        if plot_types_input:
            config['visualization']['plot_types'] = [t.strip() for t in plot_types_input.split(',')]
        else:
            config['visualization']['plot_types'] = available_types

        # 그래프 크기
        current_size = f"{config['visualization']['figsize'][0]}x{config['visualization']['figsize'][1]}"
        figsize_input = input(f"그래프 크기 (가로x세로) [{current_size}]: ").strip()
        if figsize_input and 'x' in figsize_input:
            try:
                width, height = figsize_input.split('x')
                config['visualization']['figsize'] = [int(width), int(height)]
            except:
                print("⚠️ 잘못된 형식입니다. 기본값을 사용합니다.")

        # 출력 형식
        format_input = input(f"출력 형식 (png/pdf/svg) [{config['output']['format']}]: ").strip().lower()
        if format_input in ['png', 'pdf', 'svg']:
            config['output']['format'] = format_input

        # DPI 설정
        dpi_input = input(f"이미지 해상도 DPI [300]: ").strip()
        if dpi_input.isdigit():
            config['visualization']['dpi'] = int(dpi_input)
        else:
            config['visualization']['dpi'] = 300

        # 고급 시각화 옵션
        print("\n🎨 고급 시각화 옵션:")

        # 이상값 표시
        show_outliers = input(f"이상값 표시 (Y/N) [N]: ").strip().upper()
        config['visualization']['show_outliers'] = (show_outliers == 'Y')

        # 평균값 표시
        show_mean = input(f"평균값 표시 (Y/N) [Y]: ").strip().upper()
        config['visualization']['show_mean'] = (show_mean != 'N')

        # 평균값 숫자 표시
        show_mean_value = input(f"평균값 숫자 표시 (Y/N) [Y]: ").strip().upper()
        config['visualization']['show_mean_value'] = (show_mean_value != 'N')

        # 클리핑 백분위수
        clip_input = input(f"데이터 클리핑 백분위수 (0.9-1.0) [0.95]: ").strip()
        try:
            clip_value = float(clip_input)
            if 0.9 <= clip_value <= 1.0:
                config['visualization']['clip_percentile'] = clip_value
            else:
                config['visualization']['clip_percentile'] = 0.95
        except:
            config['visualization']['clip_percentile'] = 0.95

        # 평균 위치 전략
        mean_strategies = ['adaptive', 'fixed_percentage', 'above_max']
        strategy_input = input(f"평균값 위치 전략 ({'/'.join(mean_strategies)}) [adaptive]: ").strip()
        if strategy_input in mean_strategies:
            config['visualization']['mean_position_strategy'] = strategy_input
        else:
            config['visualization']['mean_position_strategy'] = 'adaptive'

        print(f"\n✅ 설정 완료!")
        print(f"📊 플롯 타입: {', '.join(config['visualization']['plot_types'])}")
        print(f"📐 그래프 크기: {config['visualization']['figsize'][0]}x{config['visualization']['figsize'][1]}")
        print(f"🎯 해상도: {config['visualization']['dpi']} DPI")
        print(f"📁 출력: {config['output']['directory']}")

    except (EOFError, KeyboardInterrupt):
        print("\n⚡ 기본값을 사용합니다.")

    return config


def load_data(config):
    """데이터 로드"""
    csv_path = Path(config['data']['path'])

    try:
        df = pd.read_csv(csv_path)
        print(f"✅ 데이터 로드 성공: {len(df):,}개 행")
        return df
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {csv_path}")
        print("🔧 테스트 데이터를 생성합니다...")

        # 테스트 데이터 생성
        np.random.seed(42)
        n_samples = 10000

        df = pd.DataFrame({
            'zone_type': np.random.choice(['도시', '농어촌'], n_samples),
            'public_total_walking_time_seconds': np.random.normal(450, 180, n_samples),
            'drt_total_walking_time_seconds': np.random.normal(360, 150, n_samples),
            'public_onboard_time_seconds': np.random.normal(900, 300, n_samples),
            'drt_onboard_time_seconds': np.random.normal(810, 270, n_samples),
            'public_waiting_time_seconds': np.random.normal(1350, 450, n_samples),
            'drt_waiting_time_seconds': np.random.normal(1080, 360, n_samples),
            'public_total_time_seconds': np.random.normal(2700, 600, n_samples),
            'drt_total_trip_time_seconds': np.random.normal(2250, 540, n_samples)
        })

        print(f"✅ 테스트 데이터 생성 완료: {len(df):,}개 행")
        return df


def run_analysis(config):
    """분석 실행"""
    print("🚀 셔클 효과 분석 시작...")

    # 출력 디렉토리 생성
    output_dir = Path(config['output']['directory'])
    output_dir.mkdir(exist_ok=True)
    print(f"📁 출력 디렉토리: {output_dir}")

    # 1. 데이터 로드
    df_analysis = load_data(config)

    # 2. 지역별 데이터 전처리
    print("\n📊 데이터 전처리 중...")

    # 도시지역
    df_city = df_analysis[df_analysis.zone_type == "도시"].copy()
    city_processed_data = prepare_data_func(df_city, config['analysis']['time_columns'], config['analysis']['convert_to_minutes'])

    # 농어촌지역
    df_rural = df_analysis[df_analysis.zone_type == "농어촌"].copy()
    rural_processed_data = prepare_data_func(df_rural, config['analysis']['time_columns'], config['analysis']['convert_to_minutes'])

    print(f"✅ 도시지역 처리 완료: {len(df_city):,}개 행")
    print(f"✅ 농어촌지역 처리 완료: {len(df_rural):,}개 행")

    # 3. 시각화 생성
    if config['visualization']['save_plots'] or config['visualization']['show_plots']:
        print("\n🎨 시각화 생성 중...")

        # 시각화 타입별 함수 매핑
        if USE_OPTIMIZED:
            plot_functions = {
                'boxplot': create_grouped_boxplot,
                'violin': create_grouped_violin_plot,
                'boxen': create_grouped_boxen_plot
            }
        else:
            plot_functions = {
                'boxplot': create_grouped_boxplot_basic,
                'violin': create_grouped_boxplot_basic,  # 기본 모드에서는 boxplot만 사용
                'boxen': create_grouped_boxplot_basic
            }

        plot_types = config['visualization'].get('plot_types', ['boxplot', 'violin', 'boxen'])
        regions_data = [("city", city_processed_data, "도시지역"), ("rural", rural_processed_data, "농어촌지역")]

        # DPI 설정
        dpi = config['visualization'].get('dpi', 300)

        for region, processed_data, region_name in regions_data:
            for plot_type in plot_types:
                plot_func = plot_functions.get(plot_type, create_grouped_boxplot_func)

                try:
                    # 시각화 옵션 준비
                    plot_kwargs = {
                        'figsize': tuple(config['visualization']['figsize']),
                        'show_outliers': config['visualization'].get('show_outliers', False),
                        'show_mean': config['visualization'].get('show_mean', True),
                        'show_mean_value': config['visualization'].get('show_mean_value', True),
                        'clip_percentile': config['visualization'].get('clip_percentile', 0.95),
                        'mean_position_strategy': config['visualization'].get('mean_position_strategy', 'adaptive')
                    }

                    fig, _ = plot_func(
                        processed_data,
                        f"셔클 도입 효과 분석({region_name}) - {plot_type.upper()}",
                        "시간(분)",
                        **plot_kwargs
                    )

                    if config['visualization']['save_plots']:
                        plot_path = output_dir / f"{region}_{plot_type}_analysis.{config['output']['format']}"
                        fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                        print(f"💾 {region_name} {plot_type} 저장: {plot_path} (DPI: {dpi})")

                    if config['visualization']['show_plots']:
                        plt.show()
                    else:
                        plt.close(fig)

                except Exception as e:
                    print(f"⚠️ {region_name} {plot_type} 생성 실패: {e}")
                    # 기본 boxplot으로 대체
                    if plot_type != 'boxplot':
                        try:
                            basic_kwargs = {
                                'figsize': tuple(config['visualization']['figsize'])
                            }

                            fig, _ = create_grouped_boxplot_func(
                                processed_data,
                                f"셔클 도입 효과 분석({region_name}) - {plot_type.upper()}",
                                "시간(분)",
                                **basic_kwargs
                            )

                            if config['visualization']['save_plots']:
                                plot_path = output_dir / f"{region}_{plot_type}_analysis.{config['output']['format']}"
                                fig.savefig(plot_path, dpi=dpi, bbox_inches='tight')
                                print(f"💾 {region_name} {plot_type} (기본) 저장: {plot_path}")

                            plt.close(fig)
                        except Exception as fallback_e:
                            print(f"⚠️ {region_name} {plot_type} 기본 생성도 실패: {fallback_e}")

    # 4. 통계 분석
    print("\n📈 통계 분석 수행 중...")

    results = []

    for region, processed_data in [("도시", city_processed_data), ("농어촌", rural_processed_data)]:
        for category, data in processed_data.items():
            if "대중교통" in data.columns and "셔클" in data.columns:
                improvement = calculate_improvement_func(
                    baseline_data=data["대중교통"],
                    improved_data=data["셔클"]
                )

                results.append({
                    '지역': region,
                    '카테고리': category,
                    '대중교통_평균': data["대중교통"].mean(),
                    '셔클_평균': data["셔클"].mean(),
                    '평균_개선량': improvement['평균_개선량'],
                    '평균_개선율': improvement['평균_개선율'],
                    '중앙값_개선량': improvement['중앙값_개선량'],
                    '중앙값_개선율': improvement['중앙값_개선율']
                })

    # 결과를 DataFrame으로 변환하고 저장
    results_df = pd.DataFrame(results)
    results_path = output_dir / "analysis_results.csv"
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"💾 분석 결과 저장: {results_path}")

    # 5. 요약 출력
    print("\n" + "="*60)
    print("🏆 분석 결과 요약")
    print("="*60)
    print(results_df.round(2).to_string(index=False))

    # 6. 데이터 파일 저장
    print("\n📁 데이터 파일 저장 중...")

    # 카테고리명 영어 변환
    category_mapping = {
        "도보시간": "walking_time",
        "탑승시간": "onboard_time",
        "대기시간": "waiting_time",
        "총 이동시간": "total_time"
    }

    for region, processed_data in [("city", city_processed_data), ("rural", rural_processed_data)]:
        for category, data in processed_data.items():
            eng_category = category_mapping.get(category, category)
            filename = f"{eng_category}_{region}.csv"
            data_path = output_dir / filename
            data.to_csv(data_path, index=False, encoding='utf-8-sig')
            print(f"💾 {filename} 저장 완료")

    print(f"\n✅ 모든 분석 완료! 결과는 {output_dir}에 저장되었습니다.")

    # 출력 디렉토리 내용 확인
    print(f"\n📋 저장된 파일 목록:")
    for file_path in sorted(output_dir.glob("*")):
        if file_path.is_file():
            print(f"  • {file_path.name}")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="셔클 효과 분석 도구")
    parser.add_argument("--interactive", action="store_true", help="대화형 모드")

    args = parser.parse_args()

    if args.interactive:
        config = interactive_config()
    else:
        config = get_default_config()
        print("📋 기본 설정으로 분석을 수행합니다.")

    run_analysis(config)


if __name__ == "__main__":
    main()