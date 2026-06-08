#!/usr/bin/env python3
"""
测试 EmotionVisualizer 的所有方法
"""

import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from EmotionAnalysis.emotion_visualizer import EmotionVisualizer
    print("✓ 成功导入 EmotionVisualizer")
    
    # 创建可视化器
    visualizer = EmotionVisualizer(output_dir='.')
    
    # 创建测试数据 - 声学特征
    acoustic_features = {
        'f0': {
            'mean_f0': 150.0,
            'min_f0': 80.0,
            'max_f0': 220.0,
            'std_f0': 30.0,
            'range_f0': 140.0
        },
        'formants': {
            'f1': 500.0,
            'f2': 1500.0,
            'f3': 2500.0,
            'f4': 3500.0
        },
        'energy': {
            'rms_mean': 0.1,
            'rms_max': 0.3,
            'rms_min': 0.01,
            'total_energy': 100.0
        },
        'quality': {
            'hnr': 25.0,
            'jitter': 1.5,
            'shimmer': 2.0
        },
        'spectral': {
            'spectral_centroid_mean': 1500.0,
            'spectral_bandwidth_mean': 2000.0,
            'spectral_rolloff_85_mean': 3000.0
        },
        'temporal': {
            'voiced_ratio': 0.7,
            'num_voiced_segments': 10,
            'mean_segment_duration': 0.5
        },
        'mfcc': {
            f'mfcc_{i}_mean': float(i * 0.5) for i in range(13)
        }
    }
    
    # 创建测试数据 - 情绪结果
    emotion_result = {
        'dominant_emotion': 'happy',
        'confidence': 0.85,
        'intensity': 0.7,
        'secondary_emotion': 'surprise',
        'secondary_confidence': 0.6,
        'emotion_scores': {
            'happy': 0.85,
            'sad': 0.1,
            'angry': 0.05,
            'neutral': 0.5,
            'fear': 0.02,
            'surprise': 0.6,
            'disgust': 0.01
        }
    }
    
    # 创建测试数据 - 时间线结果
    timeline_result = {
        'dominant_overall': 'happy',
        'distribution': {'happy': 0.6, 'neutral': 0.3, 'sad': 0.1},
        'num_changes': 3,
        'timeline': [
            {'dominant_emotion': 'happy', 'confidence': 0.9, 'intensity': 0.8, 'emotion_scores': emotion_result['emotion_scores']},
            {'dominant_emotion': 'neutral', 'confidence': 0.7, 'intensity': 0.5, 'emotion_scores': emotion_result['emotion_scores']},
            {'dominant_emotion': 'happy', 'confidence': 0.85, 'intensity': 0.75, 'emotion_scores': emotion_result['emotion_scores']}
        ]
    }
    
    # 测试 plot_acoustic_features
    print("\n测试 plot_acoustic_features...")
    try:
        path = visualizer.plot_acoustic_features(acoustic_features, 'test_audio.wav')
        print(f"✓ plot_acoustic_features 成功: {path}")
    except Exception as e:
        print(f"✗ plot_acoustic_features 失败: {e}")
    
    # 测试 plot_emotion_analysis (别名)
    print("\n测试 plot_emotion_analysis...")
    try:
        path = visualizer.plot_emotion_analysis(emotion_result, 'test_audio.wav')
        print(f"✓ plot_emotion_analysis 成功: {path}")
    except Exception as e:
        print(f"✗ plot_emotion_analysis 失败: {e}")
    
    # 测试 generate_comprehensive_report
    print("\n测试 generate_comprehensive_report...")
    try:
        path = visualizer.generate_comprehensive_report(
            acoustic_features,
            emotion_result,
            timeline_result,
            'test_audio.wav',
            {'serial_time': 2.5, 'parallel_time': 1.0, 'speedup': 2.5}
        )
        print(f"✓ generate_comprehensive_report 成功: {path}")
    except Exception as e:
        print(f"✗ generate_comprehensive_report 失败: {e}")
    
    # 测试 plot_emotion_summary
    print("\n测试 plot_emotion_summary...")
    try:
        path = visualizer.plot_emotion_summary(emotion_result, 'test_audio.wav')
        print(f"✓ plot_emotion_summary 成功: {path}")
    except Exception as e:
        print(f"✗ plot_emotion_summary 失败: {e}")
    
    # 测试 plot_emotion_distribution
    print("\n测试 plot_emotion_distribution...")
    try:
        path = visualizer.plot_emotion_distribution(emotion_result['emotion_scores'], 'test_audio.wav')
        print(f"✓ plot_emotion_distribution 成功: {path}")
    except Exception as e:
        print(f"✗ plot_emotion_distribution 失败: {e}")
    
    # 测试 plot_emotion_timeline
    print("\n测试 plot_emotion_timeline...")
    try:
        path = visualizer.plot_emotion_timeline(timeline_result, 'test_audio.wav')
        print(f"✓ plot_emotion_timeline 成功: {path}")
    except Exception as e:
        print(f"✗ plot_emotion_timeline 失败: {e}")
    
    print("\n" + "="*60)
    print("所有可视化测试完成！")
    print("="*60)
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
