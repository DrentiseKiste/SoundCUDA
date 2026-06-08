#!/usr/bin/env python3
"""
测试修复后的人声分析功能
"""

import numpy as np
import librosa
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from VoiceAnalysis.acoustic_features import AcousticFeatureExtractor
    print("✓ 成功导入 AcousticFeatureExtractor")
    
    # 创建测试音频 - 一个简单的正弦波
    print("\n创建测试音频...")
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # 生成一个复合信号（基频 + 共振峰）
    audio = np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)
    audio = librosa.util.normalize(audio)
    
    # 创建特征提取器
    print("创建特征提取器...")
    extractor = AcousticFeatureExtractor(sr=sr)
    
    # 测试共振峰提取
    print("\n测试共振峰提取...")
    try:
        formant_features = extractor.extract_formant_features(audio, sr)
        print(f"✓ 共振峰特征提取成功:")
        for key, val in formant_features.items():
            print(f"  {key}: {val}")
    except Exception as e:
        print(f"✗ 共振峰提取失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试基频提取
    print("\n测试基频提取...")
    try:
        f0_features = extractor.extract_f0_features(audio, sr)
        print(f"✓ 基频特征提取成功")
    except Exception as e:
        print(f"✗ 基频提取失败: {e}")
    
    # 测试所有特征提取
    print("\n测试所有特征提取...")
    try:
        all_features = extractor.extract_all_features(audio, sr)
        print(f"✓ 所有特征提取成功")
        print(f"  特征类别: {list(all_features.keys())}")
        
        # 检查是否有 NaN/Inf
        has_nan = False
        for category, features in all_features.items():
            for key, val in features.items():
                if isinstance(val, float):
                    if np.isnan(val) or np.isinf(val):
                        print(f"  ⚠ {category}.{key}: {val} (无效值)")
                        has_nan = True
        
        if not has_nan:
            print("  ✓ 所有值都是有效数值（无 NaN/Inf）")
    except Exception as e:
        print(f"✗ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
except Exception as e:
    print(f"✗ 导入模块失败: {e}")
    import traceback
    traceback.print_exc()
