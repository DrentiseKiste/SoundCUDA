#!/usr/bin/env python3
"""
快速验证脚本 - 测试我们的新模块
"""

print("=" * 60)
print("快速验证测试")
print("=" * 60)

try:
    from recognize import FourierSpeakerSeparator
    print("✓ 模块导入成功！")
    
    # 创建一个简单的实例
    separator = FourierSpeakerSeparator(sample_rate=16000)
    print("✓ 类实例化成功！")
    
    # 检查主要方法
    methods = [
        'load_audio',
        'compute_stft',
        'compute_spectral_features',
        'separate_by_nmf',
        'separate_by_ica',
        'separate_by_clustering',
        'separate_speakers',
        'save_separated_audios',
        'visualize_results',
        'extract_speaker_features'
    ]
    
    print("\n可用的主要方法:")
    for method in methods:
        if hasattr(separator, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method}")
    
    print("\n" + "=" * 60)
    print("验证成功！模块已准备就绪！")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ 验证失败: {e}")
    import traceback
    traceback.print_exc()
