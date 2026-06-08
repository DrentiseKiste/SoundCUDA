"""
人声分析示例脚本

演示如何使用VoiceAnalysis模块进行人声分析
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VoiceAnalysis import VoiceAnalyzer, AcousticFeatureExtractor, EmotionAnalyzer, ParallelProcessor


def example_basic_analysis():
    """基础分析示例"""
    print("\n" + "=" * 60)
    print("示例1: 基础人声分析")
    print("=" * 60)
    
    analyzer = VoiceAnalyzer(use_parallel=True)
    
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'testvoices', 'test_pack', 'Eng', 'OSR_us_000_0010_8k.wav'
    )
    
    if os.path.exists(test_file):
        result = analyzer.analyze_audio(test_file, generate_report=True)
        
        if result:
            print(analyzer.get_feature_summary(result))
            
            analyzer.export_results(result)
    else:
        print(f"测试文件不存在: {test_file}")
        print("请修改测试文件路径")


def example_feature_extraction():
    """单独特征提取示例"""
    print("\n" + "=" * 60)
    print("示例2: 单独声学特征提取")
    print("=" * 60)
    
    import librosa
    
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'testvoices', 'test_pack', 'Eng', 'OSR_us_000_0010_8k.wav'
    )
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    wav, sr = librosa.load(test_file, sr=None)
    
    extractor = AcousticFeatureExtractor()
    
    features = extractor.extract_all_features(wav[:int(sr*3)], sr)
    
    print("\n基频特征:")
    for key, value in features['f0'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\n共振峰特征:")
    for key, value in features['formants'].items():
        print(f"  {key}: {value:.2f}")
    
    print("\n频谱特征:")
    for key, value in features['spectral'].items():
        print(f"  {key}: {value:.2f}")


def example_emotion_analysis():
    """单独情绪分析示例"""
    print("\n" + "=" * 60)
    print("示例3: 单独情绪分析")
    print("=" * 60)
    
    import librosa
    
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'testvoices', 'test_pack', 'Eng', 'OSR_us_000_0010_8k.wav'
    )
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    wav, sr = librosa.load(test_file, sr=None)
    
    extractor = AcousticFeatureExtractor()
    emotion_analyzer = EmotionAnalyzer()
    
    features = extractor.extract_all_features(wav[:int(sr*3)], sr)
    
    emotion_result = emotion_analyzer.analyze_emotion(features)
    
    print("\n情绪分析结果:")
    print(f"  主导情绪: {emotion_result['dominant_emotion']}")
    print(f"  置信度: {emotion_result['confidence']:.2%}")
    print(f"  强度: {emotion_result['intensity']:.2%}")
    print(f"  次要情绪: {emotion_result['secondary_emotion']}")
    
    print("\n各情绪得分:")
    for emotion, score in emotion_result['emotion_scores'].items():
        print(f"  {emotion}: {score:.2%}")
    
    print("\n情绪指示器:")
    for indicator, value in emotion_result['emotion_indicators'].items():
        print(f"  {indicator}: {value}")


def example_parallel_performance():
    """并行性能对比示例"""
    print("\n" + "=" * 60)
    print("示例4: 并行性能对比")
    print("=" * 60)
    
    test_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'testvoices', 'test_pack', 'Eng', 'OSR_us_000_0010_8k.wav'
    )
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return
    
    analyzer = VoiceAnalyzer(use_parallel=True)
    
    performance = analyzer.compare_parallel_performance(test_file)
    
    print("\n性能对比结果:")
    print(f"  串行时间: {performance['serial_time']:.2f}秒")
    print(f"  并行时间: {performance['parallel_time']:.2f}秒")
    print(f"  加速比: {performance['speedup']:.2f}x")
    print(f"  并行效率: {performance['efficiency']:.2%}")


def example_batch_analysis():
    """批量分析示例"""
    print("\n" + "=" * 60)
    print("示例5: 批量分析")
    print("=" * 60)
    
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'testvoices', 'test_pack', 'Eng'
    )
    
    if not os.path.exists(test_dir):
        print(f"测试目录不存在: {test_dir}")
        return
    
    audio_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')][:5]
    
    if len(audio_files) == 0:
        print("未找到音频文件")
        return
    
    audio_files = [os.path.join(test_dir, f) for f in audio_files]
    
    print(f"找到 {len(audio_files)} 个音频文件")
    
    analyzer = VoiceAnalyzer(use_parallel=True)
    
    results = analyzer.analyze_batch(audio_files, generate_reports=False)
    
    print("\n批量分析结果摘要:")
    for i, result in enumerate(results):
        if result:
            print(f"\n文件 {i+1}: {os.path.basename(result['audio_file'])}")
            print(f"  时长: {result['duration']:.2f}秒")
            print(f"  主导情绪: {result['emotion_result']['dominant_emotion']}")
            print(f"  分析时间: {result['analysis_time']:.2f}秒")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("VoiceAnalysis 模块使用示例")
    print("=" * 60)
    
    examples = [
        ("基础人声分析", example_basic_analysis),
        ("声学特征提取", example_feature_extraction),
        ("情绪分析", example_emotion_analysis),
        ("并行性能对比", example_parallel_performance),
        ("批量分析", example_batch_analysis)
    ]
    
    print("\n可用示例:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n运行所有示例...")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n示例 '{name}' 执行失败: {e}")


if __name__ == "__main__":
    main()