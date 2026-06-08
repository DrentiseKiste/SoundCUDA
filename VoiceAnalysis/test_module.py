"""
VoiceAnalysis 简单测试脚本
"""

import os
import sys
import traceback

print("=" * 60)
print("VoiceAnalysis 模块测试")
print("=" * 60)

try:
    # 测试导入
    print("\n[测试1] 导入模块...")
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from VoiceAnalysis import VoiceAnalyzer, AcousticFeatureExtractor, EmotionAnalyzer
    print("✓ 导入成功")
    
    # 测试初始化
    print("\n[测试2] 初始化分析器...")
    analyzer = VoiceAnalyzer(use_parallel=True)
    print("✓ 初始化成功")
    
    # 测试音频文件
    test_dir = os.path.join(os.path.dirname(__file__), '..', 'testvoices', 'test_pack', 'Eng')
    audio_file = os.path.join(test_dir, 'OSR_us_000_0010_8k.wav')
    
    if not os.path.exists(audio_file):
        print(f"\n[警告] 测试文件不存在: {audio_file}")
        print("搜索其他测试文件...")
        
        # 搜索其他wav文件
        test_dir_parent = os.path.join(os.path.dirname(__file__), '..', 'testvoices')
        for root, dirs, files in os.walk(test_dir_parent):
            for file in files:
                if file.endswith('.wav'):
                    audio_file = os.path.join(root, file)
                    print(f"找到测试文件: {audio_file}")
                    break
            if os.path.exists(audio_file):
                break
    
    if os.path.exists(audio_file):
        # 测试音频分析
        print(f"\n[测试3] 分析音频: {os.path.basename(audio_file)}")
        result = analyzer.analyze_audio(audio_file, generate_report=True)
        
        if result:
            print("\n✓ 分析完成！")
            
            # 显示摘要
            print("\n" + "=" * 60)
            print("分析结果摘要")
            print("=" * 60)
            print(analyzer.get_feature_summary(result))
            
            # 导出结果
            print("\n[测试4] 导出结果...")
            output_file = analyzer.export_results(result)
            print(f"✓ 结果已导出: {output_file}")
            
            # 性能对比
            print("\n[测试5] 并行性能对比...")
            performance = analyzer.compare_parallel_performance(audio_file)
            print(f"\n性能对比结果:")
            print(f"  串行时间: {performance['serial_time']:.2f}秒")
            print(f"  并行时间: {performance['parallel_time']:.2f}秒")
            print(f"  加速比: {performance['speedup']:.2f}x")
            print(f"  并行效率: {performance['efficiency']:.2%}")
            
        else:
            print("\n✗ 分析失败")
    else:
        print(f"\n✗ 未找到测试音频文件")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗ 错误: {e}")
    print("\n详细错误信息:")
    traceback.print_exc()
