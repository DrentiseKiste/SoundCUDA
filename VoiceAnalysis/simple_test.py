"""
最简单的测试脚本
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("开始导入VoiceAnalysis模块...")

try:
    from VoiceAnalysis.voice_analyzer import VoiceAnalyzer
    print("✓ VoiceAnalyzer导入成功")

    analyzer = VoiceAnalyzer(use_parallel=False)
    print("✓ 分析器初始化成功")

    test_file = os.path.join(
        os.path.dirname(__file__),
        '..',
        'testvoices',
        'test_pack',
        'Eng',
        'OSR_us_000_0010_8k.wav'
    )

    print(f"测试文件: {test_file}")
    print(f"文件存在: {os.path.exists(test_file)}")

    if os.path.exists(test_file):
        print("\n开始分析...")
        result = analyzer.analyze_audio(test_file, generate_report=True)

        if result:
            print("\n✓ 分析成功！")
            print(analyzer.get_feature_summary(result))
        else:
            print("\n✗ 分析失败")
    else:
        print("测试文件不存在")

except Exception as e:
    print(f"\n✗ 发生错误: {e}")
    import traceback
    traceback.print_exc()
