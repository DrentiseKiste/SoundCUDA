"""
测试人声分析的日志功能
"""

import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from VoiceAnalysis import VoiceAnalyzer

def test_logging():
    """测试日志功能"""
    print("=" * 60)
    print("测试 VoiceAnalysis 日志功能")
    print("=" * 60)
    
    # 创建分析器，指定日志目录
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    print(f"\n日志目录: {log_dir}")
    
    # 查找测试音频文件
    test_file = None
    test_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testvoices', 'test_pack', 'Eng'),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testvoices')
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.endswith('.wav'):
                    test_file = os.path.join(test_dir, file)
                    break
        if test_file:
            break
    
    if not test_file:
        print("未找到测试音频文件！")
        return
    
    print(f"测试文件: {test_file}")
    
    # 创建分析器并测试
    print("\n初始化分析器...")
    analyzer = VoiceAnalyzer(use_parallel=True, log_dir=log_dir)
    
    print("\n开始分析音频...")
    result = analyzer.analyze_audio(test_file, generate_report=True)
    
    if result:
        print("\n✓ 分析成功！")
        print(f"  主导情绪: {result['emotion_result']['dominant_emotion']}")
        print(f"  置信度: {result['emotion_result']['confidence']:.2%}")
        print(f"  总耗时: {result['analysis_time']:.2f}秒")
        
        # 导出结果
        print("\n测试导出功能...")
        export_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "test_export.json"
        )
        analyzer.export_results(result, export_file)
        print(f"✓ 结果已导出: {export_file}")
    else:
        print("\n✗ 分析失败！")
    
    print("\n" + "=" * 60)
    print("日志测试完成！")
    print("=" * 60)
    print(f"\n请检查日志目录: {log_dir}")
    print("查看生成的日志文件确认功能正常。")

if __name__ == "__main__":
    test_logging()
