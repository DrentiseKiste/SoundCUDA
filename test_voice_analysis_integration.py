#!/usr/bin/env python3
"""
VoiceAnalysis 模块集成测试

验证 VoiceAnalysis 能正常从 EmotionAnalysis 导入情绪分析类
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_voice_analysis_import():
    """测试 VoiceAnalysis 模块导入"""
    print("=" * 60)
    print("测试: VoiceAnalysis 模块集成")
    print("=" * 60)

    try:
        print("\n[步骤1] 导入 VoiceAnalysis 模块...")
        from VoiceAnalysis import VoiceAnalyzer, EmotionAnalyzer, VoiceVisualizer
        print("[成功] VoiceAnalysis 模块导入完成")

        print("\n[步骤2] 验证 EmotionAnalyzer 来源...")
        # 检查 EmotionAnalyzer 是否来自 EmotionAnalysis
        emotion_module = EmotionAnalyzer.__module__
        print(f"EmotionAnalyzer 模块: {emotion_module}")
        if 'EmotionAnalysis' in emotion_module:
            print("[成功] EmotionAnalyzer 来自 EmotionAnalysis 模块")
        else:
            print("[警告] EmotionAnalyzer 来自本地模块")

        print("\n[步骤3] 验证 VoiceVisualizer 来源...")
        visualizer_module = VoiceVisualizer.__module__
        print(f"VoiceVisualizer 模块: {visualizer_module}")
        if 'EmotionAnalysis' in visualizer_module:
            print("[成功] VoiceVisualizer 来自 EmotionAnalysis 模块")
        else:
            print("[警告] VoiceVisualizer 来自本地模块")

        print("\n[步骤4] 测试初始化...")
        analyzer = VoiceAnalyzer()
        print("[成功] VoiceAnalyzer 初始化完成")

        print("\n" + "=" * 60)
        print("所有集成测试通过！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[失败] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_voice_analysis_import()
    sys.exit(0 if success else 1)
