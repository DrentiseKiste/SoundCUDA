#!/usr/bin/env python3
"""
情绪分析模块测试脚本

功能：
- 测试 EmotionAnalyzer 初始化
- 测试 EmotionVisualizer 初始化
- 验证模块可以正常导入
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_emotion_analyzer():
    """测试情绪分析器"""
    print("=" * 60)
    print("测试 1: EmotionAnalyzer 初始化")
    print("=" * 60)
    
    try:
        from EmotionAnalysis.emotion_analyzer import EmotionAnalyzer
        analyzer = EmotionAnalyzer()
        print(f"[成功] EmotionAnalyzer 初始化完成")
        print(f"支持的情绪类别: {analyzer.emotion_categories}")
        return True
    except Exception as e:
        print(f"[失败] EmotionAnalyzer 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_emotion_visualizer():
    """测试情绪可视化器"""
    print("\n" + "=" * 60)
    print("测试 2: EmotionVisualizer 初始化")
    print("=" * 60)
    
    try:
        from EmotionAnalysis.emotion_visualizer import EmotionVisualizer
        visualizer = EmotionVisualizer()
        print(f"[成功] EmotionVisualizer 初始化完成")
        print(f"情绪颜色映射: {visualizer.emotion_colors}")
        return True
    except Exception as e:
        print(f"[失败] EmotionVisualizer 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_imports():
    """测试模块导入"""
    print("\n" + "=" * 60)
    print("测试 3: 模块导入")
    print("=" * 60)
    
    try:
        # 测试从包中导入
        from EmotionAnalysis import EmotionAnalyzer, EmotionVisualizer
        print(f"[成功] 从 EmotionAnalysis 包中导入类")
        
        # 测试版本号
        import EmotionAnalysis
        print(f"[成功] EmotionAnalysis 模块版本: {EmotionAnalysis.__version__}")
        
        return True
    except Exception as e:
        print(f"[失败] 模块导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("EmotionAnalysis 模块测试")
    print("=" * 60)
    
    results = []
    results.append(("EmotionAnalyzer 初始化", test_emotion_analyzer()))
    results.append(("EmotionVisualizer 初始化", test_emotion_visualizer()))
    results.append(("模块导入测试", test_module_imports()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
