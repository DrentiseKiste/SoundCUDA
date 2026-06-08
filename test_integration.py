#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合集成测试 - 验证所有模块正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_module_imports():
    """测试所有模块导入"""
    print("=" * 70)
    print("测试 1: 模块导入测试")
    print("=" * 70)
    
    modules_to_test = [
        ("DeNoise.denoiser", "Denoiser"),
        ("Gender.GPU_GenderRecognition", "GPUGenderAnalyzer"),
        ("recognize.enhanced_speaker_diarization", "SpeakerDiarization"),
        ("FourierSeparation.fourier_speaker_separator", "FourierSpeakerSeparator"),
        ("VoiceAnalysis.voice_analyzer", "VoiceAnalyzer"),
        ("EmotionAnalysis.emotion_analyzer", "EmotionAnalyzer"),
        ("EmotionAnalysis.emotion_visualizer", "EmotionVisualizer")
    ]
    
    all_passed = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"[OK] 成功导入: {module_name}.{class_name}")
        except Exception as e:
            print(f"[FAIL] 导入失败: {module_name}.{class_name} - {e}")
            all_passed = False
    
    return all_passed


def test_recognize_backwards_compatibility():
    """测试 recognize 模块的向后兼容性"""
    print("\n" + "=" * 70)
    print("测试 2: recognize 模块向后兼容性")
    print("=" * 70)
    
    try:
        from recognize import FourierSpeakerSeparator
        print("[OK] recognize.FourierSpeakerSeparator 导入成功 (向后兼容)")
        return True
    except Exception as e:
        print(f"[FAIL] recognize.FourierSpeakerSeparator 导入失败 - {e}")
        return False


def test_voiceanalysis_backwards_compatibility():
    """测试 VoiceAnalysis 模块的向后兼容性"""
    print("\n" + "=" * 70)
    print("测试 3: VoiceAnalysis 模块向后兼容性")
    print("=" * 70)
    
    try:
        from VoiceAnalysis import VoiceAnalyzer, EmotionAnalyzer
        print("[OK] VoiceAnalysis.VoiceAnalyzer 导入成功")
        print("[OK] VoiceAnalysis.EmotionAnalyzer 导入成功 (来自 EmotionAnalysis 模块)")
        return True
    except Exception as e:
        print(f"[FAIL] VoiceAnalysis 模块导入失败 - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_project_structure():
    """测试项目结构"""
    print("\n" + "=" * 70)
    print("测试 4: 项目结构检查")
    print("=" * 70)
    
    required_dirs = [
        "DeNoise",
        "Gender",
        "VoiceAnalysis",
        "EmotionAnalysis",
        "FourierSeparation",
        "recognize",
        "front",
        "testvoices"
    ]
    
    all_passed = True
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"[OK] 存在目录: {dir_name}")
        else:
            print(f"[FAIL] 缺少目录: {dir_name}")
            all_passed = False
    
    return all_passed


def main():
    """主测试函数"""
    print("\n")
    print("+" + "-" * 68 + "+")
    print("|" + " " * 23 + "项目整合测试" + " " * 31 + "|")
    print("+" + "-" * 68 + "+")
    
    results = []
    
    results.append(("模块导入", test_module_imports()))
    results.append(("recognize 模块兼容性", test_recognize_backwards_compatibility()))
    results.append(("VoiceAnalysis 模块兼容性", test_voiceanalysis_backwards_compatibility()))
    results.append(("项目结构检查", test_project_structure()))
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results:
        status = "通过" if result else "失败"
        print(f"  {test_name:<40} {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("所有测试通过！项目结构整理完成！")
    else:
        print("部分测试失败，请检查错误信息")
    print("=" * 70)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
