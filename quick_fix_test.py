#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""快速验证修复"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试 VoiceAnalyzer 导入...")
    from VoiceAnalysis import VoiceAnalyzer
    print("[OK] VoiceAnalyzer 导入成功")

    print("\n测试 VoiceAnalyzer 初始化...")
    analyzer = VoiceAnalyzer()
    print("[OK] VoiceAnalyzer 初始化成功")

    print("\n测试 GUI 模块导入...")
    from front.Audio_Processing_App import MainWindow
    print("[OK] GUI 模块导入成功")

    print("\n" + "=" * 50)
    print("所有测试通过！修复成功！")
    print("=" * 50)

except Exception as e:
    print(f"[FAIL] 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
