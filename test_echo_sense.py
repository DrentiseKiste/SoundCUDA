#!/usr/bin/env python3
"""
测试 EchoSense 应用
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 测试导入
try:
    from front.Audio_Processing_App import MainWindow
    print("✓ 应用模块导入成功！")
    
    # 检查标题是否正确
    import inspect
    source = inspect.getsource(MainWindow.__init__)
    if 'EchoSense' in source:
        print("✓ 窗口标题已修改为 'EchoSense'")
    else:
        print("✗ 窗口标题未修改")
    
    # 检查图标设置
    if 'QIcon' in source and 'EchoSense.jpg' in source:
        print("✓ 图标设置已添加")
    else:
        print("✗ 图标设置未添加")
        
    print("\n修改内容:")
    print("1. 窗口标题: '音频处理综合应用 v2.0' -> 'EchoSense'")
    print("2. 添加图标设置: 使用 EchoSense.jpg")
    
except Exception as e:
    print(f"✗ 测试失败: {e}")
    import traceback
    traceback.print_exc()
