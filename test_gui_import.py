#!/usr/bin/env python3
"""
测试 GUI 应用能否正常启动
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("测试导入 Audio_Processing_App...")
    
    from front.Audio_Processing_App import (
        MainWindow,
        VoiceAnalysisWorkerThread,
        VoiceAnalysisTab,
        MplCanvas
    )
    
    print("✓ MainWindow")
    print("✓ VoiceAnalysisWorkerThread")
    print("✓ VoiceAnalysisTab")
    print("✓ MplCanvas")
    
    print("\n✓ 所有组件导入成功！")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
