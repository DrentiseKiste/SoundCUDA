#!/usr/bin/env python3
"""
测试 MplCanvas 修复
"""

import sys
sys.path.insert(0, 'front')

try:
    from Audio_Processing_App import MplCanvas
    
    canvas = MplCanvas(width=5, height=4, dpi=100)
    
    print(f"canvas.fig 属性存在: {hasattr(canvas, 'fig')}")
    print(f"canvas.axes 属性存在: {hasattr(canvas, 'axes')}")
    
    if hasattr(canvas, 'fig'):
        print("✓ MplCanvas 修复成功！")
    else:
        print("✗ MplCanvas 修复失败！")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
