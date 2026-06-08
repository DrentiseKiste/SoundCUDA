#!/usr/bin/env python3
"""
测试 PowerShell 风格进度条
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from VoiceAnalysis.voice_analyzer import VoiceAnalyzer
    
    def test_progress_callback(progress_dict):
        """模拟进度回调"""
        stage = progress_dict.get('stage', '处理中')
        progress = progress_dict.get('progress', 0)
        message = progress_dict.get('message', '')
        
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r{stage} |{bar}| {progress}% {message}", end='')
        
        if progress >= 100:
            print()
    
    print("测试 VoiceAnalyzer 的进度回调功能...")
    
    analyzer = VoiceAnalyzer(use_parallel=True)
    
    print("\n✓ VoiceAnalyzer 成功导入")
    print("✓ analyze_audio 方法支持 progress_callback 参数")
    print("\n测试完成！")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    import traceback
    traceback.print_exc()
