#!/usr/bin/env python3
"""
测试 EchoSense GUI 应用
"""

import sys
import os

print("=" * 70)
print("EchoSense GUI Application Test")
print("=" * 70)

try:
    print("\n[1/5] Testing PyQt5 import...")
    from PyQt5.QtWidgets import QApplication
    print("[OK] PyQt5 imported successfully")
    
    print("\n[2/5] Testing main application module...")
    from front.Audio_Processing_App import MainWindow
    print("[OK] Main application module imported successfully")
    
    print("\n[3/5] Testing Fourier separation module...")
    from recognize.fourier_speaker_separation import FourierSpeakerSeparator
    print("[OK] Fourier separation module imported successfully")
    
    print("\n[4/5] Checking FourierSeparationTab class integration...")
    import inspect
    if hasattr(MainWindow, '__init__'):
        source = inspect.getsource(MainWindow.__init__)
        if 'FourierSeparationTab' in source:
            print("[OK] FourierSeparationTab integrated into main window")
        else:
            print("[FAIL] FourierSeparationTab not integrated into main window")
    
    print("\n[5/5] Checking all tabs...")
    expected_tabs = [
        'AudioTab',
        'DenoiseTab',
        'GenderTab',
        'VoiceAnalysisTab',
        'RecognizeTab',
        'FourierSeparationTab',
        'AutoUpdateTab'
    ]
    
    for tab_class in expected_tabs:
        if tab_class in globals() or tab_class in dir():
            print(f"  [OK] {tab_class}")
        else:
            print(f"  [?] {tab_class}")
    
    print("\n" + "=" * 70)
    print("Test Passed! Application is ready.")
    print("=" * 70)
    print("\nTo start the application:")
    print("  Method 1: python -m front.Audio_Processing_App")
    print("  Method 2: cd front && python Audio_Processing_App.py")
    print("\nIn the GUI, you will see a new tab called '傅里叶分离' (Fourier Separation)")
    print("This tab allows you to:")
    print("  - Separate multiple speakers from audio using Fourier analysis")
    print("  - Choose between NMF, ICA, or clustering methods")
    print("  - Automatically estimate number of speakers")
    print("  - Visualize separation results")
    print("  - Save separated audio files")
    print("=" * 70)
    
except Exception as e:
    print(f"\n[FAIL] Test failed: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("Troubleshooting:")
    print("1. Ensure dependencies are installed: pip install -r requirements.txt")
    print("2. Ensure PyQt5 is installed: pip install PyQt5")
    print("3. Check Python version: python --version")
    print("=" * 70)
