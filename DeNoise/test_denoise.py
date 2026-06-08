"""
简单的降噪测试脚本
"""

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import time

# 测试音频文件
input_file = r"C:\Users\27946\Desktop\Sound\testvoices\20250610yingyu1tingli01.mp3"
output_file = r"C:\Users\27946\Desktop\Sound\testvoices\test_denoised.mp3"

try:
    print("加载音频文件...")
    audio, sr = librosa.load(input_file, sr=16000)
    print(f"音频长度: {len(audio)/sr:.2f}秒")
    
    print("提取噪声样本...")
    noise_sample = audio[:int(sr * 0.5)]  # 使用前0.5秒作为噪声样本
    
    print("开始降噪...")
    start_time = time.time()
    
    denoised_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_sample,
        n_fft=2048,
        hop_length=512,
        time_mask_smooth_ms=120,
        freq_mask_smooth_hz=120,
        stationary=False,
        prop_decrease=0.9
    )
    
    end_time = time.time()
    print(f"降噪完成，耗时: {end_time - start_time:.2f}秒")
    
    print("保存降噪后的音频...")
    sf.write(output_file, denoised_audio, sr)
    print(f"音频已保存至: {output_file}")
    
    print("测试成功！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
