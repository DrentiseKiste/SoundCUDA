"""
简单的降噪测试脚本
"""

import os
import numpy as np
import librosa
import soundfile as sf
from denoiser import Denoiser

def test_denoise():
    print("开始测试降噪功能...")
    
    # 测试文件
    input_file = r"C:\Users\27946\Desktop\debate.wav"
    output_file = r"C:\Users\27946\Desktop\debate_denoised.wav"
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 文件不存在: {input_file}")
        return
    
    # 创建降噪器
    denoiser = Denoiser(algorithm='advanced')
    
    try:
        # 加载音频
        print(f"加载音频文件: {input_file}")
        audio, sr = denoiser.load_audio(input_file)
        
        # 降噪
        print("开始降噪处理...")
        denoised_audio = denoiser.denoise(audio, sr)
        
        # 保存结果
        print(f"保存降噪结果到: {output_file}")
        denoiser.save_audio(denoised_audio, sr, output_file)
        
        # 分析结果
        print("\n分析降噪效果:")
        
        # 计算能量保留率
        energy_original = np.sum(np.abs(audio) ** 2)
        energy_denoised = np.sum(np.abs(denoised_audio) ** 2)
        energy_ratio = energy_denoised / energy_original
        print(f"能量保留率: {energy_ratio:.2f}")
        
        # 检测断续情况
        frame_size = 512
        energy_frames_original = np.array([np.sum(np.abs(audio[i:i+frame_size])**2) 
                                         for i in range(0, len(audio)-frame_size, frame_size)])
        energy_frames_denoised = np.array([np.sum(np.abs(denoised_audio[i:i+frame_size])**2) 
                                          for i in range(0, len(denoised_audio)-frame_size, frame_size)])
        
        energy_change_original = np.std(energy_frames_original) / np.mean(energy_frames_original)
        energy_change_denoised = np.std(energy_frames_denoised) / np.mean(energy_frames_denoised)
        
        print(f"原始音频能量变化率: {energy_change_original:.2f}")
        print(f"降噪后音频能量变化率: {energy_change_denoised:.2f}")
        
        if energy_change_denoised > energy_change_original * 1.5:
            print("⚠️  警告: 可能存在严重的断续现象")
        elif energy_change_denoised > energy_change_original * 1.2:
            print("⚠️  注意: 可能存在轻微的断续现象")
        else:
            print("✅ 良好: 未检测到明显的断续现象")
        
        print("\n测试完成！")
        print(f"原始文件: {input_file}")
        print(f"降噪后文件: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_denoise()
