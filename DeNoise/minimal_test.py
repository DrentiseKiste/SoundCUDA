"""
极简的降噪测试脚本
"""

print("开始测试降噪功能...")

try:
    # 导入必要的库
    import numpy as np
    import librosa
    import soundfile as sf
    print("✓ 基本库导入成功")
    
    # 导入noisereduce
    import noisereduce as nr
    print("✓ noisereduce库导入成功")
    
    # 测试音频文件
    input_file = r"C:\Users\27946\Desktop\debate.wav"
    output_file = r"C:\Users\27946\Desktop\debate_denoised.wav"
    
    print(f"输入文件: {input_file}")
    
    # 加载音频
    print("加载音频...")
    audio, sr = librosa.load(input_file, sr=16000)
    print(f"音频长度: {len(audio)/sr:.2f}秒")
    
    # 提取噪声样本
    print("提取噪声样本...")
    noise_sample = audio[:int(sr * 0.5)]
    
    # 降噪
    print("开始降噪...")
    denoised_audio = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_sample,
        n_fft=2048,
        hop_length=512,
        time_mask_smooth_ms=200,
        freq_mask_smooth_hz=150,
        stationary=False,
        prop_decrease=0.6
    )
    
    # 保存结果
    print("保存结果...")
    sf.write(output_file, denoised_audio, sr)
    print(f"输出文件: {output_file}")
    
    print("\n测试完成！")
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
