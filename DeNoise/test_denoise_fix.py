"""
测试降噪功能
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from DeNoise.denoiser import Denoiser
import librosa
import numpy as np

print("=" * 60)
print("测试降噪功能")
print("=" * 60)

# 创建降噪器
denoiser = Denoiser(algorithm='advanced')

# 查找测试音频文件
test_file = None
test_dirs = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testvoices', 'test_pack', 'Eng'),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'testvoices')
]

for test_dir in test_dirs:
    if os.path.exists(test_dir):
        for file in os.listdir(test_dir):
            if file.endswith('.wav'):
                test_file = os.path.join(test_dir, file)
                break
    if test_file:
        break

if not test_file:
    print("未找到测试音频文件！")
    sys.exit(1)

print(f"测试文件: {test_file}")

# 加载音频
audio, sr = librosa.load(test_file, sr=16000)
print(f"音频长度: {len(audio)/sr:.2f}秒")

# 测试降噪
print("\n开始降噪...")
try:
    denoised_audio = denoiser.denoise(audio, sr)
    print(f"降噪成功！降噪后长度: {len(denoised_audio)/sr:.2f}秒")
    
    # 计算降噪效果
    original_energy = np.mean(np.abs(audio))
    denoised_energy = np.mean(np.abs(denoised_audio))
    reduction_ratio = (original_energy - denoised_energy) / original_energy * 100
    
    print(f"\n降噪效果:")
    print(f"  原始能量: {original_energy:.4f}")
    print(f"  降噪后能量: {denoised_energy:.4f}")
    print(f"  能量减少: {reduction_ratio:.1f}%")
    
    print("\n" + "=" * 60)
    print("测试成功！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()