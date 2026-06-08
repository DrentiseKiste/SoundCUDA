"""
音频降噪主模块
功能：整合多种降噪算法，提供统一的接口

支持的算法：
1. 谱减法（传统算法）
2. 基于noisereduce的高级降噪（推荐）
3. 基于FRCRN的深度学习降噪（强力）
"""

import numpy as np
import librosa
import time
from concurrent.futures import ThreadPoolExecutor

class Denoiser:
    """音频降噪器"""
    def __init__(self, algorithm='advanced'):
        """初始化降噪器
        
        Args:
            algorithm: 降噪算法选择
                'basic': 谱减法（传统算法）
                'advanced': noisereduce高级降噪（推荐）
                'deep': FRCRN深度学习降噪（强力）
        """
        self.algorithm = algorithm
        self.sr = 16000
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        
        # 加载相应的降噪模块
        if algorithm == 'advanced':
            try:
                import noisereduce as nr
                self.nr = nr
                self.use_nr = True
                print("✓ 已加载noisereduce高级降噪模块")
            except ImportError:
                print("✗ noisereduce库未安装，使用谱减法")
                self.use_nr = False
        elif algorithm == 'deep':
            try:
                import torch
                self.torch = torch
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print("✓ 已加载深度学习降噪模块")
            except ImportError:
                print("✗ torch库未安装，使用谱减法")
                self.algorithm = 'basic'
        
    def load_audio(self, audio_file):
        """加载音频文件"""
        print(f"正在加载音频: {audio_file}")
        y, sr = librosa.load(audio_file, sr=self.sr)
        print(f"音频长度: {len(y)/sr:.2f}秒, 采样率: {sr}Hz")
        return y, sr
    
    def denoise_basic(self, audio, noise_profile=None):
        """使用谱减法降噪"""
        if noise_profile is None:
            # 估计噪声
            noise_duration = 1.0
            noise_samples = int(noise_duration * self.sr)
            noise = audio[:noise_samples]
            noise_stft = librosa.stft(noise, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
            noise_power = np.abs(noise_stft) ** 2
            noise_profile = np.mean(noise_power, axis=1)
        
        # 计算STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 谱减法
        magnitude = np.maximum(magnitude - 0.3 * noise_profile[:, None], 0)
        
        # 重建信号
        stft_denoised = magnitude * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length, win_length=self.win_length)
        
        return audio_denoised
    
    def denoise_advanced(self, audio, sr):
        """使用noisereduce高级降噪"""
        if not hasattr(self, 'nr'):
            return self.denoise_basic(audio)
        
        # 改进噪声样本提取：寻找音频中能量较低的部分作为噪声样本
        # 计算音频能量
        energy = np.array([np.sum(np.abs(audio[i:i+2048])**2) for i in range(0, len(audio)-2048, 512)])
        # 选择能量最低的20%作为噪声样本
        noise_indices = np.argsort(energy)[:int(len(energy)*0.2)]
        if len(noise_indices) > 0:
            # 取第一个噪声段
            start_idx = noise_indices[0] * 512
            noise_sample = audio[start_idx:start_idx+2048]
        else:
            #  fallback：使用前0.5秒
            noise_sample = audio[:int(sr * 0.5)]
        
        # 使用noisereduce降噪，优化参数以保护人声
        denoised_audio = self.nr.reduce_noise(
            y=audio,
            sr=sr,
            y_noise=noise_sample,
            stationary=False,
            prop_decrease=0.6,  # 降低降噪强度，保护人声
            n_std_thresh_stationary=1.5,  # 噪声检测阈值
            use_tqdm=False
        )
        
        return denoised_audio
    
    def denoise_deep(self, audio):
        """使用FRCRN深度学习降噪"""
        # 这里可以集成FRCRN模型
        # 由于模型训练需要大量数据，这里暂时使用高级降噪
        return self.denoise_advanced(audio, self.sr)
    
    def denoise(self, audio, sr=None):
        """统一的降噪接口"""
        if sr is None:
            sr = self.sr
        
        start_time = time.time()
        
        if self.algorithm == 'advanced' and self.use_nr:
            denoised_audio = self.denoise_advanced(audio, sr)
        elif self.algorithm == 'deep':
            denoised_audio = self.denoise_deep(audio)
        else:
            denoised_audio = self.denoise_basic(audio)
        
        end_time = time.time()
        print(f"降噪完成，耗时: {end_time - start_time:.2f}秒")
        
        return denoised_audio
    
    def save_audio(self, audio, sr, output_file):
        """保存音频文件"""
        import soundfile as sf
        sf.write(output_file, audio, sr)
        print(f"降噪后的音频已保存为: {output_file}")

def main():
    """主函数"""
    # 创建降噪器
    denoiser = Denoiser(algorithm='advanced')
    
    # 输入输出文件
    input_file = r"C:\Users\27946\Desktop\debate.wav"
    output_file = r"C:\Users\27946\Desktop\debate_denoised.wav"
    
    try:
        # 加载音频
        audio, sr = denoiser.load_audio(input_file)
        
        # 降噪
        denoised_audio = denoiser.denoise(audio, sr)
        
        # 保存结果
        denoiser.save_audio(denoised_audio, sr, output_file)
        
        print("\n降噪任务完成！")
        print(f"原始音频: {input_file}")
        print(f"降噪后音频: {output_file}")
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保输入音频文件存在且格式正确")

if __name__ == "__main__":
    main()
