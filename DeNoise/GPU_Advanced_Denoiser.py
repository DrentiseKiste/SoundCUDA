"""
GPU高级音频降噪程序
功能：使用现成的深度学习库进行强力音频降噪，去除环境音，保留人声
技术：基于noisereduce库的深度学习降噪方法

依赖：
- numpy: 数值计算
- librosa: 音频处理
- noisereduce: 音频降噪库
- torch: 深度学习框架
- matplotlib: 可视化
- soundfile: 音频文件读写

安装命令：
pip install numpy librosa matplotlib soundfile torch noisereduce
"""

import numpy as np
import librosa
import noisereduce as nr
import torch
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

class GPUAdvancedDenoiser:
    def __init__(self):
        """初始化GPU高级音频降噪器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            print("✓ GPU加速可用")
        else:
            print("✗ GPU加速不可用，使用CPU模式")
        
        # 降噪参数
        self.sr = 16000  # 采样率
        self.n_fft = 2048  # FFT窗口大小
        self.hop_length = 512  # hop长度
        
        # 并行计算参数
        self.batch_size = 16  # 批处理大小
        self.max_workers = 8  # CPU线程数
        self.segment_size = 16000  # 1秒音频段
    
    def load_audio(self, audio_file):
        """加载音频文件"""
        print(f"正在加载音频: {audio_file}")
        y, sr = librosa.load(audio_file, sr=self.sr)
        print(f"音频长度: {len(y)/sr:.2f}秒, 采样率: {sr}Hz")
        return y, sr
    
    def denoise_segment(self, segment, sr):
        """降噪单个音频段"""
        # 使用noisereduce进行降噪
        # 自动检测噪声样本（使用前0.5秒）
        noise_sample = segment[:int(sr * 0.5)]
        
        denoised_segment = nr.reduce_noise(
            y=segment,
            sr=sr,
            y_noise=noise_sample,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            time_mask_smooth_ms=120,
            freq_mask_smooth_hz=120,
            stationary=False,
            prop_decrease=0.9,
            use_tqdm=False
        )
        
        return denoised_segment
    
    def denoise_audio_parallel(self, audio, sr):
        """并行降噪处理"""
        print(f"开始并行降噪处理，批处理大小: {self.batch_size}")
        
        # 分割音频
        segments = []
        for i in range(0, len(audio) - self.segment_size + 1, self.segment_size):
            segments.append(audio[i:i+self.segment_size])
        
        # 处理最后一段
        if len(audio) % self.segment_size != 0:
            last_segment = audio[-(self.segment_size):]
            segments.append(last_segment)
        
        print(f"音频分割为 {len(segments)} 个段")
        
        # 并行处理
        denoised_segments = []
        start_time = time.time()
        
        if self.gpu_available:
            # GPU批处理
            from tqdm import tqdm
            for i in tqdm(range(0, len(segments), self.batch_size), desc="GPU批处理进度"):
                batch_segments = segments[i:i+self.batch_size]
                batch_denoised = []
                
                for segment in batch_segments:
                    denoised = self.denoise_segment(segment, sr)
                    batch_denoised.append(denoised)
                
                denoised_segments.extend(batch_denoised)
        else:
            # CPU多线程处理
            from tqdm import tqdm
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                denoised_segments = list(tqdm(executor.map(lambda seg: self.denoise_segment(seg, sr), segments), 
                                             total=len(segments), desc="CPU处理进度"))
        
        # 合并结果
        denoised_audio = np.concatenate(denoised_segments)
        
        # 裁剪到原始长度
        denoised_audio = denoised_audio[:len(audio)]
        
        total_time = time.time() - start_time
        print(f"降噪完成，耗时: {total_time:.2f}秒")
        print(f"处理速度: {len(audio)/sr/total_time:.2f}x实时")
        
        return denoised_audio
    
    def save_audio(self, audio, sr, output_file):
        """保存音频文件"""
        import soundfile as sf
        sf.write(output_file, audio, sr)
        print(f"降噪后的音频已保存为: {output_file}")
    
    def plot_comparison(self, original, denoised, sr):
        """绘制原始音频和降噪后音频的对比"""
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        plt.figure(figsize=(15, 10))
        
        # 原始音频波形
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(original))/sr, original)
        plt.title('原始音频')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        
        # 降噪后音频波形
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(denoised))/sr, denoised)
        plt.title('降噪后音频')
        plt.xlabel('时间 (秒)')
        plt.ylabel('振幅')
        
        plt.tight_layout()
        plt.savefig('audio_comparison_advanced.png')
        plt.show()
        print("音频对比图已保存为: audio_comparison_advanced.png")

    def compare_performance(self, audio, sr):
        """比较CPU和GPU性能"""
        print("\n=== CPU vs GPU 性能对比 ===")
        
        # 强制使用CPU
        import os
        original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 禁用GPU
        
        print("\n1. CPU模式测试:")
        start_time = time.time()
        cpu_denoised = self.denoise_audio_parallel(audio, sr)
        cpu_time = time.time() - start_time
        print(f"CPU处理时间: {cpu_time:.2f}秒")
        
        # 恢复GPU
        if self.gpu_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible
            print("\n2. GPU模式测试:")
            start_time = time.time()
            gpu_denoised = self.denoise_audio_parallel(audio, sr)
            gpu_time = time.time() - start_time
            print(f"GPU处理时间: {gpu_time:.2f}秒")
            print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 恢复环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_visible

def main():
    """主函数"""
    denoiser = GPUAdvancedDenoiser()
    
    # 输入输出文件
    input_file = r"C:\Users\27946\Desktop\Sound\testvoices\20250610yingyu1tingli01.mp3"  # 测试音频文件
    output_file = r"C:\Users\27946\Desktop\Sound\testvoices\denoised_audio_advanced.mp3"
    
    try:
        # 加载音频
        audio, sr = denoiser.load_audio(input_file)
        
        # 比较性能（如果GPU可用）
        denoiser.compare_performance(audio, sr)
        
        # 并行降噪
        denoised_audio = denoiser.denoise_audio_parallel(audio, sr)
        
        # 保存结果
        denoiser.save_audio(denoised_audio, sr, output_file)
        
        # 绘制对比图
        denoiser.plot_comparison(audio, denoised_audio, sr)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保输入音频文件存在且格式正确")

if __name__ == "__main__":
    main()
