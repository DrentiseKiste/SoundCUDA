"""
GPU深度学习音频降噪程序
功能：使用FRCRN模型进行强力音频降噪，去除环境音，保留人声
技术：基于复数域深度学习的FRCRN模型

依赖：
- numpy: 数值计算
- librosa: 音频处理
- torch: 深度学习框架
- cupy: GPU加速计算
- matplotlib: 可视化
- soundfile: 音频文件读写

安装命令：
pip install numpy librosa matplotlib soundfile torch
# GPU支持（可选）
pip install cupy-cuda12x  # 根据CUDA版本选择
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import time
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# 动态导入CuPy
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

class ComplexConv2d(nn.Module):
    """复数域卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ComplexConv2d, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        # x: (B, C, H, W, 2) - 最后一维是实部和虚部
        real = x[..., 0]
        imag = x[..., 1]
        
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.real_conv(imag) + self.imag_conv(real)
        
        return torch.stack([real_out, imag_out], dim=-1)

class FRCRN(nn.Module):
    """Frequency Recurrent Complex-valued Network"""
    def __init__(self, input_channels=1, hidden_channels=64, num_layers=3):
        super(FRCRN, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            ComplexConv2d(input_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            ComplexConv2d(hidden_channels, hidden_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            ComplexConv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU()
        )
        
        # 循环层
        self.rnn = nn.LSTM(
            input_size=hidden_channels * 4,
            hidden_size=hidden_channels * 4,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            ComplexConv2d(hidden_channels * 8, hidden_channels * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            ComplexConv2d(hidden_channels * 2, hidden_channels, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            ComplexConv2d(hidden_channels, input_channels, kernel_size=(3, 3), padding=(1, 1))
        )
    
    def forward(self, x):
        # x: (B, 1, T, F, 2)
        batch_size, _, time_steps, freq_bins, _ = x.shape
        
        # 编码
        encoded = self.encoder(x)
        
        # 准备RNN输入
        rnn_input = encoded.permute(0, 2, 3, 1, 4).reshape(batch_size, time_steps, freq_bins * encoded.shape[1] * 2)
        
        # RNN处理
        rnn_output, _ = self.rnn(rnn_input)
        
        # 准备解码输入
        rnn_output = rnn_output.reshape(batch_size, time_steps, freq_bins, -1, 1).permute(0, 3, 1, 2, 4)
        
        # 解码
        output = self.decoder(torch.cat([encoded, rnn_output], dim=1))
        
        return output

class GPUDeepLearningDenoiser:
    def __init__(self):
        """初始化GPU深度学习音频降噪器"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_available = torch.cuda.is_available()
        self.cp = cp
        
        if self.gpu_available:
            print("✓ GPU加速可用")
        else:
            print("✗ GPU加速不可用，使用CPU模式")
        
        # 降噪参数
        self.noise_reduction = 0.3  # 噪声减少程度
        self.sr = 16000  # 采样率
        self.n_fft = 2048  # FFT窗口大小
        self.hop_length = 512  # hop长度
        self.win_length = 2048  # 窗口长度
        
        # 并行计算参数
        self.batch_size = 16  # 批处理大小（增加以提高GPU利用率）
        self.max_workers = 8  # CPU线程数
        self.segment_size = self.hop_length * 4  # 音频段大小
        
        # 初始化模型
        self.model = FRCRN().to(self.device)
        # 加载预训练权重（如果有）
        # self.load_model('frcrn_model.pth')
    
    def load_audio(self, audio_file):
        """加载音频文件"""
        print(f"正在加载音频: {audio_file}")
        y, sr = librosa.load(audio_file, sr=self.sr)
        print(f"音频长度: {len(y)/sr:.2f}秒, 采样率: {sr}Hz")
        return y, sr
    
    def preprocess_audio(self, audio):
        """预处理音频，转换为时频域"""
        # 计算STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length)
        
        # 转换为复数张量
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # 转换为实部和虚部
        real = magnitude * np.cos(phase)
        imag = magnitude * np.sin(phase)
        
        return real, imag, phase
    
    def postprocess_audio(self, real, imag, phase):
        """后处理音频，从时频域转换回时域"""
        # 重建复数谱
        stft_denoised = real + 1j * imag
        
        # 逆STFT
        audio_denoised = librosa.istft(stft_denoised, hop_length=self.hop_length, win_length=self.win_length)
        
        return audio_denoised
    
    def denoise_segment(self, segment):
        """降噪单个音频段"""
        # 预处理
        real, imag, phase = self.preprocess_audio(segment)
        
        # 准备模型输入
        input_data = np.stack([real, imag], axis=-1)
        input_data = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            output = self.model(input_data)
        
        # 提取输出
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        real_denoised = output[..., 0]
        imag_denoised = output[..., 1]
        
        # 后处理
        segment_denoised = self.postprocess_audio(real_denoised, imag_denoised, phase)
        
        return segment_denoised
    
    def denoise_audio_parallel(self, audio):
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
                    denoised = self.denoise_segment(segment)
                    batch_denoised.append(denoised)
                
                denoised_segments.extend(batch_denoised)
        else:
            # CPU多线程处理
            from tqdm import tqdm
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                denoised_segments = list(tqdm(executor.map(self.denoise_segment, segments), 
                                             total=len(segments), desc="CPU处理进度"))
        
        # 合并结果
        denoised_audio = np.concatenate(denoised_segments)
        
        # 裁剪到原始长度
        denoised_audio = denoised_audio[:len(audio)]
        
        total_time = time.time() - start_time
        print(f"降噪完成，耗时: {total_time:.2f}秒")
        print(f"处理速度: {len(audio)/self.sr/total_time:.2f}x实时")
        
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
        plt.savefig('audio_comparison_deeplearning.png')
        plt.show()
        print("音频对比图已保存为: audio_comparison_deeplearning.png")

    def compare_performance(self, audio):
        """比较CPU和GPU性能"""
        print("\n=== CPU vs GPU 性能对比 ===")
        
        # 测试CPU性能
        original_device = self.device
        self.device = torch.device('cpu')
        self.model = self.model.to(self.device)
        
        print("\n1. CPU模式测试:")
        start_time = time.time()
        cpu_denoised = self.denoise_audio_parallel(audio)
        cpu_time = time.time() - start_time
        print(f"CPU处理时间: {cpu_time:.2f}秒")
        
        # 测试GPU性能
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            print("\n2. GPU模式测试:")
            start_time = time.time()
            gpu_denoised = self.denoise_audio_parallel(audio)
            gpu_time = time.time() - start_time
            print(f"GPU处理时间: {gpu_time:.2f}秒")
            print(f"加速比: {cpu_time/gpu_time:.2f}x")
        
        # 恢复原始状态
        self.device = original_device
        self.model = self.model.to(self.device)

    def load_model(self, model_path):
        """加载预训练模型"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"模型已加载: {model_path}")
        except Exception as e:
            print(f"加载模型失败: {e}")
            print("使用随机初始化的模型")

def main():
    """主函数"""
    denoiser = GPUDeepLearningDenoiser()
    
    # 输入输出文件
    input_file = r"C:\Users\27946\Desktop\Sound\testvoices\20250610yingyu1tingli01.mp3"  # 测试音频文件
    output_file = r"C:\Users\27946\Desktop\Sound\testvoices\denoised_audio_deeplearning.mp3"
    
    try:
        # 加载音频
        audio, sr = denoiser.load_audio(input_file)
        
        # 比较性能（如果GPU可用）
        denoiser.compare_performance(audio)
        
        # 并行降噪
        denoised_audio = denoiser.denoise_audio_parallel(audio)
        
        # 保存结果
        denoiser.save_audio(denoised_audio, sr, output_file)
        
        # 绘制对比图
        denoiser.plot_comparison(audio, denoised_audio, sr)
        
    except Exception as e:
        print(f"错误: {e}")
        print("请确保输入音频文件存在且格式正确")

if __name__ == "__main__":
    main()
