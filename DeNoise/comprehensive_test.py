"""
全面的降噪效果测试脚本
功能：测试不同类型音频样本的降噪效果，评估人声保护情况
"""

import os
import numpy as np
import librosa
import soundfile as sf
from denoiser import Denoiser
import matplotlib.pyplot as plt

class DenoiseTester:
    """降噪效果测试器"""
    def __init__(self):
        self.denoiser = Denoiser(algorithm='advanced')
        self.test_files = []
        self.results = []
    
    def add_test_file(self, file_path, description):
        """添加测试文件"""
        if os.path.exists(file_path):
            self.test_files.append((file_path, description))
            print(f"添加测试文件: {description} - {file_path}")
        else:
            print(f"警告: 文件不存在: {file_path}")
    
    def calculate_snr(self, clean, noisy):
        """计算信噪比"""
        signal_power = np.mean(clean ** 2)
        noise_power = np.mean((clean - noisy) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def calculate_pesq(self, clean, noisy, sr):
        """计算PESQ分数（感知语音质量评估）"""
        try:
            from pesq import pesq
            return pesq(sr, clean, noisy, 'wb')
        except ImportError:
            print("警告: pesq库未安装，跳过PESQ计算")
            return None
    
    def test_file(self, file_path, description):
        """测试单个文件"""
        print(f"\n=== 测试: {description} ===")
        print(f"文件: {file_path}")
        
        # 加载音频
        audio, sr = self.denoiser.load_audio(file_path)
        
        # 降噪
        denoised_audio = self.denoiser.denoise(audio, sr)
        
        # 保存结果
        output_file = f"{os.path.splitext(file_path)[0]}_denoised.wav"
        self.denoiser.save_audio(denoised_audio, sr, output_file)
        
        # 计算评估指标
        snr_before = self.calculate_snr(audio, audio)  # 原始音频的SNR
        snr_after = self.calculate_snr(audio, denoised_audio)  # 降噪后的SNR
        
        # 计算PESQ（如果可用）
        pesq_score = self.calculate_pesq(audio, denoised_audio, sr)
        
        # 分析音频能量
        energy_original = np.sum(np.abs(audio) ** 2)
        energy_denoised = np.sum(np.abs(denoised_audio) ** 2)
        energy_ratio = energy_denoised / energy_original
        
        # 检测断续情况（通过能量变化）
        frame_size = 512
        energy_frames_original = np.array([np.sum(np.abs(audio[i:i+frame_size])**2) 
                                         for i in range(0, len(audio)-frame_size, frame_size)])
        energy_frames_denoised = np.array([np.sum(np.abs(denoised_audio[i:i+frame_size])**2) 
                                          for i in range(0, len(denoised_audio)-frame_size, frame_size)])
        
        # 计算能量变化率
        energy_change_original = np.std(energy_frames_original) / np.mean(energy_frames_original)
        energy_change_denoised = np.std(energy_frames_denoised) / np.mean(energy_frames_denoised)
        
        # 评估断续情况
        if energy_change_denoised > energy_change_original * 1.5:
            discontinuity = "严重"
        elif energy_change_denoised > energy_change_original * 1.2:
            discontinuity = "轻微"
        else:
            discontinuity = "无"
        
        # 保存结果
        result = {
            'file': file_path,
            'description': description,
            'output_file': output_file,
            'snr_before': snr_before,
            'snr_after': snr_after,
            'snr_improvement': snr_after - snr_before,
            'pesq': pesq_score,
            'energy_ratio': energy_ratio,
            'discontinuity': discontinuity,
            'energy_change_original': energy_change_original,
            'energy_change_denoised': energy_change_denoised
        }
        
        self.results.append(result)
        
        # 打印结果
        print(f"SNR改善: {result['snr_improvement']:.2f} dB")
        if pesq_score:
            print(f"PESQ分数: {pesq_score:.2f}")
        print(f"能量保留率: {energy_ratio:.2f}")
        print(f"断续情况: {discontinuity}")
        
        return result
    
    def run_all_tests(self):
        """运行所有测试"""
        print("开始全面降噪效果测试...")
        print(f"共测试 {len(self.test_files)} 个文件")
        
        for file_path, description in self.test_files:
            self.test_file(file_path, description)
        
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n=== 测试报告 ===")
        print(f"测试文件数量: {len(self.results)}")
        
        # 计算平均指标
        avg_snr_improvement = np.mean([r['snr_improvement'] for r in self.results])
        avg_energy_ratio = np.mean([r['energy_ratio'] for r in self.results])
        
        print(f"平均SNR改善: {avg_snr_improvement:.2f} dB")
        print(f"平均能量保留率: {avg_energy_ratio:.2f}")
        
        # 统计断续情况
        discontinuity_counts = {'无': 0, '轻微': 0, '严重': 0}
        for r in self.results:
            discontinuity_counts[r['discontinuity']] += 1
        
        print("断续情况统计:")
        for level, count in discontinuity_counts.items():
            print(f"  {level}: {count} 个文件")
        
        # 生成可视化
        self.generate_visualization()
    
    def generate_visualization(self):
        """生成测试结果可视化"""
        if not self.results:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # SNR改善条形图
        descriptions = [r['description'] for r in self.results]
        snr_improvements = [r['snr_improvement'] for r in self.results]
        
        axes[0].bar(descriptions, snr_improvements)
        axes[0].set_title('各测试文件的SNR改善')
        axes[0].set_ylabel('SNR改善 (dB)')
        axes[0].tick_params(axis='x', rotation=45, ha='right')
        
        # 能量保留率和断续情况
        energy_ratios = [r['energy_ratio'] for r in self.results]
        discontinuity_scores = []
        for r in self.results:
            if r['discontinuity'] == '无':
                discontinuity_scores.append(0)
            elif r['discontinuity'] == '轻微':
                discontinuity_scores.append(1)
            else:
                discontinuity_scores.append(2)
        
        axes[1].scatter(energy_ratios, discontinuity_scores, s=100)
        axes[1].set_title('能量保留率 vs 断续情况')
        axes[1].set_xlabel('能量保留率')
        axes[1].set_ylabel('断续情况 (0=无, 1=轻微, 2=严重)')
        axes[1].set_yticks([0, 1, 2])
        axes[1].set_yticklabels(['无', '轻微', '严重'])
        
        # 添加标签
        for i, desc in enumerate(descriptions):
            axes[1].annotate(desc, (energy_ratios[i], discontinuity_scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig('denoise_test_results.png')
        print("测试结果可视化已保存为: denoise_test_results.png")

def main():
    """主函数"""
    tester = DenoiseTester()
    
    # 添加测试文件
    # 1. 测试文件1：用户提供的文件
    tester.add_test_file(r"C:\Users\27946\Desktop\nt.wav", "用户测试文件")
    
    # 2. 测试文件2：LibriSpeech样本
    libri_file = r"C:\Users\27946\Desktop\Sound\testvoices\LibriSpeech\0.wav"
    if os.path.exists(libri_file):
        tester.add_test_file(libri_file, "LibriSpeech样本")
    
    # 3. 测试文件3：中文语音样本
    chn_file = r"C:\Users\27946\Desktop\Sound\testvoices\zhthchs30\A13_1.wav"
    if os.path.exists(chn_file):
        tester.add_test_file(chn_file, "中文语音样本")
    
    # 运行测试
    tester.run_all_tests()

if __name__ == "__main__":
    main()
