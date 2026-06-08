#!/usr/bin/env python3
"""
基于傅里叶分析的多说话人声音分离系统 - 增强基频版本

核心改进：
1. 加强基频(F0)特征提取 - 男女声最主要区别
2. 使用YIN算法进行基频估计
3. 添加性别特征到聚类中
4. 改进滑动窗口参数
"""

import os
import numpy as np
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.decomposition import FastICA, NMF
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FourierSpeakerSeparator:
    """基于傅里叶分析的说话人分离器 - 增强基频版本"""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 2048, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = int(1.5 * sample_rate)  # 1.5秒窗口
        self.window_overlap = int(0.3 * sample_rate)  # 0.3秒重叠
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        return audio, sr
    
    def compute_stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return stft, magnitude, phase
    
    def compute_f0(self, audio: np.ndarray) -> np.ndarray:
        """使用PYIN算法估计基频"""
        f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C1'),  # 32.7 Hz
            fmax=librosa.note_to_hz('C7'),  # 2093 Hz
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        f0 = np.nan_to_num(f0, nan=0)
        return f0
    
    def extract_frame_features_with_f0(self, magnitude: np.ndarray, f0: np.ndarray) -> np.ndarray:
        """提取包含基频的帧特征"""
        n_frames = magnitude.shape[1]
        features = []
        
        freq_bins = librosa.fft_frequencies(sr=self.sample_rate, n_fft=self.n_fft)
        
        for t in range(n_frames):
            frame_mag = magnitude[:, t]
            
            spectral_centroid = np.sum(frame_mag * freq_bins) / (np.sum(frame_mag) + 1e-10)
            spectral_centroid_norm = spectral_centroid / self.sample_rate
            
            spectral_bandwidth = np.sqrt(np.sum(((freq_bins - spectral_centroid) ** 2) * frame_mag) / (np.sum(frame_mag) + 1e-10))
            spectral_bandwidth_norm = spectral_bandwidth / self.sample_rate
            
            energy = np.sum(frame_mag ** 2)
            energy_norm = energy / np.max(magnitude) ** 2 if np.max(magnitude) > 0 else 0
            
            zcr = np.sum(np.abs(np.diff(np.sign(frame_mag)))) / (2 * len(frame_mag))
            
            frame_f0 = f0[t] if t < len(f0) else 0
            f0_norm = frame_f0 / 500.0  # 归一化到500Hz
            
            low_freq_energy = np.sum(frame_mag[:int(len(frame_mag)*0.1)])
            high_freq_energy = np.sum(frame_mag[int(len(frame_mag)*0.5):])
            freq_balance = (high_freq_energy + 1e-10) / (low_freq_energy + high_freq_energy + 1e-10)
            
            frame_features = np.array([
                spectral_centroid_norm,
                spectral_bandwidth_norm,
                energy_norm,
                zcr,
                f0_norm,
                freq_balance
            ])
            features.append(frame_features)
        
        return np.array(features)
    
    def estimate_n_speakers_from_f0(self, f0: np.ndarray, voiced_frames: np.ndarray) -> int:
        """基于基频估计说话人数量"""
        valid_f0 = f0[(f0 > 0) & (voiced_frames > 0.5)]
        
        if len(valid_f0) < 20:
            return 1
        
        f0_mean = np.mean(valid_f0)
        f0_std = np.std(valid_f0)
        
        if f0_std > 30:
            male_f0 = valid_f0[valid_f0 < 170]
            female_f0 = valid_f0[valid_f0 >= 170]
            
            if len(male_f0) > 10 and len(female_f0) > 10:
                return 2
        
        return 1
    
    def cluster_with_f0(self, features: np.ndarray, f0: np.ndarray, max_speakers: int = 3) -> Tuple[np.ndarray, int]:
        """结合基频的聚类"""
        if len(features) < 15:
            return np.zeros(len(features), dtype=int), 1
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        inertias = []
        sil_scores = []
        
        for n in range(2, min(max_speakers, 4) + 1):
            try:
                kmeans = KMeans(n_clusters=n, random_state=42, n_init=5, max_iter=100)
                labels = kmeans.fit_predict(features_scaled)
                
                inertias.append(kmeans.inertia_)
                if len(np.unique(labels)) > 1:
                    sil_scores.append(silhouette_score(features_scaled, labels))
                else:
                    sil_scores.append(-1)
            except:
                inertias.append(float('inf'))
                sil_scores.append(-1)
        
        if not inertias or max(sil_scores) < 0.1:
            return np.zeros(len(features), dtype=int), 1
        
        best_n = 2
        if len(sil_scores) > 0 and max(sil_scores) > 0.2:
            best_sil_idx = np.argmax(sil_scores)
            best_n = min(max(best_sil_idx + 2, 2), max_speakers)
        
        kmeans = KMeans(n_clusters=best_n, random_state=42, n_init=5, max_iter=100)
        labels = kmeans.fit_predict(features_scaled)
        
        if best_n == 2:
            cluster_centers = kmeans.cluster_centers_
            f0_idx = 4
            if abs(cluster_centers[0, f0_idx] - cluster_centers[1, f0_idx]) < 0.05:
                return np.zeros(len(features), dtype=int), 1
        
        return labels, best_n
    
    def match_speakers_across_windows(self, prev_centers: np.ndarray, curr_centers: np.ndarray) -> np.ndarray:
        if prev_centers.shape[0] == 0 or curr_centers.shape[0] == 0:
            return np.arange(curr_centers.shape[0])
        
        cost_matrix = np.zeros((prev_centers.shape[0], curr_centers.shape[0]))
        for i in range(prev_centers.shape[0]):
            for j in range(curr_centers.shape[0]):
                cost_matrix[i, j] = np.linalg.norm(prev_centers[i] - curr_centers[j])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = np.zeros(curr_centers.shape[0], dtype=int)
        for i, j in zip(row_ind, col_ind):
            mapping[j] = i
        
        for j in range(curr_centers.shape[0]):
            if j not in col_ind:
                mapping[j] = prev_centers.shape[0] + list(set(range(curr_centers.shape[0])) - set(col_ind)).index(j)
        
        return mapping
    
    def separate_with_temporal_dynamics(self, audio: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, List[int]]:
        n_samples = len(audio)
        stft, magnitude, phase = self.compute_stft(audio)
        f0 = self.compute_f0(audio)
        
        all_labels = np.zeros(magnitude.shape[1], dtype=int)
        all_n_speakers = []
        
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        voiced_frames = (rms > 0.02).astype(float)
        
        frame_features = self.extract_frame_features_with_f0(magnitude, f0)
        
        prev_centers = np.array([])
        global_speaker_id = 0
        speaker_centers = {}
        
        start = 0
        while start < n_samples:
            end = min(start + self.window_size, n_samples)
            
            frame_start = start // self.hop_length
            frame_end = min(frame_start + (self.window_size // self.hop_length) + 2, magnitude.shape[1])
            
            if frame_end - frame_start < 8:
                start += self.window_overlap
                continue
            
            window_features = frame_features[frame_start:frame_end]
            window_f0 = f0[frame_start:frame_end]
            
            window_n_speakers = self.estimate_n_speakers_from_f0(window_f0, voiced_frames[frame_start:frame_end])
            
            if window_n_speakers == 1:
                window_labels, window_n_speakers = self.cluster_with_f0(window_features, window_f0, max_speakers=3)
            
            else:
                window_labels, window_n_speakers = self.cluster_with_f0(window_features, window_f0, max_speakers=3)
            
            all_n_speakers.append(window_n_speakers)
            
            centers = []
            for i in range(window_n_speakers):
                mask = window_labels == i
                if np.any(mask):
                    centers.append(np.mean(window_features[mask], axis=0))
                else:
                    centers.append(np.zeros(window_features.shape[1]))
            centers = np.array(centers)
            
            if prev_centers.size > 0:
                mapping = self.match_speakers_across_windows(prev_centers, centers)
                
                remapped_labels = np.zeros(len(window_labels), dtype=int)
                for i in range(window_n_speakers):
                    new_id = mapping[i]
                    if new_id < len(speaker_centers):
                        remapped_labels[window_labels == i] = new_id
                    else:
                        remapped_labels[window_labels == i] = global_speaker_id
                        speaker_centers[global_speaker_id] = centers[i]
                        global_speaker_id += 1
                
                window_labels = remapped_labels
            else:
                for i in range(window_n_speakers):
                    speaker_centers[global_speaker_id + i] = centers[i]
                window_labels = window_labels + global_speaker_id
                global_speaker_id += window_n_speakers
            
            if frame_end <= len(all_labels):
                all_labels[frame_start:frame_end] = window_labels[:frame_end - frame_start]
            else:
                all_labels[frame_start:] = window_labels[:len(all_labels) - frame_start]
            
            prev_centers = centers
            start += self.window_overlap
        
        n_total_speakers = len(speaker_centers)
        if n_total_speakers == 0:
            n_total_speakers = 1
        
        separated_magnitudes = []
        for speaker_id in range(n_total_speakers):
            mask = all_labels == speaker_id
            speaker_mag = np.zeros_like(magnitude)
            speaker_mag[:, mask] = magnitude[:, mask] * 0.85
            separated_magnitudes.append(speaker_mag)
        
        separated_audios = [librosa.istft(mag * np.exp(1j * phase), hop_length=self.hop_length) 
                           for mag in separated_magnitudes]
        
        return separated_audios, all_labels, all_n_speakers, n_total_speakers
    
    def get_speaker_statistics(self, audio: np.ndarray) -> Dict:
        stats = {}
        stats['duration'] = len(audio) / self.sample_rate
        
        try:
            f0, _, _ = librosa.pyin(audio, fmin=60, fmax=400, sr=self.sample_rate)
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                stats['mean_f0'] = float(np.mean(valid_f0))
                stats['std_f0'] = float(np.std(valid_f0))
            else:
                stats['mean_f0'] = 0
                stats['std_f0'] = 0
        except:
            stats['mean_f0'] = 0
            stats['std_f0'] = 0
        
        rms = librosa.feature.rms(y=audio)[0]
        stats['mean_rms'] = float(np.mean(rms))
        
        if stats['mean_f0'] > 0:
            if stats['mean_f0'] > 190:
                stats['gender_hint'] = "女性"
            elif stats['mean_f0'] < 150:
                stats['gender_hint'] = "男性"
            else:
                stats['gender_hint'] = "中性"
        else:
            stats['gender_hint'] = "未知"
        
        return stats
    
    def extract_speaker_features(self, audio: np.ndarray) -> Dict:
        return self.get_speaker_statistics(audio)
    
    def separate_speakers(self, audio_path: str, method: str = 'temporal', 
                         n_speakers: Optional[int] = None) -> Dict:
        audio, sr = self.load_audio(audio_path)
        
        target_n = n_speakers if n_speakers is not None else 2
        
        if method == 'temporal':
            separated_audios, labels, n_speakers_history, n_total = self.separate_with_temporal_dynamics(audio)
            if n_speakers is not None:
                n_total = n_speakers
        elif method == 'nmf':
            _, magnitude, phase = self.compute_stft(audio)
            model = NMF(n_components=target_n, init='nndsvd', random_state=42, max_iter=200)
            W = model.fit_transform(magnitude)
            H = model.components_
            separated_magnitudes = [np.outer(W[:, i], H[i, :]) for i in range(target_n)]
            separated_audios = [librosa.istft(mag * np.exp(1j * phase), hop_length=self.hop_length) 
                               for mag in separated_magnitudes]
            n_total = target_n
            labels = None
            n_speakers_history = [target_n]
        elif method == 'clustering':
            _, magnitude, phase = self.compute_stft(audio)
            f0 = self.compute_f0(audio)
            features = self.extract_frame_features_with_f0(magnitude, f0)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            kmeans = KMeans(n_clusters=target_n, random_state=42, n_init=5, max_iter=100)
            labels = kmeans.fit_predict(features_scaled)
            
            separated_magnitudes = []
            for i in range(target_n):
                mask = labels == i
                speaker_mag = np.zeros_like(magnitude)
                speaker_mag[:, mask] = magnitude[:, mask]
                separated_magnitudes.append(speaker_mag)
            
            separated_audios = [librosa.istft(mag * np.exp(1j * phase), hop_length=self.hop_length) 
                               for mag in separated_magnitudes]
            n_total = target_n
            n_speakers_history = [target_n]
        else:
            raise ValueError(f"未知方法: {method}")
        
        _, magnitude, phase = self.compute_stft(audio)
        
        speaker_stats = []
        for i, audio_segment in enumerate(separated_audios):
            stats = self.get_speaker_statistics(audio_segment)
            stats['speaker_id'] = i + 1
            speaker_stats.append(stats)
        
        result = {
            'original_audio': audio,
            'sample_rate': sr,
            'separated_audios': separated_audios,
            'n_speakers': n_total,
            'method': method,
            'speaker_labels': labels,
            'n_speakers_history': n_speakers_history,
            'speaker_stats': speaker_stats,
            'magnitude': magnitude,
            'phase': phase
        }
        
        return result
    
    def save_separated_audios(self, result: Dict, output_dir: str = 'output'):
        os.makedirs(output_dir, exist_ok=True)
        
        for i, audio in enumerate(result['separated_audios']):
            output_path = os.path.join(output_dir, f'speaker_{i+1}.wav')
            sf.write(output_path, audio, result['sample_rate'])
        
        import json
        stats_file = os.path.join(output_dir, 'speaker_statistics.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result['speaker_stats'], f, indent=2, ensure_ascii=False)


def main():
    print("=" * 60)
    print("基于傅里叶分析的多说话人声音分离系统 - 增强基频版本")
    print("=" * 60)
    
    separator = FourierSpeakerSeparator(sample_rate=16000)
    
    audio_path = input("\n请输入音频文件路径: ").strip()
    if not audio_path or not os.path.exists(audio_path):
        print("无效的文件路径")
        return
    
    print("\n可用的分离方法:")
    print("1. Temporal (时间动态分离 - 推荐)")
    print("2. NMF (非负矩阵分解)")
    print("3. Clustering (全局聚类)")
    
    method_choice = input("请选择分离方法 (1-3, 默认1): ").strip()
    method_map = {'1': 'temporal', '2': 'nmf', '3': 'clustering'}
    method = method_map.get(method_choice, 'temporal')
    
    n_speakers_input = input("请输入说话人数量 (可选，按Enter自动估计): ").strip()
    n_speakers = int(n_speakers_input) if n_speakers_input else None
    
    print(f"\n正在使用 {method} 方法分离说话人...")
    result = separator.separate_speakers(audio_path, method=method, n_speakers=n_speakers)
    
    output_dir = 'separated_output'
    separator.save_separated_audios(result, output_dir)
    
    print(f"\n检测到 {result['n_speakers']} 个说话人")
    for i, stats in enumerate(result['speaker_stats']):
        print(f"\n说话人 {i+1}:")
        print(f"  性别预测: {stats['gender_hint']}")
        print(f"  平均基频: {stats['mean_f0']:.1f} Hz")
        print(f"  平均能量: {stats['mean_rms']:.4f}")
        print(f"  时长: {stats['duration']:.2f} 秒")
    
    if 'n_speakers_history' in result and len(result['n_speakers_history']) > 0:
        print(f"\n时间窗口说话人数变化: {result['n_speakers_history'][:10]}...")
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()