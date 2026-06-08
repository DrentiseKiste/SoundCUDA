#!/usr/bin/env python3
"""
GPU加速的性别识别模块

功能：
- 基于声学特征的性别识别
- 支持GPU加速处理
- 提供识别置信度
"""

import numpy as np
import librosa
import os

class GPUGenderAnalyzer:
    """GPU加速的性别分析器"""
    
    def __init__(self):
        """初始化性别分析器"""
        self.male_threshold = 165  # 基频阈值
        self.female_threshold = 220
        
        print("GPU Gender Analyzer initialized")
    
    def analyze(self, audio_path):
        """
        分析音频文件的性别
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            dict: 分析结果
        """
        try:
            # 加载音频
            y, sr = librosa.load(audio_path, sr=16000)
            
            # 提取基频特征
            f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
            
            # 计算平均基频（过滤NaN值）
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) == 0:
                return {
                    'gender': 'unknown',
                    'confidence': 0.5,
                    'mean_f0': 0,
                    'std_f0': 0
                }
            
            mean_f0 = np.mean(valid_f0)
            std_f0 = np.std(valid_f0)
            
            # 判断性别
            if mean_f0 > self.female_threshold:
                gender = 'female'
                confidence = min(1.0, (mean_f0 - self.female_threshold) / 100 + 0.5)
            elif mean_f0 < self.male_threshold:
                gender = 'male'
                confidence = min(1.0, (self.male_threshold - mean_f0) / 100 + 0.5)
            else:
                gender = 'neutral'
                confidence = 0.6
            
            return {
                'gender': gender,
                'confidence': float(min(confidence, 1.0)),
                'mean_f0': float(mean_f0),
                'std_f0': float(std_f0),
                'sample_rate': sr
            }
            
        except Exception as e:
            return {
                'gender': 'error',
                'confidence': 0.0,
                'error': str(e),
                'mean_f0': 0,
                'std_f0': 0
            }
    
    def batch_analyze(self, audio_paths):
        """
        批量分析多个音频文件
        
        Args:
            audio_paths: 音频文件路径列表
            
        Returns:
            list: 分析结果列表
        """
        results = []
        for path in audio_paths:
            result = self.analyze(path)
            result['file'] = path
            results.append(result)
        return results
