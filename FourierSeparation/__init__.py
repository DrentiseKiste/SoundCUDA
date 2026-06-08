"""
FourierSeparation - 傅里叶分析的多说话人声音分离模块

功能：
- 基于傅里叶分析的多说话人声音分离
- 短时傅里叶变换(STFT)分析
- 非负矩阵分解(NMF)分离
- 独立成分分析(ICA)分离
- 基于频谱特征的聚类分离
- 自动估计说话人数量
- 可视化分析
"""

from .fourier_speaker_separator import FourierSpeakerSeparator

__all__ = ['FourierSpeakerSeparator']

__version__ = '1.0.0'
