"""
VoiceAnalysis - 人声分析模块

功能：
- 基础声学特征提取（基频、共振峰、频谱特征等）
- 情绪/情感分析（来自 EmotionAnalysis 模块）
- 并行计算优化
- 可视化报告生成
- 完整的日志记录系统
"""

from .acoustic_features import AcousticFeatureExtractor
from .voice_analyzer import VoiceAnalyzer
from .parallel_processor import ParallelProcessor
from .logger import (
    setup_logger,
    log_analysis_start,
    log_analysis_complete,
    log_error,
    log_performance,
    log_features,
    log_emotion_distribution,
    log_gui_operation,
    log_parallel_info
)

# 从 EmotionAnalysis 模块导入情绪分析相关类
try:
    from EmotionAnalysis.emotion_analyzer import EmotionAnalyzer
    from EmotionAnalysis.emotion_visualizer import EmotionVisualizer as VoiceVisualizer
except ImportError:
    # 如果 EmotionAnalysis 不可用，从本地导入（向后兼容）
    from .emotion_analyzer import EmotionAnalyzer
    from .visualizer import VoiceVisualizer

__all__ = [
    'AcousticFeatureExtractor',
    'EmotionAnalyzer',
    'VoiceAnalyzer',
    'ParallelProcessor',
    'VoiceVisualizer',
    'setup_logger',
    'log_analysis_start',
    'log_analysis_complete',
    'log_error',
    'log_performance',
    'log_features',
    'log_emotion_distribution',
    'log_gui_operation',
    'log_parallel_info'
]

__version__ = '2.1.0'