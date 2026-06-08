# EchoSense - 音频处理综合分析系统

EchoSense 是一个基于 Python 的音频处理综合分析系统，提供音频降噪、性别识别、人声分析、说话人分离等多种功能。

---

##  目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [核心模块原理](#核心模块原理)
- [安装指南](#安装指南)
- [使用方法](#使用方法)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [开发说明](#开发说明)
- [许可证](#许可证)

---

## ✨ 功能特性

| 功能模块 | 描述 | 核心算法 |
|---------|------|---------|
| **音频降噪** | 支持多种降噪算法，保护人声质量 | 谱减法、noisereduce、深度学习 |
| **性别识别** | 基于基频特征的性别分析 | PYIN基频估计 + 阈值判断 |
| **人声分析** | 完整的声学特征提取和情绪分析 | 并行特征提取 + 机器学习分类 |
| **说话人分离** | 多说话人音频分离 | 傅里叶变换 + K-Means聚类/NMF/ICA |
| **可视化报告** | 生成专业的分析报告和图表 | Matplotlib可视化 |

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        EchoSense 系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    GUI 前端层                           │   │
│  │            Audio_Processing_App.py                      │   │
│  │  (PyQt5 单窗口应用，集成所有功能模块)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    业务逻辑层                            │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │ DeNoise  │ │  Gender  │ │VoiceAnalysis│ │ Fourier │   │   │
│  │  │  降噪    │ │ 性别识别 │ │  人声分析   │ │Separation│   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  │         │            │           │            │         │   │
│  └─────────┼────────────┼───────────┼────────────┼─────────┘   │
│            │            │           │            │              │
│            ▼            ▼           ▼            ▼              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    数据处理层                            │   │
│  │        librosa / numpy / scipy / sklearn               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    数据存储层                            │   │
│  │  logs/ (日志) | models/ (模型) | output/ (输出结果)     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔬 核心模块原理

### 1. 音频降噪模块 (DeNoise)

**原理概述**: 基于短时傅里叶变换(STFT)的频谱减法和自适应噪声抑制

**三种算法对比**:

| 算法 | 原理 | 优势 | 适用场景 |
|------|------|------|----------|
| **谱减法** | 估计噪声谱并从信号谱中减去 | 快速、轻量 | 低噪声环境 |
| **noisereduce** | 自适应噪声估计和频域滤波 | 智能、保护人声 | 推荐通用场景 |
| **深度学习** | FRCRN神经网络降噪 | 强力降噪 | 高噪声环境 |

**核心处理流程**:
```
原始音频 → STFT变换 → 噪声估计 → 谱减法处理 → ISTFT逆变换 → 降噪音频
```

### 2. 性别识别模块 (Gender)

**原理概述**: 基于基频(F0)特征的性别分类

**基频阈值设定**:
- 男性: F0 < 165 Hz
- 女性: F0 > 220 Hz
- 中性: 165 Hz ≤ F0 ≤ 220 Hz

**置信度计算**:
```
置信度 = min(1.0, |F0 - 阈值| / 100 + 0.5)
```

### 3. 人声分析模块 (VoiceAnalysis)

**原理概述**: 多维度声学特征提取与情绪分析

**提取的声学特征**:

| 特征类别 | 特征名称 | 描述 |
|---------|---------|------|
| **基频特征** | Mean F0, F0 Range, F0 Std | 音调高低和变化 |
| **频谱特征** | Spectral Centroid, Bandwidth, Roll-off | 频谱能量分布 |
| **能量特征** | RMS Energy, Dynamic Range | 声音强度 |
| **语音质量** | HNR, Jitter, Shimmer | 嗓音清晰度和稳定性 |
| **MFCC特征** | 13个MFCC系数 | 语音识别核心特征 |
| **时域特征** | ZCR, Voiced Ratio | 时间维度特性 |

**情绪分析流程**:
```
音频片段 → 特征提取 → 情绪分类模型 → 情绪概率分布 → 主导情绪判定
```

### 4. 傅里叶说话人分离模块 (FourierSeparation)

**原理概述**: 基于傅里叶分析和机器学习的多说话人分离

**三种分离方法**:

| 方法 | 原理 | 优势 |
|------|------|------|
| **聚类方法** | K-Means聚类 + 基频特征 | 自动估计说话人数，效果好 |
| **NMF** | 非负矩阵分解 | 数学模型优雅 |
| **ICA** | 独立成分分析 | 盲源分离经典算法 |

**核心步骤**:
1. STFT变换获取频谱
2. PYIN算法估计基频F0
3. 提取帧级特征（频谱质心、能量、F0等）
4. K-Means聚类识别说话人
5. 时频掩码应用
6. ISTFT重构分离音频

---

##  安装指南

### 环境要求
- Python 3.8+
- Windows 11

### 安装依赖

```bash
# 安装核心依赖
pip install numpy librosa matplotlib scikit-learn soundfile

# 安装PyQt5 (GUI)
pip install pyqt5

# 安装降噪库
pip install noisereduce

# 安装可选依赖（GPU加速）
pip install torch torchvision  # 深度学习降噪


---

##  使用方法

### 1. 启动GUI应用

```bash
cd Sound/front
python Audio_Processing_App.py
```

### 2. 使用流程

**步骤1: 加载音频文件**
- 拖拽音频文件到指定区域，或点击"选择文件"按钮
- 支持格式: WAV, MP3, FLAC, OGG

**步骤2: 选择功能模块**

| 标签页 | 功能 |
|--------|------|
| **音频导入** | 加载和管理音频文件 |
| **音频降噪** | 选择降噪算法和参数，执行降噪 |
| **性别识别** | 分析音频中的性别特征 |
| **人声分析** | 完整声学特征提取和情绪分析 |
| **说话人分离** | 分离多说话人音频 |

**步骤3: 配置参数**
- 根据需求调整各项参数
- 使用默认参数可获得较好效果

**步骤4: 执行分析**
- 点击对应按钮开始处理
- 查看进度条和控制台输出
- 分析完成后可导出结果

### 3. 命令行使用

#### 音频降噪示例
```python
from DeNoise.denoiser import Denoiser

# 创建降噪器（使用高级降噪算法）
denoiser = Denoiser(algorithm='advanced')

# 加载并降噪
audio, sr = librosa.load('input.wav', sr=16000)
denoised_audio = denoiser.denoise(audio, sr)

# 保存结果
import soundfile as sf
sf.write('output_denoised.wav', denoised_audio, sr)
```

#### 人声分析示例
```python
from VoiceAnalysis.voice_analyzer import VoiceAnalyzer

# 创建分析器（启用并行处理）
analyzer = VoiceAnalyzer(use_parallel=True)

# 分析音频
result = analyzer.analyze_audio('input.wav', generate_report=True)

# 获取特征摘要
print(analyzer.get_feature_summary(result))

# 导出结果
analyzer.export_results(result, 'analysis_result.json')
```

#### 说话人分离示例
```python
from FourierSeparation.fourier_speaker_separator import FourierSpeakerSeparator

# 创建分离器
separator = FourierSpeakerSeparator(sample_rate=16000)

# 分离说话人（自动估计人数）
result = separator.separate_speakers('input.wav', method='clustering')

# 保存分离结果
separator.save_separated_audios(result, output_dir='separated')
```

---

## 📁 项目结构

```
Sound/                              # 项目根目录
├── DeNoise/                        # 音频降噪模块
│   ├── __init__.py
│   ├── denoiser.py                 # 降噪器主类
│   ├── GPU_Advanced_Denoiser.py    # GPU高级降噪
│   ├── GPU_DeepLearning_Denoiser.py# 深度学习降噪
│   └── test_denoise.py             # 测试脚本
├── EmotionAnalysis/                # 情绪分析模块
│   ├── __init__.py
│   ├── emotion_analyzer.py         # 情绪分析核心
│   └── emotion_visualizer.py       # 情绪可视化
├── FourierSeparation/              # 傅里叶说话人分离模块
│   ├── __init__.py
│   └── fourier_speaker_separator.py# 说话人分离器
├── Gender/                         # 性别识别模块
│   ├── __init__.py
│   ├── GenderRecognition.py        # 性别识别
│   ├── GPU_GenderRecognition.py    # GPU加速版本
│   └── ParallelGenderRecognition.py# 并行处理版本
├── VoiceAnalysis/                  # 人声分析模块
│   ├── __init__.py
│   ├── acoustic_features.py        # 声学特征提取
│   ├── voice_analyzer.py           # 人声分析器主类
│   ├── parallel_processor.py       # 并行处理器
│   └── logger.py                   # 日志系统
├── front/                          # GUI前端
│   ├── Audio_Processing_App.py     # 主应用程序
│   └── 人声分析功能说明.md          # 功能说明文档
├── models/                         # 模型文件
│   └── XVector/                    # x-vector说话人模型
├── logs/                           # 日志目录
├── testvoices/                     # 测试音频文件
├── PROJECT_ORGANIZATION_SUMMARY.md # 项目结构说明
├── requirements.txt                # 依赖列表
└── README.md                       # 项目说明文档
```

---

## 🛠️ 技术栈

| 分类 | 技术 | 版本 |
|------|------|------|
| **语言** | Python | 3.8+ |
| **GUI框架** | PyQt5 | 5.15+ |
| **音频处理** | librosa | 0.10+ |
| **数值计算** | NumPy | 1.24+ |
| **机器学习** | scikit-learn | 1.3+ |
| **可视化** | Matplotlib | 3.7+ |
| **降噪** | noisereduce | 2.0+ |
| **深度学习** | PyTorch | 2.0+ (可选) |

---

## 📝 开发说明

### 代码规范
- 遵循 PEP 8 代码风格
- 使用 Google 风格的文档字符串
- 模块间保持低耦合
- 关键函数添加单元测试

### 扩展开发

#### 添加新功能模块
1. 在项目根目录创建新模块目录
2. 创建 `__init__.py` 导出公共API
3. 实现核心功能类
4. 在 `front/Audio_Processing_App.py` 中集成GUI

#### 添加新降噪算法
1. 在 `DeNoise/denoiser.py` 中添加新算法分支
2. 实现算法逻辑
3. 在GUI中添加算法选项

### 测试命令

```bash
# 运行综合测试
python test_integration.py

# 测试人声分析模块
python test_voice_analysis.py

# 测试情绪分析模块
python test_emotion_module.py

# 测试GUI集成
python test_gui_integration.py

