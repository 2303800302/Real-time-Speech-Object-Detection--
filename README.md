# 认知症筛查系统

本项目旨在为认知症患者提供实时语音识别与物体检测分析。系统集成了YOLOv11物体检测模型、Vosk离线语音识别引擎和DeepSeek API，用于智能理解与关键词提取。系统还包括针对Cookie Theft测试的语言分析与认知评估功能。

## 功能简介

- **YOLOv11物体检测**：实时检测视频中的物体，并通过锚框显示检测结果。
- **Vosk离线语音识别**：在无需网络的情况下进行语音识别，确保隐私保护。
- **DeepSeek智能理解**：通过DeepSeek API，提取语音中的关键词，自动识别和纠正语音识别错误。
- **Cookie Theft分析**：通过分析用户语音中的关键词，评估可能的认知症迹象。
- **认知症评估**：基于语音转录和检测结果，生成认知症筛查报告。
- **关键词与锚框显示**：识别语音中的关键词，控制锚框的显示与隐藏。
- **自动保存音频记录与报告**：会话期间的音频记录和分析报告可以自动保存。
- **自定义语音触发窗口**：根据识别到的关键词触发语音控制窗口，设定触发时长。

## 系统架构

1. **Vosk语音识别**：负责将音频信号转换为文本。
2. **YOLOv11目标检测**：在视频流中实时识别并显示物体。
3. **DeepSeek关键词提取**：从语音识别的文本中提取关键物体类别。
4. **认知症筛查与分析**：基于关键词和转录文本生成认知症评估报告。

## 环境要求

### 必须安装的依赖

- Python 3.8及以上版本
- OpenCV
- PyTorch
- Vosk
- PyAudio
- requests
- jieba
- scikit-learn
- pyttsx3
- tkinter

### 安装步骤

1. 克隆项目到本地：

    ```bash
    git clone https://github.com/your-username/recognition-system.git
    cd recognition-system
    ```

2. 创建虚拟环境（推荐使用Anaconda）：

    ```bash
    conda create -n recognition-system python=3.8
    conda activate recognition-system
    ```

3. 安装必要的依赖：

    ```bash
    pip install -r requirements.txt
    ```

4. 下载Vosk中文模型：

    从[https://alphacephei.com/vosk/models](https://alphacephei.com/vosk/models)下载`vosk-model-cn-0.22`，并将其解压至项目根目录。

5. 设置DeepSeek API密钥：

    通过DeepSeek平台获取API密钥，并将其配置到系统中。你可以在`.env`文件中设置密钥，或者在启动时手动输入。

    ```bash
    DEEPSEEK_API_KEY=your-api-key-here
    ```

## 使用方法

1. **运行系统**：

    在命令行中运行以下命令启动系统：

    ```bash
    python main.py
    ```

    系统将启动并打开图形界面。你可以使用摄像头实时进行物体检测，也可以通过语音识别进行控制。

2. **使用语音控制**：

    按下 **🎤 开始识别** 按钮，系统将开始监听用户的语音并进行识别。当用户说出特定物体名称时，系统将自动检测并在视频中显示相应的锚框。

3. **进行认知症评估**：

    在系统运行时，你可以随时通过 **🧠 认知评估** 按钮对当前会话进行认知症评估。系统将根据语音和视频检测结果生成评估报告。

4. **手动输入**：

    你也可以通过 **手动输入** 框输入关键词，触发相应的物体检测和锚框显示。

## 项目结构
'''
recognition-system/
│
├── new_main.py # 主程序入口
├── detector.py # YOLOv11模型和视频处理
├── model_calculation.py # 认知症评估模型和特征提取
├── statistics.py # 会话统计与分析
├── requirements.txt # 依赖项
├── yolo11n.pt
├── vosk-model-cn-0.22/ # Vosk语音模型
├── .env # 存储API密钥（可选）
└── README.md # 项目的说明文件
'''

## 注意事项

- **摄像头**：系统需要一个正常工作的摄像头来进行视频流捕捉和物体检测。
- **音频输入**：系统依赖麦克风输入音频用于语音识别。
- **DeepSeek API密钥**：如果没有API密钥，系统将默认使用简单的关键词匹配功能。
- **网络连接**：如果使用DeepSeek API进行智能理解，系统需要网络连接。



