"""
认知症筛查系统 - 集成Vosk离线识别和DeepSeek API
主要改进：
1. 使用vosk-model-cn-0.22进行离线中文语音识别
2. 集成DeepSeek API进行智能关键词提取
3. 优化语音识别的容错处理
4. 支持模糊语音内容的智能理解
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import threading
import queue
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFont, ImageDraw
import json
import os
import logging
from pathlib import Path
from collections import defaultdict
import pyttsx3
import requests

# 新增导入
import vosk
import pyaudio
import wave

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 解决OpenCV中文显示问题
def cv2_put_chinese_text(img, text, position, font_size=20, color=(0, 255, 0)):
    """在OpenCV图像上显示中文文字"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    try:
        font_path = "C:/Windows/Fonts/simhei.ttf"
        if not os.path.exists(font_path):
            font_path = "C:/Windows/Fonts/msyh.ttc"
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    draw.text(position, text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class YOLOv11Detector:
    """YOLO11目标检测器（保持不变）"""
    def __init__(self, model_path='yolo11n.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"成功加载模型: {model_path}")
            self.class_names = self.model.names
            logger.info(f"检测类别数: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            try:
                self.model = YOLO('yolov8n.pt')
                self.model.to(self.device)
                self.class_names = self.model.names
                logger.info("使用备用模型 YOLOv8n")
            except:
                raise Exception("无法加载YOLO模型")

    def detect(self, frame, conf_threshold=0.5):
        """执行目标检测"""
        try:
            results = self.model(frame, conf=conf_threshold, verbose=False)
            detections = []
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.class_names[cls] if cls < len(self.class_names) else f"class_{cls}"
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'class': class_name,
                            'confidence': conf
                        })
            return detections
        except Exception as e:
            logger.error(f"检测错误: {e}")
            return []

class VoskDeepSeekRecognizer:
    """集成Vosk离线识别和DeepSeek API的语音识别器"""

    # YOLO类别映射
    YOLO_CHINESE_MAP = {
        'person': '人', 'bicycle': '自行车', 'car': '汽车', 'motorcycle': '摩托车',
        'airplane': '飞机', 'bus': '公交车', 'train': '火车', 'truck': '卡车',
        'boat': '船', 'traffic light': '红绿灯', 'fire hydrant': '消防栓',
        'stop sign': '停止标志', 'parking meter': '停车计时器', 'bench': '长椅',
        'bird': '鸟', 'cat': '猫', 'dog': '狗', 'horse': '马', 'sheep': '羊',
        'cow': '牛', 'elephant': '大象', 'bear': '熊', 'zebra': '斑马',
        'giraffe': '长颈鹿', 'backpack': '背包', 'umbrella': '雨伞',
        'handbag': '手提包', 'tie': '领带', 'suitcase': '手提箱',
        'frisbee': '飞盘', 'skis': '滑雪板', 'snowboard': '单板滑雪',
        'sports ball': '球', 'kite': '风筝', 'baseball bat': '棒球棒',
        'baseball glove': '棒球手套', 'skateboard': '滑板', 'surfboard': '冲浪板',
        'tennis racket': '网球拍', 'bottle': '瓶子', 'wine glass': '酒杯',
        'cup': '杯子', 'fork': '叉子', 'knife': '刀', 'spoon': '勺子',
        'bowl': '碗', 'banana': '香蕉', 'apple': '苹果', 'sandwich': '三明治',
        'orange': '橙子', 'broccoli': '西兰花', 'carrot': '胡萝卜',
        'hot dog': '热狗', 'pizza': '披萨', 'donut': '甜甜圈', 'cake': '蛋糕',
        'chair': '椅子', 'couch': '沙发', 'potted plant': '盆栽', 'bed': '床',
        'dining table': '餐桌', 'toilet': '马桶', 'tv': '电视',
        'laptop': '笔记本电脑', 'mouse': '鼠标', 'remote': '遥控器',
        'keyboard': '键盘', 'cell phone': '手机', 'microwave': '微波炉',
        'oven': '烤箱', 'toaster': '烤面包机', 'sink': '水槽',
        'refrigerator': '冰箱', 'book': '书', 'clock': '时钟', 'vase': '花瓶',
        'scissors': '剪刀', 'teddy bear': '泰迪熊', 'hair drier': '吹风机',
        'toothbrush': '牙刷'
    }

    def __init__(self, vosk_model_path="vosk-model-cn-0.22", deepseek_api_key=None):
        """初始化Vosk + DeepSeek识别器"""
        self.vosk_model_path = vosk_model_path
        self.deepseek_api_key = deepseek_api_key or os.getenv('DEEPSEEK_API_KEY')

        # 初始化Vosk
        self.vosk_model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.transcript_queue = queue.Queue()
        self.keywords_queue = queue.Queue()

        # 控制变量
        self.is_listening = False
        self.audio_thread = None
        self.process_thread = None

        # Cookie Theft相关关键词
        self.cookie_theft_keywords = [
            '水', '水龙头', '水槽', '溢出', '流水', '女人', '妈妈', '母亲', '女士',
            '男孩', '孩子', '儿子', '小孩', '女孩', '女儿', '凳子', '椅子', '板凳',
            '摔倒', '倒下', '跌倒', '饼干', '曲奇', '罐子', '饼干罐', '柜子', '橱柜',
            '厨房', '盘子', '碟子', '餐具', '窗户', '窗帘', '拿', '偷', '够', '伸手'
        ]

        # 创建反向映射
        self.chinese_to_english = {v: k for k, v in self.YOLO_CHINESE_MAP.items()}

        # 初始化组件
        self._init_vosk()
        self._test_deepseek_api()

    def _init_vosk(self):
        """初始化Vosk模型"""
        try:
            if not os.path.exists(self.vosk_model_path):
                logger.error(f"Vosk模型文件不存在: {self.vosk_model_path}")
                logger.info("请下载vosk-model-cn-0.22模型")
                return False

            self.vosk_model = vosk.Model(self.vosk_model_path)
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, 16000)
            logger.info("Vosk模型初始化成功")
            return True

        except Exception as e:
            logger.error(f"Vosk初始化失败: {e}")
            return False

    def _test_deepseek_api(self):
        """测试DeepSeek API连接"""
        if not self.deepseek_api_key:
            logger.warning("未设置DeepSeek API Key，将使用简单关键词匹配")
            return False

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "测试"}],
                    "max_tokens": 5
                },
                timeout=5
            )

            if response.status_code == 200:
                logger.info("DeepSeek API连接成功")
                return True
            else:
                logger.warning(f"DeepSeek API测试失败: {response.status_code}")
                return False

        except Exception as e:
            logger.warning(f"DeepSeek API连接失败: {e}")
            return False

    def start_listening(self):
        """开始语音识别"""
        if not self.vosk_model:
            logger.error("Vosk模型未初始化")
            return False

        self.is_listening = True

        # 启动音频捕获线程
        self.audio_thread = threading.Thread(target=self._audio_capture_loop, daemon=True)
        self.audio_thread.start()

        # 启动语音处理线程
        self.process_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.process_thread.start()

        logger.info("开始Vosk语音识别")
        return True

    def stop_listening(self):
        """停止语音识别"""
        self.is_listening = False
        logger.info("停止语音识别")

    def _audio_capture_loop(self):
        """音频捕获循环"""
        try:
            # 初始化PyAudio
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=4000
            )

            logger.info("开始音频捕获")

            while self.is_listening:
                try:
                    data = stream.read(4000, exception_on_overflow=False)
                    self.audio_queue.put(data)
                except Exception as e:
                    logger.error(f"音频捕获错误: {e}")
                    break

        except Exception as e:
            logger.error(f"音频初始化失败: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
                p.terminate()
            except:
                pass

    def _process_audio_loop(self):
        """音频处理循环"""
        while self.is_listening:
            try:
                # 获取音频数据（非阻塞）
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()

                    # Vosk识别
                    if self.recognizer.AcceptWaveform(audio_data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()

                        if text:
                            logger.info(f"Vosk识别结果: {text}")
                            self.transcript_queue.put(text)

                            # 使用DeepSeek提取关键词
                            self._extract_keywords_with_llm(text)

                else:
                    time.sleep(0.01)  # 避免CPU占用过高

            except Exception as e:
                logger.error(f"语音处理错误: {e}")
                time.sleep(0.1)

    def _extract_keywords_with_llm(self, text):
        """使用DeepSeek LLM提取关键词"""
        try:
            if self.deepseek_api_key:
                # 使用DeepSeek API
                keywords = self._call_deepseek_api(text)
            else:
                # 降级到简单匹配
                keywords = self._simple_keyword_extraction(text)

            if keywords:
                self.keywords_queue.put(keywords)
                logger.info(f"提取关键词: {keywords}")

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            # 降级处理
            keywords = self._simple_keyword_extraction(text)
            if keywords:
                self.keywords_queue.put(keywords)

    def _call_deepseek_api(self, text):
        """调用DeepSeek API进行关键词提取"""
        try:
            # 构建YOLO类别列表
            yolo_classes_str = ', '.join(self.YOLO_CHINESE_MAP.values())

            prompt = f"""
任务：从语音识别文本中提取YOLO目标检测可识别的物体关键词。

语音文本（可能包含识别错误、方言、口音导致的问题）：{text}

YOLO支持的物体类别：{yolo_classes_str}

请注意：
1. 语音识别可能有错误，请根据发音相似性推测用户真实意图
2. 只提取确实存在的物体类别，忽略动作、形容词等
3. 考虑同音字、近音字的情况（如：被子→杯子，椅子→椅子）
4. 如果没有匹配的物体，返回"无"

输出格式：只返回中文物体名称，多个用逗号分隔，不要解释。
示例：人,椅子,杯子
"""

            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.deepseek_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 50,  # 限制输出长度
                    "temperature": 0.1  # 降低随机性
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                llm_output = result['choices'][0]['message']['content'].strip()
                logger.info(f"DeepSeek输出: {llm_output}")

                # 解析LLM输出
                return self._parse_llm_output(llm_output, text)
            else:
                logger.error(f"DeepSeek API错误: {response.status_code}")
                return self._simple_keyword_extraction(text)

        except Exception as e:
            logger.error(f"DeepSeek API调用失败: {e}")
            return self._simple_keyword_extraction(text)

    def _parse_llm_output(self, llm_output, original_text):
        """解析LLM输出的关键词"""
        keywords = {'yolo_classes': [], 'special': []}

        if llm_output == "无" or not llm_output:
            return keywords

        # 解析中文物体名称
        chinese_objects = [obj.strip() for obj in llm_output.split(',') if obj.strip()]

        for chinese_obj in chinese_objects:
            # 转换为英文YOLO类别
            english_class = self.chinese_to_english.get(chinese_obj)
            if english_class:
                keywords['yolo_classes'].append(english_class)
                logger.info(f"LLM匹配: {chinese_obj} -> {english_class}")

        # 检查Cookie Theft特殊关键词
        for keyword in self.cookie_theft_keywords:
            if keyword in original_text:
                keywords['special'].append(keyword)

        return keywords

    def _simple_keyword_extraction(self, text):
        """简单的关键词匹配（备用方案）"""
        keywords = {'yolo_classes': [], 'special': []}

        # 检查YOLO物体类别
        for chinese, english in self.chinese_to_english.items():
            if chinese in text:
                keywords['yolo_classes'].append(english)

        # 检查特殊关键词
        for keyword in self.cookie_theft_keywords:
            if keyword in text:
                keywords['special'].append(keyword)

        return keywords

    def manual_input(self, text):
        """手动输入文本"""
        logger.info(f"手动输入: {text}")
        self.transcript_queue.put(text)
        self._extract_keywords_with_llm(text)

    def get_latest_transcript(self):
        """获取最新的转录文本"""
        transcripts = []
        while not self.transcript_queue.empty():
            transcripts.append(self.transcript_queue.get())
        return transcripts

    def get_latest_keywords(self):
        """获取最新的关键词"""
        all_keywords = {'yolo_classes': [], 'special': []}
        while not self.keywords_queue.empty():
            keywords = self.keywords_queue.get()
            all_keywords['yolo_classes'].extend(keywords.get('yolo_classes', []))
            all_keywords['special'].extend(keywords.get('special', []))

        # 去重
        all_keywords['yolo_classes'] = list(set(all_keywords['yolo_classes']))
        all_keywords['special'] = list(set(all_keywords['special']))

        return all_keywords

class CognitiveAssessmentApp:
    """认知评估系统主应用（更新版）"""

    def __init__(self):
        """初始化应用"""
        self.root = tk.Tk()
        self.root.title("认知症筛查系统 - YOLO11 + Vosk + DeepSeek")
        self.root.geometry("1400x900")

        # 设置样式
        self.setup_styles()

        # 初始化组件
        self.detector = YOLOv11Detector('yolo11n.pt')

        # 从环境变量或配置获取API密钥
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_key:
            # 弹出对话框获取API密钥
            deepseek_key = self.get_api_key()

        self.recognizer = VoskDeepSeekRecognizer(
            vosk_model_path="vosk-model-cn-0.22",
            deepseek_api_key=deepseek_key
        )

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # 视频相关
        self.cap = None
        self.video_thread = None
        self.is_running = False
        self.current_frame = None

        # 数据存储
        self.all_detections = []
        self.active_keywords = {'yolo_classes': [], 'special': []}
        self.matched_objects = []
        self.session_data = {
            'start_time': None,
            'end_time': None,
            'transcripts': [],
            'detections': defaultdict(int),
            'matches': [],
            'statistics': {}
        }

        # 控制标志
        self.show_all_boxes = False

        # 创建UI
        self.create_widgets()

        # 启动更新循环
        self.update_display()

        # 窗口关闭处理
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_api_key(self):
        """获取DeepSeek API密钥"""
        dialog = tk.Toplevel(self.root)
        dialog.title("配置DeepSeek API")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="请输入DeepSeek API密钥：").pack(pady=10)
        ttk.Label(dialog, text="(留空将使用简单关键词匹配)", font=('Arial', 9)).pack()

        api_key_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=api_key_var, width=50, show="*")
        entry.pack(pady=10)
        entry.focus()

        result = {"key": None}

        def on_ok():
            result["key"] = api_key_var.get().strip()
            dialog.destroy()

        def on_skip():
            result["key"] = None
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=20)

        ttk.Button(btn_frame, text="确定", command=on_ok).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="跳过", command=on_skip).pack(side='left', padx=5)

        entry.bind('<Return>', lambda e: on_ok())

        dialog.wait_window()
        return result["key"]

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')

    def create_widgets(self):
        """创建界面组件（基本保持原样，添加API状态显示）"""
        # 顶部标题
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(title_frame, text="🧠 认知症筛查系统", style='Title.TLabel').pack()

        # API状态显示
        api_status = "DeepSeek API: 已连接" if self.recognizer.deepseek_api_key else "DeepSeek API: 未配置（使用简单匹配）"
        vosk_status = "Vosk: 已就绪" if self.recognizer.vosk_model else "Vosk: 未就绪"

        ttk.Label(title_frame, text=f"基于YOLO11 + Vosk + DeepSeek | {vosk_status} | {api_status}").pack()

        # 其余UI组件保持不变...
        # （这里省略了大部分UI代码，因为与原版基本相同）

        # 主内容区
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 左侧：视频区域
        left_frame = ttk.LabelFrame(main_frame, text="📹 视频检测", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5)

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack()

        # 视频控制
        video_controls = ttk.Frame(left_frame)
        video_controls.pack(pady=10)

        self.btn_camera = ttk.Button(video_controls, text="📷 开启摄像头", command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=0, padx=5)

        ttk.Button(video_controls, text="📁 上传视频", command=self.load_video).grid(row=0, column=1, padx=5)

        self.show_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(video_controls, text="显示所有检测框",
                       variable=self.show_all_var,
                       command=self.toggle_show_all).grid(row=0, column=2, padx=5)

        # 检测信息
        detection_frame = ttk.LabelFrame(left_frame, text="检测到的物体", padding=5)
        detection_frame.pack(fill='both', expand=True, pady=10)

        self.detection_text = tk.Text(detection_frame, height=10, width=50)
        self.detection_text.pack(fill='both', expand=True)

        # 右侧：语音区域
        right_frame = ttk.LabelFrame(main_frame, text="🎤 语音识别 (Vosk + DeepSeek)", padding=10)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=5)

        # 语音控制
        voice_controls = ttk.Frame(right_frame)
        voice_controls.pack(pady=10)

        self.btn_voice = ttk.Button(voice_controls, text="🎤 开始识别", command=self.toggle_voice)
        self.btn_voice.grid(row=0, column=0, padx=5)

        self.voice_status = ttk.Label(voice_controls, text="未开始", style='Error.TLabel')
        self.voice_status.grid(row=0, column=1, padx=5)

        # 手动输入框
        manual_frame = ttk.LabelFrame(right_frame, text="手动输入（备用方案）", padding=5)
        manual_frame.pack(fill='x', pady=10)

        self.manual_entry = ttk.Entry(manual_frame, width=40, font=('Arial', 11))
        self.manual_entry.pack(side='left', padx=5)
        self.manual_entry.bind('<Return>', lambda e: self.submit_manual_text())

        ttk.Button(manual_frame, text="提交", command=self.submit_manual_text).pack(side='left')

        # 转录显示
        transcript_frame = ttk.LabelFrame(right_frame, text="语音内容", padding=5)
        transcript_frame.pack(fill='both', expand=True, pady=5)

        self.transcript_text = tk.Text(transcript_frame, height=6, width=50)
        self.transcript_text.pack(fill='both', expand=True)

        # 关键词显示
        keywords_frame = ttk.LabelFrame(right_frame, text="DeepSeek提取的关键词", padding=5)
        keywords_frame.pack(fill='both', expand=True, pady=5)

        self.keywords_text = tk.Text(keywords_frame, height=4, width=50)
        self.keywords_text.pack(fill='both', expand=True)

        # 匹配结果
        match_frame = ttk.LabelFrame(right_frame, text="当前显示的物体", padding=5)
        match_frame.pack(fill='both', expand=True, pady=5)

        self.match_text = tk.Text(match_frame, height=4, width=50)
        self.match_text.pack(fill='both', expand=True)

        # 底部控制
        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(bottom_frame, text="💾 保存结果", command=self.save_results).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="📊 查看统计", command=self.show_statistics).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="🔄 清除关键词", command=self.clear_keywords).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="🔄 重置会话", command=self.reset_session).pack(side='left', padx=5)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪 - Vosk离线识别 + DeepSeek智能理解", style='Status.TLabel')
        self.status_bar.pack(side='bottom', fill='x', padx=10, pady=2)

        # 配置网格
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    # 其余方法保持原样，只需要修改调用recognizer的地方
    def toggle_voice(self):
        """切换语音识别状态"""
        if not self.recognizer.is_listening:
            if self.recognizer.start_listening():
                self.btn_voice.config(text="⏹ 停止识别")
                self.voice_status.config(text="Vosk识别中...", style='Success.TLabel')

                if self.recognizer.deepseek_api_key:
                    self.status_bar.config(text="正在使用Vosk + DeepSeek进行语音识别和关键词提取")
                else:
                    self.status_bar.config(text="正在使用Vosk + 简单匹配进行语音识别")
            else:
                messagebox.showerror("错误", "无法启动语音识别，请检查Vosk模型文件")
        else:
            self.recognizer.stop_listening()
            self.btn_voice.config(text="🎤 开始识别")
            self.voice_status.config(text="已停止", style='Error.TLabel')

    def submit_manual_text(self):
        """提交手动输入的文本"""
        text = self.manual_entry.get()
        if text:
            self.recognizer.manual_input(text)
            self.manual_entry.delete(0, 'end')
            self.status_bar.config(text=f"已提交给DeepSeek处理: {text}")

    # 其余方法基本保持不变...
    # (省略其他方法的重复代码，因为大部分与原版相同)

    def clear_keywords(self):
        """清除关键词"""
        self.active_keywords = {'yolo_classes': [], 'special': []}
        self.matched_objects = []
        self.keywords_text.delete('1.0', 'end')
        self.match_text.delete('1.0', 'end')
        self.status_bar.config(text="关键词已清除")

    def toggle_show_all(self):
        """切换是否显示所有检测框"""
        self.show_all_boxes = self.show_all_var.get()

    def toggle_camera(self):
        """切换摄像头状态"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_running = True
                self.session_data['start_time'] = datetime.now()
                self.btn_camera.config(text="⏹ 关闭摄像头")
                self.status_bar.config(text="摄像头已开启")
                self.video_thread = threading.Thread(target=self.process_video, daemon=True)
                self.video_thread.start()
            else:
                messagebox.showerror("错误", "无法打开摄像头")
                self.cap = None
        else:
            self.stop_video()
            self.btn_camera.config(text="📷 开启摄像头")

    def load_video(self):
        """加载视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if file_path:
            self.stop_video()
            self.cap = cv2.VideoCapture(file_path)
            if self.cap.isOpened():
                self.is_running = True
                self.session_data['start_time'] = datetime.now()
                self.status_bar.config(text=f"已加载: {os.path.basename(file_path)}")
                self.video_thread = threading.Thread(target=self.process_video, daemon=True)
                self.video_thread.start()
            else:
                messagebox.showerror("错误", "无法打开视频文件")

    def stop_video(self):
        """停止视频"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None

    def process_video(self):
        """处理视频流"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                detections = self.detector.detect(frame)
                self.all_detections = detections
                annotated_frame = self.draw_selective_annotations(frame, detections)
                self.current_frame = annotated_frame
                for det in detections:
                    self.session_data['detections'][det['class']] += 1
                time.sleep(0.03)
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def draw_selective_annotations(self, frame, detections):
        """选择性绘制检测框"""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']

            should_display = False
            if self.show_all_boxes:
                should_display = True
                is_matched = class_name in self.active_keywords['yolo_classes']
            else:
                if class_name in self.active_keywords['yolo_classes']:
                    should_display = True
                    is_matched = True
                else:
                    continue

            color = (0, 255, 0) if is_matched else (0, 0, 255)
            thickness = 3 if is_matched else 2

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            chinese_name = self.recognizer.YOLO_CHINESE_MAP.get(class_name, class_name)
            label = f"{chinese_name} {confidence:.2f}"
            annotated = cv2_put_chinese_text(annotated, label, (x1, y1 - 5), font_size=16, color=color)

            if is_matched and class_name not in self.matched_objects:
                self.matched_objects.append(class_name)
                self.session_data['matches'].append({
                    'time': datetime.now().isoformat(),
                    'object': class_name,
                    'confidence': confidence
                })

        return annotated

    def update_display(self):
        """更新显示内容"""
        # 更新视频
        if self.current_frame is not None:
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            height, width = frame_rgb.shape[:2]
            max_width = 640
            if width > max_width:
                scale = max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk

        # 更新语音转录
        new_transcripts = self.recognizer.get_latest_transcript()
        for transcript in new_transcripts:
            self.transcript_text.insert('end', f"[{datetime.now().strftime('%H:%M:%S')}] {transcript}\n")
            self.transcript_text.see('end')
            self.session_data['transcripts'].append({
                'time': datetime.now().isoformat(),
                'text': transcript
            })

        # 更新关键词
        new_keywords = self.recognizer.get_latest_keywords()
        if new_keywords['yolo_classes'] or new_keywords['special']:
            self.active_keywords['yolo_classes'].extend(new_keywords['yolo_classes'])
            self.active_keywords['special'].extend(new_keywords['special'])
            self.active_keywords['yolo_classes'] = list(set(self.active_keywords['yolo_classes']))
            self.active_keywords['special'] = list(set(self.active_keywords['special']))

            # 显示关键词
            self.keywords_text.delete('1.0', 'end')
            self.keywords_text.insert('1.0', "DeepSeek提取的YOLO物体: ")
            for kw in self.active_keywords['yolo_classes']:
                chinese = self.recognizer.YOLO_CHINESE_MAP.get(kw, kw)
                self.keywords_text.insert('end', f"[{chinese}] ")

            if self.active_keywords['special']:
                self.keywords_text.insert('end', "\n\n特殊词汇: ")
                for kw in self.active_keywords['special']:
                    self.keywords_text.insert('end', f"[{kw}] ")

        # 更新检测列表
        self.detection_text.delete('1.0', 'end')
        if self.all_detections:
            detection_count = defaultdict(int)
            for det in self.all_detections:
                detection_count[det['class']] += 1

            self.detection_text.insert('1.0', "所有检测到的物体：\n")
            for class_name, count in detection_count.items():
                chinese_name = self.recognizer.YOLO_CHINESE_MAP.get(class_name, class_name)
                is_active = class_name in self.active_keywords['yolo_classes']
                marker = "✅" if is_active else "⭕"
                self.detection_text.insert('end', f"{marker} {chinese_name}: {count}个\n")
        else:
            self.detection_text.insert('1.0', "未检测到物体")

        # 更新匹配结果
        self.match_text.delete('1.0', 'end')
        if self.matched_objects:
            self.match_text.insert('1.0', "正在显示的物体:\n")
            for obj in self.matched_objects:
                chinese_name = self.recognizer.YOLO_CHINESE_MAP.get(obj, obj)
                self.match_text.insert('end', f"✓ {chinese_name}\n")
        else:
            if self.active_keywords['yolo_classes']:
                self.match_text.insert('1.0', "等待检测匹配的物体...")
            else:
                self.match_text.insert('1.0', "请说出或输入物体名称")

        self.root.after(100, self.update_display)

    def save_results(self):
        """保存测试结果"""
        if not self.session_data['start_time']:
            messagebox.showwarning("警告", "没有数据可保存")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialfile=f"cognitive_test_vosk_deepseek_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if file_path:
            self.session_data['end_time'] = datetime.now()
            save_data = {
                'test_info': {
                    'start_time': self.session_data['start_time'].isoformat(),
                    'end_time': self.session_data['end_time'].isoformat(),
                    'duration': (self.session_data['end_time'] - self.session_data['start_time']).total_seconds(),
                    'voice_engine': 'Vosk + DeepSeek',
                    'deepseek_used': bool(self.recognizer.deepseek_api_key)
                },
                'transcripts': self.session_data['transcripts'],
                'detections': dict(self.session_data['detections']),
                'matches': self.session_data['matches'],
                'keywords': self.active_keywords,
                'statistics': self.calculate_statistics()
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("成功", f"结果已保存到:\n{file_path}")

    def calculate_statistics(self):
        """计算统计数据"""
        stats = {
            'total_transcripts': len(self.session_data['transcripts']),
            'total_detections': sum(self.session_data['detections'].values()),
            'unique_objects': len(self.session_data['detections']),
            'total_matches': len(self.matched_objects),
            'active_keywords': len(self.active_keywords['yolo_classes']),
            'match_rate': len(self.matched_objects) / len(self.active_keywords['yolo_classes'])
                         if self.active_keywords['yolo_classes'] else 0,
            'deepseek_enhancement': bool(self.recognizer.deepseek_api_key)
        }
        return stats

    def show_statistics(self):
        """显示统计信息"""
        stats = self.calculate_statistics()
        engine_info = "Vosk离线识别 + DeepSeek智能提取" if stats['deepseek_enhancement'] else "Vosk离线识别 + 简单匹配"

        message = f"""
测试统计信息：

语音引擎: {engine_info}
语音转录次数: {stats['total_transcripts']}
检测物体总数: {stats['total_detections']}
不同物体种类: {stats['unique_objects']}

激活的关键词: {stats['active_keywords']}
成功显示的物体: {stats['total_matches']}
匹配率: {stats['match_rate']:.2%}

特殊关键词: {len(self.active_keywords['special'])}
        """
        messagebox.showinfo("统计信息", message)

    def reset_session(self):
        """重置会话"""
        if messagebox.askyesno("确认", "确定要重置所有数据吗？"):
            self.all_detections = []
            self.active_keywords = {'yolo_classes': [], 'special': []}
            self.matched_objects = []
            self.session_data = {
                'start_time': datetime.now() if self.is_running else None,
                'end_time': None,
                'transcripts': [],
                'detections': defaultdict(int),
                'matches': [],
                'statistics': {}
            }

            self.transcript_text.delete('1.0', 'end')
            self.keywords_text.delete('1.0', 'end')
            self.detection_text.delete('1.0', 'end')
            self.match_text.delete('1.0', 'end')
            self.status_bar.config(text="会话已重置")

    def on_closing(self):
        """窗口关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.stop_video()
            self.recognizer.stop_listening()
            cv2.destroyAllWindows()
            self.root.destroy()

    def run(self):
        """运行应用"""
        logger.info("系统启动 - Vosk + DeepSeek 版本")
        self.root.mainloop()


def main():
    """主函数"""
    print("="*70)
    print("认知症筛查系统 - YOLO11 + Vosk离线识别 + DeepSeek智能理解")
    print("="*70)
    print(f"Python版本: {os.sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    print("\n依赖检查:")
    try:
        import vosk
        print("✅ Vosk已安装")
    except ImportError:
        print("❌ Vosk未安装，请运行: pip install vosk")
        return

    try:
        import pyaudio
        print("✅ PyAudio已安装")
    except ImportError:
        print("❌ PyAudio未安装，请运行: pip install pyaudio")
        return

    print("\n模型检查:")
    if os.path.exists("vosk-model-cn-0.22"):
        print("✅ Vosk中文模型已就绪")
    else:
        print("❌ 缺少Vosk中文模型")
        print("请从 https://alphacephei.com/vosk/models 下载 vosk-model-cn-0.22")
        print("解压到当前目录")

    print("\nAPI配置:")
    if os.getenv('DEEPSEEK_API_KEY'):
        print("✅ DeepSeek API密钥已配置（环境变量）")
    else:
        print("⚠️ 未配置DeepSeek API密钥，将在启动时询问")
        print("获取API密钥: https://platform.deepseek.com/")

    print("="*70)
    print("启动说明:")
    print("1. Vosk进行离线语音识别，保护隐私")
    print("2. DeepSeek API进行智能关键词提取和纠错")
    print("3. 支持方言、口音、识别错误的智能理解")
    print("4. 默认只显示语音识别到的物体")
    print("5. 可手动输入文本作为备用方案")
    print("="*70)

    try:
        app = CognitiveAssessmentApp()
        app.run()
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"\n错误: {e}")
        print("\n故障排除:")
        print("1. 检查vosk-model-cn-0.22模型文件")
        print("2. 检查PyAudio和麦克风权限")
        print("3. 检查网络连接（用于DeepSeek API）")
        print("4. 检查摄像头权限")


if __name__ == "__main__":
    main()