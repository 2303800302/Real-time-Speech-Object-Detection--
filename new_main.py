"""
主程序模块 - main.py
认知症筛查系统主程序，集成所有功能模块
"""

import cv2
import numpy as np
import torch
import threading
import queue
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import json
import os
import logging
from pathlib import Path
import pyttsx3
import requests

# 新增导入
import vosk
import pyaudio
import wave

# 导入自定义模块
from detector import YOLOv11Detector, VideoProcessor, BoundingBoxManager, cv2_put_chinese_text
from statistics import SessionStatistics, CookieTheftAnalyzer, AudioRecorder
from model_calculation import CognitiveAssessmentModel, FeatureExtractor, ModelEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VoskDeepSeekRecognizer:
    """集成Vosk离线识别和DeepSeek API的语音识别器"""

    # YOLO类别的中英文映射
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
        self.transcript_queue = queue.Queue()
        self.keywords_queue = queue.Queue()
        self.audio_data_queue = queue.Queue()  # 保存音频数据

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
                    self.audio_data_queue.put(data)  # 同时保存原始音频数据
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
                if not self.audio_queue.empty():
                    audio_data = self.audio_queue.get()

                    # Vosk识别
                    if self.recognizer.AcceptWaveform(audio_data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '').strip()

                        if text:
                            logger.info(f"Vosk识别结果: {text}")
                            self.transcript_queue.put({
                                'text': text,
                                'timestamp': datetime.now().isoformat(),
                                'audio_data': self._get_recent_audio_data()
                            })

                            # 使用DeepSeek提取关键词
                            self._extract_keywords_with_llm(text)

                else:
                    time.sleep(0.01)

            except Exception as e:
                logger.error(f"语音处理错误: {e}")
                time.sleep(0.1)

    def _get_recent_audio_data(self):
        """获取最近的音频数据"""
        audio_chunks = []
        chunk_count = 8  # 约2秒
        for _ in range(min(chunk_count, self.audio_data_queue.qsize())):
            if not self.audio_data_queue.empty():
                audio_chunks.append(self.audio_data_queue.get())
        return b''.join(audio_chunks) if audio_chunks else None

    def _extract_keywords_with_llm(self, text):
        """使用DeepSeek LLM提取关键词"""
        try:
            if self.deepseek_api_key:
                keywords = self._call_deepseek_api(text)
            else:
                keywords = self._simple_keyword_extraction(text)

            if keywords:
                self.keywords_queue.put(keywords)
                logger.info(f"提取关键词: {keywords}")

        except Exception as e:
            logger.error(f"关键词提取失败: {e}")
            keywords = self._simple_keyword_extraction(text)
            if keywords:
                self.keywords_queue.put(keywords)

    def _call_deepseek_api(self, text):
        """调用DeepSeek API进行关键词提取"""
        try:
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
                    "max_tokens": 50,
                    "temperature": 0.1
                },
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                llm_output = result['choices'][0]['message']['content'].strip()
                logger.info(f"DeepSeek输出: {llm_output}")
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

        chinese_objects = [obj.strip() for obj in llm_output.split(',') if obj.strip()]

        for chinese_obj in chinese_objects:
            english_class = self.chinese_to_english.get(chinese_obj)
            if english_class:
                keywords['yolo_classes'].append(english_class)
                logger.info(f"LLM匹配: {chinese_obj} -> {english_class}")

        for keyword in self.cookie_theft_keywords:
            if keyword in original_text:
                keywords['special'].append(keyword)

        return keywords

    def _simple_keyword_extraction(self, text):
        """简单的关键词匹配（备用方案）"""
        keywords = {'yolo_classes': [], 'special': []}
        for chinese, english in self.chinese_to_english.items():
            if chinese in text:
                keywords['yolo_classes'].append(english)
        for keyword in self.cookie_theft_keywords:
            if keyword in text:
                keywords['special'].append(keyword)
        return keywords

    def manual_input(self, text):
        """手动输入文本"""
        logger.info(f"手动输入: {text}")
        self.transcript_queue.put({
            'text': text,
            'timestamp': datetime.now().isoformat(),
            'audio_data': None
        })
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
        all_keywords['yolo_classes'] = list(set(all_keywords['yolo_classes']))
        all_keywords['special'] = list(set(all_keywords['special']))
        return all_keywords

class CognitiveAssessmentApp:
    """认知评估系统主应用"""

    def __init__(self):
        """初始化应用"""
        self.root = tk.Tk()
        self.root.title("认知症筛查系统 - YOLO11 + Vosk + DeepSeek + Cookie Theft分析")
        self.root.geometry("1600x1000")

        # 设置样式
        self.setup_styles()

        # 初始化组件
        self.detector = YOLOv11Detector('yolo11n.pt')
        self.video_processor = VideoProcessor(self.detector)

        # 获取API密钥
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        if not deepseek_key:
            deepseek_key = self.get_api_key()

        self.recognizer = VoskDeepSeekRecognizer(
            vosk_model_path="vosk-model-cn-0.22",
            deepseek_api_key=deepseek_key
        )

        # 初始化统计和模型组件
        self.statistics = SessionStatistics()
        self.cognitive_model = CognitiveAssessmentModel()

        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 150)

        # 状态变量
        self.current_frame = None
        self.active_keywords = {'yolo_classes': [], 'special': []}
        self.matched_objects = []
        self.show_all_boxes = False

        # ★ 修改：语音触发窗口默认时长（秒）
        self.voice_trigger_window = 2.0  # 增加到2秒

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
        dialog.geometry("500x250")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="请输入DeepSeek API密钥：", font=('Arial', 12)).pack(pady=15)
        ttk.Label(dialog, text="(留空将使用简单关键词匹配)", font=('Arial', 9), foreground='gray').pack()
        ttk.Label(dialog, text="获取API密钥: https://platform.deepseek.com/", font=('Arial', 9), foreground='blue').pack(pady=5)

        api_key_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=api_key_var, width=60, show="*")
        entry.pack(pady=15)
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

        ttk.Button(btn_frame, text="确定", command=on_ok).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="跳过", command=on_skip).pack(side='left', padx=10)

        entry.bind('<Return>', lambda e: on_ok())

        dialog.wait_window()
        return result["key"]

    def setup_styles(self):
        """设置界面样式"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Status.TLabel', font=('Arial', 10))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')

    def create_widgets(self):
        """创建界面组件"""
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=10, pady=5)

        ttk.Label(title_frame, text="🧠 认知症筛查系统 - Cookie Theft测试", style='Title.TLabel').pack()

        api_status = "DeepSeek API: 已连接" if self.recognizer.deepseek_api_key else "DeepSeek API: 未配置"
        vosk_status = "Vosk: 已就绪" if self.recognizer.vosk_model else "Vosk: 未就绪"
        ttk.Label(title_frame, text=f"YOLO11 + {vosk_status} + {api_status} + Cookie Theft分析").pack()

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)

        left_frame = ttk.LabelFrame(main_frame, text="📹 视频检测与锚框控制", padding=10)
        left_frame.grid(row=0, column=0, sticky='nsew', padx=5)

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack()

        video_controls = ttk.Frame(left_frame)
        video_controls.pack(pady=10)

        self.btn_camera = ttk.Button(video_controls, text="📷 开启摄像头", command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=0, padx=5)

        ttk.Button(video_controls, text="📁 上传视频", command=self.load_video).grid(row=0, column=1, padx=5)

        bbox_controls = ttk.LabelFrame(left_frame, text="锚框显示控制", padding=5)
        bbox_controls.pack(fill='x', pady=10)

        ttk.Label(bbox_controls, text="显示时长(秒):").grid(row=0, column=0, padx=5)
        self.duration_var = tk.DoubleVar(value=5.0)
        duration_scale = ttk.Scale(bbox_controls, from_=1.0, to=30.0,
                                   variable=self.duration_var, orient='horizontal')
        duration_scale.grid(row=0, column=1, padx=5, sticky='ew')

        self.duration_label = ttk.Label(bbox_controls, text="5.0秒")
        self.duration_label.grid(row=0, column=2, padx=5)
        duration_scale.configure(command=self.update_duration_display)

        # ★ 新增：语音触发窗口时长控制
        ttk.Label(bbox_controls, text="触发窗口(秒):").grid(row=1, column=0, padx=5)
        self.gate_duration_var = tk.DoubleVar(value=self.voice_trigger_window)
        gate_scale = ttk.Scale(bbox_controls, from_=0.5, to=5.0,
                              variable=self.gate_duration_var, orient='horizontal')
        gate_scale.grid(row=1, column=1, padx=5, sticky='ew')

        self.gate_label = ttk.Label(bbox_controls, text=f"{self.voice_trigger_window}秒")
        self.gate_label.grid(row=1, column=2, padx=5)
        gate_scale.configure(command=self.update_gate_duration)

        self.show_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bbox_controls, text="显示所有检测框",
                        variable=self.show_all_var,
                        command=self.toggle_show_all).grid(row=2, column=0, columnspan=3, pady=5)

        bbox_controls.columnconfigure(1, weight=1)

        detection_frame = ttk.LabelFrame(left_frame, text="检测统计", padding=5)
        detection_frame.pack(fill='both', expand=True, pady=10)

        self.detection_text = tk.Text(detection_frame, height=8, width=50)
        self.detection_text.pack(fill='both', expand=True)

        middle_frame = ttk.LabelFrame(main_frame, text="🎤 语音识别 (Vosk + DeepSeek)", padding=10)
        middle_frame.grid(row=0, column=1, sticky='nsew', padx=5)

        voice_controls = ttk.Frame(middle_frame)
        voice_controls.pack(pady=10)

        self.btn_voice = ttk.Button(voice_controls, text="🎤 开始识别", command=self.toggle_voice)
        self.btn_voice.grid(row=0, column=0, padx=5)

        self.voice_status = ttk.Label(voice_controls, text="未开始", style='Error.TLabel')
        self.voice_status.grid(row=0, column=1, padx=5)

        manual_frame = ttk.LabelFrame(middle_frame, text="手动输入", padding=5)
        manual_frame.pack(fill='x', pady=10)

        self.manual_entry = ttk.Entry(manual_frame, width=40, font=('Arial', 11))
        self.manual_entry.pack(side='left', padx=5)
        self.manual_entry.bind('<Return>', lambda e: self.submit_manual_text())

        ttk.Button(manual_frame, text="提交", command=self.submit_manual_text).pack(side='left')

        # ★ 新增：测试按钮（可选）
        ttk.Button(manual_frame, text="测试锚框", command=self.test_bbox_display).pack(side='left', padx=5)

        transcript_frame = ttk.LabelFrame(middle_frame, text="语音转录", padding=5)
        transcript_frame.pack(fill='both', expand=True, pady=5)

        self.transcript_text = tk.Text(transcript_frame, height=6, width=40)
        scroll1 = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.transcript_text.yview)
        self.transcript_text.configure(yscrollcommand=scroll1.set)
        self.transcript_text.pack(side="left", fill='both', expand=True)
        scroll1.pack(side="right", fill="y")

        keywords_frame = ttk.LabelFrame(middle_frame, text="提取的关键词", padding=5)
        keywords_frame.pack(fill='both', expand=True, pady=5)

        self.keywords_text = tk.Text(keywords_frame, height=4, width=40)
        self.keywords_text.pack(fill='both', expand=True)

        right_frame = ttk.LabelFrame(main_frame, text="📊 Cookie Theft分析", padding=10)
        right_frame.grid(row=0, column=2, sticky='nsew', padx=5)

        analysis_frame = ttk.LabelFrame(right_frame, text="语言分析结果", padding=5)
        analysis_frame.pack(fill='both', expand=True, pady=5)

        self.analysis_text = tk.Text(analysis_frame, height=12, width=45)
        scroll2 = ttk.Scrollbar(analysis_frame, orient="vertical", command=self.analysis_text.yview)
        self.analysis_text.configure(yscrollcommand=scroll2.set)
        self.analysis_text.pack(side="left", fill='both', expand=True)
        scroll2.pack(side="right", fill="y")

        assessment_frame = ttk.LabelFrame(right_frame, text="认知评估", padding=5)
        assessment_frame.pack(fill='both', expand=True, pady=5)

        self.assessment_text = tk.Text(assessment_frame, height=8, width=45)
        self.assessment_text.pack(fill='both', expand=True)

        bottom_frame = ttk.Frame(self.root)
        bottom_frame.pack(fill='x', padx=10, pady=5)

        ttk.Button(bottom_frame, text="▶️ 开始会话", command=self.start_session).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="💾 保存完整报告", command=self.save_comprehensive_report).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="📊 查看详细统计", command=self.show_detailed_statistics).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="🧠 认知评估", command=self.perform_cognitive_assessment).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="🔄 清除关键词", command=self.clear_keywords).pack(side='left', padx=5)
        ttk.Button(bottom_frame, text="🔄 重置会话", command=self.reset_session).pack(side='left', padx=5)

        self.status_bar = ttk.Label(self.root, text="就绪 - 请开始会话", style='Status.TLabel')
        self.status_bar.pack(side='bottom', fill='x', padx=10, pady=2)

        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.columnconfigure(2, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def update_duration_display(self, value):
        """更新持续时间显示"""
        duration = float(value)
        self.duration_label.config(text=f"{duration:.1f}秒")
        self.detector.set_bbox_display_duration(duration)

    def update_gate_duration(self, value):
        """更新语音触发窗口时长"""
        duration = float(value)
        self.voice_trigger_window = duration
        self.gate_label.config(text=f"{duration:.1f}秒")
        self.detector.bbox_manager.set_gate_duration(duration)
        logger.info(f"语音触发窗口时长设置为: {duration:.1f}秒")

    def start_session(self):
        """开始新会话"""
        session_id = self.statistics.start_session()
        self.status_bar.config(text=f"会话已开始: {session_id}")
        messagebox.showinfo("会话开始", f"新会话已开始\n会话ID: {session_id}\n音频将保存到: audio_records/{session_id}/")

    def toggle_camera(self):
        """切换摄像头状态"""
        if not self.video_processor.is_running:
            if self.video_processor.start_camera():
                self.btn_camera.config(text="⏹ 关闭摄像头")
                self.status_bar.config(text="摄像头已开启")
            else:
                messagebox.showerror("错误", "无法打开摄像头")
        else:
            self.video_processor.stop_video()
            self.btn_camera.config(text="📷 开启摄像头")
            self.status_bar.config(text="摄像头已关闭")

    def load_video(self):
        """加载视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv"), ("所有文件", "*.*")]
        )
        if file_path:
            if self.video_processor.load_video(file_path):
                self.status_bar.config(text=f"已加载: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("错误", "无法打开视频文件")

    def toggle_show_all(self):
        """切换是否显示所有检测框"""
        self.show_all_boxes = self.show_all_var.get()

    def toggle_voice(self):
        """切换语音识别状态"""
        if not self.recognizer.is_listening:
            if self.recognizer.start_listening():
                self.btn_voice.config(text="⏹ 停止识别")
                self.voice_status.config(text="Vosk识别中...", style='Success.TLabel')
                if self.recognizer.deepseek_api_key:
                    self.status_bar.config(text="Vosk + DeepSeek 语音识别已启动")
                else:
                    self.status_bar.config(text="Vosk + 简单匹配 语音识别已启动")
            else:
                messagebox.showerror("错误", "无法启动语音识别，请检查Vosk模型文件")
        else:
            self.recognizer.stop_listening()
            self.btn_voice.config(text="🎤 开始识别")
            self.voice_status.config(text="已停止", style='Error.TLabel')

    def submit_manual_text(self):
        """提交手动输入的文本 - 修复版"""
        text = self.manual_entry.get()
        if text:
            # ★ 关键修改：开启语音触发窗口
            logger.info(f"手动输入触发，开启Gate窗口 {self.voice_trigger_window}秒")
            self.detector.bbox_manager.open_gate(self.voice_trigger_window)

            # 清除已显示记录
            self.detector.bbox_manager.clear_displayed_objects()

            # 处理输入
            self.recognizer.manual_input(text)
            self.manual_entry.delete(0, 'end')
            self.status_bar.config(text=f"手动输入已提交: {text}")

    def test_bbox_display(self):
        """测试锚框显示功能"""
        # 直接设置测试关键词
        test_keywords = ['person', 'chair', 'cup', 'bottle']
        self.active_keywords['yolo_classes'] = test_keywords

        # 开启语音触发窗口（较长时间以便测试）
        self.detector.bbox_manager.open_gate(10.0)

        # 显示关键词
        self._display_keywords()

        # 更新状态栏
        self.status_bar.config(text="测试模式：已添加测试关键词，Gate窗口开启10秒")
        logger.info(f"测试模式激活：关键词 {test_keywords}")

        # 弹出提示
        messagebox.showinfo("测试模式",
                          "已激活测试模式\n"
                          "关键词：人、椅子、杯子、瓶子\n"
                          "Gate窗口：10秒\n"
                          "如果画面中有这些物体，应该会显示绿色锚框")

    def clear_keywords(self):
        """清除关键词"""
        self.active_keywords = {'yolo_classes': [], 'special': []}
        self.matched_objects = []
        self.keywords_text.delete('1.0', 'end')
        self.detector.clear_all_bboxes()
        self.detector.bbox_manager.clear_displayed_objects()
        # ★ 同时关闭语音触发窗口
        self.detector.bbox_manager.close_gate()
        self.status_bar.config(text="关键词和锚框已清除，Gate已关闭")

    def reset_session(self):
        """重置会话"""
        if messagebox.askyesno("确认", "确定要重置所有数据吗？"):
            self.statistics.reset_session()
            self.clear_keywords()

            self.transcript_text.delete('1.0', 'end')
            self.analysis_text.delete('1.0', 'end')
            self.assessment_text.delete('1.0', 'end')
            self.detection_text.delete('1.0', 'end')

            self.detector.bbox_manager.clear_displayed_objects()
            self.detector.bbox_manager.close_gate()  # ★ 关闭Gate
            self.status_bar.config(text="会话已重置")

    def update_display(self):
        """更新显示内容"""
        if self.video_processor.is_running:
            frame = self.video_processor.get_frame()
            if frame is not None:
                annotated_frame, detections = self.video_processor.process_frame(
                    frame, self.active_keywords['yolo_classes'],
                    self.show_all_boxes, self.recognizer.YOLO_CHINESE_MAP
                )
                self.current_frame = annotated_frame

                for detection in detections:
                    self.statistics.add_detection(detection)

                self._display_frame(annotated_frame)

        self._update_speech_analysis()
        self._update_bbox_statistics()
        self.root.after(100, self.update_display)

    def _display_frame(self, frame):
        """显示视频帧"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = frame_rgb.shape[:2]
        max_width = 500
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=imgtk)
        self.video_label.image = imgtk

    def _update_speech_analysis(self):
        """更新语音分析 - 修复版"""
        new_transcripts = self.recognizer.get_latest_transcript()

        for transcript_data in new_transcripts:
            text = transcript_data['text']
            audio_data = transcript_data.get('audio_data')
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.transcript_text.insert('end', f"[{timestamp}] {text}\n")
            self.transcript_text.see('end')

            # 进行分析
            analysis = self.statistics.add_transcript(text, audio_data)
            self._display_analysis_result(analysis)

        # 获取新的关键词
        new_keywords = self.recognizer.get_latest_keywords()

        if new_keywords['yolo_classes'] or new_keywords['special']:
            # ★ 关键修改：开启语音触发窗口
            logger.info(f"检测到新关键词，开启Gate窗口 {self.voice_trigger_window}秒")
            self.detector.bbox_manager.open_gate(self.voice_trigger_window)

            # 清除所有锚框
            logger.info("清除已有锚框")
            self.detector.bbox_manager.clear_all_boxes()

            # 更新关键词
            self.active_keywords['yolo_classes'] = new_keywords['yolo_classes']
            self.active_keywords['special'] = new_keywords['special']

            # 去重
            self.active_keywords['yolo_classes'] = list(set(self.active_keywords['yolo_classes']))
            self.active_keywords['special'] = list(set(self.active_keywords['special']))

            # 显示新的关键词
            self._display_keywords()

            # 更新状态栏
            self.status_bar.config(text=f"新关键词已激活，Gate窗口开启 {self.voice_trigger_window}秒")

    def _display_analysis_result(self, analysis):
        """显示Cookie Theft分析结果"""
        if not analysis:
            return

        summary = analysis.get('summary', {})
        percentages = analysis.get('percentages', {})

        result_text = f"""
📝 最新分析结果 [{datetime.now().strftime('%H:%M:%S')}]
    
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总字数: {summary.get('total_words', 0)}

📊 各项指标:
• 错误表述: {summary.get('error_rate', '0%')}
• 不流畅表述: {summary.get('disfluency_rate', '0%')}
• 结构支持词汇: {summary.get('support_structure_rate', '0%')}
• 重复内容: {summary.get('repetition_rate', '0%')}
• 有效信息: {summary.get('valid_information_rate', '0%')}
• 解释性表述: {summary.get('interpretive_rate', '0%')}
• 无关词汇: {summary.get('irrelevant_rate', '0%')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.analysis_text.insert('1.0', result_text)
        self.analysis_text.see('1.0')

    def _display_keywords(self):
        """显示关键词"""
        self.keywords_text.delete('1.0', 'end')

        if self.active_keywords['yolo_classes']:
            self.keywords_text.insert('end', "🎯 YOLO物体关键词:\n")
            for kw in self.active_keywords['yolo_classes']:
                chinese = self.recognizer.YOLO_CHINESE_MAP.get(kw, kw)
                self.keywords_text.insert('end', f"  • {chinese} ({kw})\n")

        if self.active_keywords['special']:
            self.keywords_text.insert('end', "\n🍪 Cookie Theft特殊词汇:\n")
            for kw in self.active_keywords['special']:
                self.keywords_text.insert('end', f"  • {kw}\n")

    def _update_bbox_statistics(self):
        """更新锚框统计"""
        bbox_stats = self.detector.get_bbox_statistics()
        self.statistics.update_bbox_statistics(bbox_stats)

        self.detection_text.delete('1.0', 'end')

        # 获取Gate状态
        gate_status = "开启" if self.detector.bbox_manager.is_gate_open() else "关闭"
        gate_remaining = ""
        if self.detector.bbox_manager.is_gate_open() and self.detector.bbox_manager.gate_open_until:
            remaining = (self.detector.bbox_manager.gate_open_until - datetime.now()).total_seconds()
            if remaining > 0:
                gate_remaining = f" (剩余{remaining:.1f}秒)"

        detection_info = f"""
📊 检测统计:
• 当前活跃锚框: {bbox_stats['currently_active']}
• 总创建锚框: {bbox_stats['total_created']}
• 锚框显示时长: {bbox_stats['display_duration']}秒
• Gate状态: {gate_status}{gate_remaining}

🎯 当前关键词:
"""
        self.detection_text.insert('1.0', detection_info)

        # 显示当前关键词
        for obj in self.active_keywords['yolo_classes']:
            chinese_name = self.recognizer.YOLO_CHINESE_MAP.get(obj, obj)
            self.detection_text.insert('end', f"  ✅ {chinese_name}\n")

    def perform_cognitive_assessment(self):
        """执行认知评估"""
        comprehensive_analysis = self.statistics.get_comprehensive_analysis()
        if not comprehensive_analysis:
            messagebox.showwarning("警告", "没有数据可分析，请先进行语音录制")
            return
        assessment_result = self.cognitive_model.predict_cognitive_status(comprehensive_analysis)
        self._display_cognitive_assessment(assessment_result)

    def _display_cognitive_assessment(self, assessment_result):
        """显示认知评估结果"""
        self.assessment_text.delete('1.0', 'end')
        assessment_type = assessment_result.get('assessment_type', 'unknown')

        if assessment_type == 'rule_based':
            assessment_text = f"""
🧠 认知评估结果 (基于规则)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
评估等级: {assessment_result.get('assessment', 'unknown')}
风险得分: {assessment_result.get('risk_score', 0):.2%}

📋 建议: {assessment_result.get('recommendation', '')}

🔍 风险因素:
"""
            risk_factors = assessment_result.get('risk_factors', {})
            for factor, present in risk_factors.items():
                status = "⚠️ 是" if present else "✅ 否"
                factor_name = {
                    'high_error_rate': '高错误率',
                    'high_disfluency': '高不流畅率',
                    'low_valid_info': '低有效信息',
                    'high_irrelevant': '高无关内容',
                    'excessive_repetition': '过度重复'
                }.get(factor, factor)
                assessment_text += f"  • {factor_name}: {status}\n"
        else:
            assessment_text = f"""
🧠 认知评估结果 (机器学习模型)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
预测结果: {assessment_result.get('prediction', 'unknown')}
置信度: {assessment_result.get('confidence', 0):.2%}

🔍 重要特征:
"""
            feature_importance = assessment_result.get('feature_importance', {})
            for feature, importance in list(feature_importance.items())[:5]:
                assessment_text += f"  • {feature}: {importance:.3f}\n"

        assessment_text += f"\n⏰ 评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        self.assessment_text.insert('1.0', assessment_text)

    def save_comprehensive_report(self):
        """保存综合报告"""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")],
            initialfile=f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        if filepath:
            saved_path = self.statistics.save_session_report(filepath)
            if saved_path:
                messagebox.showinfo("保存成功", f"综合报告已保存到:\n{saved_path}")
            else:
                messagebox.showerror("保存失败", "报告保存失败，请检查文件路径")

    def show_detailed_statistics(self):
        """显示详细统计信息"""
        comprehensive_analysis = self.statistics.get_comprehensive_analysis()
        if not comprehensive_analysis:
            messagebox.showwarning("警告", "没有数据可显示")
            return

        stats_window = tk.Toplevel(self.root)
        stats_window.title("详细统计信息")
        stats_window.geometry("800x600")

        text_frame = ttk.Frame(stats_window)
        text_frame.pack(fill='both', expand=True, padx=10, pady=10)

        stats_text = tk.Text(text_frame, wrap='word')
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=stats_text.yview)
        stats_text.configure(yscrollcommand=scrollbar.set)

        stats_text.pack(side="left", fill='both', expand=True)
        scrollbar.pack(side="right", fill="y")

        stats_content = self._generate_detailed_stats_content(comprehensive_analysis)
        stats_text.insert('1.0', stats_content)

        ttk.Button(stats_window, text="关闭", command=stats_window.destroy).pack(pady=10)

    def _generate_detailed_stats_content(self, analysis):
        """生成详细统计内容"""
        session_info = analysis.get('session_info', {})
        lang_summary = analysis.get('language_analysis_summary', {})
        detection_summary = analysis.get('detection_summary', {})
        audio_info = analysis.get('audio_info', {})

        content = f"""
🧠 认知症筛查系统 - 详细统计报告
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 会话信息:
• 开始时间: {session_info.get('start_time', 'N/A')}
• 结束时间: {session_info.get('end_time', 'N/A')}
• 持续时间: {session_info.get('duration_formatted', 'N/A')}

🎤 音频信息:
• 会话ID: {audio_info.get('session_id', 'N/A')}
• 音频保存目录: {audio_info.get('audio_dir', 'N/A')}
• 音频片段数量: {audio_info.get('segments_count', 0)}

📝 语言分析汇总:
• 总字符数: {lang_summary.get('total_characters', 0)}
• 转录次数: {lang_summary.get('transcript_count', 0)}

📊 Cookie Theft分析结果:
"""
        breakdown = lang_summary.get('detailed_breakdown', {})
        for metric, value in breakdown.items():
            metric_name = {
                'error_rate': '错误表述百分比',
                'disfluency_rate': '不流畅表述百分比',
                'support_structure_rate': '结构支持词汇百分比',
                'repetition_rate': '重复内容百分比',
                'valid_information_rate': '有效信息百分比',
                'interpretive_rate': '解释性表述百分比',
                'irrelevant_rate': '无关词汇百分比'
            }.get(metric, metric)
            content += f"• {metric_name}: {value}\n"

        content += f"""

🎯 视觉检测统计:
• 总检测次数: {detection_summary.get('total_detections', 0)}
• 独特物体种类: {detection_summary.get('unique_objects', 0)}
• 成功匹配次数: {detection_summary.get('successful_matches', 0)}
• 匹配成功率: {detection_summary.get('match_rate', 0):.2f}%

📦 物体检测详情:
"""
        detection_breakdown = detection_summary.get('detection_breakdown', {})
        for obj, count in detection_breakdown.items():
            chinese_name = self.recognizer.YOLO_CHINESE_MAP.get(obj, obj)
            content += f"• {chinese_name} ({obj}): {count}次\n"

        content += f"""

⏰ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return content

    def on_closing(self):
        """窗口关闭事件"""
        if messagebox.askokcancel("退出", "确定要退出系统吗？"):
            self.video_processor.stop_video()
            self.recognizer.stop_listening()
            cv2.destroyAllWindows()
            self.root.destroy()

    def run(self):
        """运行应用"""
        logger.info("系统启动 - 完整版Cookie Theft分析系统")
        self.root.mainloop()


def main():
    """主函数"""
    print("="*80)
    print("🧠 认知症筛查系统 - Cookie Theft测试分析")
    print("YOLO11 + Vosk离线识别 + DeepSeek智能理解 + 完整语言分析")
    print("="*80)

    print(f"Python版本: {os.sys.version}")
    print(f"OpenCV版本: {cv2.__version__}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

    print("\n📋 依赖检查:")
    dependencies = [
        ('vosk', 'Vosk语音识别'),
        ('pyaudio', 'PyAudio音频处理'),
        ('jieba', 'Jieba中文分词'),
        ('sklearn', 'Scikit-learn机器学习'),
        ('requests', 'Requests网络请求')
    ]

    missing_deps = []
    for dep, desc in dependencies:
        try:
            __import__(dep)
            print(f"✅ {desc} - 已安装")
        except ImportError:
            print(f"❌ {desc} - 未安装")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n⚠️ 缺少依赖，请安装: pip install {' '.join(missing_deps)}")
        return

    print("\n📁 模型检查:")
    if os.path.exists("vosk-model-cn-0.22"):
        print("✅ Vosk中文模型 - 已就绪")
    else:
        print("❌ Vosk中文模型 - 缺失")
        print("   请从 https://alphacephei.com/vosk/models 下载 vosk-model-cn-0.22")

    print("\n🔑 API配置:")
    if os.getenv('DEEPSEEK_API_KEY'):
        print("✅ DeepSeek API密钥 - 已配置")
    else:
        print("⚠️ DeepSeek API密钥 - 未配置（启动时将询问）")

    print("\n📂 目录准备:")
    os.makedirs("audio_records", exist_ok=True)
    print("✅ 音频记录目录 - 已创建")

    print("\n" + "="*80)
    print("🚀 系统功能:")
    print("• Vosk离线语音识别，保护隐私")
    print("• DeepSeek API智能关键词提取和纠错")
    print("• Cookie Theft测试完整8步分析")
    print("• 可配置锚框显示时间和语音触发窗口")
    print("• 自动保存音频记录")
    print("• 实时认知评估")
    print("• 详细统计报告生成")
    print("="*80)

    try:
        app = CognitiveAssessmentApp()
        app.run()
    except Exception as e:
        logger.error(f"系统启动失败: {e}")
        print(f"\n❌ 错误: {e}")
        print("\n🔧 故障排除:")
        print("1. 检查所有依赖是否正确安装")
        print("2. 确保vosk-model-cn-0.22模型文件存在")
        print("3. 检查摄像头和麦克风权限")
        print("4. 确保网络连接正常（DeepSeek API）")

if __name__ == "__main__":
    main()