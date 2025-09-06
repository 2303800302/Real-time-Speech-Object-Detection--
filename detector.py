"""
目标检测模块 - detector.py
负责YOLO目标检测和锚框管理
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import logging
from datetime import datetime, timedelta
from PIL import Image, ImageFont, ImageDraw
import os

logger = logging.getLogger(__name__)

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

class BoundingBoxManager:
    """锚框管理器 - 控制锚框显示时间 + 语音触发窗口（gate）"""

    def __init__(self, display_duration=5.0, overlap_threshold=0.5):
        """
        初始化锚框管理器
        Args:
            display_duration: 锚框显示持续时间（秒）
            overlap_threshold: 重叠阈值（IoU），超过此值认为是同一物体
        """
        self.display_duration = display_duration
        self.overlap_threshold = overlap_threshold
        self.active_boxes = {}  # {object_id: {'bbox': ..., 'expire_time': ..., 'class': ..., 'confidence': ...}}
        self.box_counter = 0
        self.recently_expired = {}  # {class_name: expire_time} 记录最近过期的锚框（冷却）

        # ★ 新增：语音触发窗口（仅在窗口开启时才允许“创建新锚框”）
        self.gate_open_until = None
        self.gate_default_duration = 1.0  # 秒，可按需调整

    # ---------- Gate 控制 ----------
    def open_gate(self, duration=None):
        """开启语音触发窗口，在窗口内允许创建新锚框"""
        if duration is None:
            duration = self.gate_default_duration
        self.gate_open_until = datetime.now() + timedelta(seconds=duration)
        logger.info(f"语音触发窗口已开启，持续 {duration:.2f}s")

    def close_gate(self):
        """关闭语音触发窗口，不允许创建新锚框"""
        self.gate_open_until = None
        logger.info("语音触发窗口已关闭")

    def is_gate_open(self):
        """当前是否处于语音触发窗口内"""
        return self.gate_open_until is not None and datetime.now() <= self.gate_open_until

    def set_gate_duration(self, duration):
        """设置默认语音触发窗口时长"""
        self.gate_default_duration = float(duration)
        logger.info(f"语音触发窗口默认时长设置为: {self.gate_default_duration:.2f}s")

    # ---------- 常规控制 ----------
    def set_display_duration(self, duration):
        """设置锚框显示持续时间"""
        self.display_duration = duration
        logger.info(f"锚框显示时间设置为: {duration}秒")

    def _calculate_iou(self, box1, box2):
        """
        计算两个边界框的IoU（Intersection over Union）
        Args:
            box1, box2: [x1, y1, x2, y2]格式的边界框
        Returns:
            float: IoU值
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _find_overlapping_box(self, new_bbox, class_name):
        """
        查找与新边界框重叠的现有锚框
        Args:
            new_bbox: 新的边界框 [x1, y1, x2, y2]
            class_name: 物体类别名称
        Returns:
            str or None: 重叠的锚框ID，如果没有重叠返回None
        """
        for box_id, box_info in self.active_boxes.items():
            if box_info['class'] != class_name:
                continue
            iou = self._calculate_iou(new_bbox, box_info['bbox'])
            if iou > self.overlap_threshold:
                return box_id
        return None

    def add_detection(self, detection, keyword_matched=False):
        """
        将本帧检测结果尝试写入锚框系统。
        规则：
        - 若与已有锚框重叠：允许“更新位置”（不延长寿命），无论 gate 是否开启；
        - 若无重叠：仅当 (keyword_matched == True 且 gate 开启) 才能“创建新锚框”；
        """
        new_bbox = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        current_time = datetime.now()

        # 先看是否与已有锚框重叠 —— 允许更新位置（即使 gate 已关闭）
        overlapping_box_id = self._find_overlapping_box(new_bbox, class_name)
        if overlapping_box_id:
            old_box = self.active_boxes[overlapping_box_id]
            old_confidence = old_box['confidence']
            old_bbox = old_box['bbox']

            total_conf = old_confidence + confidence
            if total_conf <= 0:
                total_conf = 1e-6
            weight_old = old_confidence / total_conf
            weight_new = confidence / total_conf

            updated_bbox = [
                int(old_bbox[0] * weight_old + new_bbox[0] * weight_new),
                int(old_bbox[1] * weight_old + new_bbox[1] * weight_new),
                int(old_bbox[2] * weight_old + new_bbox[2] * weight_new),
                int(old_bbox[3] * weight_old + new_bbox[3] * weight_new)
            ]

            self.active_boxes[overlapping_box_id].update({
                'bbox': updated_bbox,
                'confidence': max(old_confidence, confidence),
                'updated_time': current_time,
                'update_count': old_box.get('update_count', 0) + 1
            })
            return  # 已更新，直接返回

        # 若无重叠：仅在 gate 开启 & 匹配关键词 时允许创建新锚框
        if not keyword_matched or not self.is_gate_open():
            return

        # 冷却（可保留，避免刚过期马上重建；gate 机制已经足够严格，这里作为额外保险）
        if class_name in self.recently_expired:
            if current_time < self.recently_expired[class_name]:
                return
            else:
                del self.recently_expired[class_name]

        # 创建新锚框
        self.box_counter += 1
        expire_time = current_time + timedelta(seconds=self.display_duration)
        box_id = f"box_{self.box_counter}_{class_name}"
        self.active_boxes[box_id] = {
            'bbox': new_bbox,
            'class': class_name,
            'confidence': confidence,
            'expire_time': expire_time,
            'added_time': current_time,
            'update_count': 0
        }
        logger.info(f"添加新锚框: {class_name} (显示{self.display_duration}秒)")

    def get_active_boxes(self):
        """获取当前有效的锚框，同时清理过期锚框并记录冷却"""
        current_time = datetime.now()
        cooldown_time = 2.0  # 秒

        expired_boxes = []
        for box_id, box_info in self.active_boxes.items():
            if current_time > box_info['expire_time']:
                expired_boxes.append(box_id)
                class_name = box_info['class']
                self.recently_expired[class_name] = current_time + timedelta(seconds=cooldown_time)

        for box_id in expired_boxes:
            logger.debug(f"移除过期锚框: {self.active_boxes[box_id]['class']}")
            del self.active_boxes[box_id]

        return list(self.active_boxes.values())

    def clear_all_boxes(self):
        """清除所有锚框和记录"""
        self.active_boxes.clear()
        self.recently_expired.clear()
        self.box_counter = 0
        logger.info("清除所有锚框")

    # 兼容旧调用：仅清除冷却期记录
    def clear_displayed_objects(self):
        self.recently_expired.clear()
        logger.info("清除冷却期记录")

    def get_box_stats(self):
        """获取锚框统计信息"""
        return {
            'total_created': self.box_counter,
            'currently_active': len(self.active_boxes),
            'display_duration': self.display_duration
        }

class YOLOv11Detector:
    """YOLO11目标检测器"""

    def __init__(self, model_path='yolo11n.pt'):
        """初始化YOLO11检测器"""
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
            logger.info("尝试下载默认模型...")
            try:
                self.model = YOLO('yolov8n.pt')  # 备用模型
                self.model.to(self.device)
                self.class_names = self.model.names
                logger.info("使用备用模型 YOLOv8n")
            except:
                raise Exception("无法加载YOLO模型，请检查网络连接或模型文件")

        # 初始化锚框管理器
        self.bbox_manager = BoundingBoxManager()

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

    def set_bbox_display_duration(self, duration):
        """设置锚框显示持续时间"""
        self.bbox_manager.set_display_duration(duration)

    def update_detections_with_keywords(self, detections, active_keywords):
        """根据关键词更新检测结果到锚框管理器"""
        for detection in detections:
            keyword_matched = detection['class'] in active_keywords
            self.bbox_manager.add_detection(detection, keyword_matched)

    def draw_annotations(self, frame, detections, active_keywords, show_all_boxes=False,
                         chinese_map=None):
        """
        绘制检测结果
        Args:
            frame: 输入帧
            detections: 所有检测结果
            active_keywords: 激活的关键词列表
            show_all_boxes: 是否显示所有检测框
            chinese_map: 中英文映射字典
        """
        annotated = frame.copy()

        if chinese_map is None:
            chinese_map = {}

        # 更新锚框管理器（★ 创建新锚框仅在 gate 开启时发生）
        self.update_detections_with_keywords(detections, active_keywords)

        if show_all_boxes:
            # 显示所有检测框（实时检测）
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                class_name = det['class']
                confidence = det['confidence']
                is_matched = class_name in active_keywords
                color = (0, 255, 0) if is_matched else (0, 0, 255)
                thickness = 3 if is_matched else 2
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                chinese_name = chinese_map.get(class_name, class_name)
                label = f"{chinese_name} {confidence:.2f}"
                annotated = cv2_put_chinese_text(
                    annotated, label, (x1, y1 - 5), font_size=16, color=color
                )

        # 显示处于生命周期内的“持续锚框”
        active_boxes = self.bbox_manager.get_active_boxes()
        for box_info in active_boxes:
            x1, y1, x2, y2 = box_info['bbox']
            class_name = box_info['class']
            confidence = box_info['confidence']
            remaining_time = (box_info['expire_time'] - datetime.now()).total_seconds()
            alpha = max(0.3, remaining_time / self.bbox_manager.display_duration)
            color = (0, int(255 * alpha), 0)  # 绿色
            thickness = 4

            update_count = box_info.get('update_count', 0)
            marker = f"🔄({update_count})" if update_count > 0 else "🎯"

            chinese_name = chinese_map.get(class_name, class_name)
            label = f"{marker}{chinese_name} {confidence:.2f} ({remaining_time:.1f}s)"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            annotated = cv2_put_chinese_text(
                annotated, label, (x1, y1 - 35), font_size=16, color=color
            )

            # 置信度条
            bar_width = min(100, x2 - x1 - 10)
            bar_height = 8
            bar_x = x1 + 5
            bar_y = y1 + 5
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + bar_width, bar_y + bar_height),
                          (50, 50, 50), -1)
            conf_width = int(bar_width * confidence)
            conf_color = (0, int(255 * confidence), int(255 * (1 - confidence)))
            cv2.rectangle(annotated, (bar_x, bar_y),
                          (bar_x + conf_width, bar_y + bar_height),
                          conf_color, -1)

        return annotated

    def get_bbox_statistics(self):
        """获取锚框统计信息"""
        return self.bbox_manager.get_box_stats()

    def clear_all_bboxes(self):
        """清除所有锚框"""
        self.bbox_manager.clear_all_boxes()

class VideoProcessor:
    """视频处理器"""

    def __init__(self, detector):
        """
        初始化视频处理器
        Args:
            detector: YOLO检测器实例
        """
        self.detector = detector
        self.cap = None
        self.is_running = False

    def start_camera(self, camera_id=0):
        """启动摄像头"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if self.cap.isOpened():
                self.is_running = True
                logger.info("摄像头启动成功")
                return True
            else:
                logger.error("无法打开摄像头")
                return False
        except Exception as e:
            logger.error(f"摄像头启动失败: {e}")
            return False

    def load_video(self, video_path):
        """加载视频文件"""
        try:
            self.cap = cv2.VideoCapture(video_path)
            if self.cap.isOpened():
                self.is_running = True
                logger.info(f"视频文件加载成功: {video_path}")
                return True
            else:
                logger.error(f"无法打开视频文件: {video_path}")
                return False
        except Exception as e:
            logger.error(f"视频文件加载失败: {e}")
            return False

    def stop_video(self):
        """停止视频"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info("视频已停止")

    def get_frame(self):
        """获取一帧图像"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def process_frame(self, frame, active_keywords, show_all_boxes=False, chinese_map=None):
        """处理单帧图像"""
        detections = self.detector.detect(frame)
        annotated_frame = self.detector.draw_annotations(
            frame, detections, active_keywords, show_all_boxes, chinese_map
        )
        return annotated_frame, detections
