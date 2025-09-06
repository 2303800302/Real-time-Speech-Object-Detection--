"""
统计分析模块 - statistics.py
负责Cookie Theft测试的语音分析和统计计算
"""

import re
import os
import json
import wave
import logging
from datetime import datetime
from collections import defaultdict, Counter
from pathlib import Path
import jieba
import jieba.posseg as pseg

logger = logging.getLogger(__name__)


class AudioRecorder:
    """音频录制器"""

    def __init__(self, sample_rate=16000, channels=1):
        """
        初始化音频录制器
        Args:
            sample_rate: 采样率
            channels: 声道数
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.audio_dir = self._create_audio_directory()
        self.current_session = None
        self.audio_segments = []

    def _create_audio_directory(self):
        """创建音频保存目录"""
        audio_dir = Path("audio_records")
        audio_dir.mkdir(exist_ok=True)
        logger.info(f"音频保存目录: {audio_dir.absolute()}")
        return audio_dir

    def start_session(self, session_id=None):
        """开始新的录音会话"""
        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.current_session = session_id
        session_dir = self.audio_dir / session_id
        session_dir.mkdir(exist_ok=True)

        self.audio_segments = []
        logger.info(f"开始录音会话: {session_id}")
        return session_dir

    def save_audio_segment(self, audio_data, segment_id=None):
        """保存音频片段"""
        if not self.current_session:
            logger.error("未开始录音会话")
            return None

        if segment_id is None:
            segment_id = len(self.audio_segments)

        filename = f"segment_{segment_id:03d}.wav"
        filepath = self.audio_dir / self.current_session / filename

        try:
            with wave.open(str(filepath), 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)

            self.audio_segments.append({
                'id': segment_id,
                'filename': filename,
                'filepath': str(filepath),
                'timestamp': datetime.now().isoformat(),
                'size': len(audio_data)
            })

            logger.info(f"保存音频片段: {filename}")
            return filepath

        except Exception as e:
            logger.error(f"保存音频失败: {e}")
            return None

    def get_session_info(self):
        """获取当前会话信息"""
        return {
            'session_id': self.current_session,
            'audio_dir': str(self.audio_dir / self.current_session) if self.current_session else None,
            'segments_count': len(self.audio_segments),
            'segments': self.audio_segments
        }


class CookieTheftAnalyzer:
    """Cookie Theft测试分析器"""

    def __init__(self):
        """初始化分析器"""
        # 初始化jieba分词
        jieba.initialize()

        # 定义分析规则
        self._init_analysis_rules()

    def _init_analysis_rules(self):
        """初始化分析规则"""
        # 第二步：错误表述模式
        self.error_patterns = {
            'gender_confusion': ['男孩.*父亲', '女孩.*母亲', '男.*女', '女.*男'],
            'pronoun_errors': ['他们的.*他的', '她的.*他的', '我们的.*我的'],
            'vague_references': ['这个东西', '那个东西', '什么东西', '这玩意', '那玩意']
        }

        # 第三步：不流畅表述
        self.disfluency_patterns = {
            'fillers': ['嗯', '呃', '啊', '这个', '那个', '就是'],
            'false_starts': ['这.*这个', '那.*那个'],
            'repetitions': r'(.+)\1+',  # 重复模式
        }

        # 第四步：结构支持词汇
        self.structural_support = {
            'task_related': ['就这些了', '这是什么意思', '对', '嗯', '不知道'],
            'descriptive_void': ['这个是', '好像是', '似乎是'],
            'modal_particles': ['呢', '呀', '吧', '啊'],
            'non_specific': ['什么的', '类似的', '之类的'],
            'progressive_words': ['此外', '还', '也', '另外'],
            'conjunctions': ['和', '因为', '或者', '在这个时候', '然后'],
            'articles': ['一个', '这个', '那个', '这', '那'],
            'clear_pronouns': ['她', '他的', '他们的', '它的']
        }

        # 第六步：有效信息关键词（图片中可能出现的元素）
        self.valid_content_keywords = [
            # 人物
            '男孩', '女孩', '孩子', '小孩', '儿童', '妈妈', '母亲', '女人', '女士',
            # 厨房用品
            '厨房', '水槽', '水龙头', '水', '盘子', '碟子', '餐具', '柜子', '橱柜',
            '饼干', '曲奇', '饼干罐', '罐子', '食物',
            # 家具
            '椅子', '凳子', '板凳', '桌子', '台子',
            # 窗户
            '窗户', '窗帘', '玻璃',
            # 动作（有效的图片信息）
            '站', '坐', '爬', '够', '拿', '伸手', '倒', '流', '溢出'
        ]

        # 第七步：解释性表述模式
        self.interpretive_patterns = [
            '想要', '打算', '准备', '可能', '应该', '估计', '觉得', '认为',
            '关系', '家庭', '亲情', '危险', '小心', '担心'
        ]

        # 第八步：无关词汇
        self.irrelevant_words = [
            '看不清', '看不出来', '看不见', '不清楚', '模糊', '不知道'
        ]

    def analyze_transcript(self, text):
        """
        分析转录文本，执行Cookie Theft测试的8步分析
        Args:
            text: 转录文本
        Returns:
            dict: 详细分析结果
        """
        # 预处理文本
        cleaned_text = self._preprocess_text(text)

        # 第一步：计算总字数
        total_chars = self._count_total_characters(cleaned_text)

        # 分词处理
        words = list(jieba.cut(cleaned_text))
        word_pos = list(pseg.cut(cleaned_text))

        # 执行8步分析
        analysis_result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'total_characters': total_chars,
            'step1': self._step1_total_count(cleaned_text),
            'step2': self._step2_error_expressions(cleaned_text, words),
            'step3': self._step3_disfluent_expressions(cleaned_text, words),
            'step4': self._step4_structural_support(cleaned_text, words),
            'step5': self._step5_repetitive_content(cleaned_text, words),
            'step6': self._step6_valid_information(cleaned_text, words),
            'step7': self._step7_interpretive_expressions(cleaned_text, words),
            'step8': self._step8_irrelevant_words(cleaned_text, words)
        }

        # 计算百分比
        analysis_result['percentages'] = self._calculate_percentages(analysis_result)

        # 生成摘要
        analysis_result['summary'] = self._generate_summary(analysis_result)

        return analysis_result

    def _preprocess_text(self, text):
        """预处理文本"""
        # 移除儿化音
        text = re.sub(r'([^儿])儿(?=[^a-zA-Z]|$)', r'\1', text)

        # 移除多余的空格和标点
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        return text

    def _count_total_characters(self, text):
        """计算总字数（不包括标点和空格）"""
        # 移除标点和空格
        clean_text = re.sub(r'[^\u4e00-\u9fff]', '', text)
        return len(clean_text)

    def _step1_total_count(self, text):
        """第一步：精确转录录音，计算总字数"""
        total_chars = self._count_total_characters(text)
        return {
            'total_characters': total_chars,
            'description': '从被试完全理解任务并开始描述相关信息开始转录'
        }

    def _step2_error_expressions(self, text, words):
        """第二步：挑出文本中的错误表述"""
        errors = {
            'gender_confusion': [],
            'pronoun_errors': [],
            'vague_references': [],
            'total_errors': 0
        }

        # 检查性别混淆
        for pattern in self.error_patterns['gender_confusion']:
            matches = re.findall(pattern, text)
            errors['gender_confusion'].extend(matches)

        # 检查代词错误
        for pattern in self.error_patterns['pronoun_errors']:
            matches = re.findall(pattern, text)
            errors['pronoun_errors'].extend(matches)

        # 检查模糊指代
        for pattern in self.error_patterns['vague_references']:
            if pattern in text:
                errors['vague_references'].append(pattern)

        errors['total_errors'] = (len(errors['gender_confusion']) +
                                  len(errors['pronoun_errors']) +
                                  len(errors['vague_references']))

        return errors

    def _step3_disfluent_expressions(self, text, words):
        """第三步：挑出文本中不流畅的表述"""
        disfluencies = {
            'fillers': [],
            'false_starts': [],
            'repetitions': [],
            'total_disfluencies': 0
        }

        # 检查填充词
        for word in words:
            if word in self.disfluency_patterns['fillers']:
                disfluencies['fillers'].append(word)

        # 检查错误开始
        for pattern in self.disfluency_patterns['false_starts']:
            matches = re.findall(pattern, text)
            disfluencies['false_starts'].extend(matches)

        # 检查重复
        repetition_matches = re.findall(self.disfluency_patterns['repetitions'], text)
        disfluencies['repetitions'] = repetition_matches

        disfluencies['total_disfluencies'] = (len(disfluencies['fillers']) +
                                              len(disfluencies['false_starts']) +
                                              len(disfluencies['repetitions']))

        return disfluencies

    def _step4_structural_support(self, text, words):
        """第四步：挑出提供结构支持的字词"""
        support_words = {
            'task_related': [],
            'descriptive_void': [],
            'modal_particles': [],
            'non_specific': [],
            'progressive_words': [],
            'conjunctions': [],
            'articles': [],
            'clear_pronouns': [],
            'total_support': 0
        }

        # 检查各类结构支持词
        for category, word_list in self.structural_support.items():
            for word in word_list:
                if word in text:
                    support_words[category].append(word)

        # 计算总数
        total = sum(len(words) for key, words in support_words.items()
                    if key != 'total_support')
        support_words['total_support'] = total

        return support_words

    def _step5_repetitive_content(self, text, words):
        """第五步：挑出重复描述的内容"""
        # 使用n-gram方法检测重复
        repetitions = {
            'repeated_phrases': [],
            'repeated_words': [],
            'total_repetitions': 0
        }

        # 检查重复的词
        word_counts = Counter(words)
        repeated_words = {word: count for word, count in word_counts.items()
                          if count > 1 and len(word) > 1}

        repetitions['repeated_words'] = repeated_words

        # 检查重复的短语（2-3个字的组合）
        bigrams = [words[i] + words[i + 1] for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = {phrase: count for phrase, count in bigram_counts.items()
                            if count > 1}

        repetitions['repeated_phrases'] = repeated_bigrams
        repetitions['total_repetitions'] = len(repeated_words) + len(repeated_bigrams)

        return repetitions

    def _step6_valid_information(self, text, words):
        """第六步：挑出表述图片有效信息"""
        valid_info = {
            'valid_keywords': [],
            'total_valid': 0
        }

        for keyword in self.valid_content_keywords:
            if keyword in text:
                valid_info['valid_keywords'].append(keyword)

        valid_info['total_valid'] = len(valid_info['valid_keywords'])

        return valid_info

    def _step7_interpretive_expressions(self, text, words):
        """第七步：挑出解释图片信息的表述"""
        interpretive = {
            'interpretive_words': [],
            'total_interpretive': 0
        }

        for pattern in self.interpretive_patterns:
            if pattern in text:
                interpretive['interpretive_words'].append(pattern)

        interpretive['total_interpretive'] = len(interpretive['interpretive_words'])

        return interpretive

    def _step8_irrelevant_words(self, text, words):
        """第八步：统计剩余无关字词"""
        irrelevant = {
            'irrelevant_words': [],
            'total_irrelevant': 0
        }

        for word in self.irrelevant_words:
            if word in text:
                irrelevant['irrelevant_words'].append(word)

        irrelevant['total_irrelevant'] = len(irrelevant['irrelevant_words'])

        return irrelevant

    def _calculate_percentages(self, analysis):
        """计算各项百分比"""
        total_chars = analysis['total_characters']

        if total_chars == 0:
            return {key: 0.0 for key in [
                'error_percentage', 'disfluency_percentage', 'support_percentage',
                'repetition_percentage', 'valid_info_percentage',
                'interpretive_percentage', 'irrelevant_percentage'
            ]}

        return {
            'error_percentage': (analysis['step2']['total_errors'] / total_chars) * 100,
            'disfluency_percentage': (analysis['step3']['total_disfluencies'] / total_chars) * 100,
            'support_percentage': (analysis['step4']['total_support'] / total_chars) * 100,
            'repetition_percentage': (analysis['step5']['total_repetitions'] / total_chars) * 100,
            'valid_info_percentage': (analysis['step6']['total_valid'] / total_chars) * 100,
            'interpretive_percentage': (analysis['step7']['total_interpretive'] / total_chars) * 100,
            'irrelevant_percentage': (analysis['step8']['total_irrelevant'] / total_chars) * 100
        }

    def _generate_summary(self, analysis):
        """生成分析摘要"""
        percentages = analysis['percentages']

        return {
            'total_words': analysis['total_characters'],
            'error_rate': f"{percentages['error_percentage']:.2f}%",
            'disfluency_rate': f"{percentages['disfluency_percentage']:.2f}%",
            'support_structure_rate': f"{percentages['support_percentage']:.2f}%",
            'repetition_rate': f"{percentages['repetition_percentage']:.2f}%",
            'valid_information_rate': f"{percentages['valid_info_percentage']:.2f}%",
            'interpretive_rate': f"{percentages['interpretive_percentage']:.2f}%",
            'irrelevant_rate': f"{percentages['irrelevant_percentage']:.2f}%",
            'analysis_timestamp': datetime.now().isoformat()
        }


class SessionStatistics:
    """会话统计管理器"""

    def __init__(self):
        """初始化统计管理器"""
        self.reset_session()
        self.analyzer = CookieTheftAnalyzer()
        self.audio_recorder = AudioRecorder()

    def reset_session(self):
        """重置会话数据"""
        self.session_data = {
            'start_time': None,
            'end_time': None,
            'transcripts': [],
            'detections': defaultdict(int),
            'matches': [],
            'audio_segments': [],
            'language_analysis': [],
            'bbox_statistics': {
                'total_shown': 0,
                'display_duration': 5.0,
                'current_active': 0
            }
        }

    def start_session(self):
        """开始新会话"""
        self.session_data['start_time'] = datetime.now()
        session_id = self.session_data['start_time'].strftime("%Y%m%d_%H%M%S")

        # 开始音频录制会话
        audio_dir = self.audio_recorder.start_session(session_id)
        self.session_data['audio_dir'] = str(audio_dir)

        logger.info(f"开始新会话: {session_id}")
        return session_id

    def add_transcript(self, text, audio_data=None):
        """添加转录文本和音频"""
        timestamp = datetime.now()

        # 保存音频
        audio_path = None
        if audio_data:
            audio_path = self.audio_recorder.save_audio_segment(audio_data)

        # 进行Cookie Theft分析
        analysis = self.analyzer.analyze_transcript(text)

        # 添加到会话数据
        transcript_entry = {
            'timestamp': timestamp.isoformat(),
            'text': text,
            'audio_path': str(audio_path) if audio_path else None,
            'analysis': analysis
        }

        self.session_data['transcripts'].append(transcript_entry)
        self.session_data['language_analysis'].append(analysis)

        logger.info(f"添加转录和分析: {len(text)}字符")
        return analysis

    def add_detection(self, detection):
        """添加检测结果"""
        self.session_data['detections'][detection['class']] += 1

    def add_match(self, class_name, confidence):
        """添加匹配结果"""
        match_entry = {
            'timestamp': datetime.now().isoformat(),
            'object': class_name,
            'confidence': confidence
        }
        self.session_data['matches'].append(match_entry)

    def update_bbox_statistics(self, bbox_stats):
        """更新锚框统计"""
        self.session_data['bbox_statistics'].update(bbox_stats)

    def get_comprehensive_analysis(self):
        """获取综合分析结果"""
        if not self.session_data['start_time']:
            return None

        # 计算会话时长
        end_time = datetime.now()
        duration = (end_time - self.session_data['start_time']).total_seconds()

        # 汇总语言分析
        language_summary = self._summarize_language_analysis()

        # 汇总检测统计
        detection_summary = self._summarize_detection_stats()

        # 音频信息
        audio_info = self.audio_recorder.get_session_info()

        return {
            'session_info': {
                'start_time': self.session_data['start_time'].isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'duration_formatted': f"{duration // 60:.0f}分{duration % 60:.0f}秒"
            },
            'language_analysis_summary': language_summary,
            'detection_summary': detection_summary,
            'audio_info': audio_info,
            'bbox_statistics': self.session_data['bbox_statistics'],
            'raw_data': {
                'transcripts': len(self.session_data['transcripts']),
                'total_detections': sum(self.session_data['detections'].values()),
                'unique_objects': len(self.session_data['detections']),
                'total_matches': len(self.session_data['matches'])
            }
        }

    def _summarize_language_analysis(self):
        """汇总语言分析结果"""
        if not self.session_data['language_analysis']:
            return {}

        analyses = self.session_data['language_analysis']

        # 汇总所有百分比
        avg_percentages = {}
        for key in ['error_percentage', 'disfluency_percentage', 'support_percentage',
                    'repetition_percentage', 'valid_info_percentage',
                    'interpretive_percentage', 'irrelevant_percentage']:
            values = [a['percentages'][key] for a in analyses if 'percentages' in a]
            avg_percentages[key] = sum(values) / len(values) if values else 0

        # 总字数
        total_characters = sum(a['total_characters'] for a in analyses)

        return {
            'total_characters': total_characters,
            'average_percentages': avg_percentages,
            'transcript_count': len(analyses),
            'detailed_breakdown': {
                'error_rate': f"{avg_percentages.get('error_percentage', 0):.2f}%",
                'disfluency_rate': f"{avg_percentages.get('disfluency_percentage', 0):.2f}%",
                'support_structure_rate': f"{avg_percentages.get('support_percentage', 0):.2f}%",
                'repetition_rate': f"{avg_percentages.get('repetition_percentage', 0):.2f}%",
                'valid_information_rate': f"{avg_percentages.get('valid_info_percentage', 0):.2f}%",
                'interpretive_rate': f"{avg_percentages.get('interpretive_percentage', 0):.2f}%",
                'irrelevant_rate': f"{avg_percentages.get('irrelevant_percentage', 0):.2f}%"
            }
        }

    def _summarize_detection_stats(self):
        """汇总检测统计"""
        return {
            'total_detections': sum(self.session_data['detections'].values()),
            'unique_objects': len(self.session_data['detections']),
            'detection_breakdown': dict(self.session_data['detections']),
            'successful_matches': len(self.session_data['matches']),
            'match_rate': (len(self.session_data['matches']) /
                           max(1, sum(self.session_data['detections'].values()))) * 100
        }

    def save_session_report(self, filepath=None):
        """保存会话报告"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"session_report_{timestamp}.json"

        comprehensive_analysis = self.get_comprehensive_analysis()

        # 包含详细的原始数据
        report = {
            'comprehensive_analysis': comprehensive_analysis,
            'detailed_transcripts': self.session_data['transcripts'],
            'detailed_detections': dict(self.session_data['detections']),
            'detailed_matches': self.session_data['matches'],
            'bbox_statistics': self.session_data['bbox_statistics']
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"会话报告已保存: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"保存会话报告失败: {e}")
            return None