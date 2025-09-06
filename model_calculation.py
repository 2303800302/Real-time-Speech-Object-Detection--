"""
模型计算模块 - model_calculation.py
预留模块，用于未来的机器学习模型计算和认知评估算法
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """特征提取器 - 从语音和视觉数据中提取特征"""

    def __init__(self):
        """初始化特征提取器"""
        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_language_features(self, language_analysis):
        """
        从语言分析结果中提取特征
        Args:
            language_analysis: Cookie Theft分析结果
        Returns:
            dict: 提取的特征
        """
        if not language_analysis or 'percentages' not in language_analysis:
            return {}

        features = {
            # 基础统计特征
            'total_characters': language_analysis.get('total_characters', 0),
            'word_count_normalized': language_analysis.get('total_characters', 0) / 100,  # 归一化

            # 错误相关特征
            'error_percentage': language_analysis['percentages'].get('error_percentage', 0),
            'disfluency_percentage': language_analysis['percentages'].get('disfluency_percentage', 0),

            # 结构特征
            'support_percentage': language_analysis['percentages'].get('support_percentage', 0),
            'repetition_percentage': language_analysis['percentages'].get('repetition_percentage', 0),

            # 内容质量特征
            'valid_info_percentage': language_analysis['percentages'].get('valid_info_percentage', 0),
            'interpretive_percentage': language_analysis['percentages'].get('interpretive_percentage', 0),
            'irrelevant_percentage': language_analysis['percentages'].get('irrelevant_percentage', 0),

            # 计算的复合特征
            'error_fluency_ratio': self._safe_divide(
                language_analysis['percentages'].get('error_percentage', 0),
                language_analysis['percentages'].get('disfluency_percentage', 0)
            ),
            'content_structure_ratio': self._safe_divide(
                language_analysis['percentages'].get('valid_info_percentage', 0),
                language_analysis['percentages'].get('support_percentage', 0)
            ),
            'efficiency_score': self._calculate_efficiency_score(language_analysis['percentages'])
        }

        return features

    def extract_visual_features(self, detection_summary, bbox_statistics):
        """
        从视觉检测结果中提取特征
        Args:
            detection_summary: 检测统计摘要
            bbox_statistics: 锚框统计信息
        Returns:
            dict: 提取的特征
        """
        features = {
            # 检测基础特征
            'total_detections': detection_summary.get('total_detections', 0),
            'unique_objects': detection_summary.get('unique_objects', 0),
            'match_rate': detection_summary.get('match_rate', 0),

            # 锚框特征
            'bbox_display_duration': bbox_statistics.get('display_duration', 5.0),
            'total_bbox_created': bbox_statistics.get('total_created', 0),
            'avg_bbox_per_detection': self._safe_divide(
                bbox_statistics.get('total_created', 0),
                detection_summary.get('total_detections', 0)
            ),

            # 计算的复合特征
            'detection_efficiency': self._safe_divide(
                detection_summary.get('unique_objects', 0),
                detection_summary.get('total_detections', 0)
            ),
            'interaction_success_rate': detection_summary.get('match_rate', 0) / 100
        }

        return features

    def extract_temporal_features(self, session_info, transcripts):
        """
        从时间序列数据中提取特征
        Args:
            session_info: 会话信息
            transcripts: 转录历史
        Returns:
            dict: 时间特征
        """
        features = {
            # 会话时长特征
            'session_duration_seconds': session_info.get('duration_seconds', 0),
            'session_duration_minutes': session_info.get('duration_seconds', 0) / 60,

            # 语音活动特征
            'transcript_count': len(transcripts),
            'avg_transcript_length': np.mean([len(t.get('text', '')) for t in transcripts]) if transcripts else 0,
            'speech_rate': self._calculate_speech_rate(session_info, transcripts),

            # 时间分布特征
            'speech_density': self._calculate_speech_density(session_info, transcripts),
            'pause_frequency': self._estimate_pause_frequency(transcripts)
        }

        return features

    def combine_features(self, language_features, visual_features, temporal_features):
        """
        组合所有特征
        Args:
            language_features: 语言特征
            visual_features: 视觉特征
            temporal_features: 时间特征
        Returns:
            dict: 组合特征向量
        """
        combined = {}

        # 合并所有特征
        combined.update(language_features)
        combined.update(visual_features)
        combined.update(temporal_features)

        # 添加交互特征
        combined.update(self._calculate_interaction_features(
            language_features, visual_features, temporal_features
        ))

        return combined

    def _safe_divide(self, numerator, denominator):
        """安全除法，避免除零错误"""
        return numerator / denominator if denominator != 0 else 0

    def _calculate_efficiency_score(self, percentages):
        """计算效率得分"""
        valid_info = percentages.get('valid_info_percentage', 0)
        irrelevant = percentages.get('irrelevant_percentage', 0)
        errors = percentages.get('error_percentage', 0)

        # 效率 = 有效信息 - 无关内容 - 错误
        return max(0, valid_info - irrelevant - errors)

    def _calculate_speech_rate(self, session_info, transcripts):
        """计算语音速率（字符/分钟）"""
        total_chars = sum(len(t.get('text', '')) for t in transcripts)
        duration_minutes = session_info.get('duration_seconds', 1) / 60
        return total_chars / duration_minutes if duration_minutes > 0 else 0

    def _calculate_speech_density(self, session_info, transcripts):
        """计算语音密度（转录次数/分钟）"""
        transcript_count = len(transcripts)
        duration_minutes = session_info.get('duration_seconds', 1) / 60
        return transcript_count / duration_minutes if duration_minutes > 0 else 0

    def _estimate_pause_frequency(self, transcripts):
        """估计停顿频率"""
        if len(transcripts) < 2:
            return 0

        # 简单的停顿估计：基于转录间隔
        pause_indicators = sum(1 for t in transcripts if '...' in t.get('text', '') or '，' in t.get('text', ''))
        return pause_indicators / len(transcripts)

    def _calculate_interaction_features(self, lang_feat, vis_feat, temp_feat):
        """计算交互特征"""
        return {
            # 语言-视觉交互
            'language_visual_alignment': self._safe_divide(
                vis_feat.get('match_rate', 0),
                lang_feat.get('valid_info_percentage', 1)
            ),

            # 时间-效率交互
            'temporal_efficiency': self._safe_divide(
                lang_feat.get('efficiency_score', 0),
                temp_feat.get('session_duration_minutes', 1)
            ),

            # 综合表现指标
            'overall_coherence': self._calculate_coherence_score(lang_feat, vis_feat, temp_feat)
        }

    def _calculate_coherence_score(self, lang_feat, vis_feat, temp_feat):
        """计算整体连贯性得分"""
        # 这是一个示例评分函数，可以根据临床需求调整
        lang_score = lang_feat.get('efficiency_score', 0) / 100
        vis_score = vis_feat.get('interaction_success_rate', 0)
        temp_score = min(1.0, temp_feat.get('speech_rate', 0) / 200)  # 假设200字符/分钟为理想速率

        # 加权平均
        return (lang_score * 0.5 + vis_score * 0.3 + temp_score * 0.2)

class CognitiveAssessmentModel:
    """认知评估模型（预留框架）"""

    def __init__(self, model_type='random_forest'):
        """
        初始化认知评估模型
        Args:
            model_type: 模型类型
        """
        self.model_type = model_type
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.feature_names = []

        # 初始化模型
        self._init_model()

    def _init_model(self):
        """初始化机器学习模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        # 可以添加其他模型类型
        # elif self.model_type == 'svm':
        #     self.model = SVC()
        # elif self.model_type == 'neural_network':
        #     self.model = MLPClassifier()

        logger.info(f"初始化{self.model_type}模型")

    def prepare_features(self, comprehensive_analysis):
        """
        从综合分析结果中准备特征
        Args:
            comprehensive_analysis: 综合分析结果
        Returns:
            np.array: 特征向量
        """
        # 提取各类特征
        lang_features = self.feature_extractor.extract_language_features(
            comprehensive_analysis.get('language_analysis_summary', {})
        )

        vis_features = self.feature_extractor.extract_visual_features(
            comprehensive_analysis.get('detection_summary', {}),
            comprehensive_analysis.get('bbox_statistics', {})
        )

        temp_features = self.feature_extractor.extract_temporal_features(
            comprehensive_analysis.get('session_info', {}),
            comprehensive_analysis.get('detailed_transcripts', [])
        )

        # 组合特征
        combined_features = self.feature_extractor.combine_features(
            lang_features, vis_features, temp_features
        )

        # 转换为数组
        if not self.feature_names:
            self.feature_names = list(combined_features.keys())

        feature_vector = np.array([combined_features.get(name, 0) for name in self.feature_names])

        return feature_vector.reshape(1, -1), combined_features

    def predict_cognitive_status(self, comprehensive_analysis):
        """
        预测认知状态
        Args:
            comprehensive_analysis: 综合分析结果
        Returns:
            dict: 预测结果
        """
        if not self.is_trained:
            # 如果模型未训练，返回基于规则的评估
            return self._rule_based_assessment(comprehensive_analysis)

        # 准备特征
        feature_vector, feature_dict = self.prepare_features(comprehensive_analysis)

        # 模型预测
        prediction = self.model.predict(feature_vector)[0]
        prediction_proba = self.model.predict_proba(feature_vector)[0]

        # 特征重要性分析
        feature_importance = self._get_feature_importance(feature_dict)

        return {
            'prediction': prediction,
            'confidence': max(prediction_proba),
            'probability_distribution': prediction_proba.tolist(),
            'feature_importance': feature_importance,
            'assessment_type': 'model_based',
            'timestamp': datetime.now().isoformat()
        }

    def _rule_based_assessment(self, comprehensive_analysis):
        """
        基于规则的认知评估（当模型未训练时使用）
        Args:
            comprehensive_analysis: 综合分析结果
        Returns:
            dict: 基于规则的评估结果
        """
        lang_summary = comprehensive_analysis.get('language_analysis_summary', {})
        percentages = lang_summary.get('average_percentages', {})

        # 定义阈值（这些可以根据临床研究调整）
        risk_factors = {
            'high_error_rate': percentages.get('error_percentage', 0) > 15,
            'high_disfluency': percentages.get('disfluency_percentage', 0) > 20,
            'low_valid_info': percentages.get('valid_info_percentage', 0) < 30,
            'high_irrelevant': percentages.get('irrelevant_percentage', 0) > 25,
            'excessive_repetition': percentages.get('repetition_percentage', 0) > 15
        }

        # 计算风险得分
        risk_score = sum(risk_factors.values()) / len(risk_factors)

        # 评估结果
        if risk_score >= 0.6:
            assessment = 'high_risk'
            recommendation = '建议进一步临床评估'
        elif risk_score >= 0.3:
            assessment = 'moderate_risk'
            recommendation = '建议持续监测'
        else:
            assessment = 'low_risk'
            recommendation = '当前表现正常'

        return {
            'assessment': assessment,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'detailed_metrics': percentages,
            'assessment_type': 'rule_based',
            'timestamp': datetime.now().isoformat()
        }

    def _get_feature_importance(self, feature_dict):
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
            feature_importance = {
                name: float(score) for name, score in zip(self.feature_names, importance_scores)
            }
            # 按重要性排序
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}

    def train_model(self, training_data, labels):
        """
        训练模型（预留接口）
        Args:
            training_data: 训练数据
            labels: 标签
        """
        logger.info("模型训练功能需要临床数据集支持")
        # 这里可以实现模型训练逻辑
        # self.model.fit(training_data, labels)
        # self.is_trained = True
        pass

    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"模型已保存: {filepath}")
        except Exception as e:
            logger.error(f"模型保存失败: {e}")

    def load_model(self, filepath):
        """加载模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = model_data['is_trained']

            logger.info(f"模型已加载: {filepath}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")

class ModelEvaluator:
    """模型评估器"""

    def __init__(self):
        """初始化评估器"""
        pass

    def evaluate_prediction_quality(self, predictions, ground_truth=None):
        """
        评估预测质量
        Args:
            predictions: 预测结果列表
            ground_truth: 真实标签（如果有的话）
        Returns:
            dict: 评估结果
        """
        if ground_truth is not None:
            # 如果有真实标签，计算准确性指标
            return self._calculate_accuracy_metrics(predictions, ground_truth)
        else:
            # 否则进行内部一致性评估
            return self._calculate_consistency_metrics(predictions)

    def _calculate_accuracy_metrics(self, predictions, ground_truth):
        """计算准确性指标"""
        # 这需要真实的临床数据来实现
        logger.info("准确性评估需要临床标准数据")
        return {}

    def _calculate_consistency_metrics(self, predictions):
        """计算一致性指标"""
        if not predictions:
            return {}

        # 计算预测的一致性和稳定性
        risk_scores = [p.get('risk_score', 0) for p in predictions if 'risk_score' in p]

        if risk_scores:
            return {
                'mean_risk_score': np.mean(risk_scores),
                'std_risk_score': np.std(risk_scores),
                'consistency_score': 1 - (np.std(risk_scores) / (np.mean(risk_scores) + 1e-6))
            }

        return {}

# 预留的扩展接口
class AdvancedModelingFramework:
    """高级建模框架（预留）"""

    def __init__(self):
        """初始化高级建模框架"""
        self.models = {}
        self.ensemble_weights = {}

    def add_model(self, name, model):
        """添加模型到集成"""
        self.models[name] = model

    def ensemble_predict(self, comprehensive_analysis):
        """集成预测"""
        # 预留集成学习接口
        pass

    def longitudinal_analysis(self, session_history):
        """纵向分析（多次会话的变化趋势）"""
        # 预留纵向分析接口
        pass