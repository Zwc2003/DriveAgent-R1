# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from io import BytesIO
import os
import re
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image as PILImage
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict, Features, Value, Image
import pandas as pd  
from transformers import Qwen2VLForConditionalGeneration
import numpy as np
import math
from collections import defaultdict

# from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["speed_accuracy", "traj_accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'speed_accuracy', 'traj_accuracy', 'format', 'rarity_bonus'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    scene_list: Optional[str] = field(
        default=None,
        metadata={"help": "训练时要使用的场景列表文件路径"},
    )
    use_3_Stage_CoT: Optional[bool] = field(
        default=False,
        metadata={"help": "是否使用3阶段CoT"},
    )
    enable_tool_calls: Optional[bool] = field(
        default=False,
        metadata={"help": "是否启用工具调用功能"},
    )
    max_tool_calls: Optional[int] = field(
        default=4,
        metadata={"help": "每个生成路径的最大工具调用次数"},
    )
    mode: Optional[str] = field(
        default="no-tool",
        metadata={"help": "no-tool, mixed, adaptive, no-image"},
    )
    freeze_mode: Optional[str] = field(
        default="none",
        metadata={"help": "Part of the model to freeze. One of ['none', 'vision_encoder', 'vision_projector', 'vision_encoder_except_projector']"},
    )
    include_tool_in_sys_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "是否在系统提示词中包含工具介绍"},
    )
    base_mix_sft: Optional[bool] = field(
        default=False,
        metadata={"help": "是否基于mix-sft的权重进行RL训练"},
    )
    entropy_bonus_weight: Optional[float] = field(
        default=0.1,
        metadata={"help": "批内动作稀有度奖励的权重，用于鼓励多样性、对抗模式坍塌"},
    )
    pos_class_freq_json: Optional[str] = field(
        default=None,
        metadata={"help": "按位置-类别的频率JSON文件路径，用于构建位置-类别权重"},
    )
    pos_class_gamma: Optional[float] = field(
        default=0.5,
        metadata={"help": "将频率转换为权重时的指数gamma，w∝(freq+eps)^(-gamma)"},
    )
    pos_weight_clip_min: Optional[float] = field(
        default=0.7,
        metadata={"help": "权重下限(裁剪)"},
    )
    pos_weight_clip_max: Optional[float] = field(
        default=1.3,
        metadata={"help": "权重上限(裁剪)"},
    )
    weight_rarity_by_accuracy: Optional[bool] = field(
        default=True,
        metadata={"help": "是否用准确性得分对稀有度奖励进行加权"},
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the chat template jinja file."},
    )
    use_surround_views_in_no_tool: Optional[bool] = field(
        default=False,
        metadata={"help": "在 no-tool 模式下是否在提示中提供6个环视图像(front_left, front, front_right, back_left, back, back_right)"},
    )



    

VALID_SPEED_ACTIONS = {'accelerate', 'decelerate', 'keep speed', 'stop'}
VALID_TRAJECTORY_ACTIONS = {'straight', 'left turn', 'right turn'}

# 位置-类别权重与控制（全局变量，由main中读取并设置）
SPEED_POS_WEIGHTS = None  # List[Dict[class->weight]] 长度=4
TRAJ_POS_WEIGHTS = None   # List[Dict[class->weight]] 长度=4

def _build_pos_class_weights_from_freq(
    freq_json_path: str,
    gamma: float = 1.0,
    eps: float = 1e-6,
    clip_min: float = 0.7,
    clip_max: float = 1.3,
):
    """从频率JSON构建权重（每一位置的类别权重，平均为1）
    JSON schema:
    {
      "speed_frequencies_by_pos": [ {class: freq, ...} x4 ],
      "trajectory_frequencies_by_pos": [ {class: freq, ...} x4 ]
    }
    """
    global SPEED_POS_WEIGHTS, TRAJ_POS_WEIGHTS
    try:
        with open(freq_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        def _freqs_to_weights(freq_dicts, valid_keys):
            weights_by_pos = []
            for pos_dict in freq_dicts:
                # 先计算位置上的均值频率
                mean_p = 0.0
                for k in valid_keys:
                    mean_p += float(pos_dict.get(k, 0.0))
                mean_p = mean_p / max(len(valid_keys), 1)

                # 基础权重：以“与位置均值频率的比值”为核心，避免直接倒数过激
                # raw_w = ((mean_p + eps) / (p + eps))^gamma
                raw = {}
                for k in valid_keys:
                    p = float(pos_dict.get(k, 0.0))
                    base = ((mean_p + eps) / (p + eps)) ** gamma
                    # 裁剪
                    shrunk = max(clip_min, min(clip_max, base))
                    raw[k] = shrunk

                # 归一化为均值为1，避免改变整体尺度
                mean_w = sum(raw.values()) / max(len(raw), 1)
                if mean_w <= 0:
                    weights_by_pos.append({k: 1.0 for k in valid_keys})
                else:
                    weights_by_pos.append({k: (raw[k] / mean_w) for k in valid_keys})
            return weights_by_pos

        speed_keys = ["accelerate", "decelerate", "keep speed", "stop"]
        traj_keys = ["straight", "left turn", "right turn"]

        SPEED_POS_WEIGHTS = _freqs_to_weights(data.get("speed_frequencies_by_pos", [{}]*4), speed_keys)
        TRAJ_POS_WEIGHTS = _freqs_to_weights(data.get("trajectory_frequencies_by_pos", [{}]*4), traj_keys)

        print("已加载位置-类别权重(含收缩与裁剪):")
        print("SPEED_POS_WEIGHTS:", SPEED_POS_WEIGHTS)
        print("TRAJ_POS_WEIGHTS:", TRAJ_POS_WEIGHTS)
    except Exception as e:
        print(f"加载位置-类别频率JSON失败: {e}")
        SPEED_POS_WEIGHTS = None
        TRAJ_POS_WEIGHTS = None

def _normalize_action(action):
    """将动作标准化为(速度,轨迹)元组"""
    if isinstance(action, str):
        parts = [p.strip().lower() for p in action.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1])
    return ("", "")

def _calculate_edit_distance_score(pred_seq, true_seq, element_match_scorer):
    """通用编辑距离计算序列相似度"""
    if not pred_seq or not true_seq:
        return 0.0
    
    # 调低删除/插入成本以鼓励非逐位对齐
    del_insert_cost = 0.6


    m, n = len(pred_seq), len(true_seq)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 允许element_match_scorer接收(i-1,j-1)位置参数以融入位置权重
            try:
                match_score = element_match_scorer(pred_seq[i-1], true_seq[j-1], i-1, j-1)
            except TypeError:
                try:
                    match_score = element_match_scorer(pred_seq[i-1], true_seq[j-1], j-1)
                except TypeError:
                    match_score = element_match_scorer(pred_seq[i-1], true_seq[j-1])
            replace_cost = 1.0 - match_score
            
            dp[i][j] = min(
                dp[i-1][j] + del_insert_cost,          # 删除
                dp[i][j-1] + del_insert_cost,          # 插入
                dp[i-1][j-1] + replace_cost # 替换
            )

    max_possible_distance = max(m, n)
    if max_possible_distance == 0:
        return 1.0
    
    normalized_distance = dp[m][n] / max_possible_distance
    similarity_score = 1.0 - normalized_distance
    return similarity_score

def _get_action_sequences(completions, solution):
    """Generator to parse and yield action sequences from completions."""
    completion_contents = [completion[0]["content"] for completion in completions]
    for content, sol in zip(completion_contents, solution):
        pred_seq, true_seq = None, None
        try:
            # Always try to parse prediction
            content_match = re.search(r'<meta actions>(.*?)</meta actions>', content, re.DOTALL)
            if content_match:
                pred_seq_str = content_match.group(1).strip()
                if pred_seq_str.startswith('[') and pred_seq_str.endswith(']'):
                    pred_seq_str = pred_seq_str.replace("'", '"').replace('""', '"')
                    try:
                        pred_seq = eval(pred_seq_str)
                    except Exception:
                        pred_seq = None # Failed to parse pred_seq

            # Only parse solution if it exists
            if sol is not None:
                sol_match = re.search(r'<meta actions>(.*?)</meta actions>', sol, re.DOTALL)
                if sol_match:
                    true_seq_str = sol_match.group(1).strip().replace("'", '"').replace('""', '"')
                    try:
                        true_seq = eval(true_seq_str)
                    except Exception:
                        true_seq = None # Failed to parse true_seq
                
                # If we have a solution, we can validate. If validation fails, nullify both.
                if pred_seq is not None and true_seq is not None:
                    valid_actions = True
                    for action in pred_seq + true_seq:
                        try:
                            speed, traj = _normalize_action(action)
                            if speed not in VALID_SPEED_ACTIONS or traj not in VALID_TRAJECTORY_ACTIONS:
                                valid_actions = False
                                break
                        except Exception:
                            valid_actions = False
                            break
                    if not valid_actions:
                        pred_seq, true_seq = None, None
        except Exception:
            pred_seq, true_seq = None, None # Catch any other errors
        
        yield content, sol, pred_seq, true_seq

# 0.7
def speed_accuracy_reward(completions, solution, **kwargs):
    """基于编辑距离计算速度预测的准确性奖励"""
    sample_idx = kwargs.get("sample_idx", -1)
    gen_idx = kwargs.get("gen_idx", -1)
    rewards = []
    
    def speed_element_match_scorer(pred_speed, true_speed, pred_idx=None, true_idx=None):
        base = 0.0
        if pred_speed == true_speed:
            base = 1.0

        # 融入位置-类别权重（按真值位置）
        if true_idx is not None and true_idx >= 0 and SPEED_POS_WEIGHTS is not None and true_idx < len(SPEED_POS_WEIGHTS):
            wmap = SPEED_POS_WEIGHTS[true_idx]
            if true_speed in wmap:
                base *= float(wmap[true_speed])
        return base

    monitor_max_strict_trend = []
    for idx, (content, sol, pred_seq, true_seq) in enumerate(_get_action_sequences(completions, solution)):

        speed_score = 0.0
        if pred_seq is not None and true_seq is not None and len(pred_seq) == 4:
            # 分别提取速度和轨迹序列
            pred_speed_seq = [_normalize_action(action)[0] for action in pred_seq]
            true_speed_seq = [_normalize_action(action)[0] for action in true_seq]
            
            # 仅使用“位置权重已融入匹配分”的编辑距离
            strict_score_ed = _calculate_edit_distance_score(pred_speed_seq, true_speed_seq, speed_element_match_scorer)
            monitor_max_strict_trend.append(strict_score_ed)
            speed_score = strict_score_ed
        else:
            monitor_max_strict_trend.append(0.0)
        
        # 记录
        log_path = os.getenv("LOG_PATH")
        if log_path:
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Speed Accuracy (Sample {sample_idx[idx] if isinstance(sample_idx, list) else sample_idx}, Gen {gen_idx[idx] if isinstance(gen_idx, list) else gen_idx}, LocalIdx {idx}): {speed_score:.4f}  -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
                    f.flush()
            except Exception as e:
                print(f"打印日志时出错: {e}")

        # 返回加权分数
        rewards.append(0.7 * speed_score)
        
    # 将监控数据挂到函数对象（供训练器采集绘图）
    try:
        setattr(speed_accuracy_reward, "_monitor_speed_maxStrictEdTrend", monitor_max_strict_trend)
    except Exception:
        pass
    return rewards
# 0.3
def traj_accuracy_reward(completions, solution, **kwargs):
    """基于编辑距离计算轨迹预测的准确性奖励"""
    sample_idx = kwargs.get("sample_idx", -1)
    gen_idx = kwargs.get("gen_idx", -1)
    rewards = []

    def traj_element_match_scorer(pred_traj, true_traj, pred_idx=None, true_idx=None):
        base = 0.0
        if pred_traj == true_traj:
            base=1.0
        # 融入位置-类别权重（按真值位置）
        if true_idx is not None and true_idx >= 0 and TRAJ_POS_WEIGHTS is not None and true_idx < len(TRAJ_POS_WEIGHTS):
            wmap = TRAJ_POS_WEIGHTS[true_idx]
            if true_traj in wmap:
                base *= float(wmap[true_traj])
        return base

    monitor_max_strict_trend = []
    for idx, (content, sol, pred_seq, true_seq) in enumerate(_get_action_sequences(completions, solution)):
        
        traj_score = 0.0
        if pred_seq is not None and true_seq is not None and len(pred_seq) == 4:
            # 分别提取轨迹序列
            pred_traj_seq = [_normalize_action(action)[1] for action in pred_seq]
            true_traj_seq = [_normalize_action(action)[1] for action in true_seq]

            strict_score_ed = _calculate_edit_distance_score(pred_traj_seq, true_traj_seq, traj_element_match_scorer)
            monitor_max_strict_trend.append(strict_score_ed)
            traj_score = strict_score_ed
        else:
            monitor_max_strict_trend.append(0.0)
        
        # 记录日志
        log_path = os.getenv("LOG_PATH")
        if log_path:
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            try:
                with open(log_path, "a") as f:
                    f.write(f"------------- {current_time} Trajectory Accuracy (Sample {sample_idx[idx] if isinstance(sample_idx, list) else sample_idx}, Gen {gen_idx[idx] if isinstance(gen_idx, list) else gen_idx}, LocalIdx {idx}): {traj_score:.4f}-------------\n")
                    f.flush()
            except Exception as e:
                print(f"打印日志时出错: {e}")
        
        # 返回加权分数
        rewards.append(0.3 * traj_score)
        
    try:
        setattr(traj_accuracy_reward, "_monitor_traj_maxStrictEdTrend", monitor_max_strict_trend)
    except Exception:
        pass
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion follows the three-stage format"""
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for idx, content in enumerate(completion_contents):
       
        # 正常图像：按原来的复杂要求评分
        score = 0.0

        # 基础结构检查：有效的think标签 + meta actions
        # The backreference \1 ensures that the opening and closing tags match (e.g., <think_with_tools>...</think_with_tools>).
        think_pattern = r"<(think_no_tools|think_with_tools)>.*?</\1>\s*<meta actions>.*?</meta actions>"
        if re.search(think_pattern, content, re.DOTALL):
            score = 0.4  # 基础结构分

            # 统计三阶段标签是否出现 (不要求顺序)
            stages_present = 0
            for tag in ["description", "reasoning", "prediction"]:
                if re.search(rf"<{tag}>.*?</{tag}>", content, re.DOTALL):
                    stages_present += 1
            
            # 按存在的阶段加分, 每个阶段0.2分, 上限0.6分
            score += min(stages_present, 3) * 0.2

            # 当三阶段都存在且顺序正确时给满分
            if stages_present == 3:
                full_pattern = r"<(think_no_tools|think_with_tools)>.*?" \
                                r"<description>.*?</description>.*?" \
                                r"<reasoning>.*?</reasoning>.*?" \
                                r"<prediction>.*?</prediction>.*?" \
                                r"</\1>\s*<meta actions>.*?</meta actions>"
                if re.search(full_pattern, content, re.DOTALL):
                    score = 1.0
        
        rewards.append(score*0.2)
    
    return rewards



reward_funcs_registry = {
    "speed_accuracy": speed_accuracy_reward,
    "traj_accuracy": traj_accuracy_reward,
    "format": format_reward,
}


def generate_system_prompt(enable_tool_calls=False):
    """生成系统提示词，根据工具库自动生成工具描述"""
    if not enable_tool_calls:
        
        return "You are an expert in autonomous driving assistant."
    
    # 从工具库获取所有工具信息
    try:
        # 先尝试直接导入
        try:
            from open_r1.tools import list_tools, get_tool
        except ImportError:
            # 如果直接导入失败，尝试相对导入
            try:
                from .tools import list_tools, get_tool
            except ImportError:
                # 最后尝试绝对导入
                from tools import list_tools, get_tool

        
        import inspect
        
        # 添加调试信息
        all_tools = list_tools()
        print(f"正在生成工具描述，可用工具列表: {all_tools}, 数量: {len(all_tools)}")
        
        # 基础提示词
        base_prompt = """
You are an autonomous driving assistant capable of using tools. 
When you require external information or processing capabilities to aid your reasoning, you can use the following tools:

Available Tools:
{tools_description}
"""
        
        # 生成工具描述
        tools_description = ""
        for tool_name in list_tools():
            tool_func = get_tool(tool_name)
            if not tool_func:
                continue
            
            try:
                # 添加工具名称
                tools_description += f"- {tool_name}:\n"
                
                # 获取工具文档
                doc = inspect.getdoc(tool_func) or "No description available."
                
                # 添加工具描述
                tools_description += f"  {doc}\n"

                
                tools_description += "\n\n"
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        # print("tools_description:",tools_description)
        # 返回完整提示词
        return base_prompt.format(tools_description=tools_description)
    except Exception as e:
        # 返回不含工具描述的基础提示词
        return base_prompt.format(tools_description="No tools available. Error occurred when generating tool descriptions.")


def main(script_args, training_args, model_args):
    # 生成系统提示词
    enable_tool_calls = getattr(script_args, "enable_tool_calls", False)

    mode = getattr(script_args, "mode", "no-tool")

    include_tool_in_sys_prompt = getattr(script_args, "include_tool_in_sys_prompt", False)

    SYSTEM_PROMPT = generate_system_prompt(include_tool_in_sys_prompt)

    # 全局变量设置
    global SPEED_POS_WEIGHTS, TRAJ_POS_WEIGHTS
    entropy_bonus_weight = float(getattr(script_args, "entropy_bonus_weight", 0.1))
    # print(f"批内稀有度奖励权重: entropy_bonus_weight={ENTROPY_BONUS_WEIGHT}")

    # 构建位置-类别权重（可选）
    pos_json = getattr(script_args, "pos_class_freq_json", None)
    pos_gamma = float(getattr(script_args, "pos_class_gamma", 1.0) or 1.0)
    if pos_json is not None:
        clip_min = float(getattr(script_args, "pos_weight_clip_min", 0.7) or 0.7)
        clip_max = float(getattr(script_args, "pos_weight_clip_max", 1.3) or 1.3)
        _build_pos_class_weights_from_freq(
            pos_json,
            gamma=pos_gamma,
            clip_min=clip_min,
            clip_max=clip_max,
        )



    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    features = Features({
        'vid': Value('string'),
        'solution': Value('string'),
        'tag': Value('string'),
        'tag_en': Value('string'),
        'navigation': Value('string'),
        'decision': Value('string'),
        'new_navigation': Value('string'),
        'key_event': Value('string'),
        'cam_front_left': Image(decode=True),
        'cam_front': Image(decode=True),
        'cam_front_right': Image(decode=True),
        'cam_back_left': Image(decode=True),
        'cam_back': Image(decode=True),
        'cam_back_right': Image(decode=True),
        'cam_global': Image(decode=True),
        'history_coords': Value('string'),
        'future_coords': Value('string'),
        'speed': Value('string'),

        '1s_ago_cam_front_left': Image(decode=True),
        '1s_ago_cam_front': Image(decode=True),
        '1s_ago_cam_front_right': Image(decode=True),
        '1s_ago_cam_back_left': Image(decode=True),
        '1s_ago_cam_back': Image(decode=True),
        '1s_ago_cam_back_right': Image(decode=True),

        '2s_ago_cam_front_left': Image(decode=True),
        '2s_ago_cam_front': Image(decode=True),
        '2s_ago_cam_front_right': Image(decode=True),
        '2s_ago_cam_back_left': Image(decode=True),
        '2s_ago_cam_back': Image(decode=True),
        '2s_ago_cam_back_right': Image(decode=True),

        '3s_ago_cam_front_left': Image(decode=True),
        '3s_ago_cam_front': Image(decode=True),
        '3s_ago_cam_front_right': Image(decode=True),
        '3s_ago_cam_back_left': Image(decode=True),
        '3s_ago_cam_back': Image(decode=True),
        '3s_ago_cam_back_right': Image(decode=True),

        '4s_ago_cam_front_left': Image(decode=True),
        '4s_ago_cam_front': Image(decode=True),
        '4s_ago_cam_front_right': Image(decode=True),
        '4s_ago_cam_back_left': Image(decode=True),
        '4s_ago_cam_back': Image(decode=True),
        '4s_ago_cam_back_right': Image(decode=True),

        '5s_ago_cam_front_left': Image(decode=True),
        '5s_ago_cam_front': Image(decode=True),
        '5s_ago_cam_front_right': Image(decode=True),
        '5s_ago_cam_back_left': Image(decode=True),
        '5s_ago_cam_back': Image(decode=True),
        '5s_ago_cam_back_right': Image(decode=True),
    })



    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, features=features)
    dataset = dataset.shuffle(seed=42)


    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    base_mix_sft = getattr(script_args, "base_mix_sft", False)

    if base_mix_sft:
        
        QUESTION_TEMPLATE = """
<context>
You are provided with a front view image of the car camera from the current frame.
The navigation command is: {navigation_command}.
Your current speed is: {speed} km/h.
</context>
<task>
Your task is to analyze the driving scenario and construct a natural, logical reasoning process and predict meta actions. The following three stages are guidelines for structuring your thoughts:
    1. <description>: Briefly describe the key aspects of the driving environment relevant to the decision. (<4 sentences)
    2. <reasoning>: Briefly explain logically how the perceived situation help you to predict the meta actions. (<4 sentences)
    3. <prediction>: Based on reasoning, briefly explain how you arrived at the meta actions. (<2 sentences)
Finally, output the meta actions for the current frame and the next 3 frames(2s interval between frames).
Output ONE composite action for each frame, each consisting of one speed token from {{"Stop","Keep Speed","Accelerate","Decelerate"}} and one trajectory token from {{"Left Turn","Right Turn","Straight"}}
Note that Left Turn and Right Turn include both major turns and minor adjustments like lane changes or small heading corrections.
</task>
<rule>
* Your response should be concise, prefer short, direct sentences.
* First check if context suffices for prediction, then choose <think_with_tools> or <think_no_tools> mode.
* Output the final meta action prediction in <meta actions> </meta actions> tags.
</rule>
<tool_usage>
Under <think_with_tools>, you need to consider the following content:
- You are recommeded to call tools multiple times for better understanding (max 3 calls)
- Please call the tool meaningfully, as each call to the tool incurs additional costs. Do not call the tool for the sake of calling it
</tool_usage>
-example_for_insufficient_visual_context:
    <think_with_tools>  
        <description>  
            ...(you can call tools here to get more information as you need)
        </description>
        <reasoning> 
            ...(you can call tools here to get more information as you need)
        </reasoning>
        <prediction> 
            ...(you can call tools here to get more information as you need)
        </prediction>
    </think_with_tools>
    <meta actions>['speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token']</meta actions>

-example_for_sufficient_visual_context:
    <think_no_tools>  
        <description>
            ...
        </description>
        <reasoning>
            ...
        </reasoning>
        <prediction>
            ...
        </prediction>
    </think_no_tools>
    <meta actions>['speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token']</meta actions>
"""

    elif mode=="no-tool" or mode=="no-image":
        QUESTION_TEMPLATE = """
    <context>
    You are provided with a front view image of the car camera from the current frame.
    The navigation command is: {navigation_command}.
    Your current speed is: {speed} km/h.
    </context>
    <task>
    Your task is to analyze the driving scenario and construct a natural, logical reasoning process and predict meta actions. The following three stages are guidelines for structuring your thoughts:
    1. <description>: Briefly describe the key aspects of the driving environment based on the image you see. e.g., lane position, road layout, traffic signals, other vehicles or pedestrians. use<description></description> to end your current stage thinking process.
    2. <reasoning>: Interpret the scene, integrating the speed, the navigation command and visual information. Explain logically how the perceived situation help you to predict the meta actions. use<reasoning></reasoning> to end your current stage thinking process.
    3. <prediction>: Based on reasoning, explain how you arrived at the meta actions(<100 words). use<prediction></prediction> to end your current stage thinking process.
    
    Finally, output the meta actions for the current frame and the next 3 frames(2s interval between frames), use<meta actions></meta actions> to embed the meta actions.
    Output ONE composite action for each frame, each consisting of one speed token from {{"Stop","Keep Speed","Accelerate","Decelerate"}} and one trajectory token from {{"Left Turn","Right Turn","Straight"}}
    Note that Left Turn and Right Turn include both major turns and minor adjustments like lane changes or small heading corrections.
    </task>
    example:
        <think_no_tools>  
            <description>  
                ......
            </description>
            <reasoning> 
                ...
            </reasoning>
            <prediction> 
                ...
            </prediction>
        </think_no_tools>
        <meta actions>['speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token', 'speed token, trajectory token']
        </meta actions>
"""



    def make_conversation_image(example):

        navigation_command = example["new_navigation"]
        speed = example["speed"]
        
        # 构造用户消息内容：根据是否启用6环视，在 no-tool 模式插入6张图像
        user_content = []
        if mode == "no-tool" and getattr(script_args, "use_surround_views_in_no_tool", False):
            user_content.extend([
                {"type": "text", "text": "front_left view:"},
                {"type": "image"},  # front_left
                {"type": "text", "text": "front view:"},
                {"type": "image"},  # front
                {"type": "text", "text": "front_right view:"},
                {"type": "image"},  # front_right
                {"type": "text", "text": "back_left view:"},
                {"type": "image"},  # back_left
                {"type": "text", "text": "back view:"},
                {"type": "image"},  # back
                {"type": "text", "text": "back_right view:"},
                {"type": "image"},  # back_right
            ])
        else:
            user_content.extend([
                {"type": "text", "text": "The front view:"},
                {"type": "image"},
            ])

        user_content.append({
            "type": "text",
            "text": QUESTION_TEMPLATE.format(
                navigation_command=navigation_command,
                speed=speed
            )
        })

        # 创建对话格式
        example["prompt"] = [
            {
                "role": "system", 
                "content": [
                    {"type": "text", "text": SYSTEM_PROMPT}
                ],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        return example


    if "cam_front" in dataset["train"].features:
        dataset = dataset.map(make_conversation_image)   
    else:
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        enable_tool_calls =True,
        mode = mode,
        freeze_mode=script_args.freeze_mode,
        base_mix_sft = getattr(script_args, "base_mix_sft", False),
        entropy_bonus_weight = entropy_bonus_weight,
        weight_rarity_by_accuracy = script_args.weight_rarity_by_accuracy,
        chat_template_path=script_args.chat_template_path,
        use_surround_views_in_no_tool=getattr(script_args, "use_surround_views_in_no_tool", False),
    )

    # Train and push the model to the Hub
    trainer.train()

    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)