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
from qwen_vl_utils import process_vision_info
import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object
import torch
import torch.utils.data
import torch.nn.functional as F
import transformers
import warnings
from unittest.mock import patch
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
import datetime
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.import_utils import is_vllm_available

from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl import GRPOTrainer

import copy
from open_r1.tools.utils import execute_tool_call

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
import torch.nn as nn
from torch.utils.data import Sampler
import json
import time
import gc
import weakref
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
from PIL import Image
import random
import re
import logging

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

logger = logging.getLogger(__name__)

def _normalize_action(action):
    """将动作标准化为(速度,轨迹)元组"""
    if isinstance(action, str):
        parts = [p.strip().lower() for p in action.split(',')]
        if len(parts) == 2:
            return (parts[0], parts[1])
    return ("", "")

class SharedImageManager:
    """管理共享图像资源，避免重复存储相同的图像"""
    
    def __init__(self):
        self.shared_images = {}  # sample_id -> {view_name: image}
        self.ref_counts = {}     # sample_id -> count
        
    def get_shared_images(self, sample_id, all_view_images):
        """获取或创建共享图像，返回图像字典"""
        if sample_id not in self.shared_images:
            # 第一次创建，存储图像引用
            self.shared_images[sample_id] = {}

            for view_name, img in all_view_images.items():
                if view_name == "front":
                    self.shared_images[sample_id]["front"] = img
                elif view_name in ["front_left", "front_right", "back_left", "back", "back_right", "global"]:
                    img_id = f"{view_name}"
                    self.shared_images[sample_id][img_id] = img
                elif view_name in ["1s_ago_front_left", "1s_ago_front", "1s_ago_front_right", "1s_ago_back_left", "1s_ago_back", "1s_ago_back_right"]:
                    img_id = f"{view_name}"
                    self.shared_images[sample_id][img_id] = img
                elif view_name in ["2s_ago_front_left", "2s_ago_front", "2s_ago_front_right", "2s_ago_back_left", "2s_ago_back", "2s_ago_back_right"]:
                    img_id = f"{view_name}"
                    self.shared_images[sample_id][img_id] = img

            self.ref_counts[sample_id] = 0
        
        # 增加引用计数
        self.ref_counts[sample_id] += 1
        return self.shared_images[sample_id]
    
    def release_shared_images(self, sample_id):
        """释放共享图像的引用"""
        if sample_id in self.ref_counts:
            self.ref_counts[sample_id] -= 1
            if self.ref_counts[sample_id] <= 0:
                # 没有引用了，可以清理
                if sample_id in self.shared_images:
                    for img_id in self.shared_images[sample_id]:
                        self.shared_images[sample_id][img_id] = None
                    del self.shared_images[sample_id]
                if sample_id in self.ref_counts:
                    del self.ref_counts[sample_id]
    
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process()
        memory_info = process.memory_info()
        gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
        return {
            'cpu_memory_gb': memory_info.rss / 1024**3,
            'gpu_memory_gb': gpu_memory,
            'shared_images_count': sum(len(imgs) for imgs in self.shared_images.values()),
            'active_samples': len(self.ref_counts)
        }

class Qwen2VLGRPOVLLMTrainerModified(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        # qwen2-vl related params
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
        enable_tool_calls: Optional[bool]=False,
        mode: str = "no-tool", # no-tool, mixed, adaptive, no-image
        freeze_mode: str = "none",
        base_mix_sft: Optional[bool]=False,
        entropy_bonus_weight: Optional[float]=0.1,
        weight_rarity_by_accuracy: Optional[bool]=True,
        chat_template_path: Optional[str] = None,
        use_surround_views_in_no_tool: Optional[bool] = False,
    ):
        self.mode = mode
        self.base_mix_sft = base_mix_sft
        self.entropy_bonus_weight = entropy_bonus_weight
        self.weight_rarity_by_accuracy = weight_rarity_by_accuracy
        logger.info(f"Training mode: {self.mode}, entropy_bonus_weight: {entropy_bonus_weight}, weight_rarity_by_accuracy: {weight_rarity_by_accuracy}")
        self.enable_tool_calls = enable_tool_calls
        self.use_surround_views_in_no_tool = bool(use_surround_views_in_no_tool)
        
        # 初始化共享图像管理器
        self.shared_image_manager = SharedImageManager()
        
        # 如果是no-image模式，预创建空白图像
        if self.mode == "no-image":
            self.blank_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)  # 白色空白图像
        
        # For plotting metrics
        self.metrics_history = defaultdict(list)
        self.logged_steps = []

        # 分布监控：位置-动作的滑动窗口统计（默认每50步聚合一次）
        self._dist_window_size = 20
        self._dist_window_start_step = 0
        speed_vocab = ["accelerate", "decelerate", "keep speed", "stop"]
        traj_vocab = ["straight", "left turn", "right turn"]
        # 初始化当前窗口计数
        self._speed_counts_window = [dict((k, 0) for k in speed_vocab) for _ in range(4)]
        self._traj_counts_window = [dict((k, 0) for k in traj_vocab) for _ in range(4)]
        
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id or "qwen2-5-vl" in model_id:
                model_init_kwargs["torch_dtype"] = torch.bfloat16
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if freeze_mode == "vision_encoder":
            logger.info("Freezing the entire vision encoder.")
            for param in model.visual.parameters():
                param.requires_grad = False
        elif freeze_mode == "vision_projector":
            logger.info("Freezing the vision projector only.")
            for param in model.visual.merger.parameters():
                param.requires_grad = False
        elif freeze_mode == "vision_encoder_except_projector":
            logger.info("Freezing vision encoder, but keeping projector trainable.")
            for param in model.visual.parameters():
                param.requires_grad = False
            for param in model.visual.merger.parameters():
                param.requires_grad = True
        elif freeze_mode != "none":
            warnings.warn(f"Unknown freeze_mode: {freeze_mode}. Not freezing any part of the model.")

        # 记录可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable parameters: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")


        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            elif "Qwen2.5-VL" in model_id or "qwen2-5-vl" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model, **model_init_kwargs
                )
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(
                    model_id, **model_init_kwargs
                )
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(
                    model_id, **model_init_kwargs
                )
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen" in model_id or "Aria" in model_id or "qwen":
                processing_class = AutoProcessor.from_pretrained(model_id,use_fast=True,max_pixels = 259200, min_pixels = 6272)
                if chat_template_path:
                    with open(chat_template_path, "r") as f:
                        chat_template = f.read()
                    processing_class.tokenizer.chat_template = chat_template
                    logger.info(f"Loaded chat template from {chat_template_path}")
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "qwen" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
                    # processing_class.max_pixels = max_pixels
                    # processing_class.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(
                    model.config._name_or_path, padding_side="left"
                )
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
         # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.0,  # Temperature for generation sampling
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)
        self.use_vllm = args.use_vllm

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            # 为所有进程创建sampling_params，而不仅仅是主进程
       
            self.sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=self.max_completion_length+self.max_prompt_length,
                stop_token_ids=[processing_class.tokenizer.eos_token_id],
                repetition_penalty=1.1,
            )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if vllm_device == "auto":
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                # Check that the requested device is available
                if (
                    vllm_device.split(":")[0] == "cuda"
                    and int(vllm_device.split(":")[1]) >= torch.cuda.device_count()
                ):
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {
                    f"cuda:{idx}" for idx in range(self.accelerator.num_processes)
                }:
                    warnings.warn(
                        f"The requested device {vllm_device} is also used for training. This may lead to unexpected "
                        "behavior. It is recommended to use a dedicated device for vLLM."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1
                )
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
                    return_value=None,
                )
                with world_size_patch, profiling_patch:
                    logger.info(f"vLLM is running on: {vllm_device}")
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                        dtype=torch.bfloat16,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                        enforce_eager=True,
                        limit_mm_per_prompt={"image": 8}, 
                        max_model_len=(self.max_completion_length+self.max_prompt_length),
                        mm_processor_kwargs={"max_pixels": 259200,"min_pixels": 6272},
                        trust_remote_code =True
                    )

            
            self._last_loaded_step = 0
            self.accelerator.wait_for_everyone()
        else:
            raise ValueError(
                "GRPOVLLMTrainerModified only supports vllm generation, please set --use_vllm True"
            )

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
        assistant_mask=None,
    ):
        """
        计算给定模型对输入序列的 per-token log probabilities。
        只计算assistant生成部分的token概率（如果提供了assistant_mask）
        
        Args:
            model: 要计算 logprobs 的模型 (参考模型或策略模型)
            input_ids: 输入 token ids (B, L)
            attention_mask: 注意力掩码 (B, L)
            pixel_values: 图像特征 (B, ...)
            image_grid_thw: 图像网格信息 (B, ...)
            assistant_mask: 标识assistant角色生成内容的掩码 (B, L-1)
            
        Returns:
            per_token_logps: 每个 token 的 log probability (B, L-1)
        """
        # 确保输入在正确设备上
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)
        pixel_values = pixel_values.to(model.device)
        image_grid_thw = image_grid_thw.to(model.device)
        
        # 获取模型输出
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits  # (B, L, V)
        
        # 计算 log probabilities，去掉最后一个时间步(它预测的是下一个token)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        
        # 使用错位的input_ids (从第二个token开始)
        # 因为logits[i]预测的是input_ids[i+1]
        shifted_input_ids = input_ids[:, 1:]  # (B, L-1)
        
        # 如果提供了assistant_mask，处理logits和input_ids以匹配它
        if assistant_mask is not None:
            # 确保所有张量的第二维度长度匹配
            if assistant_mask.size(1) != logits.size(1):

                min_len = min(assistant_mask.size(1), logits.size(1))
                if assistant_mask.size(1) > min_len:
                    assistant_mask = assistant_mask[:, :min_len]
                if logits.size(1) > min_len:
                    logits = logits[:, :min_len, :]
                    shifted_input_ids = shifted_input_ids[:, :min_len]
        
        # 计算所有token的log概率
        log_probs = logits.log_softmax(dim=-1)  # (B, L-1, V)
        
        # 收集每个token对应的log概率
        per_token_logps = []
        for i, (log_probs_row, input_ids_row) in enumerate(zip(log_probs, shifted_input_ids)):
            # 提取每个token的log概率
            token_log_probs = torch.gather(
                log_probs_row, 
                dim=1, 
                index=input_ids_row.unsqueeze(1)
            ).squeeze(1)  # (L-1)
            
            per_token_logps.append(token_log_probs)
        
        # 将每个样本的token log probs堆叠成一个批次
        return torch.stack(per_token_logps)

    def _prepare_inputs_original(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        原始的输入准备方法，不支持工具调用
        该方法目前被禁用，只支持工具调用版本
        """
        raise ValueError(
            "目前只支持使用工具调用的版本，请设置 enable_tool_calls=True。"
            "如需使用原始版本，请参考代码历史。"
        )

    def create_role_masks(self, input_ids, dialogue_history):
        """
        根据对话历史创建不同角色的掩码，只关注assistant角色
        支持多轮对话，能够识别每个<|im_start|>assistant和对应的<|im_end|>匹配对
        
        Args:
            input_ids: 输入token的张量 [batch_size, seq_len]
            dialogue_history: 包含对话历史的列表，每个元素是一个对话路径的历史
            
        Returns:
            role_masks: 一个字典，包含不同角色的掩码，形状为 [batch_size, seq_len-1]
        """
        batch_size, seq_len = input_ids.shape
        
        # 创建掩码，只关注assistant
        assistant_mask = torch.zeros(batch_size, seq_len-1, device=input_ids.device)
        
        # 获取各种角色的标记ID
        tokenizer = self.processing_class.tokenizer
        im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        assistant_marker = tokenizer.encode("assistant", add_special_tokens=False)
        im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
        
        # 对每个样本单独处理
        for i in range(batch_size):
            # 完整的token序列
            tokens = input_ids[i].tolist()
            
            # 在序列中查找所有角色标记的位置
            role_positions = []  # [(位置, "role_type", "start/end"), ...]
            
            # 查找所有<|im_start|>位置
            for j in range(len(tokens) - len(im_start) + 1):
                if tokens[j:j+len(im_start)] == im_start:
                    # 检查下一个token段是否是角色名称
                    role_end = j + len(im_start)
                    # 检查是否是assistant
                    if role_end + len(assistant_marker) <= len(tokens) and tokens[role_end:role_end+len(assistant_marker)] == assistant_marker:
                        # 找到了<|im_start|>assistant
                        role_positions.append((j, "assistant", "start"))
                    # 可以添加其他角色的检查，但我们只关心assistant
            
            # 查找所有<|im_end|>位置
            for j in range(len(tokens) - len(im_end) + 1):
                if tokens[j:j+len(im_end)] == im_end:
                    role_positions.append((j, "any", "end"))
            
            # 按位置排序
            role_positions.sort(key=lambda x: x[0])
            
            # 跟踪当前正在处理的角色
            current_role = None
            assistant_regions = []  # [(start, end), ...]
            
            # 扫描所有角色标记，正确匹配每对开始/结束标记
            j = 0
            while j < len(role_positions):
                pos, role, marker = role_positions[j]
                
                if marker == "start":
                    # 找到了一个角色的开始
                    current_role = role
                    if role == "assistant":
                        # 记录assistant内容的开始位置（跳过<|im_start|>assistant标记）
                        start_pos = pos + len(im_start) + len(assistant_marker)
                        
                        # 查找对应的结束标记
                        end_pos = None
                        for k in range(j+1, len(role_positions)):
                            if role_positions[k][2] == "end":
                                end_pos = role_positions[k][0]
                                j = k  # 跳到这个结束标记
                                break
                        
                        if end_pos is not None:
                            # 找到了匹配的结束标记，记录这个assistant区域
                            assistant_regions.append((start_pos, end_pos))
       
                        else:
                            # 没有找到结束标记，使用序列末尾
                            assistant_regions.append((start_pos, len(tokens)))
    
                
                elif marker == "end":
                    # 找到了一个角色的结束，重置当前角色
                    current_role = None
                
                j += 1
            
            # 标记所有找到的assistant区域
            for start_pos, end_pos in assistant_regions:
                start_idx = min(start_pos, seq_len-1)
                end_idx = min(end_pos, seq_len-1)
                
                if start_idx < seq_len-1:
                    assistant_mask[i, start_idx:end_idx] = 1.0
            
            # 如果上述方法未找到任何assistant区域，尝试备用方法
            if len(assistant_regions) == 0:
                logger.debug(f"Sample {i}: using fallback method for assistant region detection")
                
                # 备用方法1: 搜索文本中的标记
                full_text = tokenizer.decode(tokens, skip_special_tokens=False)
                assistant_sections = []
                
                start = 0
                while True:
                    start_tag = "<|im_start|>assistant"
                    end_tag = "<|im_end|>"
                    
                    start = full_text.find(start_tag, start)
                    if start == -1:
                        break
                        
                    # 找到开始标记，跳过标记本身
                    content_start = start + len(start_tag)
                    # 找到对应的结束标记
                    end = full_text.find(end_tag, content_start)
                    if end == -1:
                        end = len(full_text)
                        
                    # 记录这段assistant内容
                    assistant_sections.append((content_start, end))
                    # 从结束位置之后继续搜索
                    start = end + len(end_tag)
                
                # 处理找到的每段assistant内容
                for text_start, text_end in assistant_sections:
                    # 提取assistant内容
                    assistant_text = full_text[text_start:text_end].strip()
                    if not assistant_text:
                        continue
                        
                    # 将文本编码为token
                    assistant_tokens = tokenizer.encode(assistant_text, add_special_tokens=False)
                    
                    # 在token序列中查找这段内容
                    for j in range(len(tokens) - len(assistant_tokens) + 1):
                        if tokens[j:j+len(assistant_tokens)] == assistant_tokens:
                            end_j = min(j + len(assistant_tokens), seq_len-1)
                            if j < seq_len-1:
                                assistant_mask[i, j:end_j] = 1.0
                                logger.debug(f"Fallback method found assistant region: {j} - {end_j}, length: {end_j - j}")
                

        
        return {
            "assistant_mask": assistant_mask,
            "user_mask": torch.zeros_like(assistant_mask)  # 保持接口一致，但不再使用
        }


    def _prepare_inputs_with_tool_calls(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """准备带有工具调用的输入数据
        
        重要说明:
        1. 为了解决多进程环境下的路径混淆问题，我们为每个路径分配全局唯一ID
        2. 全局路径ID = 进程ID * 样本数量 * num_generations + 本地索引
        3. 每个路径记录其所属的原始进程ID和样本索引
        4. 在多轮工具调用和最终奖励计算中，始终使用全局ID进行路径跟踪
        这确保了即使在复杂的交互流程中，不同进程的路径也不会混淆
        """
        from open_r1.tools.utils import execute_tool_call
        device = self.accelerator.device



        # --------------------------------------------------
        # 1. 准备初始输入
        # --------------------------------------------------
        prompts_data = [x["prompt"] for x in inputs]  # 结构化对话历史
        
        # 提取所有视角的图像字典
        all_view_images_list = []


        for x in inputs:
            view_images = {}

            # 提取当前六个视角图像和标定文件路径
            view_keys = {
                "front_left": "cam_front_left",
                "front": "cam_front",
                "front_right": "cam_front_right",
                "back_left": "cam_back_left",
                "back": "cam_back",
                "back_right": "cam_back_right",
                "global":"cam_global",

                "1s_ago_front_left": "1s_ago_cam_front_left",
                "1s_ago_front": "1s_ago_cam_front",
                "1s_ago_front_right": "1s_ago_cam_front_right",
                "1s_ago_back_left": "1s_ago_cam_back_left",
                "1s_ago_back": "1s_ago_cam_back",
                "1s_ago_back_right": "1s_ago_cam_back_right",

                "2s_ago_front_left": "2s_ago_cam_front_left",
                "2s_ago_front": "2s_ago_cam_front",
                "2s_ago_front_right": "2s_ago_cam_front_right",
                "2s_ago_back_left": "2s_ago_cam_back_left",
                "2s_ago_back": "2s_ago_cam_back",
                "2s_ago_back_right": "2s_ago_cam_back_right",

            }


            has_all_views = all(view_keys[vk] in x for vk in view_keys)


            if has_all_views:
                for view_name, image_key in view_keys.items():
                    view_images[view_name] = x[image_key]     
                all_view_images_list.append(view_images)

            else:
                all_view_images_list.append({})

        # 处理初始提示文本，应用聊天模板
        initial_prompts_text = [
            maybe_apply_chat_template({"prompt": prompt}, self.processing_class)["prompt"]
            for prompt in prompts_data
        ]

        # 准备初始输入所需的图像列表
        # 在 no-tool 且启用 use_surround_views_in_no_tool 时，按顺序提供6张环视图像
        # 顺序需与 grpo.py 中 user_content 的插图顺序一致：
        # front_left, front, front_right, back_left, back, back_right
        initial_images_list = []
        for x in inputs:
            if self.mode == "no-image":
                initial_images_list.append(self.blank_image)
            elif self.mode == "no-tool" and self.use_surround_views_in_no_tool:
                imgs = []
                imgs.append(x.get("cam_front_left"))
                imgs.append(x.get("cam_front"))
                imgs.append(x.get("cam_front_right"))
                imgs.append(x.get("cam_back_left"))
                imgs.append(x.get("cam_back"))
                imgs.append(x.get("cam_back_right"))
                initial_images_list.append(imgs)
            elif "cam_front" in x:
                initial_images_list.append(x["cam_front"])
            else:
                initial_images_list.append(None)

        
        # 处理初始提示，支持单图或多图
        prompt_inputs = self.processing_class(
            text=copy.deepcopy(initial_prompts_text),
            images=initial_images_list,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
        prompt_ids, prompt_mask = prompt_inputs["input_ids"].to(device), prompt_inputs["attention_mask"].to(device)
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
        
        # 加载模型权重到 vLLM (如果需要)
        if self.state.global_step != self._last_loaded_step:
            with unwrap_model_for_generation(
                self.model,
                self.accelerator,
                gather_deepspeed3_params=False,
            ) as unwrapped_model:
                if is_compiled_module(unwrapped_model):
                    state_dict = unwrapped_model._orig_mod.state_dict()
                else:
                    state_dict = unwrapped_model.state_dict()
            if self.accelerator.is_main_process:
                llm_model = (
                    self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                )
                llm_model.load_weights(state_dict.items())
            self._last_loaded_step = self.state.global_step
        
        # --------------------------------------------------
        # 2. 使用类似PathState的类进行多轮交互
        # --------------------------------------------------
        class PathState:
            """跟踪每个生成路径的状态，处理对话历史、图像和工具调用"""
            
            def __init__(self, initial_dialogue, sample_id, shared_image_manager,
                         force_tool_use=False, force_no_tool_use=False, global_step=0, mode="adaptive", base_mix_sft=False,
                         use_surround_views_in_no_tool=False):
                # 使用浅拷贝而不是深拷贝，减少内存占用
                self.dialogue_history = copy.deepcopy(initial_dialogue)
                
                # 使用共享图像管理器
                self.sample_id = sample_id
                self.shared_image_manager = shared_image_manager
                self.shared_images = shared_image_manager.get_shared_images(sample_id, {})  # 将在后面设置
                self.private_images = {}  # 存储路径私有的图像（如工具调用生成的图像）
                self.image_order = []
                self.thinking_mode = None
                
                # 浅拷贝标定信息
                self.original_process_id = -1  # 记录该路径所属的原始进程ID
                
                # 新增: 存储所有assistant生成的文本
                self.combined_assistant_text = ""

                # 初始化图像顺序
                if mode == "no-tool" and use_surround_views_in_no_tool:
                    # 与 grpo.py 的用户提示中 <image> 顺序保持一致
                    self.image_order.extend([
                        "front_left",
                        "front",
                        "front_right",
                        "back_left",
                        "back",
                        "back_right",
                    ])
                else:
                    self.image_order.append("front")
                
                self.tool_call_count = 0  # 总工具调用次数
                self.valuable_tool_call_count = 0  # 有价值的工具调用次数
                self.accumulated_tool_reward = 0.0
                self.is_finished = False
                
                # 记录各种工具的调用情况，用于计算多样性奖励
                self.tool_calls_by_type = {}  # 工具名称 -> 调用次数
                
                # 测试模式：添加标记跟踪工具调用测试状态
                self.test_tool_call_executed = False
                self.force_tool_use = force_tool_use
                self.force_no_tool_use = force_no_tool_use
                self.has_used_tool = False  # 跟踪是否使用过工具
                self.random_seed = None  # 为生成设置随机种子
                self.tool_use_penalty = 0.0

                # 通过在用户提示词末尾添加暗示来触发思考模式
                if mode == "mixed":
                    hint_text = ""
                    if force_tool_use:
                        hint_text = "\nHint: Now Organize Your thought with <think_with_tools></think_with_tools>"
                    else:
                        hint_text = "\nHint: Now Organize Your thought with <think_no_tools></think_no_tools>"
                    
                    # 找到对话历史中的最后一个用户消息并添加提示
                    if self.dialogue_history and self.dialogue_history[-1]["role"] == "user":
                        user_content = self.dialogue_history[-1]["content"]
                        # 用户内容是一个列表，我们找到最后一个文本部分并追加提示
                        if user_content and isinstance(user_content, list):
                            for i in range(len(user_content) - 1, -1, -1):
                                if user_content[i].get("type") == "text":
                                    user_content[i]["text"] += hint_text
                                    break
            
            def set_shared_images(self, shared_images):
                """设置共享图像引用"""
                self.shared_images = shared_images
            
            def get_image(self, img_id):
                """获取图像，先从私有图像中查找，再从共享图像中查找"""
                if img_id in self.private_images:
                    return self.private_images[img_id]
                elif img_id in self.shared_images:
                    return self.shared_images[img_id]
                return None
            
            def add_private_image(self, img_id, image):
                """添加路径私有的图像"""
                self.private_images[img_id] = image
            
            def cleanup(self):
                """清理资源"""
                # 清理私有图像
                for img_id in list(self.private_images.keys()):
                    self.private_images[img_id] = None
                self.private_images.clear()
                
                # 释放共享图像引用
                if hasattr(self, 'shared_image_manager') and hasattr(self, 'sample_id'):
                    self.shared_image_manager.release_shared_images(self.sample_id)

            def get_vllm_input_prompt(self, processor):
                """获取用于vLLM生成的提示"""
                formatted = maybe_apply_chat_template({"prompt": self.dialogue_history}, processor)
                return formatted["prompt"]

            def get_images_for_vllm(self):
                """获取当前的图像列表"""
                images = []
                for img_id in self.image_order:
                    img = self.get_image(img_id)
                    images.append(img)
                return images
            
            def append_assistant_message(self, text_content):
                """追加助手消息，并更新combined_assistant_text"""
                # 检查对话历史是否为空，以及最后一个消息是否是assistant角色
                if self.dialogue_history and self.dialogue_history[-1]["role"] == "assistant":
                    # 如果最后一个消息是assistant，直接在其content列表中追加
                    self.dialogue_history[-1]["content"].append({"type": "text", "text": text_content})
                else:
                    # 否则添加一个新的assistant消息
                    self.dialogue_history.append({
                        "role": "assistant", 
                        "content": [{"type": "text", "text": text_content}]
                    })
                
                # 更新combined_assistant_text，如果已有内容则添加空格分隔
                if self.combined_assistant_text:
                    self.combined_assistant_text += " " + text_content
                else:
                    self.combined_assistant_text = text_content
                
                if self.force_tool_use is None:
                    if "<think_with_tools>" in text_content:
                        self.force_tool_use = True
                    elif "<think_no_tools>" in text_content:
                        self.force_tool_use = False
                    else:
                        self.force_tool_use = None
                    self.force_no_tool_use = (not self.force_tool_use)


            def add_tool_result(self, tool_result, tool_reward, image_ref=None):
                """处理工具调用结果，支持图像和文本输出，放入新的user角色消息
                
                Args:
                    tool_result: 工具调用结果对象
                    tool_reward: 工具调用奖励值
                    image_ref: 如果工具返回了图像，这是分配给该图像的引用ID
                """
                self.tool_call_count += 1
                self.accumulated_tool_reward += tool_reward
                
                # 记录工具类型调用次数，用于多样性奖励计算
                if tool_result.tool_name not in self.tool_calls_by_type:
                    self.tool_calls_by_type[tool_result.tool_name] = 0
                self.tool_calls_by_type[tool_result.tool_name] += 1
                
                # 创建新的user角色消息
                self.dialogue_history.append({
                    "role": "user",
                    "content": []
                })
                
                # 获取新创建的user消息
                user_message = self.dialogue_history[-1]
                

                if tool_result.error:
                    # 错误情况
                    tool_result.error = f"\n tool_call_count: {self.tool_call_count}" + tool_result.error
                    result_text = f'<tool_result>\n{tool_result.error}\n</tool_result>'
                    user_message["content"].append({"type": "text", "text": result_text})
                    user_message["content"].append({"type": "text", "text": f'Continue to think, don\'t forget to end your reasoning stage with </description> or </reasoning> or </prediction> before you start a new stage.'})
                else:
                    # 成功情况
                    if tool_result.has_image and tool_result.image_output is not None and image_ref is not None:
                        # 如果有图像输出，分三部分添加：开始标签、图像、结束标签
                        output_text= f"\n tool_call_count: {self.tool_call_count}"
                        output_text += f'You can access the returned image through reference id:{image_ref}:'
                        user_message["content"].append({"type": "text", "text": f'<tool_result>\n{output_text}'})
                        user_message["content"].append({"type": "image"})
                        user_message["content"].append({"type": "text", "text": f'\n</tool_result>\n'})
                        user_message["content"].append({"type": "text", "text": f'Continue to think, don\'t forget to end your reasoning stage with </description> or </reasoning> or </prediction> before you start a new stage.'})
                    else:
                        # 如果没有图像输出，只添加文本结果
                        output_text = f"\n tool_call_count: {self.tool_call_count}\n"
                        if tool_result.text_output:
                            output_text += tool_result.text_output
                            output_text += "\nThe returned image is None!"
                        result_text = f'<tool_result>\n{output_text}\n</tool_result>'
                        user_message["content"].append({"type": "text", "text": result_text})
                        user_message["content"].append({"type": "text", "text": f'Continue to think, don\'t forget to end your reasoning stage with </description> or </reasoning> or </prediction> before you start a new stage.'})


        # 为每个提示创建路径，每个样本重复num_generations次
        all_paths = []
        # 全局路径索引 = 进程编号 * 样本数量 * num_generations + 本地索引
        # 这确保每个进程创建的路径有唯一ID
        base_path_idx = self.accelerator.process_index * len(prompts_data) * self.num_generations
        
        # 为每个样本创建共享图像
        for i in range(len(prompts_data)):
            sample_id = f"process_{self.accelerator.process_index}_sample_{i}"
            
            if self.mode == "no-image":
                # no-image模式：创建空白图像字典  
                blank_view_images = {"front": self.blank_image}
                shared_images = self.shared_image_manager.get_shared_images(sample_id, blank_view_images)
            else:
                shared_images = self.shared_image_manager.get_shared_images(sample_id, all_view_images_list[i])
            
            for j in range(self.num_generations):
                # 将每组路径分为强制使用工具和禁止使用工具两类
                if self.mode == "no-tool" or self.mode == "no-image":
                    force_tool_use = False
                    force_no_tool_use = (not force_tool_use)
                elif self.mode == "mixed": # mixed
                    force_tool_use = (j % 2 == 0)
                    force_no_tool_use = (not force_tool_use)
                elif self.mode == "adaptive":
                    force_tool_use = None
                    force_no_tool_use = None
                

                # 全局唯一路径ID
                global_path_idx = base_path_idx + i * self.num_generations + j
                
                # 确保独特的随机种子: GPU编号 * 1000 + 样本编号 * num_generations + 复制编号
                random_seed = self.accelerator.process_index * 1000 + i * self.num_generations + j + int(time.time()) % 1000
                
                path = PathState(
                    prompts_data[i],  # 不再使用深拷贝
                    sample_id,
                    self.shared_image_manager,
                    force_tool_use=force_tool_use,
                    force_no_tool_use=force_no_tool_use,
                    global_step=self.state.global_step,
                    mode=self.mode,
                    base_mix_sft = self.base_mix_sft,
                    use_surround_views_in_no_tool=self.use_surround_views_in_no_tool
                )
                path.set_shared_images(shared_images)  # 设置共享图像引用
                path.random_seed = random_seed
                path.global_path_id = global_path_idx  # 存储全局路径ID
                path.original_process_id = self.accelerator.process_index  # 直接设置原始进程
                path.local_sample_idx = i  # 记录本地样本索引，方便后续收集阶段使用
                path.local_gen_idx = j     # 记录生成索引
                all_paths.append(path)

        # 记录本地索引到全局路径ID的映射
        local_to_global_path_map = {idx: path.global_path_id for idx, path in enumerate(all_paths)}
        # 记录全局路径ID到本地索引的映射  
        global_to_local_path_map = {path.global_path_id: idx for idx, path in enumerate(all_paths)}

        # --------------------------------------------------
        # 3. 执行多轮交互生成
        # --------------------------------------------------
        max_tool_calls = 3  # 最大工具调用次数
        active_paths = list(range(len(all_paths)))  # 初始时使用本地索引跟踪活跃路径
        any_active_globally = True  # 全局是否有活跃路径，用于控制所有进程的循环

        round_num = 0
        while any_active_globally:  # 使用全局标志控制循环
            round_num += 1
            logger.debug(f"Round {round_num}: {len(active_paths)} active paths")
            
            # 准备当前批次的vLLM输入，添加进程ID标记
            vllm_inputs = []
            sampling_params_list = []
            input_process_ids = []  # 记录每个输入来自哪个进程
            path_global_ids = []    # 记录每个输入对应的全局路径ID

            for local_idx in active_paths:
                path = all_paths[local_idx]
                # 只处理属于当前进程的路径
                if path.original_process_id != self.accelerator.process_index:
                    continue
                    
                prompt = path.get_vllm_input_prompt(self.processing_class)
                images = path.get_images_for_vllm()
                # 在 no-tool + use_surround_views_in_no_tool 情况下，确保初始 images 已为6图；
                # path.image_order 默认包含 "front"，这里无需额外处理，初始阶段已传入。
                
                # Tokenize and truncate before sending to vLLM
                prompt_ids = self.processing_class.tokenizer(prompt, add_special_tokens=False)["input_ids"]
                if self.max_prompt_length is not None and len(prompt_ids) > self.max_prompt_length:
                    prompt_ids = prompt_ids[-self.max_prompt_length:]

                # 添加包含全局路径ID的输入
                input_dict = {
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": {"image": images},
                    "process_id": self.accelerator.process_index,  # 添加进程ID
                    "path_global_id": path.global_path_id,         # 使用全局路径ID
                    "local_idx": local_idx                         # 保留本地索引以便后续处理
                }
                vllm_inputs.append(input_dict)
                input_process_ids.append(self.accelerator.process_index)
                path_global_ids.append(path.global_path_id)
                
                # 为每个路径创建独立的sampling_params
                sp = copy.deepcopy(self.sampling_params)
                sp.stop = ["</call_tool>"]
                if path.random_seed is not None:
                    sp.seed = path.random_seed
                sampling_params_list.append(sp)

            # 收集所有进程的数据和进程ID
            all_vllm_inputs = gather_object(vllm_inputs if vllm_inputs else [None])
            all_sampling_params = gather_object(sampling_params_list if sampling_params_list else [None])
            all_process_ids = gather_object(input_process_ids if input_process_ids else [None])
            all_path_global_ids = gather_object(path_global_ids if path_global_ids else [None])

            # 在主进程进行生成
            if self.accelerator.is_main_process:
                process_counts = {}
                for pid in all_process_ids:
                    if pid is not None:
                        process_counts[pid] = process_counts.get(pid, 0) + 1
                
                # 收集全局路径ID信息，用于调试
                all_global_ids = []
                for global_id in all_path_global_ids:
                    if global_id is not None:
                        all_global_ids.append(global_id)
                
                # 过滤None值，只处理有效输入
                valid_inputs = []
                valid_params = []
                valid_indices = []
                
                for i, (input_data, param) in enumerate(zip(all_vllm_inputs, all_sampling_params)):
                    if input_data is not None and param is not None:
                        valid_inputs.append(input_data)
                        valid_params.append(param)
                        valid_indices.append(i)
                
                # 只有在有效输入存在时才调用生成
                if valid_inputs:
                    try:
                        outputs = self.llm.generate(
                            valid_inputs, 
                            sampling_params=valid_params,
                            use_tqdm=False
                        )
                    except ValueError as e:
                        if "is too long to fit into the model" in str(e):
                            outputs = []  # 设置为空列表，以便后续逻辑可以处理
                        else:
                            # 如果是其他ValueError，则重新引发异常
                            raise e
                            
                    all_completion_ids = [[token_id for token_id in req_output.outputs[0].token_ids] 
                                         for req_output in outputs]
                    
                    # 对输出结果按全局路径ID和进程ID分组
                    completion_ids_by_id = {}
                    if all_completion_ids:
                        for idx, i in enumerate(valid_indices):
                            pid = all_process_ids[i]
                            global_id = all_path_global_ids[i]
                            input_info = all_vllm_inputs[i]
                            
                            if global_id is not None and pid is not None and input_info is not None:
                                completion_ids_by_id[global_id] = {
                                    "completion": all_completion_ids[idx],
                                    "process_id": pid,
                                    "local_idx": input_info.get("local_idx", -1)
                                }          
                else:
                    completion_ids_by_id = {}
            else:
                completion_ids_by_id = {}

            # 广播结果到所有进程
            completion_ids_by_id = broadcast_object_list([completion_ids_by_id], from_process=0)[0]

            # 处理生成结果，确保每个进程只处理自己的路径
            next_active = []  # 存储本地索引

            for local_idx in active_paths:
                path = all_paths[local_idx]
                global_id = path.global_path_id
                
                # 检查是否有该路径的结果
                if global_id not in completion_ids_by_id:
                    continue
                
                result_data = completion_ids_by_id[global_id]
                completion_tokens = result_data["completion"]
                completion_text = self.processing_class.decode(completion_tokens, skip_special_tokens=False)

                unwanted_tokens = [self.processing_class.tokenizer.eos_token, self.processing_class.tokenizer.pad_token, self.processing_class.tokenizer.unk_token]
                unwanted_tokens = [token for token in unwanted_tokens if token is not None]
                for token in unwanted_tokens:
                    completion_text = completion_text.replace(token, "")  
                
                
                # 始终添加新的assistant消息
                path.append_assistant_message(completion_text)

                # 禁止工具调用的直接结束
                if path.force_no_tool_use:
                    call_tool = False
                    path.is_finished = True
                
                # 检查是否有工具调用或需要测试工具调用
                call_tool = False
                if "<call_tool>" in completion_text and "</call_tool>" in completion_text:
                    if completion_text.rstrip().endswith("</call_tool>"):
                        call_tool = True
                        path.has_used_tool = True
                
                if call_tool and path.tool_call_count < max_tool_calls:
                    # 执行工具调用
                    try:
                        # 合并共享图像和私有图像用于工具调用
                        combined_images = {}
                        combined_images.update(path.shared_images)
                        combined_images.update(path.private_images)
                        
                        # 执行工具调用
                        # 从当前样本中获取 vid 并透传，便于工具加载高分辨率图像
                        current_vid = inputs[path.local_sample_idx].get("vid") if isinstance(inputs, list) and path.local_sample_idx < len(inputs) else None
                        tool_result, tool_reward = execute_tool_call(
                            completion_text, combined_images, vid=current_vid
                        )
                        
                        # 如果工具返回了新图像，将其添加到私有图像字典中
                        new_image_id = None
                        if tool_result.has_image and tool_result.image_output is not None:
                            new_image_id = f"tool_image_{path.tool_call_count}"
                            path.add_private_image(new_image_id, tool_result.image_output)
                            path.image_order.append(new_image_id)
                            
                        
                        # 添加工具调用结果到对话历史，包括图像引用ID
                        path.add_tool_result(tool_result, tool_reward, image_ref=new_image_id)
                        next_active.append(local_idx)  # 保持活跃
                    except Exception as e:
                        path.is_finished = True

                else:
                    # 无工具调用或已达到最大调用次数
                    path.is_finished = True


            active_paths = next_active  # 更新活跃路径

            # 使用 all_reduce 高效同步全局活跃状态，取代原有的 gather + broadcast
            has_active_tensor = torch.tensor(1.0 if active_paths else 0.0, device=self.accelerator.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(has_active_tensor, op=torch.distributed.ReduceOp.SUM)
            any_active_globally = has_active_tensor.item() > 0

            # 定期清理GPU缓存，避免内存碎片化
            if round_num % 2 == 0:  # 每两轮清理一次
                torch.cuda.empty_cache()
                gc.collect()

        # --------------------------------------------------
        # 4. 最终准备收集这个批次中各个路径的输出（优化版本）
        # --------------------------------------------------


        # 确保路径按样本和生成索引排序，以便正确地对奖励进行分组
        all_paths.sort(key=lambda p: (p.local_sample_idx, p.local_gen_idx))

        # 准备用于奖励计算的扁平化列表
        completions = [[{"role": "assistant", "content": p.combined_assistant_text}] for p in all_paths]
        solutions = [inputs[p.local_sample_idx].get("solution") for p in all_paths]
        vids = [inputs[p.local_sample_idx].get("vid") for p in all_paths]
        future_coords_list = [inputs[p.local_sample_idx].get("future_coords") for p in all_paths]

        # --- 新增：统计每个位置-动作的分布（窗口聚合），用于监控模式坍塌 ---
        # 说明：解析<meta actions>中前4个预测，
        # - 速度取 {accelerate, decelerate, keep speed, stop}
        # - 轨迹取 {straight, left turn, right turn}
        # 将每步的计数累积到当前窗口，满窗口或越过边界时输出占比到metrics。
        try:
            import re as _re

            def _normalize_action_pair(action_text: str):
                parts = [x.strip().lower() for x in action_text.split(',')]
                if len(parts) != 2:
                    return None, None
                speed_raw, traj_raw = parts[0], parts[1]

                # 速度归一
                if speed_raw in ["accelerate", "acc", "+acc"]:
                    speed = "accelerate"
                elif speed_raw in ["decelerate", "dec", "brake", "slow down", "slowing down", "slowdown"]:
                    speed = "decelerate"
                elif speed_raw in ["keep speed", "keep", "maintain speed", "steady", "cruise", "cruising"]:
                    speed = "keep speed"
                elif speed_raw in ["stop", "halt"]:
                    speed = "stop"
                else:
                    speed = None

                # 轨迹归一
                if traj_raw in ["straight", "go straight", "forward"]:
                    traj = "straight"
                elif traj_raw in ["left turn", "left", "turn left", "slight left", "lane change left", "merge left"]:
                    traj = "left turn"
                elif traj_raw in ["right turn", "right", "turn right", "slight right", "lane change right", "merge right"]:
                    traj = "right turn"
                else:
                    traj = None

                return speed, traj

            # 计数字典（四个位置）本步增量
            speed_vocab = ["accelerate", "decelerate", "keep speed", "stop"]
            traj_vocab = ["straight", "left turn", "right turn"]
            pos_num = 4
            speed_counts_step = [dict((k, 0) for k in speed_vocab) for _ in range(pos_num)]
            traj_counts_step = [dict((k, 0) for k in traj_vocab) for _ in range(pos_num)]

            for path in all_paths:
                text = path.combined_assistant_text or ""
                m = _re.search(r"<meta actions>(.*?)</meta actions>", text, _re.DOTALL)
                if not m:
                    continue
                seq_str = m.group(1).strip()
                # 容错：统一引号
                seq_str = seq_str.replace("'", '"').replace('""', '"')
                try:
                    # 使用 eval 的原逻辑，但放在受控作用域
                    parsed = eval(seq_str, {"__builtins__": {}}, {})
                    if not isinstance(parsed, (list, tuple)):
                        continue
                    for idx, act in enumerate(parsed[:pos_num]):
                        if not isinstance(act, str):
                            continue
                        s, t = _normalize_action_pair(act)
                        if s in speed_counts_step[idx]:
                            speed_counts_step[idx][s] += 1
                        if t in traj_counts_step[idx]:
                            traj_counts_step[idx][t] += 1
                except Exception:
                    # 解析失败直接跳过
                    pass

            # 累加到窗口
            for pos in range(pos_num):
                for k in speed_vocab:
                    self._speed_counts_window[pos][k] += speed_counts_step[pos][k]
                for k in traj_vocab:
                    self._traj_counts_window[pos][k] += traj_counts_step[pos][k]

            # 若达到窗口边界，输出占比并重置窗口
            cur_step = self.state.global_step
            if cur_step - self._dist_window_start_step + 1 >= self._dist_window_size:
                # 输出占比
                for pos in range(pos_num):
                    total_s = sum(self._speed_counts_window[pos].values())
                    total_t = sum(self._traj_counts_window[pos].values())
                    for k in speed_vocab:
                        val = (self._speed_counts_window[pos][k] / total_s) if total_s > 0 else float('nan')
                        key = f"dist_speed_pos{pos+1}_{k.replace(' ', '_')}"
                        self._metrics.setdefault(key, []).append(val)
                    for k in traj_vocab:
                        val = (self._traj_counts_window[pos][k] / total_t) if total_t > 0 else float('nan')
                        key = f"dist_traj_pos{pos+1}_{k.replace(' ', '_')}"
                        self._metrics.setdefault(key, []).append(val)

                # 重置窗口，并追加一个轻微抖动点保证线段可视
                self._speed_counts_window = [dict((k, 0) for k in speed_vocab) for _ in range(pos_num)]
                self._traj_counts_window = [dict((k, 0) for k in traj_vocab) for _ in range(pos_num)]
                self._dist_window_start_step = cur_step + 1
        except Exception as e:
            logger.warning(f"Error calculating action entropy: {e}")

        # 批量计算所有奖励
        all_reward_values = {}
        for reward_func in self.reward_funcs:
            reward_name = reward_func.__name__
            try:
                reward_kwargs = {
                    "solution": solutions,
                    "vids": vids,
                    "sample_idx": [p.local_sample_idx for p in all_paths],
                    "gen_idx": [p.local_gen_idx for p in all_paths],
                    "future_coords": future_coords_list,
                }
                
                # 批量调用奖励函数
                reward_values = reward_func(completions=completions, **reward_kwargs)
                all_reward_values[reward_name] = reward_values

                # 记录指标
                metric_name = f"reward_{reward_name}"
                if metric_name not in self._metrics:
                    self._metrics[metric_name] = []
                self._metrics[metric_name].extend(reward_values)

                # 新增：收集奖励函数内部的监控曲线，便于绘图
                # 速度：
                if hasattr(reward_func, "_monitor_speed_maxStrictEdTrend"):
                    try:
                        monitor_vals = getattr(reward_func, "_monitor_speed_maxStrictEdTrend")
                        if monitor_vals is not None:
                            if "monitor_speed_maxStrictEdTrend" not in self._metrics:
                                self._metrics["monitor_speed_maxStrictEdTrend"] = []
                            self._metrics["monitor_speed_maxStrictEdTrend"].extend(monitor_vals)
                    except Exception:
                        pass

                # 轨迹：
                if hasattr(reward_func, "_monitor_traj_maxStrictEdTrend"):
                    try:
                        monitor_vals = getattr(reward_func, "_monitor_traj_maxStrictEdTrend")
                        if monitor_vals is not None:
                            if "monitor_traj_maxStrictEdTrend" not in self._metrics:
                                self._metrics["monitor_traj_maxStrictEdTrend"] = []
                            self._metrics["monitor_traj_maxStrictEdTrend"].extend(monitor_vals)
                    except Exception:
                        pass

            except Exception as e:
                logger.error(f"Error calculating reward {reward_name}: {e}", exc_info=True)
                all_reward_values[reward_name] = [0.0] * len(all_paths)

        # --- 新增：自适应工具使用奖励/惩罚 ---
        tool_usage_rewards = []
        speed_rewards = all_reward_values.get("speed_accuracy_reward", [0.0] * len(all_paths))
        traj_rewards = all_reward_values.get("traj_accuracy_reward", [0.0] * len(all_paths))
        
        # 计算自适应工具奖励
        tool_usage_rewards = self._calculate_adaptive_tool_rewards(
            all_paths, speed_rewards, traj_rewards
        )
        
        # 添加到奖励值字典中

        all_reward_values["tool_usage_reward"] = tool_usage_rewards
        reward_metric_name = "reward_tool_usage_reward"
        if reward_metric_name not in self._metrics:
            self._metrics[reward_metric_name] = []
        self._metrics[reward_metric_name].extend(tool_usage_rewards)
                    
                

        # 聚合所有奖励
        total_rewards_tensor = torch.zeros(len(all_paths), device=self.accelerator.device)
        for reward_name, values in all_reward_values.items():
            total_rewards_tensor += torch.tensor(values, device=self.accelerator.device)
        
        # --- 新增: 准确性加权的稀有度奖励 (全局批次版) ---
        if self.entropy_bonus_weight > 0 and all_paths:
            try:
                # 1. 提取所有本地样本的动作序列
                local_pred_speed_seqs, local_pred_traj_seqs = [], []
                for path in all_paths:
                    speed_seq, traj_seq = [], []
                    text = path.combined_assistant_text or ""
                    match = re.search(r'<meta actions>(.*?)</meta actions>', text, re.DOTALL)
                    if match:
                        seq_str = match.group(1).strip().replace("'", '"').replace('""', '"')
                        try:
                            parsed_seq = eval(seq_str)
                            for act_pair in parsed_seq:
                                s, t = _normalize_action(act_pair)
                                speed_seq.append(s)
                                traj_seq.append(t)
                        except Exception:
                            pass
                    local_pred_speed_seqs.append(speed_seq)
                    local_pred_traj_seqs.append(traj_seq)

                # 2. 从所有进程收集动作序列，以计算全局频率
                local_action_sequences = [{'speed': s, 'traj': t} for s, t in zip(local_pred_speed_seqs, local_pred_traj_seqs)]
                # gather_object返回一个嵌套列表，例如 [[proc0_results], [proc1_results], ...]
                global_action_sequences_nested = gather_object(local_action_sequences)
                # 将其展平为一个包含所有进程结果的列表
                # 安全地将嵌套列表展平，以处理来自不同进程的潜在异常数据类型
                global_action_sequences = []
                for sublist in global_action_sequences_nested:
                    if isinstance(sublist, list):
                        global_action_sequences.extend(sublist)
                    elif isinstance(sublist, dict):
                        # 处理当gather_object返回单个dict而不是list[dict]时的边缘情况
                        global_action_sequences.append(sublist)
                    elif sublist is not None:
                        # 只有主进程记录警告，避免日志刷屏
                        if self.accelerator.is_main_process:
                            logger.warning(f"Received unexpected data of type '{type(sublist)}' during action sequence gathering. Expected a list or dict. Skipping this data.")
                
                global_pred_speed_seqs = [item['speed'] for item in global_action_sequences]
                global_pred_traj_seqs = [item['traj'] for item in global_action_sequences]

                # 3. 基于全局序列计算批内频率
                pos_num = 4
                speed_counts = [defaultdict(int) for _ in range(pos_num)]
                traj_counts = [defaultdict(int) for _ in range(pos_num)]
                for seq in global_pred_speed_seqs:
                    for i in range(min(pos_num, len(seq))):
                        if seq[i]: speed_counts[i][seq[i]] += 1
                for seq in global_pred_traj_seqs:
                    for i in range(min(pos_num, len(seq))):
                        if seq[i]: traj_counts[i][seq[i]] += 1

                def _get_probs(counts):
                    return [
                        {k: v / sum(pos.values()) if sum(pos.values()) > 0 else 0 for k, v in pos.items()}
                        for pos in counts
                    ]
                speed_probs = _get_probs(speed_counts)
                traj_probs = _get_probs(traj_counts)

                # 4. 为每个本地样本计算加权稀有度奖励 (使用全局频率)
                rarity_bonuses = []
                eps = 1e-6
                
                speed_accuracy = all_reward_values.get("speed_accuracy_reward", [0.0] * len(all_paths))
                traj_accuracy = all_reward_values.get("traj_accuracy_reward", [0.0] * len(all_paths))

                for i in range(len(all_paths)):
                    speed_seq, traj_seq = local_pred_speed_seqs[i], local_pred_traj_seqs[i] # 使用本地序列
                    raw_bonus, num_actions = 0.0, 0
                    
                    for k in range(min(pos_num, len(speed_seq))):
                        if speed_seq[k]:
                            p = speed_probs[k].get(speed_seq[k], 0.0) # 使用全局概率
                            raw_bonus += -math.log(p + eps)
                            num_actions += 1
                    for k in range(min(pos_num, len(traj_seq))):
                        if traj_seq[k]:
                            p = traj_probs[k].get(traj_seq[k], 0.0) # 使用全局概率
                            raw_bonus += -math.log(p + eps)
                            num_actions += 1
                    
                    avg_raw_bonus = (raw_bonus / num_actions) if num_actions > 0 else 0.0
                    
                    if self.weight_rarity_by_accuracy:
                        # 关键：用准确性加权
                        accuracy_score = (speed_accuracy[i] + traj_accuracy[i])
                        # 准确性得分在[0,1]之间，作为权重
                        accuracy_weight = max(0, min(1, accuracy_score))
                        final_bonus = avg_raw_bonus * accuracy_weight
                    else:
                        # 不加权，直接使用原始稀有度奖励
                        final_bonus = avg_raw_bonus

                    rarity_bonuses.append(final_bonus)

                # 5. 添加到总奖励中并监控
                rarity_bonuses_tensor = torch.tensor(rarity_bonuses, device=self.accelerator.device) * self.entropy_bonus_weight
                total_rewards_tensor += rarity_bonuses_tensor
                
                metric_name = "reward_rarity_bonus_weighted"
                if metric_name not in self._metrics: 
                    self._metrics[metric_name] = []
                self._metrics[metric_name].extend(rarity_bonuses_tensor.cpu().tolist())

            except Exception as e:
                logger.error("Error calculating rarity bonus", exc_info=True)

        local_rewards = total_rewards_tensor.cpu().tolist()

        # 收集其他数据以准备模型输入
        local_dialogue_histories = [p.dialogue_history for p in all_paths]
        local_prompts = [p.get_vllm_input_prompt(self.processing_class) for p in all_paths]
        local_images = [p.get_images_for_vllm() for p in all_paths]
        local_tool_call_counts = [p.tool_call_count for p in all_paths]
        
        # 验证分组是否正确
        if all_paths:
            local_sample_indices = [p.local_sample_idx for p in all_paths]
            current_sample = None
            count_in_group = 0
            for i, sample_idx in enumerate(local_sample_indices):
                if sample_idx != current_sample:
                    if current_sample is not None:
                        if count_in_group != self.num_generations:
                            logger.warning(f"Sample {current_sample}: expected {self.num_generations} generations, but got {count_in_group}")
                    current_sample = sample_idx
                    count_in_group = 1
                else:
                    count_in_group += 1

        # 将对话处理为模型输入
        if local_prompts:
            prompt_inputs = self.processing_class(
                text=local_prompts,
                images=local_images,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            
            # 准备输入张量
            input_ids = prompt_inputs["input_ids"].to(device)
            attention_mask = prompt_inputs["attention_mask"].to(device)
            pixel_values = prompt_inputs["pixel_values"].to(device) if "pixel_values" in prompt_inputs else None
            image_grid_thw = prompt_inputs["image_grid_thw"].to(device) if "image_grid_thw" in prompt_inputs else None
            
            # 创建完成掩码（所有role为assistant生成的内容）
            role_masks = self.create_role_masks(input_ids, local_dialogue_histories)
            assistant_mask = role_masks["assistant_mask"]
            
            # 计算参考模型的log概率
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        input_ids,
                        attention_mask,
                        pixel_values,
                        image_grid_thw,
                        assistant_mask
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            pixel_values,
                            image_grid_thw,
                            assistant_mask
                        )
        else:
            # 如果没有本地数据，创建空张量
            input_ids = torch.empty(0, 0, dtype=torch.long, device=device)
            attention_mask = torch.empty(0, 0, dtype=torch.long, device=device)
            pixel_values = None
            image_grid_thw = None
            assistant_mask = torch.empty(0, 0, dtype=torch.float, device=device)
            ref_per_token_logps = torch.empty(0, 0, dtype=torch.float, device=device)
            local_rewards = []
        
        # 计算奖励和优势值
        if local_rewards:
            rewards_tensor = torch.tensor(local_rewards, device=device)

            # 确保数据长度是num_generations的整数倍
            if len(rewards_tensor) % self.num_generations != 0:
                logger.warning(f"rewards_tensor length {len(rewards_tensor)} is not a multiple of num_generations {self.num_generations}")
                # 截断到最近的完整组
                groups_count = len(rewards_tensor) // self.num_generations
                new_len = groups_count * self.num_generations
                rewards_tensor = rewards_tensor[:new_len]
                logger.debug(f"Truncated rewards_tensor to length {new_len}")

            # 计算优势值 - 现在数据已经正确分组
            if len(rewards_tensor) > 0:
                try:
                    # 重塑为[样本数, num_generations]
                    grouped_rewards = rewards_tensor.view(-1, self.num_generations)
                    
                    # 计算每个原始样本组的平均奖励和标准差
                    mean_grouped_rewards = grouped_rewards.mean(dim=1)
                    std_grouped_rewards = grouped_rewards.std(dim=1, unbiased=False)
                    
                    # 标准差可能为0，添加小的epsilon避免除零
                    std_grouped_rewards = torch.clamp(std_grouped_rewards, min=1e-8)
                    
                    # 将均值和标准差扩展回原始形状
                    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations)
                    std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations)
                    
                    # 计算优势值
                    advantages = (rewards_tensor - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
                    
                except Exception as e:
                    logger.error(f"Error calculating advantages: {e}", exc_info=True)
                    advantages = torch.zeros_like(rewards_tensor)
            else:
                advantages = torch.zeros_like(rewards_tensor)
        else:
            rewards_tensor = torch.empty(0, device=device)
            advantages = torch.empty(0, device=device)
        
        # 记录指标
        if len(rewards_tensor) > 0:
            self._metrics["reward"].append(rewards_tensor.mean().item())
            self._metrics["reward_std"].append(rewards_tensor.std().item())
        
        # 计算并记录平均工具调用次数
        if local_tool_call_counts:
            avg_tool_calls = sum(local_tool_call_counts) / len(local_tool_call_counts)
            if "avg_tool_calls" not in self._metrics:
                self._metrics["avg_tool_calls"] = []
            self._metrics["avg_tool_calls"].append(avg_tool_calls)
        
        # 清理路径资源
        for path in all_paths:
            path.cleanup()

        # 清理临时变量
        all_paths = None
        
        # 强制垃圾回收
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "assistant_mask": assistant_mask,
            "advantages": advantages,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "ref_per_token_logps": ref_per_token_logps,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算损失函数，仅对assistant角色生成的内容计算损失。
        包含KL散度项以正则化生成，同时使用优势函数来指导策略更新。
        
        注意：当前实现是对每个样本的token求平均，未来可以考虑对group内所有token求平均。
        """
        if return_outputs:
            raise ValueError("GRPOTrainer不支持返回输出")
        
        # 获取输入
        input_ids = inputs["input_ids"] 
        attention_mask = inputs["attention_mask"]
        assistant_mask = inputs["assistant_mask"]  # 使用传入的assistant_mask
        pixel_values = inputs["pixel_values"]
        image_grid_thw = inputs["image_grid_thw"]
        advantages = inputs["advantages"]
        ref_per_token_logps = inputs["ref_per_token_logps"] if "ref_per_token_logps" in inputs else None
        
        # 使用_get_per_token_logps计算当前模型的token概率
        per_token_logps = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            assistant_mask
        )
        
        # 直接使用assistant_mask作为最终掩码
        # 这确保了只有assistant生成的内容会被计算损失
        mask = assistant_mask
        
        # 如果有参考模型的log概率，计算KL散度
        if ref_per_token_logps is not None:
            # 计算模型和参考模型之间的KL散度
            # KL(q||p) = E_q[log(q/p)] = E_q[log q - log p]
            # 使用重参数化技巧: exp(ref_logp - logp) - (ref_logp - logp) - 1
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - 
                (ref_per_token_logps - per_token_logps) - 
                1
            )
        else:
            # 如果没有参考模型的log概率，KL散度为0
            logger.warning("No reference model log probabilities available, KL divergence set to 0")
            per_token_kl = torch.zeros_like(per_token_logps)
        
        # 计算重要性采样比率
        ratio = torch.exp(per_token_logps - per_token_logps.detach())

        # 策略梯度损失 (无裁剪)
        per_token_loss = -ratio * advantages.unsqueeze(1)
        
        # 加上KL散度惩罚项
        per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # 应用mask，只考虑assistant生成的部分，计算平均损失
        # 每个样本的loss是token的平均loss
        sample_losses = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        
        # 最终loss是所有样本的平均
        loss = sample_losses.mean()
            
        # 记录指标
        completion_length = (
                self.accelerator.gather_for_metrics(mask.sum(1))
                .float()
                .mean()
                .item()
            )
        self._metrics["completion_length"].append(completion_length)
        
        # 记录KL散度
        if ref_per_token_logps is not None:
            mean_kl = (
                (per_token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
        
        logger.debug(f"loss: {loss.item():.6f}, assistant tokens: {mask.sum().item()}, " +
                     (f"KL: {mean_kl.item():.6f}" if ref_per_token_logps is not None else "KL: N/A"))
        
        return loss

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """根据是否启用工具调用选择相应的输入准备方法"""
        if self.enable_tool_calls:
            return self._prepare_inputs_with_tool_calls(inputs)
        else:
            return self._prepare_inputs_original(inputs)

    def _calculate_adaptive_tool_rewards(self, all_paths, speed_rewards, traj_rewards):
        """
        自适应工具奖励计算方法
        按组(sample_id)分别计算调用工具路径的平均分s1和非调用工具路径的平均分s2，
        给每个调用工具的路径加上(s1-s2)*individual_quality的奖励
        
        
        (self_s1-avg_s2)*individual_quality
        """
        # 按sample_id分组路径
        groups = {}
        for i, path in enumerate(all_paths):
            sample_id = path.sample_id
            if sample_id not in groups:
                groups[sample_id] = []
            groups[sample_id].append((i, path))
        
        # 初始化奖励数组
        tool_usage_rewards = [0.0] * len(all_paths)
        
        # 为了避免极端情况，设置奖励范围限制
        max_reward = 0.2
        min_reward = -0.2
        
        # 统计信息（用于调试）
        debug_info = {
            'total_groups': len(groups),
            'groups_with_tools': 0,
            'avg_s1_across_groups': 0.0,
            'avg_s2_across_groups': 0.0,
            'avg_reward_diff': 0.0,
            'tool_ratios': []
        }
        
        # 对每个组分别计算奖励
        valid_groups = 0
        for sample_id, group_paths in groups.items():
            # 分离组内调用工具和未调用工具的路径
            tool_indices = []
            no_tool_indices = []
            tool_scores = []
            no_tool_scores = []
            
            for path_idx, path in group_paths:
                reward = speed_rewards[path_idx] + traj_rewards[path_idx]
                
                if path.tool_call_count > 0:
                    tool_indices.append(path_idx)
                    tool_scores.append(reward)
                else:
                    no_tool_indices.append(path_idx)
                    no_tool_scores.append(reward)
            
            # 计算组内平均分数（用于调试）；奖励计算使用当前样本的s1
            s1_avg = sum(tool_scores) / len(tool_scores) if tool_scores else 0.0  # 组内调用工具路径的平均分（仅用于统计）
            s2_avg = sum(no_tool_scores) / len(no_tool_scores) if no_tool_scores else 0.0  # 组内未调用工具路径的平均分
            s2_available = True if no_tool_scores else False
            
            # 统计信息更新
            if tool_scores:
                debug_info['groups_with_tools'] += 1
                debug_info['avg_s1_across_groups'] += s1_avg
                debug_info['avg_s2_across_groups'] += s2_avg
                # 使用当前样本的(s1 - s2)差值的组内平均作为统计
                if s2_available:
                    diffs = [(speed_rewards[idx] + traj_rewards[idx]) - s2_avg for idx in tool_indices]
                else:
                    diffs = [0.0 for idx in tool_indices]
                group_avg_diff = sum(diffs) / len(diffs) if diffs else 0.0
                debug_info['avg_reward_diff'] += group_avg_diff
                valid_groups += 1
                
            tool_ratio = len(tool_indices) / len(group_paths) if group_paths else 0.0
            debug_info['tool_ratios'].append(tool_ratio)
            
            # 计算组内每个路径的奖励
            for path_idx, path in group_paths:
                tool_usage_reward = 0.0
                
                if self.mode in ["mixed"]:
                    # Mixed模式：强制策略检查
                    if path.force_no_tool_use:
                        if path.tool_call_count > 0:  # 惩罚不一致行为
                            tool_usage_reward = -0.2
                    elif path.force_tool_use:
                        if path.tool_call_count == 0:  # 惩罚不一致行为
                            tool_usage_reward = -0.2
                            
                elif self.mode in ["no-image"]:
                    # no-image模式：不使用工具
                    if path.tool_call_count > 0:
                        tool_usage_reward = -0.2  # 惩罚意外的工具调用
                        
                elif self.mode in ['adaptive']:
                    # adaptive模式：自适应奖励机制
                    if path.force_no_tool_use:
                        if path.tool_call_count > 0:  # 惩罚不一致行为
                            tool_usage_reward = -0.2
                    elif path.force_tool_use:
                        if path.tool_call_count == 0:  # 惩罚不一致行为
                            tool_usage_reward = -0.2
                        else:
                            # 自适应奖励计算：(s1-s2) * individual_quality
                            if path.accumulated_tool_reward is not None and path.tool_call_count > 0:
                                individual_quality = path.accumulated_tool_reward / path.tool_call_count
                            else:
                                individual_quality = 0.0
                            
                            # 基于当前样本的s1（该样本的奖励）与组内s2（未使用工具平均）计算差值，并引入平滑工具成本边际
                            current_reward = speed_rewards[path_idx] + traj_rewards[path_idx]
                            base_reward_diff_current = (current_reward - s2_avg) if s2_available else 0.0
                            # 平滑负偏置：要求收益至少超过每次调用成本之和，避免零点不连续
                            tool_cost_per_call = 0.125
                            margin = path.tool_call_count * tool_cost_per_call
                            adjusted_diff = base_reward_diff_current - margin if s2_available else 0.0
                            if adjusted_diff > 0:
                                adaptive_reward = adjusted_diff * individual_quality
                            else:
                                adaptive_reward = adjusted_diff
                            
                            # 应用奖励范围限制
                            tool_usage_reward = max(min_reward, min(max_reward, adaptive_reward))
                    else:
                        # 没有按照格式选择任意的其中一种模式，应该惩罚
                        tool_usage_reward = -0.2
                            
                else:
                    pass
                
                tool_usage_rewards[path_idx] = tool_usage_reward
        
        # 计算平均统计信息
        if valid_groups > 0:
            debug_info['avg_s1_across_groups'] /= valid_groups
            debug_info['avg_s2_across_groups'] /= valid_groups
            debug_info['avg_reward_diff'] /= valid_groups
        
        # 记录调试信息
        if self.accelerator.is_main_process:
            current_step = self.state.global_step
            
            # 每10步记录一次详细信息
            if current_step % 10 == 0:
                avg_tool_ratio = sum(debug_info['tool_ratios']) / len(debug_info['tool_ratios']) if debug_info['tool_ratios'] else 0.0
                avg_tool_reward = sum(tool_usage_rewards) / len(tool_usage_rewards) if tool_usage_rewards else 0.0
                
                logger.info(f"Step {current_step}: Adaptive tool reward statistics - "
                          f"Total groups: {debug_info['total_groups']}, Groups with tools: {debug_info['groups_with_tools']}, "
                          f"Avg tool ratio: {avg_tool_ratio:.3f}, Avg s1: {debug_info['avg_s1_across_groups']:.4f}, "
                          f"Avg s2: {debug_info['avg_s2_across_groups']:.4f}, Avg reward diff: {debug_info['avg_reward_diff']:.4f}, "
                          f"Avg tool reward: {avg_tool_reward:.4f}")
        
        return tool_usage_rewards

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        is_eval = next(iter(logs.keys()), "").startswith("eval_")

        if not is_eval and self.state.is_world_process_zero:
            self.logged_steps.append(self.state.global_step)

            all_known_keys = set(self.metrics_history.keys())
            current_metric_keys = set(metrics.keys())
            all_keys = all_known_keys.union(current_metric_keys)
            
            for key in all_keys:
                # 新指标首次出现时，为历史步骤补 NaN 以对齐 x 轴
                if key not in self.metrics_history:
                    self.metrics_history[key] = [float('nan')] * (len(self.logged_steps) - 1)
                value = metrics.get(key, float('nan'))
                self.metrics_history[key].append(value)

            try:
                self._plot_metrics()
            except Exception as e:
                warnings.warn(f"Could not plot metrics. Error: {e}")

        

        # 额外绘制分布柱状图（每个窗口一个分组，包含不同动作的占比；保留历史窗口，并实时加入当前未满窗口的累计占比）
        try:
            self._plot_distribution_bars()
        except Exception as e:
            warnings.warn(f"Could not plot distribution bars. Error: {e}")

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if is_eval:
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
    
        
        self._metrics.clear()

    def _plot_metrics(self, smoothing_window=3):
        if not self.metrics_history or not self.logged_steps:
            return

        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        plot_path = os.path.join(output_dir, "training_metrics.png")

        num_metrics = len(self.metrics_history)
        cols = 3
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, (key, values) in enumerate(self.metrics_history.items()):
            ax = axes[i]
            
            # 对齐长度（某些指标动态出现导致长度短于logged_steps）
            x = self.logged_steps
            y = values
            if len(y) < len(x):
                # 右侧补NaN以对齐步数
                y = y + [float('nan')] * (len(x) - len(y))
            elif len(y) > len(x):
                y = y[:len(x)]

            # 原始数据点
            ax.plot(x, y, marker='o', linestyle='', alpha=0.3, label='Original')
            
            # 平滑处理
            if len(y) >= 1:
                series = pd.Series(y)
                # 使用中心对齐的滑动窗口，min_periods=1 确保在数据点少于窗口大小时也能生成结果
                smoothed_values = series.rolling(window=smoothing_window, min_periods=1, center=True).mean()
                ax.plot(x, smoothed_values, linestyle='-', label=f'Smoothed (window={smoothing_window})')

            ax.set_title(f"Metric: {key}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True)
            ax.legend()
        
        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(plot_path, dpi=500)
        plt.close(fig)

    def _plot_distribution_bars(self):
        if not self.metrics_history:
            return

        output_dir = self.args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        speed_actions = ["accelerate", "decelerate", "keep speed", "stop"]
        traj_actions = ["straight", "left turn", "right turn"]
        pos_num = 4
        window = self._dist_window_size

        # 从已关闭窗口的指标中恢复历史窗口数量（只统计非 NaN 的真实窗口点）
        def get_closed_len(prefix, pos, actions):
            max_len = 0
            for a in actions:
                key = f"{prefix}{pos}_{a}"
                vals = self.metrics_history.get(key, [])
                # 只统计真实记录的窗口点（过滤对齐产生的 NaN）
                non_nan = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
                max_len = max(max_len, len(non_nan))
            return max_len

        # 计算当前窗口的累计占比
        def get_current_window_ratio(counts_dict):
            total = sum(counts_dict.values())
            return {k: (counts_dict[k] / total) if total > 0 else 0.0 for k in counts_dict}

        # 画布：2行x4列
        rows, cols = 2, 4
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4.5 * rows))

        # 速度（第一行）
        for pos in range(1, pos_num + 1):
            ax = axes[0, pos - 1]
            closed_len = get_closed_len("dist_speed_pos", pos, speed_actions)
            # 收集每个动作的历史占比（仅非 NaN）
            series = {}
            for a in speed_actions:
                key = f"dist_speed_pos{pos}_{a}"
                vals_all = list(self.metrics_history.get(key, []))
                vals = [v for v in vals_all if not (isinstance(v, float) and math.isnan(v))]
                # 保证与 closed_len 对齐（极端情况下不同动作记录数不同）
                if len(vals) < closed_len:
                    vals += [0.0] * (closed_len - len(vals))
                series[a] = vals
            # 追加当前窗口占比
            cur_ratio = get_current_window_ratio(self._speed_counts_window[pos - 1])
            for a in speed_actions:
                series[a] = series[a] + [cur_ratio.get(a, 0.0)]
            # x 轴窗口标签（1-based）：1-20, 21-40, ..., lastStart-t
            num_windows = closed_len + 1
            x = np.arange(num_windows)
            closed_labels = [f"{i*window + 1}-{(i+1)*window}" for i in range(closed_len)]
            open_label = f"{closed_len*window + 1}-{self.state.global_step + 1}"
            labels = closed_labels + [open_label]
            width = 0.18
            offsets = np.linspace(-1.5*width, 1.5*width, len(speed_actions))
            for off, a in zip(offsets, speed_actions):
                ax.bar(x + off, series[a], width, label=a)
            ax.set_title(f"Speed pos{pos}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion")
            ax.grid(True, axis='y', alpha=0.3)
            if pos == 4:
                ax.legend(fontsize=8)

        # 轨迹（第二行）
        for pos in range(1, pos_num + 1):
            ax = axes[1, pos - 1]
            closed_len = get_closed_len("dist_traj_pos", pos, traj_actions)
            series = {}
            for a in traj_actions:
                key = f"dist_traj_pos{pos}_{a}"
                vals_all = list(self.metrics_history.get(key, []))
                vals = [v for v in vals_all if not (isinstance(v, float) and math.isnan(v))]
                if len(vals) < closed_len:
                    vals += [0.0] * (closed_len - len(vals))
                series[a] = vals
            cur_ratio = get_current_window_ratio(self._traj_counts_window[pos - 1])
            for a in traj_actions:
                series[a] = series[a] + [cur_ratio.get(a, 0.0)]
            num_windows = closed_len + 1
            x = np.arange(num_windows)
            closed_labels = [f"{i*window + 1}-{(i+1)*window}" for i in range(closed_len)]
            open_label = f"{closed_len*window + 1}-{self.state.global_step + 1}"
            labels = closed_labels + [open_label]
            width = 0.22
            offsets = np.linspace(-(len(traj_actions)-1)/2*width, (len(traj_actions)-1)/2*width, len(traj_actions))
            for off, a in zip(offsets, traj_actions):
                ax.bar(x + off, series[a], width, label=a)
            ax.set_title(f"Trajectory pos{pos}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Proportion")
            ax.grid(True, axis='y', alpha=0.3)
            if pos == 4:
                ax.legend(fontsize=8)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "distribution_bars.png")
        plt.savefig(plot_path, dpi=500)
        plt.close(fig)