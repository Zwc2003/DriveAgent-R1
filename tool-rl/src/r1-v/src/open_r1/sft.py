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

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 100 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill
"""

import logging
import os
import sys
import json
import warnings
import datasets
from dataclasses import dataclass, field
from typing import Optional
import torch
import transformers
from datasets import load_dataset, Features, Value, Image
from transformers import AutoTokenizer, set_seed, AutoProcessor
from transformers.trainer_utils import get_last_checkpoint
import trl
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from transformers import AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2VLConfig, Qwen2VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
AutoModelForCausalLM.register(config_class=Qwen2_5_VLConfig, model_class=Qwen2_5_VLForConditionalGeneration)
AutoModelForCausalLM.register(config_class=Qwen2VLConfig, model_class=Qwen2VLForConditionalGeneration)

from qwen_vl_utils import process_vision_info
logger = logging.getLogger(__name__)


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "The weight decay to apply (if not zero)."}
    )
    merge_lora_after_train: bool = field(
        default=True,
        metadata={"help": "Whether to merge the LoRA adapter into the base model after training."}
    )
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "The callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(default=False, metadata={"help": "Whether to overwrite the Hub revision."})
    push_to_hub_revision: bool = field(default=False, metadata={"help": "Whether to push to a Hub revision/branch."})
    freeze_mode: Optional[str] = field(
        default="none",
        metadata={"help": "Part of the model to freeze. One of ['none', 'vision_encoder', 'vision_projector', 'vision_encoder_except_projector']"}
    )
    chat_template_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the chat template jinja file."},
    )




processor = None


def convert_example(example):
    """
    将数据集中的样本转换为模型所需的格式
    
    Args:
        example: 输入样本，包含message(字符串)和图像字段
        
    Returns:
        处理后的样本，包含messages字段(Python对象)
    """
    # 解析message字符串为Python对象
    try:
        if isinstance(example["message"], str):
            messages = json.loads(example["message"])
        else:
            messages = example["message"]
    except:
        logger.error(f"无法解析message字段: {example['message'][:100]}...")
        return example
    
    # # 图像资源列表
    image_resources = [
        example.get("cam_front", None),
        example.get("tool_image_0", None),
        example.get("tool_image_1", None), 
        example.get("tool_image_2", None),
        example.get("tool_image_3", None)
    ]
    
    # 过滤掉None或空字符串
    image_resources = [img for img in image_resources if img is not None and img != ""]
    # 在过滤后添加
    logger.debug(f"样本 {example.get('vid', 'unknown')} 使用了 {len(image_resources)} 个图像")
    
    # 跟踪已处理的图像数量
    image_index = 0
    
    # 遍历messages中的所有消息和内容
    for message in messages:
        if "content" in message:
            # 处理内容列表
            if isinstance(message["content"], list):
                for i, content_item in enumerate(message["content"]):
                    # 找到类型为image的内容项
                    if isinstance(content_item, dict) and content_item.get("type") == "image":
                        # 确保还有可用图像资源
                        if image_index < len(image_resources):
                            # 用实际图像资源替换占位符
                            message["content"][i] = {
                                "type": "image", 
                                "image": image_resources[image_index]
                            }
                            image_index += 1
                        else:
                            logger.warning(f"图像资源不足，需要更多图像来填充对话")
            # 处理字符串形式的内容，保持不变
            elif isinstance(message["content"], str):
                logger.warning(f"字符串形式的内容，无法填充")
                pass
    
    # 将处理后的messages放入新字段，保留原始message字段
    example["messages"] = messages
    
    return example


def collate_fn(examples):
    # 处理文本和图像输入
    texts = [
        processor.apply_chat_template(convert_example(example)["messages"], tokenize=False, add_generation_prompt=True)
        for example in examples
    ]
    image_inputs = []
    for example in examples:
        imgs, vids = process_vision_info(example["messages"])
        image_inputs.append(imgs)
    batch = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    # 首先处理padding和图像token
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    labels[labels == image_token_id] = -100
    
    # 接下来精确识别并只保留assistant角色的token
    input_ids = batch["input_ids"]
    dialogue_histories = [convert_example(example)["messages"] for example in examples]
    
    # 获取特殊标记的ID
    tokenizer = processor.tokenizer
    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    assistant_marker = tokenizer.encode("assistant", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    # 对每个样本处理
    for i in range(len(input_ids)):
        tokens = input_ids[i].tolist()
        
        # 初始化所有token为-100，后面只恢复assistant的token
        labels[i, :] = -100
        
        # 查找所有角色标记的位置
        role_positions = []  # [(位置, "role_type", "start/end"), ...]
        
        # 查找所有<|im_start|>位置
        for j in range(len(tokens) - len(im_start) + 1):
            if tokens[j:j+len(im_start)] == im_start:
                role_end = j + len(im_start)
                # 检查是否是assistant
                if role_end + len(assistant_marker) <= len(tokens) and tokens[role_end:role_end+len(assistant_marker)] == assistant_marker:
                    role_positions.append((j, "assistant", "start"))
                # 其他角色标记为非assistant
                else:
                    role_positions.append((j, "other", "start"))
        
        # 查找所有<|im_end|>位置
        for j in range(len(tokens) - len(im_end) + 1):
            if tokens[j:j+len(im_end)] == im_end:
                role_positions.append((j, "any", "end"))
        
        # 按位置排序
        role_positions.sort(key=lambda x: x[0])
        
        # 扫描所有角色标记，识别assistant部分
        j = 0
        current_role = None
        while j < len(role_positions):
            pos, role, marker = role_positions[j]
            
            if marker == "start":
                current_role = role
                if role == "assistant":
                    # 找到assistant内容的开始位置（跳过<|im_start|>assistant标记）
                    start_pos = pos + len(im_start) + len(assistant_marker)
                    
                    # 查找对应的结束标记
                    end_pos = None
                    for k in range(j+1, len(role_positions)):
                        if role_positions[k][2] == "end":
                            end_pos = role_positions[k][0]
                            j = k  # 跳到这个结束标记
                            break
                    
                    if end_pos is not None:
                        # 找到了匹配的结束标记，恢复这个区域的标签（使用原始token ID）
                        labels[i, start_pos:end_pos] = input_ids[i, start_pos:end_pos]
                    else:
                        # 没有找到结束标记，使用序列末尾
                        labels[i, start_pos:] = input_ids[i, start_pos:]
            
            elif marker == "end":
                current_role = None
            
            j += 1
        
        # 备用方法：如果没有找到任何assistant区域，尝试使用文本搜索
        if not any(role == "assistant" for _, role, _ in role_positions):
            print("没有找到assistant区域")
    
    batch["labels"] = labels
    return batch




def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")
    print("script_args:",script_args)
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################

    features = Features({
        'vid': Value('string'),
        'message': Value('string'),
        'cam_front': Image(decode=True),
        'tool_image_0': Image(decode=True),  # 新增
        'tool_image_1': Image(decode=True),  # 新增
        'tool_image_2': Image(decode=True),  # 新增
    })


    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, features=features)

    dataset = dataset.shuffle(seed=42) # 打乱数据

    ################
    # Load tokenizer
    ################
    global processor
    if "vl" in model_args.model_name_or_path.lower():
        print("Using AutoProcessor for vision-language model.")
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code,max_pixels = 259200, min_pixels = 6272
        )
        # processor = AutoProcessor.from_pretrained(
        #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
        # )
        logger.info("Using AutoProcessor for vision-language model.")
    else:
        processor = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True,max_pixels = 259200, min_pixels = 6272
        )
        # processor = AutoTokenizer.from_pretrained(
        #     model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
        # )
        logger.info("Using AutoTokenizer for text-only model.")
    if script_args.chat_template_path:
        with open(script_args.chat_template_path, "r") as f:
            chat_template = f.read()
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.chat_template = chat_template
        else:
            processor.chat_template = chat_template
        print(f"Loaded chat template from {script_args.chat_template_path}")
    if hasattr(processor, "pad_token") and processor.pad_token is None:
        processor.pad_token = processor.eos_token
    elif hasattr(processor.tokenizer, "pad_token") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    

    
    ###################
    # Model init kwargs
    ###################
    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    torch_dtype = torch.bfloat16
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    # training_args.model_init_kwargs = model_kwargs
    from transformers import Qwen2VLForConditionalGeneration,Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )

    if training_args.freeze_mode == "vision_encoder":
        print("Freezing the entire vision encoder.")
        for param in model.visual.parameters():
            param.requires_grad = False
    elif training_args.freeze_mode == "vision_projector":
        print("Freezing the vision projector only.")
        for param in model.visual.merger.parameters():
            param.requires_grad = False
    elif training_args.freeze_mode == "vision_encoder_except_projector":
        print("Freezing vision encoder, but keeping projector trainable.")
        for param in model.visual.parameters():
            param.requires_grad = False
        for param in model.visual.merger.parameters():
            param.requires_grad = True
    elif training_args.freeze_mode != "none":
        warnings.warn(f"Unknown freeze_mode: {training_args.freeze_mode}. Not freezing any part of the model.")



    ############################
    # Initialize the SFT Trainer
    ############################
    training_args.dataset_kwargs = {
        "skip_prepare_dataset": True,
    }
    training_args.remove_unused_columns = False
    # eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset= None,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        peft_config=get_peft_config(model_args)
    )



    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["R1-V"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    #############
    # push to hub
    #############

    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)
        processor.push_to_hub(training_args.hub_model_id)




if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
