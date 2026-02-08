import os
import torch
import json
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset
import re
from datetime import datetime
from grpo import accuracy_reward, format_reward

class ModelEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.padding_side = 'left'
        self.processor.tokenizer.padding_side = 'left'
        # 设置生成参数
        self.gen_config = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 1.0,
            "top_p": 0.8
        }
        
    def process_batch(self, batch):
        # 构建输入模板
        messages = []
        for  history_meta_actions, image in zip(batch["history_meta_actions"], batch["image"]):
            message = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"""You are an autonomous driving labeller. You can access six different perspectives of car camera images from the current. 
                            These images are taken from front, front-right, front-left, back, back-left, and back-right. You can also view the historical driving behavior of the past 5 time points:{history_meta_actions}.
                            Your task is to predict the meta actions sequence for future 7 time points of the vehicle. 
                            Please follow the following thinking pattern:
                            1. Stage1—decription: imagine you are driving the car. Describe the driving scene according to weather, traffic lights, other cars or pedestrians and lane markings. Describe all important objects that you should pay attention to with their relative location to you. 
                            2. Stage2—reasoning: based on the description and historical driving actions, provide a brief description of the vehicle's recent behavior and purpose. Then point out objects that may influence potential driving decisions and considerations.
                            3. Stage3—prediction: based on the description and historical driving actions, predict the future driving action of the car in the next 7 time points (0.5-second intervals). Explain why you predict these actions. Output your predicted Meta-Actions in the format of ['Speed Action, Trajectory Action', ...] where Speed Action is one of (Accelerate, Decelerate, Maintain speed) and Trajectory Action is one of (Go Straight, Turn Left, Turn Right, Change Lane Left, Change Lane Right, Stop)."""}
                ]
            }]
            messages.append(message)
            
        # 处理输入
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        
        inputs = self.processor(
            text=texts,
            images=batch["image"],
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # 生成回答
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                **self.gen_config
            )
            
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return outputs

    def evaluate(self, dataset, batch_size=4):
        all_rewards = []
        all_format_rewards = []
        
        # 创建进度条
        progress_bar = tqdm(total=len(dataset), desc="Evaluating")
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            outputs = self.process_batch(batch)
            
            # 计算奖励
            completions = [[{"content": output}] for output in outputs]
            solutions = batch["solution"]
            
            # 计算准确度奖励
            acc_rewards = accuracy_reward(completions, solutions)
            # 计算格式奖励
            fmt_rewards = format_reward(completions)
            
            all_rewards.extend(acc_rewards)
            all_format_rewards.extend(fmt_rewards)
            
            progress_bar.update(len(batch))
        
        progress_bar.close()
        
        # 计算平均分数
        avg_reward = sum(all_rewards) / len(all_rewards)
        avg_format_reward = sum(all_format_rewards) / len(all_format_rewards)
        
        return {
            "accuracy_reward": avg_reward,
            "format_reward": avg_format_reward,
            "total_reward": avg_reward + avg_format_reward
        }

def main():
    # 配置参数
    model_path = "/cephfs/zhengwc/R1-V/src/r1-v/DriveVLM-Qwen2.5-VL-3B-SFT-RL"  # 替换为实际的模型路径
    dataset_name = "/cephfs/shared/zhengwc/datasets/DVLM-VAL"  # 替换为实际的数据集名称
    output_dir = "eval_results"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    dataset = load_dataset(dataset_name)
    dataset = dataset["validation"]
    
    # 初始化评测器
    evaluator = ModelEvaluator(model_path, device)
    
    # 运行评测
    results = evaluator.evaluate(dataset, batch_size)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_results_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation Results:")
    print(f"Accuracy Reward: {results['accuracy_reward']:.4f}")
    print(f"Format Reward: {results['format_reward']:.4f}")
    print(f"Total Reward: {results['total_reward']:.4f}")
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()