#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DetAny3D推理脚本
用于单张图像的3D目标检测推理
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
import yaml
from box import Box
import colorsys
import hashlib
import io
from torchvision.ops import box_convert

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_utils import *
from wrap_model import WrapModel
from detect_anything.utils.transforms import ResizeLongestSide

from groundingdino.util.inference import load_model
from groundingdino.util.inference import predict as dino_predict
import groundingdino.datasets.transforms as T


class DetAny3DInference:
    def __init__(self, config_path=None, device='cuda:0'):
        """
        初始化DetAny3D推理器
        
        Args:
            config_path: 配置文件路径，默认使用demo.yaml
            device: 推理设备
        """
        self.device = device
        
        # 加载配置
        if config_path is None:
            config_path = './detect_anything/configs/demo.yaml'
        
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        self.cfg = Box(cfg)
        
        # 禁用分布式
        self._disable_distributed()
        
        # 初始化模型
        self._load_models()
        
        # 设置推理参数
        self.BOX_THRESHOLD = 0.37
        self.TEXT_THRESHOLD = 0.25
        
        print(f"DetAny3D推理器初始化完成，使用设备: {device}")
    
    def _disable_distributed(self):
        pass
    
    def _load_models(self):
        """加载DetAny3D和GroundingDINO模型"""
        # 加载DetAny3D模型
        self.my_sam_model = WrapModel(self.cfg)
        
        # 加载预训练权重
        if os.path.exists(self.cfg.resume):
            checkpoint = torch.load(self.cfg.resume, map_location=self.device, weights_only=True)
            new_model_dict = self.my_sam_model.state_dict()
            for k, v in new_model_dict.items():
                if k in checkpoint['state_dict'].keys() and checkpoint['state_dict'][k].size() == new_model_dict[k].size():
                    new_model_dict[k] = checkpoint['state_dict'][k].detach()
            self.my_sam_model.load_state_dict(new_model_dict)
            print(f"加载DetAny3D权重: {self.cfg.resume}")
        else:
            print(f"警告：未找到权重文件 {self.cfg.resume}")
        
        self.my_sam_model.setup()
        self.my_sam_model.to(self.device)
        self.my_sam_model.eval()
        
        # 初始化图像变换器
        self.sam_trans = ResizeLongestSide(self.cfg.model.pad)
        
        # 加载GroundingDINO模型
        dino_config = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
        dino_weights = "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth"
        
        if os.path.exists(dino_config) and os.path.exists(dino_weights):
            self.dino_model = load_model(dino_config, dino_weights)
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()
            print(f"加载GroundingDINO模型: {dino_weights}")
        else:
            print(f"警告：未找到GroundingDINO模型文件")
            self.dino_model = None
    
    def _preprocess_image(self, img):
        """预处理图像"""
        # 图像标准化
        sam_pixel_mean = torch.Tensor(self.cfg.dataset.pixel_mean).view(-1, 1, 1)
        sam_pixel_std = torch.Tensor(self.cfg.dataset.pixel_std).view(-1, 1, 1)
        x = (img - sam_pixel_mean) / sam_pixel_std
        
        # 填充到正方形
        h, w = x.shape[-2:]
        padh = self.cfg.model.pad - h
        padw = self.cfg.model.pad - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def _preprocess_dino(self, x):
        """为GroundingDINO预处理图像"""
        x = x / 255
        IMAGENET_DATASET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        IMAGENET_DATASET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        x = (x - IMAGENET_DATASET_MEAN) / IMAGENET_DATASET_STD
        return x
    
    def _crop_hw(self, img):
        """裁剪图像到合适尺寸"""
        if img.dim() == 4:
            img = img.squeeze(0)
        h, w = img.shape[1:3]
        assert max(h, w) % 112 == 0, "target_size must be divisible by 112"
        
        # 计算裁剪后尺寸
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14
        
        # 计算裁剪区域的中心
        center_h, center_w = h // 2, w // 2
        start_h = center_h - new_h // 2
        start_w = center_w - new_w // 2
        
        # 裁剪图像
        img_cropped = img[:, start_h:start_h + new_h, start_w:start_w + new_w]
        return img_cropped.unsqueeze(0)
    
    def _convert_image_for_dino(self, img):
        """转换图像格式用于GroundingDINO"""
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        image_source = Image.fromarray(img, 'RGB')
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed
    
    def _generate_colors(self, img, num_objects):
        """生成用于可视化的颜色"""
        pixels = np.array(img).reshape(-1, 3) / 255.0
        brightness = pixels.mean(axis=1)
        prob = brightness / brightness.sum()
        
        # 采样颜色
        num_samples = min(100, pixels.shape[0])
        sampled_indices = np.random.choice(pixels.shape[0], num_samples, p=prob, replace=False)
        sampled_colors = pixels[sampled_indices]
        
        # 按亮度排序
        sampled_colors = sorted(sampled_colors, key=lambda c: colorsys.rgb_to_hsv(*c)[2])
        
        # 增强亮度
        colors = []
        for i in range(num_objects):
            if i < len(sampled_colors):
                color = self._adjust_brightness(sampled_colors[i], factor=2.0, v_min=0.4)
                colors.append([min(255, c*255) for c in color])
            else:
                # 如果对象数量超过采样颜色数量，生成随机颜色
                colors.append([np.random.randint(0, 255) for _ in range(3)])
        
        return colors
    
    def _adjust_brightness(self, color, factor=1.5, v_min=0.3):
        """调整颜色亮度"""
        r, g, b = color
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        v = max(v, v_min) * factor
        v = min(v, 1.0)
        return colorsys.hsv_to_rgb(h, s, v)
    
    def _draw_text(self, im, text, pos, scale=0.4, color='auto', font=cv2.FONT_HERSHEY_SIMPLEX, 
                   bg_color=(0, 255, 255), blend=0.33, lineType=1):
        """在图像上绘制文本"""
        text = str(text)
        pos = [int(pos[0]), int(pos[1])]
        
        if color == 'auto':
            if bg_color is not None:
                color = (0, 0, 0) if ((bg_color[0] + bg_color[1] + bg_color[2])/3) > 127.5 else (255, 255, 255)
            else:
                color = (0, 0, 0)
        
        if bg_color is not None:
            text_size, _ = cv2.getTextSize(text, font, scale, lineType)
            x_s = int(np.clip(pos[0], a_min=0, a_max=im.shape[1]))
            x_e = int(np.clip(x_s + text_size[0] - 1 + 4, a_min=0, a_max=im.shape[1]))
            y_s = int(np.clip(pos[1] - text_size[1] - 2, a_min=0, a_max=im.shape[0]))
            y_e = int(np.clip(pos[1] + 1 - 2, a_min=0, a_max=im.shape[0]))
            
            im[y_s:y_e + 1, x_s:x_e + 1, 0] = im[y_s:y_e + 1, x_s:x_e + 1, 0]*blend + bg_color[0] * (1 - blend)
            im[y_s:y_e + 1, x_s:x_e + 1, 1] = im[y_s:y_e + 1, x_s:x_e + 1, 1]*blend + bg_color[1] * (1 - blend)
            im[y_s:y_e + 1, x_s:x_e + 1, 2] = im[y_s:y_e + 1, x_s:x_e + 1, 2]*blend + bg_color[2] * (1 - blend)
            
            pos[0] = int(np.clip(pos[0] + 2, a_min=0, a_max=im.shape[1]))
            pos[1] = int(np.clip(pos[1] - 2, a_min=0, a_max=im.shape[0]))
        
        cv2.putText(im, text, tuple(pos), font, scale, color, lineType)
    
    def predict(self, image_path=None, image_data=None, text_prompt="", point_coords=None, bbox_coords=None, 
                save_result=True, output_dir="./output"):
        """
        执行推理
        
        Args:
            image_path: 输入图像路径
            image_data: 输入图像数据 (numpy array, RGB format)
            text_prompt: 文本提示词，用于GroundingDINO
            point_coords: 点坐标 [[x, y], ...]
            bbox_coords: 边界框坐标 [[x1, y1, x2, y2], ...]
            save_result: 是否保存结果
            output_dir: 输出目录
        
        Returns:
            检测结果字典
        """
        # 读取图像
        image_name_source = None
        if image_data is not None:
            img = image_data
            # 为内存中的图像数据创建一个唯一的名称
            image_name_source = f"memory_image_{hashlib.md5(img.tobytes()).hexdigest()}"
        elif image_path:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"图像文件不存在: {image_path}")
            
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_name_source = image_path
        else:
            raise ValueError("Either image_path or image_data must be provided.")
        
        original_size = tuple(img.shape[:-1])
        
        print(f"处理图像: {os.path.basename(image_name_source)}, 尺寸: {img.shape}")
        
        # 初始化检测列表
        label_list = []
        bbox_2d_list = []
        point_coords_list = []
        
        # 处理输入提示
        if point_coords is not None:
            for coord in point_coords:
                point_coords_tensor = torch.tensor(coord, dtype=torch.int).unsqueeze(0)
                point_coords_list.append(point_coords_tensor)
            label_list = ["Unknown"] * len(point_coords)
            mode = 'point'
        else:
            mode = 'box'
        
        # 如果提供了边界框坐标
        if bbox_coords is not None:
            for bbox in bbox_coords:
                bbox_2d_list.append(bbox)
                label_list.append("Unknown")
        
        # 使用GroundingDINO进行目标检测
        if self.dino_model is not None and text_prompt:
            image_source_dino, image_dino = self._convert_image_for_dino(img)
            
            boxes, logits, phrases = dino_predict(
                model=self.dino_model,
                image=image_dino,
                caption=text_prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD,
                remove_combined=False,
            )
            
            h, w, _ = image_source_dino.shape
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
            
            for i, box in enumerate(xyxy):
                if mode == 'box':
                    bbox_2d_list.append(box.to(torch.int).cpu().numpy().tolist())
                    label_list.append(phrases[i])
        
        # 检查是否有有效的检测结果
        if len(bbox_2d_list) == 0 and len(point_coords_list) == 0:
            print("警告：未检测到任何目标")
            return {"error": "No objects detected"}
        
        # 图像预处理
        raw_img = img.copy()
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
        img = img.unsqueeze(0)
        
        # 应用变换
        img = self.sam_trans.apply_image_torch(img)
        after_resize_size = tuple(img.shape[2:])  # 记录resize后的尺寸
        img = self._crop_hw(img)
        before_pad_size = tuple(img.shape[2:])
        
        # 计算裁剪的偏移量
        crop_offset_h = (after_resize_size[0] - before_pad_size[0]) // 2
        crop_offset_w = (after_resize_size[1] - before_pad_size[1]) // 2
        
        img_for_sam = self._preprocess_image(img).to(self.device)
        img_for_dino = self._preprocess_dino(img).to(self.device)
        
        image_h, image_w = int(before_pad_size[0]), int(before_pad_size[1])
        
        # 计算vit_pad_size
        if self.cfg.model.vit_pad_mask:
            vit_pad_size = (before_pad_size[0] // self.cfg.model.image_encoder.patch_size, 
                           before_pad_size[1] // self.cfg.model.image_encoder.patch_size)
        else:
            vit_pad_size = (self.cfg.model.pad // self.cfg.model.image_encoder.patch_size, 
                           self.cfg.model.pad // self.cfg.model.image_encoder.patch_size)
        
        # 准备输入数据
        if mode == 'box':
            bbox_2d_tensor = torch.tensor(bbox_2d_list)
            bbox_2d_tensor = self.sam_trans.apply_boxes_torch(bbox_2d_tensor, original_size).to(torch.int).to(self.device)
            
            input_dict = {
                "images": img_for_sam,
                'vit_pad_size': torch.tensor(vit_pad_size).to(self.device).unsqueeze(0),
                "images_shape": torch.Tensor(before_pad_size).to(self.device).unsqueeze(0),
                "image_for_dino": img_for_dino,
                "boxes_coords": bbox_2d_tensor,
            }
        else:  # point mode
            points_2d_tensor = torch.stack(point_coords_list, dim=1).to(self.device)
            points_2d_tensor = self.sam_trans.apply_coords_torch(points_2d_tensor, original_size)
            
            input_dict = {
                "images": img_for_sam,
                'vit_pad_size': torch.tensor(vit_pad_size).to(self.device).unsqueeze(0),
                "images_shape": torch.Tensor(before_pad_size).to(self.device).unsqueeze(0),
                "image_for_dino": img_for_dino,
                "point_coords": points_2d_tensor,
            }
        
        # 执行推理
        with torch.no_grad():
            ret_dict = self.my_sam_model(input_dict)
        
        # 解码结果
        K = ret_dict['pred_K']
        decoded_bboxes_pred_2d, decoded_bboxes_pred_3d = decode_bboxes(ret_dict, self.cfg, K)
        rot_mat = rotation_6d_to_matrix(ret_dict['pred_pose_6d'])
        pred_box_ious = ret_dict.get('pred_box_ious', None)
        
        # 可视化结果
        visualization_path = None
        if save_result:
            visualization_path = self._visualize_results(
                raw_img, decoded_bboxes_pred_2d, decoded_bboxes_pred_3d, 
                rot_mat, K, label_list, pred_box_ious, 
                image_name_source, output_dir, original_size, after_resize_size, before_pad_size, 
                crop_offset_h, crop_offset_w
            )
        
        # 返回结果
        results = {
            "image_path": image_path if image_path else image_name_source,
            "num_objects": len(decoded_bboxes_pred_2d),
            "bboxes_2d": decoded_bboxes_pred_2d.detach().cpu().numpy().tolist(),
            "bboxes_3d": decoded_bboxes_pred_3d.detach().cpu().numpy().tolist(),
            "labels": label_list,
            "camera_matrix": K.detach().cpu().numpy().tolist(),
            "visualization_path": visualization_path,
        }
        
        if pred_box_ious is not None:
            results["iou_scores"] = pred_box_ious.detach().cpu().numpy().tolist()
        
        print(f"检测到 {len(decoded_bboxes_pred_2d)} 个目标")
        
        return results
    
    def _visualize_results(self, img, bboxes_2d, bboxes_3d, rot_mat, K, labels, 
                          pred_box_ious, image_path, output_dir, original_size, after_resize_size, 
                          before_pad_size, crop_offset_h, crop_offset_w):
        """可视化检测结果"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成颜色
        colors = self._generate_colors(img, len(bboxes_2d))
        
        # 创建可视化图像
        vis_img = img.copy()
        
        K_np = K.detach().cpu().numpy()
        
        # 计算变换参数
        # 1. 从处理后的尺寸到resize后的尺寸（补偿crop offset）
        # 2. 从resize后的尺寸到原图尺寸（使用SAM的逆变换）
        
        # 步骤1：从处理后到resize后的变换
        # 需要添加crop offset
        
        # 步骤2：从resize后到原图的缩放比例
        resize_scale_h = original_size[0] / after_resize_size[0]
        resize_scale_w = original_size[1] / after_resize_size[1]
        
        # 调整相机内参K矩阵
        K_original = K_np.copy()
        
        # 首先补偿crop offset
        K_original[0, 0, 2] += crop_offset_w  # cx
        K_original[0, 1, 2] += crop_offset_h  # cy
        
        # 然后应用resize缩放
        K_original[0, 0, 0] *= resize_scale_w  # fx
        K_original[0, 1, 1] *= resize_scale_h  # fy
        K_original[0, 0, 2] *= resize_scale_w  # cx
        K_original[0, 1, 2] *= resize_scale_h  # cy
        
        print(f"原图尺寸: {original_size}")
        print(f"resize后尺寸: {after_resize_size}")
        print(f"处理后尺寸: {before_pad_size}")
        print(f"crop偏移: ({crop_offset_h}, {crop_offset_w})")
        print(f"resize缩放比例: 宽度 {resize_scale_w:.3f}, 高度 {resize_scale_h:.3f}")
        
        for i in range(len(bboxes_2d)):
            x, y, z, w, h, l, yaw = bboxes_3d[i].detach().cpu().numpy()
            rot_mat_i = rot_mat[i].detach().cpu().numpy()
            
            # 计算3D边界框顶点
            vertices_3d, fore_plane_center_3d = compute_3d_bbox_vertices(x, y, z, w, h, l, yaw, rot_mat_i)
            # 使用调整后的K矩阵投影到原图尺寸
            vertices_2d = project_to_image(vertices_3d, K_original.squeeze(0))
            
            color = colors[i]
            color = (int(color[0]), int(color[1]), int(color[2]))
            
            # 绘制3D边界框
            draw_bbox_2d(vis_img, vertices_2d, color=color, thickness=3)
            
            # 绘制标签和尺寸信息
            if labels[i] is not None:
                label_text = f"{labels[i]}"
                # 将2D边界框也变换到原图尺寸
                bbox_2d_orig = bboxes_2d[i].detach().cpu().numpy().copy()
                
                # 补偿crop offset
                bbox_2d_orig[0] += crop_offset_w  # x
                bbox_2d_orig[1] += crop_offset_h  # y
                
                # 应用resize缩放
                bbox_2d_orig[0] *= resize_scale_w  # x
                bbox_2d_orig[1] *= resize_scale_h  # y
                bbox_2d_orig[2] *= resize_scale_w  # w
                bbox_2d_orig[3] *= resize_scale_h  # h
                
                bbox_2d_xyxy = box_cxcywh_to_xyxy(torch.tensor(bbox_2d_orig)).numpy().tolist()
                self._draw_text(vis_img, label_text, bbox_2d_xyxy, scale=0.5, bg_color=color)
        
        # 保存结果
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{image_name}_result.jpg")
        
        vis_img_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_img_bgr)
        
        print(f"结果已保存至: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description='DetAny3D推理脚本')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--text', type=str, default="", help='文本提示词')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--output', type=str, default="./output", help='输出目录')
    parser.add_argument('--device', type=str, default="cuda:0", help='推理设备')
    parser.add_argument('--points', type=str, default=None, help='点坐标，格式：x1,y1;x2,y2')
    parser.add_argument('--boxes', type=str, default=None, help='边界框坐标，格式：x1,y1,x2,y2;x3,y3,x4,y4')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = DetAny3DInference(config_path=args.config, device=args.device)
    
    # 解析点坐标
    point_coords = None
    if args.points:
        point_coords = []
        for point in args.points.split(';'):
            x, y = map(int, point.split(','))
            point_coords.append([x, y])
    
    # 解析边界框坐标
    bbox_coords = None
    if args.boxes:
        bbox_coords = []
        for box in args.boxes.split(';'):
            x1, y1, x2, y2 = map(int, box.split(','))
            bbox_coords.append([x1, y1, x2, y2])
    
    # 执行推理
    results = inferencer.predict(
        image_path=args.image,
        text_prompt=args.text,
        point_coords=point_coords,
        bbox_coords=bbox_coords,
        save_result=True,
        output_dir=args.output
    )
    
    # 打印结果
    print("\n推理结果:")
    print(f"检测到目标数量: {results.get('num_objects', 0)}")
    if 'labels' in results:
        print(f"检测到的标签: {results['labels']}")
    
    if 'error' not in results:
        print("推理完成！")
    else:
        print(f"推理失败: {results['error']}")


if __name__ == "__main__":
    main() 