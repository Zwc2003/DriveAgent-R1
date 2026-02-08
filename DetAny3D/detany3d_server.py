#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DetAny3D Flask服务器
提供3D目标检测的HTTP API服务
"""

import os
import sys
import argparse
import base64
import io
import json
import traceback
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import threading

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import DetAny3DInference

app = Flask(__name__)

# 全局变量存储模型
MODEL = None
MODEL_LOCK = threading.Lock()

def initialize_model(config_path=None, device='cuda:0'):
    """初始化DetAny3D模型"""
    global MODEL
    try:
        print("正在初始化DetAny3D模型...")
        MODEL = DetAny3DInference(config_path=config_path, device=device)
        print("DetAny3D模型初始化成功！")
        return True
    except Exception as e:
        print(f"DetAny3D模型初始化失败: {str(e)}")
        traceback.print_exc()
        return False

def decode_image_from_base64(image_base64):
    """从base64解码图像"""
    try:
        # 移除data:image/xxx;base64,前缀（如果存在）
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # 解码base64
        image_data = base64.b64decode(image_base64)
        
        # 转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 转换为numpy数组
        image_np = np.array(image)
        
        return image_np
    except Exception as e:
        raise ValueError(f"图像解码失败: {str(e)}")

def encode_image_to_base64(image_path):
    """将图像文件编码为base64"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        return image_base64
    except Exception as e:
        raise ValueError(f"图像编码失败: {str(e)}")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': MODEL is not None
    })

@app.route('/detect', methods=['POST'])
def detect_3d_objects():
    """3D目标检测接口"""
    if MODEL is None:
        return jsonify({
            'error': 'DetAny3D模型未初始化'
        }), 500
    
    try:
        # 解析请求数据
        data = request.json
        
        if 'image' not in data:
            return jsonify({
                'error': '请求中缺少图像数据'
            }), 400
        
        image_base64 = data['image']
        text_prompt = data.get('text_prompt', '')
        
        # 解码图像
        image_np = decode_image_from_base64(image_base64)
        
        # 使用线程锁确保模型调用安全
        with MODEL_LOCK:
            print(f"处理3D检测请求，图像尺寸: {image_np.shape}, 文本提示: {text_prompt}")
            
            # 调用模型进行检测
            results = MODEL.predict(
                image_data=image_np,
                text_prompt=text_prompt,
                save_result=True,
                output_dir="/tmp/detany3d_server_output"
            )
        
        # 处理结果
        if 'error' in results:
            return jsonify({
                'error': results['error']
            }), 500
        
        # 如果有可视化结果，将其编码为base64
        response_data = {
            'num_objects': results.get('num_objects', 0),
            'bboxes_2d': results.get('bboxes_2d', []),
            'bboxes_3d': results.get('bboxes_3d', []),
            'labels': results.get('labels', []),
            'camera_matrix': results.get('camera_matrix', []),
        }
        
        # 添加IoU分数（如果有）
        if 'iou_scores' in results:
            response_data['iou_scores'] = results['iou_scores']
        
        # 编码可视化图像
        if results.get('visualization_path') and os.path.exists(results['visualization_path']):
            try:
                visualization_base64 = encode_image_to_base64(results['visualization_path'])
                response_data['visualization_image'] = visualization_base64
            except Exception as e:
                print(f"可视化图像编码失败: {str(e)}")
        
        print(f"3D检测完成，检测到 {response_data['num_objects']} 个目标")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"3D检测请求处理失败: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': f'3D检测请求处理失败: {str(e)}'
        }), 500

@app.route('/model_info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    if MODEL is None:
        return jsonify({
            'error': 'DetAny3D模型未初始化'
        }), 500
    
    return jsonify({
        'model': 'DetAny3D',
        'device': MODEL.device,
        'config': {
            'box_threshold': MODEL.BOX_THRESHOLD,
            'text_threshold': MODEL.TEXT_THRESHOLD,
        }
    })

def main():
    parser = argparse.ArgumentParser(description='DetAny3D Flask服务器')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--device', type=str, default='cuda:0', help='推理设备')
    parser.add_argument('--host', type=str, default='localhost', help='服务器主机')
    parser.add_argument('--port', type=int, default=5000, help='服务器端口')
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    args = parser.parse_args()
    
    # 初始化模型
    if not initialize_model(config_path=args.config, device=args.device):
        print("模型初始化失败，退出...")
        sys.exit(1)
    
    print(f"启动DetAny3D服务器: {args.host}:{args.port}")
    
    # 启动Flask服务器
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True  # 支持多线程
    )

if __name__ == '__main__':
    main() 