from typing import Dict, Any, Optional, List, Union, Tuple
from PIL import Image
import json
import numpy as np
import cv2
import requests
import base64
import io
import os
import logging
from .tool_registry import register_tool
import threading

logger = logging.getLogger(__name__)
# 全局变量，用于存储处理器
PROCESSOR = None

def get_processor():
    """加载并缓存Qwen-VL的处理器"""
    global PROCESSOR
    if PROCESSOR is None:
        try:
            from transformers import AutoProcessor
            # 这里的模型ID需要与您训练时使用的模型保持一致
            model_id = "model_path"
            logger.info(f"Loading processor for the first time, model ID: {model_id}")
            PROCESSOR = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, max_pixels = 259200, min_pixels = 6272)
            logger.info("Processor loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load processor: {e}", exc_info=True)
            PROCESSOR = None # 确保失败时不会缓存
    return PROCESSOR

def get_model_input_size_from_processor(image: Image.Image) -> Optional[Tuple[int, int]]:
    """使用已加载的 processor 计算该图像在模型中的实际输入尺寸(像素)。

    返回 (input_width, input_height)。失败时返回 None。
    """
    try:
        processor = get_processor()
        if processor is None:
            logger.warning("[RoI_Inspection] Processor not available; cannot compute dynamic input size.")
            return None

        # 构造最小的对话模板，仅用于让 processor 生成包含 image_grid_thw 的 inputs
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": [
                {"type": "text", "text": "describe"},
                {"image": "<image>"}
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")

        thw = inputs.get("image_grid_thw", None)
        if thw is None:
            logger.warning("[RoI_Inspection] 'image_grid_thw' not found in processor outputs.")
            return None

        # thw: [T, H, W] 的网格数，像素需乘以 14
        input_height = int(thw[0][1].item() * 14)
        input_width = int(thw[0][2].item() * 14)
        return input_width, input_height
    except Exception as e:
        logger.warning(f"[RoI_Inspection] Failed to compute input size via processor: {e}")
        return None

# 全局变量，用于存储已加载的深度估计模型
DEPTH_MODEL = None
DEPTH_MODEL_DEVICE = None

# 全局变量，用于存储已加载的DetAny3D模型
DETANY3D_MODEL = None
# 为DetAny3D模型创建一个线程锁，以确保在多线程环境下的调用安全
DETANY3D_LOCK = threading.Lock()

# 预加载深度估计模型
try:
    import sys
    import os
    import torch
    import matplotlib
    
    # 设置路径
    depth_model_path = "tool-rl/Depth-Anything-V2"
    sys.path.append(depth_model_path)
    
    # 导入模型
    from .depth_anything_v2.dpt import DepthAnythingV2
    
    # 选择设备
    if torch.cuda.is_available():
        # 在分布式训练中，每个进程都应该使用自己的GPU
        local_rank = int(os.environ.get("LOCAL_RANK", "1"))
        DEPTH_MODEL_DEVICE = f'cuda:{local_rank}'
    else:
        DEPTH_MODEL_DEVICE = 'cpu'
    
    # 选择模型配置
    encoder = 'vitl'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 加载模型
    logger.info("Preloading depth estimation model...")
    DEPTH_MODEL = DepthAnythingV2(**model_configs[encoder])
    
    checkpoint_path = os.path.join(depth_model_path, f"checkpoints/depth_anything_v2_{encoder}.pth")
    
    DEPTH_MODEL.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    DEPTH_MODEL = DEPTH_MODEL.to(DEPTH_MODEL_DEVICE).eval()
    logger.info(f"Depth estimation model loaded successfully on device: {DEPTH_MODEL_DEVICE}")
    
except Exception as e:
    logger.error("Failed to load depth estimation model", exc_info=True)
    DEPTH_MODEL = None
    DEPTH_MODEL_DEVICE = None


# DetAny3D服务器配置
DETANY3D_SERVER_URL = "http://localhost:5000"  # 可以通过环境变量配置

# 高分辨率图像根目录（可通过环境变量 HIGHRES_IMAGE_ROOT 覆盖）。
# 若未设置，将使用一个占位路径，并在找不到文件时回退到 image_dict 内的图像。
HIGHRES_IMAGE_ROOT = os.environ.get("HIGHRES_IMAGE_ROOT", "path-to-highres_images")

class ToolCallResult:
    """工具调用结果类，包含文本或图像输出"""
    def __init__(
        self, 
        text_output: Optional[str] = None, 
        image_output: Optional[Image.Image] = None,
        error: Optional[str] = None,
        tool_name: str = ""
    ):
        self.text_output = text_output
        self.image_output = image_output
        self.error = error
        self.tool_name = tool_name
    
    @property
    def has_image(self) -> bool:
        """是否包含图像输出"""
        return self.image_output is not None

@register_tool
def get_view_with_frame(
    frame_index: str,
    view_index: str,
    image_dict: Dict[str, Image.Image]
) -> ToolCallResult:
    """
--------------------------------------------------
    Tool name: get_view_with_frame

    Use cases:
    - Need to examine the environment from specific perspective and time frame
    
    Available parameters:
    frame_index: "1s_ago", "2s_ago", "3s_ago", "4s_ago", "5s_ago", "current"
    view_index: "front_left", "front_right", "back", "back_left", "back_right" (front is not recommanded)
    
    Example usage:

    <call_tool>
        <tool_name>get_view_with_frame</tool_name>
        <params>
            {"frame_index": "current", "view_index": "front_left"}
        </params>
    </call_tool>
--------------------------------------------------
    """
    # 构建图像引用键
    image_ref = f"{frame_index}_{view_index}"
    
    # 向后兼容，处理不带时间戳的旧格式
    if frame_index == "current" and view_index in image_dict:
        image_ref = view_index
    
    # 参数验证
    if image_ref not in image_dict:
        return ToolCallResult(
            error=f"image '{image_ref}' does not exist",
            tool_name="get_view_with_frame"
        )
    
    try:
        # 获取原始图像
        view_image = image_dict[image_ref]
        
        return ToolCallResult(
            text_output=f"get '{image_ref}' sucessfully!",
            image_output=view_image,
            tool_name="get_view_with_frame"
        )
    except Exception as e:
        return ToolCallResult(
            error=f"error: {str(e)}",
            tool_name="get_view_with_frame"
        )

@register_tool
def RoI_Inspection(            
    view_index: str,
    bbox: List[int],   
    description: str,
    image_dict: Dict[str, Image.Image],
    vid: Optional[str] = None,
) -> ToolCallResult:
    """
    Tool name: RoI_Inspection

    Purpose: Zoom in on specific objects or areas for detailed, high-resolution inspection
    When to use:
    - Need to examine small objects (traffic signs, license plates)
    - Want to focus on distant objects
    - Need high-resolution regional probing for finer details

    Parameters
    ----------
    view_index: "front", "front_left", "front_right", "back", "back_left", "back_right"
    bbox : List[int]
        Bounding box absolute coordinates as [x0, y0, x1, y1] where:
        - (x0, y0): upper-left corner
        - (x1, y1): lower-right corner
    description : short description of the interested area.

    Example usage
    ------------
    <call_tool>
        <tool_name>RoI_Inspection</tool_name>
        <params>
            {"view_index": "front", "bbox": [x0, y0, x1, y1], "description": "the traffic light in front"}
        </params>
    </call_tool>
    """

    # 优先尝试使用高分辨率图像（当 vid 提供且路径存在时）
    use_highres = False
    highres_image = None
    highres_path = None
    if vid is not None:
        try:
            parts = vid.split("_")
            if len(parts) >= 2:
                pre, back = parts[0], parts[1]
                candidate_dir = os.path.join(HIGHRES_IMAGE_ROOT, pre)
                candidate_name = f"{back}_{view_index}.jpg"
                candidate_path = os.path.join(candidate_dir, candidate_name)
                if os.path.exists(candidate_path):
                    highres_path = candidate_path
                    highres_image = Image.open(candidate_path).convert('RGB')
                    use_highres = True
                else:
                    logger.debug(f"[RoI_Inspection] High-resolution image not found: {candidate_path}, falling back to image_dict.")
            else:
                logger.debug(f"[RoI_Inspection] vid format unexpected (should contain underscore), received: {vid}, falling back to image_dict.")
        except Exception as _e:
            logger.debug(f"[RoI_Inspection] Error loading high-resolution image: {str(_e)}, falling back to image_dict.")

    # 回退：从 image_dict 获取图像
    if not use_highres:
        if view_index not in image_dict:
            return ToolCallResult(
                error=f"Image '{view_index}' does not exist and no highres image available (vid={vid}).",
                tool_name="RoI_Inspection"
            )

    # 验证bbox格式
    if not isinstance(bbox, list) or len(bbox) != 4:
        return ToolCallResult(
            error="bbox must be a list of 4 integers [x0, y0, x1, y1]",
            tool_name="RoI_Inspection"
        )

    # 先基于 processor 动态计算模型输入尺寸（模型输入坐标系）并校验 bbox
    try:
        _x0_in, _y0_in, _x1_in, _y1_in = bbox
        if not all(isinstance(v, (int, float)) for v in bbox):
            return ToolCallResult(
                error="All bbox coordinates must be numbers",
                tool_name="RoI_Inspection"
            )

        # 选择用于计算输入尺寸的基图像：优先使用 image_dict 的视角图
        base_image_for_input = image_dict.get(view_index, None)


        input_size = None
        if base_image_for_input is not None:
            input_size = get_model_input_size_from_processor(base_image_for_input)
        if input_size is None:
            return ToolCallResult(
                error=(
                    "Failed to compute dynamic input size via processor. "
                    f"Debug: base_image_for_input is {'set' if base_image_for_input is not None else 'None'}. "
                    "Please ensure processor is loadable and image is valid."
                ),
                tool_name="RoI_Inspection"
            )

        _input_w, _input_h = input_size
        if (
            _x0_in < 0 or _y0_in < 0 or
            _x1_in > _input_w or _y1_in > _input_h or
            _x0_in >= _x1_in or _y0_in >= _y1_in
        ):
            return ToolCallResult(
                error=(
                    f"Initial bbox out of bounds for input size ({_input_w}x{_input_h}). "
                    f"Received: ({_x0_in}, {_y0_in}, {_x1_in}, {_y1_in}). "
                    f"Expected 0 <= x0 < x1 <= {_input_w} and 0 <= y0 < y1 <= {_input_h}."
                ),
                tool_name="RoI_Inspection"
            )
    except Exception as _e:
        return ToolCallResult(
            error=f"Error validating initial bbox against dynamic input size: {str(_e)}",
            tool_name="RoI_Inspection"
        )

    try:
        x1, y1, x2, y2 = bbox

        # 选择图像源，并基于动态输入尺寸缩放到原图尺寸
        if use_highres and highres_image is not None:
            image = highres_image
            original_width, original_height = image.size
            scale_x = original_width / float(_input_w)
            scale_y = original_height / float(_input_h)
            x1_abs = int(x1 * scale_x)
            y1_abs = int(y1 * scale_y)
            x2_abs = int(x2 * scale_x)
            y2_abs = int(y2 * scale_y)
            # 按实际尺寸边界裁剪
            x1_abs = max(0, min(x1_abs, original_width - 1))
            y1_abs = max(0, min(y1_abs, original_height - 1))
            x2_abs = max(1, min(x2_abs, original_width))
            y2_abs = max(1, min(y2_abs, original_height))
        else:
            image = image_dict[view_index]
            original_width, original_height = image.size
            # 坐标缩放到原图尺寸
            x1_abs = int(x1 / float(_input_w) * original_width)
            y1_abs = int(y1 / float(_input_h) * original_height)
            x2_abs = int(x2 / float(_input_w) * original_width)
            y2_abs = int(y2 / float(_input_h) * original_height)

        # 4) 验证缩放后的坐标
        x1, y1, x2, y2 = x1_abs, y1_abs, x2_abs, y2_abs

        # 验证坐标
        if not all(isinstance(v, (int, float)) for v in bbox):
            return ToolCallResult(
                error="All bbox coordinates must be numbers",
                tool_name="RoI_Inspection"
            )
        if x1 < 0 or y1 < 0 or x2 > original_width or y2 > original_height:
            return ToolCallResult(
                error=(
                    f"Coordinates out of bounds. Image size: ({original_width}, {original_height}); "
                    f"received after scaling: ({x1}, {y1}, {x2}, {y2})"
                ),
                tool_name="RoI_Inspection"
            )
        if x1 >= x2 or y1 >= y2:
            return ToolCallResult(
                error="x0 must be < x1 and y0 must be < y1",
                tool_name="RoI_Inspection"
            )

        # 最小尺寸限制：缩放到原图坐标系后的目标框，两边都必须 > 28
        min_side = 28
        _proj_w, _proj_h = (x2 - x1), (y2 - y1)
        if _proj_w <= min_side or _proj_h <= min_side:
            return ToolCallResult(
                error=(
                    f"Cropped region too small after scaling to original image size: "
                    f"{_proj_w}x{_proj_h}. Both width and height must be > {min_side}. "
                    f"Initial bbox (in {_input_w}x{_input_h}): {bbox}; original image size: ({original_width}, {original_height})."
                ),
                tool_name="RoI_Inspection"
            )

        # 5) 执行裁剪
        cropped_image = image.crop((x1, y1, x2, y2))
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 6) 等比例缩放到 实际输入尺寸的一半
        min_width = _input_w // 2
        min_height = _input_h // 2
        target_width = max(min_width, crop_width)
        target_height = max(min_height, crop_height)
        
        # 计算缩放比例，选择使得宽度或高度先达到目标的比例
        scale_width = target_width / crop_width
        scale_height = target_height / crop_height
        scale_factor = min(scale_width, scale_height)
        
        # 计算缩放后的尺寸
        new_width = int(crop_width * scale_factor)
        new_height = int(crop_height * scale_factor)
        
        # 使用 LANCZOS 算法进行高质量缩放
        cropped_image = cropped_image.resize((new_width, new_height), Image.LANCZOS)
        
        description_text = (
            f"Cropped area from image '{view_index}'"
            + (f" using highres '{highres_path}'" if use_highres and highres_path else " from image_dict")
            + ": "
            f"box(({x1}, {y1}) – ({x2}, {y2})), "
            f"scaled from {crop_width}x{crop_height} "
            f"to {new_width}x{new_height}"
        )

        return ToolCallResult(
            text_output=description_text,
            image_output=cropped_image,
            tool_name="RoI_Inspection"
        )

    except Exception as e:
        return ToolCallResult(
            error=f"Error occurred while cropping image: {str(e)}",
            tool_name="RoI_Inspection"
        )



def encode_image_to_base64(image: Image.Image) -> str:
    """将PIL图像编码为base64字符串"""
    import io
    import base64
    
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')
    buffer.seek(0)
    
    image_bytes = buffer.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    return image_base64

def decode_image_from_base64(image_base64: str) -> Image.Image:
    """从base64字符串解码为PIL图像"""
    import io
    import base64
    
    # 移除data:image/xxx;base64,前缀（如果存在）
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # 解码base64
    image_data = base64.b64decode(image_base64)
    
    # 转换为PIL图像
    image = Image.open(io.BytesIO(image_data))
    
    return image

@register_tool
def detect_3d_objects(
    view_index: str,
    image_dict: Dict[str, Image.Image],
    object_text: Optional[str] = None,
) -> ToolCallResult:
    """
--------------------------------------------------
    Tool name: detect_3d_objects

    Performs 3D object detection on images from a specified view.

    When visualization of object position and orientation in 3D space is required.

        Usage Example:

        <call_tool>
            <tool_name>detect_3d_objects</tool_name>
            <params>
                {"view_index": "front", "object_text": "construction cone . barrier . road work sign"}
            </params>
        </call_tool>

        Args:
            view_index: The image reference ID (e.g., "front")
            object_text: Optional additional objects to detect (e.g., "construction cone . barrier"). 
                        These will be added to the default driving scenario objects.
        Default detected objects: car, truck, bus, motorcycle, bicycle, person, pedestrian, traffic sign
--------------------------------------------------
        """
    tool_name = "detect_3d_objects"

    # 1. 参数验证
    if view_index not in image_dict:
        return ToolCallResult(error=f"Image '{view_index}' does not exist", tool_name=tool_name)

    image = image_dict[view_index]

    try:
        # 2. 确保图像是RGB格式
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # 3. 将图像编码为base64
        image_base64 = encode_image_to_base64(image)

        # 4. 构建检测文本提示
        # 默认的驾驶场景对象
        default_objects = "car . truck . bus . motorcycle . bicycle . person . pedestrian . traffic sign"
        
        # 如果用户指定了额外的对象，则添加到默认对象中
        if object_text and object_text.strip():
            text_prompt = f"{default_objects} . {object_text.strip()}"
        else:
            text_prompt = default_objects
        
        # 5. 准备HTTP请求数据
        request_data = {
            'image': image_base64,
            'text_prompt': text_prompt
        }
        
        # 6. 发送HTTP请求到DetAny3D服务器
        server_url = os.environ.get('DETANY3D_SERVER_URL', DETANY3D_SERVER_URL)
        detect_url = f"{server_url}/detect"
        
        import requests
        try:
            response = requests.post(
                detect_url,
                json=request_data,
                timeout=60,  # 60秒超时
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                error_msg = f"DetAny3D server returned status {response.status_code}"
                try:
                    error_data = response.json()
                    if 'error' in error_data:
                        error_msg += f": {error_data['error']}"
                except:
                    error_msg += f": {response.text}"
                
                return ToolCallResult(
                    error=error_msg,
                    tool_name=tool_name
                )
            
            results = response.json()
            
        except requests.exceptions.ConnectionError:
            return ToolCallResult(
                error=f"Cannot connect to DetAny3D server at {server_url}. Please make sure the server is running.",
                tool_name=tool_name
            )
        except requests.exceptions.Timeout:
            return ToolCallResult(
                error="DetAny3D server request timed out (60s). The model might be processing a large request.",
                tool_name=tool_name
            )
        except requests.exceptions.RequestException as e:
            return ToolCallResult(
                error=f"HTTP request failed: {str(e)}",
                tool_name=tool_name
            )
        
        # 7. 处理检测结果
        if 'error' in results:
            return ToolCallResult(
                error=f"3D detection failed: {results['error']}",
                tool_name=tool_name
            )
        
        num_objects = results.get('num_objects', 0)
        bboxes_3d = results.get('bboxes_3d', [])
        labels = results.get('labels', [])
        camera_matrix = results.get('camera_matrix', None)
        
        # 8. 生成检测结果文本
        if num_objects == 0:
            if object_text and object_text.strip():
                output_text = f"No 3D objects detected in image '{view_index}' (searched for: {text_prompt})."
            else:
                output_text = f"No 3D objects detected in image '{view_index}' (searched for default driving objects)."
        else:
            if object_text and object_text.strip():
                output_text = f"Detected {num_objects} 3D objects in image '{view_index}' (searched for: {text_prompt}):\n"
            else:
                output_text = f"Detected {num_objects} 3D objects in image '{view_index}' (searched for default driving objects):\n"
                
            for i in range(num_objects):
                if i < len(bboxes_3d) and i < len(labels):
                    bbox_3d = bboxes_3d[i]
                    label = labels[i]
                    
                    # 提取3D边界框信息: [x, y, z, w, h, l, yaw]
                    x, y, z, w, h, l, yaw = bbox_3d
                    
                    output_text += f"  {i+1}. Class: {label}, " \
                                 f"Location: ({x:.2f}, {y:.2f}, {z:.2f}), " \
                                 f"Dimensions: {w:.2f}x{h:.2f}x{l:.2f}, " \
                                 f"Orientation (Y-axis rotation): {yaw:.2f} rad\n"
        
        # 9. 处理可视化结果图像
        vis_image = None
        if 'visualization_image' in results:
            try:
                vis_image = decode_image_from_base64(results['visualization_image'])
                output_text += "\nThe returned image includes 3D object detection results with bounding boxes and labels."
            except Exception as e:
                logger.warning(f"Error decoding visualization image: {str(e)}")
        
        return ToolCallResult(
            text_output=output_text.strip(),
            image_output=vis_image,
            tool_name=tool_name
        )
        
    except Exception as e:
        logger.error("Error occurred during 3D object detection", exc_info=True)
        return ToolCallResult(
            error=f"Error occurred during 3D object detection: {str(e)}",
            tool_name=tool_name
        )

@register_tool
def depth_estimation(
    view_index: str,
    image_dict: Dict[str, Image.Image]
) -> ToolCallResult:
    """
--------------------------------------------------
    Tool name: depth_estimation

    Generate a depth map of the image to help you assess the distance and spatial relationships of objects in the scene. The darker the blue, the farther away; the brighter the red, the closer.
    
    Usage example:

    <call_tool>
        <tool_name>depth_estimation</tool_name>
        <params>
            {"view_index": "front"}
        </params>
    </call_tool>
--------------------------------------------------
    """
    # Parameter validation
    if view_index not in image_dict:
        return ToolCallResult(
            error=f"Image '{view_index}' does not exist",
            tool_name="depth_estimation"
        )
    
    # Check if the model is loaded successfully
    if DEPTH_MODEL is None:
        return ToolCallResult(
            error="Depth estimation model is not properly loaded, cannot perform depth estimation",
            tool_name="depth_estimation"
        )
    
    try:
        # Get original image and convert to OpenCV format
        input_image = image_dict[view_index]
        raw_image = np.array(input_image)
        if raw_image.shape[2] == 4:  # If there's an Alpha channel, remove it
            raw_image = raw_image[:, :, :3]
        # PIL is RGB, convert to BGR
        raw_image = raw_image[:, :, ::-1].copy()
        
        try:
            import torch
            import matplotlib
            
            
            # 获取模型的实际设备
            device = next(DEPTH_MODEL.parameters()).device
            
            # 自定义image2tensor函数，确保使用与模型相同的设备
            def custom_image2tensor(raw_image, input_size=518, target_device=device):
                from .depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
                from torchvision.transforms import Compose
                
                transform = Compose([
                    Resize(
                        width=input_size,
                        height=input_size,
                        resize_target=False,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=14,
                        resize_method='lower_bound',
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    PrepareForNet(),
                ])
                
                h, w = raw_image.shape[:2]
                
                image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
                
                image = transform({'image': image})['image']
                image = torch.from_numpy(image).unsqueeze(0)
                
                # 明确使用模型的设备
                image = image.to(target_device)
                
                return image, (h, w)
            
            # 使用自定义的image2tensor函数
            with torch.no_grad():
                # 使用指定的输入大小
                input_size = 518
                
                # 使用自定义的image2tensor函数
                image, (h, w) = custom_image2tensor(raw_image, input_size, device)
                
                # 安全检查：确保模型和输入数据在同一设备上
                model_device = next(DEPTH_MODEL.parameters()).device
                input_device = image.device
                
                if model_device != input_device:
                    logger.debug(f"Model and input data on different devices (model: {model_device}, input: {input_device}), correcting...")
                    image = image.to(model_device)
                
                # 前向传播
                depth = DEPTH_MODEL.forward(image)
                
                # 插值到原始尺寸
                depth = torch.nn.functional.interpolate(
                    depth[:, None], (h, w), mode="bilinear", align_corners=True
                )[0, 0]
                
                # 转换为numpy数组
                depth = depth.cpu().numpy()
            
            # Process depth image
            # Invert depth map so that smaller values (closer objects) map to brighter colors
            depth = 1.0 - (depth - depth.min()) / (depth.max() - depth.min())
            depth = (depth * 255.0).astype(np.uint8)
            
            # Use color mapping
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_colored = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            # Convert back to PIL format
            result_image = Image.fromarray(depth_colored)
            
            if result_image is None:
                return ToolCallResult(
                    error="Depth estimation resulted in a None image.",
                    tool_name="depth_estimation"
                )
            return ToolCallResult(
                text_output=f"Successfully generated depth map for image '{view_index}'. Brighter areas represent closer objects, darker areas represent more distant objects.",
                image_output=result_image,
                tool_name="depth_estimation"
            )
        except Exception as e:
            return ToolCallResult(
                error=f"Depth estimation inference failed: {str(e)}",
                tool_name="depth_estimation"
            )
    except Exception as e:
        logger.error("Error occurred during depth estimation", exc_info=True)
        return ToolCallResult(
            error=f"Error occurred during depth estimation: {str(e)}",
            tool_name="depth_estimation"
        ) 

