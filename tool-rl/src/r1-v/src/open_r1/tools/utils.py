import re
import json
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
import inspect # 导入 inspect
import os
from io import BytesIO
import base64
import uuid
import requests
import time
import logging

from .tool_registry import get_tool, validate_tool_params
from .base_tools import ToolCallResult

logger = logging.getLogger(__name__)



def convert_image_to_base64(image: Image.Image) -> str:
    """将PIL Image转换为Base64编码的字符串"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def parse_tool_call(text: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    """
    解析文本中的工具调用标记，提取工具名称和参数
    
    Args:
        text: 包含工具调用的文本
        
    Returns:
        (tool_name, params, error_message) 三元组
        如果解析失败，将返回相应的错误信息
    """
    # 提取工具调用部分
    tool_call_match = re.search(r'<call_tool>\s*(.*?)\s*</call_tool>', text, re.DOTALL)
    if not tool_call_match:
        logger.debug("Tool call markers <call_tool>...</call_tool> not found")
        return None, None, "Tool call markers <call_tool>...</call_tool> not found"
    
    tool_call_content = tool_call_match.group(1)
    
    # 提取工具名称
    tool_name_match = re.search(r'<tool_name>\s*(.*?)\s*</tool_name>', tool_call_content, re.DOTALL)
    if not tool_name_match:
        logger.debug("Tool call must contain <tool_name> tag")
        return None, None, "Tool call must contain <tool_name> tag"
    
    tool_name = tool_name_match.group(1).strip()
    
    # 提取参数
    params_match = re.search(r'<params>\s*(.*?)\s*</params>', tool_call_content, re.DOTALL)
    if not params_match:
        logger.debug("Tool call must contain <params> tag")
        return None, None, "Tool call must contain <params> tag"
    
    params_text = params_match.group(1).strip()
    
    # 尝试解析JSON参数
    try:
        params = json.loads(params_text)
        if not isinstance(params, dict):
            logger.debug("Parameters must be a valid JSON object")
            return None, None, "Parameters must be a valid JSON object"
    except json.JSONDecodeError as e:
        logger.debug(f"Parameter parsing error: {str(e)}")
        return None, None, f"Parameter parsing error: {str(e)}"
    
    return tool_name, params, None

def execute_tool_call(
    text: str, 
    images_dict: Dict[str, Image.Image],
    vid: Optional[str] = None,
) -> Tuple[Optional[ToolCallResult], float, bool]:
    """
    执行文本中的工具调用
    
    Args:
        text: 包含工具调用的文本
        images_dict: 可用图像字典
        calib_dict: 校准字典
        
    Returns:
        (tool_result, tool_reward) 二元组
        tool_result: 工具调用结果或None（如果解析失败）
        tool_reward: 工具调用奖励分数 (0.0-1.0)
    """
    reward = 0.0
    
    # 解析工具调用
    tool_name, params, error = parse_tool_call(text)
    logger.debug(f"Parsing tool call: tool_name={tool_name}, params={params}")
    
    # 解析失败
    if error:
        return ToolCallResult(error=error, tool_name="unknown"), reward
    
    # 格式解析正确，奖励+0.2
    reward += 0.2
    
    # 查找工具
    tool_func = get_tool(tool_name)
    if not tool_func:
        return ToolCallResult(
            error=f"Unknown tool: '{tool_name}'", 
            tool_name=tool_name
        ), reward
    
    # 找到工具，奖励+0.2
    reward += 0.2
    
    # 验证参数
    param_errors = validate_tool_params(tool_name, params)
    if param_errors:
        logger.debug(f"Parameter validation failed for tool '{tool_name}': {param_errors}")
        error_msg = "Parameter error: " + ", ".join(f"{k}: {v}" for k, v in param_errors.items())
        return ToolCallResult(
            error=error_msg, 
            tool_name=tool_name
        ), reward
    
    # 参数验证通过，奖励+0.2
    reward += 0.2
    
            
    # 执行工具调用
    try:
        # 检查工具函数
        tool_sig = inspect.signature(tool_func)
        tool_kwargs = {"image_dict": images_dict}
        # 透传样本 vid，便于工具按需加载高分辨率图像
        if vid is not None and 'vid' in tool_sig.parameters:
            tool_kwargs['vid'] = vid


        # 使用解包传递参数
        result = tool_func(**params, **tool_kwargs)
        
        # 工具成功执行，奖励+0.4
        if not result.error:
            reward += 0.4
            
        return result, reward
    except Exception as e:
        logger.error(f"Tool execution failed for '{tool_name}': {str(e)}", exc_info=True)
        return ToolCallResult(
            error=f"Tool execution failed: {str(e)}", 
            tool_name=tool_name
        ), reward
