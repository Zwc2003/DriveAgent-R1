import inspect
from typing import Any, Callable, Dict, List, Optional

# 工具注册表
_TOOL_REGISTRY = {}

def register_tool(func_or_name: Any = None, *, name: Optional[str] = None):
    """注册一个工具函数到工具注册表。
    
    可以作为装饰器使用：
        @register_tool
        def my_tool(...):
            ...
            
        @register_tool(name="custom_name")
        def my_tool(...):
            ...
            
    也可以直接调用：
        register_tool(my_tool)
        register_tool(my_tool, name="custom_name")
    """
    def decorator(func):
        tool_name = name if name is not None else func.__name__
        _TOOL_REGISTRY[tool_name] = func
        # 保存函数签名用于参数验证
        func._tool_signature = inspect.signature(func)
        return func
    
    # 处理不同的调用形式
    if callable(func_or_name):
        return decorator(func_or_name)
    else:
        return decorator

def get_tool(tool_name: str) -> Optional[Callable]:
    """根据工具名称获取工具函数"""
    return _TOOL_REGISTRY.get(tool_name)

def list_tools() -> List[str]:
    """列出所有已注册的工具"""
    return list(_TOOL_REGISTRY.keys())

def validate_tool_params(tool_name: str, params: Dict[str, Any]) -> Dict[str, str]:
    """验证工具参数是否符合工具函数签名
    
    Args:
        tool_name: 工具名称
        params: 工具参数字典
        
    Returns:
        如果验证失败，返回错误信息字典；否则返回空字典
    """
    errors = {}
    tool_func = get_tool(tool_name)
    
    if not tool_func:
        errors["tool"] = f"Tool '{tool_name}' not found"
        return errors
    
    sig = tool_func._tool_signature
    
    # 检查必需参数
    for param_name, param in sig.parameters.items():
        # 跳过 image_dict和calib_dict，因为是由 execute_tool_call 强制添加的
        if param_name == 'image_dict' or param_name=='calib_dict':
            continue 
            
        if param.default == inspect.Parameter.empty and param_name not in params:
            errors[param_name] = f"Missing required parameter '{param_name}'"
    
    # 检查未知参数 (由 LLM 提供的参数)
    for param_name in params:
        if param_name not in sig.parameters:
            errors[param_name] = f"Unknown parameter '{param_name}'"
    
    # 可以在这里添加类型检查等其他验证逻辑
    
    return errors 