import os
import torch
import random
import numpy as np
import logging
from pathlib import Path
import yaml
from omegaconf import DictConfig, OmegaConf


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def save_config(cfg: DictConfig, save_dir: str):
    """保存配置文件"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = save_dir / "config.yaml"
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)


def load_checkpoint(model, checkpoint_path: str):
    """加载模型检查点"""
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")


def count_parameters(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device():
    """获取设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_output_dir(base_dir: str = "outputs"):
    """创建输出目录"""
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return str(output_dir)


def print_model_summary(model, input_size=(3, 224, 224)):
    """打印模型摘要"""
    try:
        from torchsummary import summary
        summary(model, input_size)
    except ImportError:
        print("torchsummary not installed. Skipping model summary.")


def calculate_model_size(model):
    """计算模型大小（MB）"""
    param_size = 0
    param_sum = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return all_size, param_sum, buffer_sum