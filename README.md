# Wave Breaking Classification with Knowledge Distillation

这是一个使用知识蒸馏技术的波浪破碎分类项目，采用DINOv2作为教师模型，自定义ViT+LSTM作为学生模型。

## 🌊 项目概述

本项目专门针对海洋波浪破碎现象的分类任务，使用知识蒸馏技术将大型DINOv2模型的知识传递给轻量级的CRNN网络。

### 特性

- **教师模型**: DINOv2 ViT-Base (86M参数)
- **学生模型**: 自定义ViT + LSTM CRNN网络 (~20M参数)
- **分类类别**: 3类波浪破碎类型
  - Plunging (卷破)
  - Spilling (溢破) 
  - Surging (涌破)
- **技术栈**: PyTorch Lightning + Hydra + WandB
- **知识蒸馏**: 结合logits蒸馏和特征蒸馏

## 📁 项目结构

```
wave_distillation/
├── configs/                 # Hydra配置文件
│   ├── config.yaml         # 主配置
│   ├── data/               # 数据配置
│   ├── model/              # 模型配置
│   ├── trainer/            # 训练器配置
│   └── logger/             # 日志配置
├── src/                    # 源代码
│   ├── data/               # 数据处理模块
│   ├── models/             # 模型定义
│   └── utils/              # 工具函数
├── main.py                 # 主运行脚本
├── requirements.txt        # 依赖文件
└── README.md              # 项目说明
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 克隆项目
git clone <your-repo-url>
cd wave_distillation

# 创建虚拟环境
conda create -n wave_distill python=3.9
conda activate wave_distill

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

将您的数据集按以下结构组织：

```
your_dataset/
├── train/
│   ├── plunging/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── spilling/
│   │   └── ...
│   └── surging/
│       └── ...
├── val/
│   ├── plunging/
│   ├── spilling/
│   └── surging/
└── test/ (可选)
    ├── plunging/
    ├── spilling/
    └── surging/
```

### 3. 配置修改

修改 `configs/data/wave_breaking_enhanced.yaml` 中的数据路径：

```yaml
data_dir: "/path/to/your/wave_dataset"  # 修改为实际路径
```

### 4. 开始训练

```bash
# 基础训练
python main.py

# 指定数据路径
python main.py data.data_dir="/path/to/your/dataset"

# 调整超参数
python main.py model.learning_rate=1e-4 data.batch_size=16

# 使用不同配置
python main.py data=wave_breaking model.temperature=4.0
```

## ⚙️ 配置说明

### 数据配置

- **wave_breaking.yaml**: 基础配置，使用标准数据增强
- **wave_breaking_enhanced.yaml**: 增强配置，包含波浪特定增强

### 模型配置

- **teacher**: DINOv2配置
- **student**: ViT+LSTM配置
- **loss_weights**: 损失权重配置

### 训练配置

- **max_epochs**: 最大训练轮数
- **callbacks**: 回调函数（检查点、早停等）
- **precision**: 混合精度训练

## 📊 监控和日志

项目集成了WandB进行实验追踪：

1. 注册WandB账号：https://wandb.ai
2. 登录：`wandb login`
3. 训练时会自动记录：
   - 损失曲线
   - 准确率指标
   - 混淆矩阵
   - 模型参数统计

## 🔧 自定义修改

### 添加新的数据格式

1. 在 `src/data/custom_dataset.py` 中添加新的加载方法
2. 创建对应的配置文件
3. 更新数据模块

### 修改模型架构

1. 编辑 `src/models/student_model.py` 中的网络结构
2. 调整 `configs/model/distillation.yaml` 中的参数
3. 重新训练

### 添加新的损失函数

1. 在 `src/models/distillation_model.py` 中定义新损失
2. 在训练步骤中集成
3. 添加对应的配置选项

## 📈 性能优化建议

### 数据加载优化

```bash
# 增加数据加载线程
python main.py data.num_workers=8

# 启用加权采样
python main.py data.use_weighted_sampling=true
```

### 训练优化

```bash
# 使用混合精度
python main.py trainer.precision="16-mixed"

# 梯度累积
python main.py trainer.accumulate_grad_batches=2

# 调整批次大小
python main.py data.batch_size=64
```

## 🐛 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   python main.py data.batch_size=16 trainer.precision="16-mixed"
   ```

2. **数据加载错误**
   - 检查数据路径是否正确
   - 确认图像格式支持
   - 验证文件夹结构

3. **模型加载失败**
   - 检查网络连接（DINOv2需要下载）
   - 确认transformers版本兼容性

### 调试模式

```bash
# 快速调试（少量数据）
python main.py trainer.fast_dev_run=true

# 详细日志
python main.py hydra.verbose=true
```

## 📝 引用

如果您使用了本项目，请引用相关论文：

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系

如有问题，请联系：[your-email@example.com]