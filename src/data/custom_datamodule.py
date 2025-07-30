import lightning as L
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from typing import Optional, List
import os
import logging

from .custom_dataset import CustomImageDataset

logger = logging.getLogger(__name__)


class CustomDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            data_format: str = "folder",
            annotation_file: Optional[str] = None,
            class_names: Optional[List[str]] = None,
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: int = 224,
            augmentation: dict = None,
            use_weighted_sampling: bool = False,
            val_split: float = 0.2,
            test_split: float = 0.1,
            nested_folders: bool = True,  # 新增参数
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.data_format = data_format
        self.annotation_file = annotation_file
        self.class_names = class_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augmentation = augmentation or {}
        self.use_weighted_sampling = use_weighted_sampling
        self.val_split = val_split
        self.test_split = test_split
        self.nested_folders = nested_folders  # 支持嵌套文件夹

        # 构建数据变换
        self.train_transform = self._build_transform(self.augmentation.get('train', []))
        self.val_transform = self._build_transform(self.augmentation.get('val', []))

        # 将在setup中设置
        self.num_classes = None
        self.class_weights = None

    def _build_transform(self, aug_config):
        """构建数据变换pipeline"""
        transform_list = []

        # 从配置构建变换
        for aug in aug_config:
            if isinstance(aug, dict):
                for name, params in aug.items():
                    try:
                        transform_class = getattr(transforms, name)
                        if params:
                            transform_list.append(transform_class(**params))
                        else:
                            transform_list.append(transform_class())
                    except AttributeError:
                        logger.warning(f"Transform {name} not found in torchvision.transforms")

        # 总是添加标准化
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return transforms.Compose(transform_list)

    def prepare_data(self):
        """数据准备阶段"""
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found")

        if self.data_format in ['csv', 'json'] and self.annotation_file:
            if not os.path.exists(self.annotation_file):
                raise FileNotFoundError(f"Annotation file {self.annotation_file} not found")

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""

        # 确定类别数量
        try:
            temp_dataset = CustomImageDataset(
                data_dir=self.data_dir,
                mode="train",
                data_format=self.data_format,
                annotation_file=self.annotation_file,
                class_names=self.class_names,
                nested_folders=self.nested_folders  # 传递嵌套参数
            )
            self.num_classes = temp_dataset.num_classes
            self.classes = temp_dataset.classes

            # 输出数据集信息
            sample_info = temp_dataset.get_sample_info()
            logger.info(f"Dataset info: {sample_info}")

        except Exception as e:
            logger.error(f"Error loading training dataset: {e}")
            if self.class_names:
                self.classes = self.class_names
                self.num_classes = len(self.class_names)
            else:
                raise ValueError("Cannot determine number of classes")

        if stage == "fit" or stage is None:
            # 加载训练集
            self.train_dataset = CustomImageDataset(
                data_dir=self.data_dir,
                mode="train",
                transform=self.train_transform,
                data_format=self.data_format,
                annotation_file=self.annotation_file,
                class_names=self.class_names,
                nested_folders=self.nested_folders
            )

            # 尝试加载验证集
            try:
                self.val_dataset = CustomImageDataset(
                    data_dir=self.data_dir,
                    mode="val",
                    transform=self.val_transform,
                    data_format=self.data_format,
                    annotation_file=self.annotation_file,
                    class_names=self.class_names,
                    nested_folders=self.nested_folders
                )
            except Exception as e:
                logger.info(f"No separate validation set found ({e}), splitting from training set")
                self._split_train_val()

            # 计算类别权重
            if self.use_weighted_sampling:
                self.class_weights = self.train_dataset.get_class_weights()

        if stage == "test" or stage is None:
            try:
                self.test_dataset = CustomImageDataset(
                    data_dir=self.data_dir,
                    mode="test",
                    transform=self.val_transform,
                    data_format=self.data_format,
                    annotation_file=self.annotation_file,
                    class_names=self.class_names,
                    nested_folders=self.nested_folders
                )
            except Exception as e:
                logger.info(f"No separate test set found ({e}), using validation set for testing")
                if hasattr(self, 'val_dataset'):
                    self.test_dataset = self.val_dataset
                else:
                    self._create_test_set()

    def _split_train_val(self):
        """从训练集分割出验证集"""
        total_size = len(self.train_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        self.train_dataset, val_subset = random_split(
            self.train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # 创建验证数据集，使用验证变换
        val_dataset_full = CustomImageDataset(
            data_dir=self.data_dir,
            mode="train",  # 使用训练模式数据，但应用验证变换
            transform=self.val_transform,
            data_format=self.data_format,
            annotation_file=self.annotation_file,
            class_names=self.class_names,
            nested_folders=self.nested_folders
        )

        # 只保留验证子集的样本
        val_indices = val_subset.indices
        val_dataset_full.samples = [val_dataset_full.samples[i] for i in val_indices]
        self.val_dataset = val_dataset_full

    def _create_test_set(self):
        """创建测试集"""
        self.test_dataset = self.val_dataset

    def train_dataloader(self):
        sampler = None
        shuffle = True

        if self.use_weighted_sampling and self.class_weights is not None:
            # 为每个样本分配权重
            sample_weights = []

            # 处理可能的Subset包装
            if hasattr(self.train_dataset, 'dataset'):
                # 这是一个Subset
                for idx in self.train_dataset.indices:
                    _, label = self.train_dataset.dataset.samples[idx]
                    sample_weights.append(self.class_weights[label])
            else:
                # 这是完整的数据集
                for _, label in self.train_dataset.samples:
                    sample_weights.append(self.class_weights[label])

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )