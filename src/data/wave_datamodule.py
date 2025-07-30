import lightning as L
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from typing import Optional
import torch
import logging

from .custom_dataset import CustomImageDataset
from .wave_transforms import WaveAugmentation, FoamEnhancement, WaterBlur, SeaColorAugmentation

logger = logging.getLogger(__name__)


class WaveDataModule(L.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 32,
            num_workers: int = 4,
            image_size: int = 224,
            use_wave_augmentation: bool = True,
            use_weighted_sampling: bool = True,
            nested_folders: bool = True,  # 新增参数，支持嵌套文件夹
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.use_wave_augmentation = use_wave_augmentation
        self.use_weighted_sampling = use_weighted_sampling
        self.nested_folders = nested_folders

        # 波浪类别（大写，与数据集保持一致）
        self.class_names = ["Plunging", "Spilling", "Surging"]
        self.num_classes = 3

        # 构建变换
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()

    def _build_train_transform(self):
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(
                self.image_size,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        # 添加波浪特定增强
        if self.use_wave_augmentation:
            transform_list.extend([
                WaveAugmentation(p=0.3),
                FoamEnhancement(p=0.2),
                WaterBlur(p=0.2),
                SeaColorAugmentation(p=0.3),
            ])

        # 添加标准增强
        transform_list.extend([
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        return transforms.Compose(transform_list)

    def _build_val_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            try:
                self.train_dataset = CustomImageDataset(
                    data_dir=self.data_dir,
                    mode="train",
                    transform=self.train_transform,
                    data_format="folder",
                    class_names=self.class_names,
                    nested_folders=self.nested_folders  # 支持嵌套文件夹
                )

                self.val_dataset = CustomImageDataset(
                    data_dir=self.data_dir,
                    mode="val",
                    transform=self.val_transform,
                    data_format="folder",
                    class_names=self.class_names,
                    nested_folders=self.nested_folders  # 支持嵌套文件夹
                )

                # 输出数据集信息
                train_info = self.train_dataset.get_sample_info()
                val_info = self.val_dataset.get_sample_info()
                logger.info(f"Training dataset: {train_info}")
                logger.info(f"Validation dataset: {val_info}")

                # 计算类别权重
                if self.use_weighted_sampling:
                    self.class_weights = self.train_dataset.get_class_weights()

            except Exception as e:
                logger.error(f"Error setting up train/val datasets: {e}")
                raise

        if stage == "test" or stage is None:
            try:
                self.test_dataset = CustomImageDataset(
                    data_dir=self.data_dir,
                    mode="test",
                    transform=self.val_transform,
                    data_format="folder",
                    class_names=self.class_names,
                    nested_folders=self.nested_folders  # 支持嵌套文件夹
                )
            except Exception as e:
                logger.info(f"No test set found ({e}), using validation set for testing")
                if hasattr(self, 'val_dataset'):
                    self.test_dataset = self.val_dataset
                else:
                    logger.warning("No validation set available for testing")

    def train_dataloader(self):
        sampler = None
        shuffle = True

        if self.use_weighted_sampling and hasattr(self, 'class_weights'):
            # 为每个样本分配权重
            sample_weights = []
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