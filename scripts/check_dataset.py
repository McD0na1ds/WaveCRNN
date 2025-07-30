#!/usr/bin/env python3
"""
验证数据集结构的脚本
"""

import os
import sys
from pathlib import Path
from PIL import Image


def check_dataset_structure(data_dir):
    """检查数据集结构"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        return False

    print(f"📁 检查数据目录: {data_dir}")

    # 检查训练和验证目录
    for split in ['train', 'val']:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"❌ {split} 目录不存在")
            return False

        print(f"\n📂 {split} 目录:")

        # 检查类别目录
        class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            print(f"❌ {split} 目录下没有类别文件夹")
            return False

        for class_dir in sorted(class_dirs):
            class_name = class_dir.name

            # 统计图像文件
            image_files = []
            for item in class_dir.iterdir():
                if item.is_file():
                    try:
                        # 尝试打开作为图像
                        with Image.open(item) as img:
                            img.verify()
                        image_files.append(item)
                    except Exception:
                        # 不是有效图像，跳过
                        continue

            print(f"   📁 {class_name}: {len(image_files)} 张图像")

            # 显示前几个文件名作为示例
            if image_files:
                print(f"      示例文件: {image_files[0].name}")
                if len(image_files) > 1:
                    print(f"      示例文件: {image_files[1].name}")

    print("\n✅ 数据集结构检查完成")
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python check_dataset.py <数据集路径>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    check_dataset_structure(dataset_path)