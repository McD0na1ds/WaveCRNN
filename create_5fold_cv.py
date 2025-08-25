#!/usr/bin/env python3
"""
脚本用于将TGRS数据集划分成5折交叉验证的数据集
格式参考datasets目录结构
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def collect_samples_by_class(tgrs_dir):
    """收集每个类别的所有样本"""
    samples_by_class = defaultdict(list)

    for item in os.listdir(tgrs_dir):
        if os.path.isdir(os.path.join(tgrs_dir, item)) and not item.startswith('.'):
            # 提取类别名称（Plunging、Spilling、Surging）
            class_name = item.split('-')[0]
            samples_by_class[class_name].append(item)

    return samples_by_class

def create_5fold_splits(samples_by_class, seed=42):
    """创建5折交叉验证的划分"""
    random.seed(seed)

    # 为每个类别创建5折划分
    folds = [defaultdict(list) for _ in range(5)]

    for class_name, samples in samples_by_class.items():
        # 打乱样本顺序
        samples_shuffled = samples.copy()
        random.shuffle(samples_shuffled)

        # 计算每折的样本数量
        fold_size = len(samples_shuffled) // 5
        remainder = len(samples_shuffled) % 5

        start_idx = 0
        for fold_idx in range(5):
            # 前remainder折多分配一个样本
            current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
            end_idx = start_idx + current_fold_size

            folds[fold_idx][class_name] = samples_shuffled[start_idx:end_idx]
            start_idx = end_idx

    return folds

def create_fold_dataset(tgrs_dir, output_dir, fold_idx, folds):
    """创建单个fold的数据集"""
    # 创建fold目录
    fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
    train_dir = os.path.join(fold_dir, "train")
    val_dir = os.path.join(fold_dir, "val")

    # 创建目录结构
    for class_name in folds[0].keys():
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # 当前fold作为验证集，其他fold作为训练集
    val_samples = folds[fold_idx]
    train_samples = defaultdict(list)

    for other_fold_idx in range(5):
        if other_fold_idx != fold_idx:
            for class_name, samples in folds[other_fold_idx].items():
                train_samples[class_name].extend(samples)

    # 复制验证集文件
    for class_name, samples in val_samples.items():
        for sample in samples:
            src = os.path.join(tgrs_dir, sample)
            dst = os.path.join(val_dir, class_name, sample)
            if os.path.exists(src):
                shutil.copytree(src, dst)
                print(f"复制到验证集: {sample} -> fold_{fold_idx}/val/{class_name}/")

    # 复制训练集文件
    for class_name, samples in train_samples.items():
        for sample in samples:
            src = os.path.join(tgrs_dir, sample)
            dst = os.path.join(train_dir, class_name, sample)
            if os.path.exists(src):
                shutil.copytree(src, dst)
                print(f"复制到训练集: {sample} -> fold_{fold_idx}/train/{class_name}/")

def print_statistics(folds):
    """打印数据集统计信息"""
    print("\n=== 5折交叉验证数据集统计 ===")

    # 总体统计
    total_by_class = defaultdict(int)
    for fold in folds:
        for class_name, samples in fold.items():
            total_by_class[class_name] += len(samples)

    print(f"\n总样本数:")
    total_samples = 0
    for class_name, count in total_by_class.items():
        print(f"  {class_name}: {count}")
        total_samples += count
    print(f"  总计: {total_samples}")

    # 每折统计
    print(f"\n每折样本数:")
    for fold_idx, fold in enumerate(folds):
        print(f"  Fold {fold_idx}:")
        fold_total = 0
        for class_name, samples in fold.items():
            print(f"    {class_name}: {len(samples)}")
            fold_total += len(samples)
        print(f"    小计: {fold_total}")

def main():
    # 设置路径
    tgrs_dir = "/Users/wuxi/PycharmProjects/WaveCRNN/TGRS"
    output_dir = "/Users/wuxi/PycharmProjects/WaveCRNN/datasets_5fold"
    seed = 42

    print(f"TGRS源目录: {tgrs_dir}")
    print(f"输出目录: {output_dir}")
    print(f"随机种子: {seed}")

    # 检查源目录是否存在
    if not os.path.exists(tgrs_dir):
        print(f"错误: 源目录不存在 {tgrs_dir}")
        return

    # 创建输出目录
    if os.path.exists(output_dir):
        print(f"警告: 输出目录已存在，将被清空 {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # 收集各类别样本
    print("\n收集各类别样本...")
    samples_by_class = collect_samples_by_class(tgrs_dir)

    print(f"发现的类别:")
    for class_name, samples in samples_by_class.items():
        print(f"  {class_name}: {len(samples)} 个样本")

    # 创建5折划分
    print(f"\n创建5折交叉验证划分...")
    folds = create_5fold_splits(samples_by_class, seed)

    # 打印统计信息
    print_statistics(folds)

    # 创建每个fold的数据集
    print(f"\n开始复制文件...")
    for fold_idx in range(5):
        print(f"\n--- 创建 Fold {fold_idx} ---")
        create_fold_dataset(tgrs_dir, output_dir, fold_idx, folds)

    print(f"\n✅ 5折交叉验证数据集创建完成!")
    print(f"输出目录: {output_dir}")
    print(f"目录结构:")
    print(f"  datasets_5fold/")
    print(f"    ├── fold_0/")
    print(f"    │   ├── train/")
    print(f"    │   │   ├── Plunging/")
    print(f"    │   │   ├── Spilling/")
    print(f"    │   │   └── Surging/")
    print(f"    │   └── val/")
    print(f"    │       ├── Plunging/")
    print(f"    │       ├── Spilling/")
    print(f"    │       └── Surging/")
    print(f"    ├── fold_1/")
    print(f"    │   └── ...")
    print(f"    ├── fold_2/")
    print(f"    │   └── ...")
    print(f"    ├── fold_3/")
    print(f"    │   └── ...")
    print(f"    └── fold_4/")
    print(f"        └── ...")

if __name__ == "__main__":
    main()
