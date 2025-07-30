import os
import shutil
import random
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
random.seed(42)

# 数据目录
data_dir = '../datasets'
output_dir = '../data'

# 创建输出目录结构
os.makedirs(output_dir, exist_ok=True)
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 类别映射（从文件名前缀提取）
classes = ['Plunging', 'Spilling', 'Surging']

# 为每个类别创建目录
for cls in classes:
    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

# 获取所有文件夹
all_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
print(f"总共找到 {len(all_folders)} 个文件夹")

# 按类别分组
folders_by_class = {cls: [] for cls in classes}
for folder in all_folders:
    for cls in classes:
        if folder.startswith(cls):
            folders_by_class[cls].append(folder)
            break

# 分割数据集
train_folders = []
val_folders = []

for cls in classes:
    folders = folders_by_class[cls]
    print(f"{cls} 类别有 {len(folders)} 个文件夹")
    
    # 按80/20分割
    train_cls, val_cls = train_test_split(folders, test_size=0.2, random_state=42)
    train_folders.extend(train_cls)
    val_folders.extend(val_cls)
    
    print(f"  - 训练集: {len(train_cls)} 个文件夹")
    print(f"  - 验证集: {len(val_cls)} 个文件夹")

# 复制文件夹到对应的训练/验证目录
def copy_folders(folders, target_dir):
    for folder in folders:
        # 确定类别
        cls = None
        for c in classes:
            if folder.startswith(c):
                cls = c
                break
        
        if cls is None:
            print(f"无法确定文件夹 {folder} 的类别")
            continue
        
        # 源路径和目标路径
        src_path = os.path.join(data_dir, folder)
        dst_path = os.path.join(target_dir, cls, folder)
        
        # 复制整个文件夹
        try:
            shutil.copytree(src_path, dst_path)
            print(f"已复制 {src_path} -> {dst_path}")
        except Exception as e:
            print(f"复制 {src_path} 时出错: {e}")

print("\n复制训练集文件夹...")
copy_folders(train_folders, train_dir)

print("\n复制验证集文件夹...")
copy_folders(val_folders, val_dir)

print("数据分割完成!")