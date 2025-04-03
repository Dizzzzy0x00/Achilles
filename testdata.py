import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 定义文件路径
vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "java", "vocab.csv")

# 检查文件是否存在
if not os.path.isfile(vocab_path):
    print(f"\x1b[31mVocabulary file not found: {vocab_path}\x1b[m")
else:
    # 加载数据集
    data = pd.read_csv(vocab_path)

    # 检查是否包含必要的列
    if 'input' not in data.columns or 'label' not in data.columns:
        print("\x1b[31mVocabulary file must contain 'input' and 'label' columns.\x1b[m")
    else:
        # 使用 train_test_split 划分数据集
        train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

        # 保存划分后的数据集
        train_path = os.path.join("data", "java", "train_vocab.csv")
        val_path = os.path.join("data", "java", "val_vocab.csv")

        #train_data.to_csv(train_path, index=False)
        val_data.to_csv(val_path, index=False)

        #print(f"训练集保存到: {train_path}")
        print(f"验证集保存到: {val_path}")