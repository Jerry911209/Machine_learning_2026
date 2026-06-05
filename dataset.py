import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter

# =====================================================================
# 1. 完全體 Dataset (支援動作名稱映射與 PyTorch 標籤修正)
# =====================================================================
class MedicalImageDataset(Dataset):
    # 定義全局的類別映射字典
    CLASS_MAP = {
        1: {"name": "Standing", "target": 0},
        2: {"name": "Sitting",  "target": 1},
        3: {"name": "Lying",    "target": 2},
        4: {"name": "Bending",  "target": 3},
        5: {"name": "Crawling", "target": 4},
        6: {"name": "Empty",    "target": 5}
    }

    def __init__(self, data_dir, label_path, transform=None):
        """
        Args:
            data_dir (str): 圖片所在的資料夾路徑 (例如: r'C:\...\rgb')
            label_path (str): 標籤檔案的完整絕對路徑 (例如: r'C:\...\labels.csv')
            transform (callable, optional): 影像預處理或資料增強的組合
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # 讀取標籤檔案
        if label_path.endswith('.csv'):
            self.labels_df = pd.read_csv(label_path)
        elif label_path.endswith('.xlsx') or label_path.endswith('.xls'):
            self.labels_df = pd.read_excel(label_path)
        else:
            raise ValueError("標籤檔案格式必須是 CSV 或 Excel 檔案")

        self.image_indices = self.labels_df['index'].values
        self.raw_labels = self.labels_df['class'].values
        
        # 💡 核心修正：將原始類別 (1~6) 轉換成 PyTorch 訓練用的連續標籤 (0~5)
        self.labels = [self.CLASS_MAP[x]["target"] for x in self.raw_labels]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_idx = self.image_indices[idx]
        img_name = f"rgb_{int(img_idx):04d}.png"
        img_path = os.path.join(self.data_dir, img_name) 
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]  # 這裡拿到的會是 0 ~ 5 的數值
        
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================================================================
# 2. 初始化與分類數量詳細統計
# =====================================================================
if __name__ == "__main__":
    
    img_size = 256
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    BATCH_SIZE = 16 
    NUM_WORKERS = 0 # Windows 環境下建議設為 0 以防多線程報錯

    print("正在初始化 Datasets 並統計動作分類...")

    train_dataset = MedicalImageDataset(
        data_dir=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_train\rgb', 
        label_path=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_train\labels.csv', 
        transform=train_transform
    )
    
    val_dataset = MedicalImageDataset(
        data_dir=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_valdidate\rgb', 
        label_path=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_valdidate\labels.csv', 
        transform=val_test_transform
    )
    
    test_dataset = MedicalImageDataset(
        data_dir=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_test\rgb', 
        label_path=r'C:\Users\jerry\OneDrive\桌面\git\Topics\機器學習\Posture_train_valdidate_test\Posture_test\labels.csv', 
        transform=val_test_transform
    )

    # 📊 定義精美統計函數
    def pprint_dataset_stats(dataset, name):
        counts = Counter(dataset.raw_labels)  # 使用原始標籤進行統計
        print(f"\n==================== {name} 姿勢分類統計 ====================")
        print(f"總張數: {len(dataset)}")
        
        # 依照原始 1~6 順序依序印出
        for orig_cls in sorted(MedicalImageDataset.CLASS_MAP.keys()):
            cls_name = MedicalImageDataset.CLASS_MAP[orig_cls]["name"]
            mapped_target = MedicalImageDataset.CLASS_MAP[orig_cls]["target"]
            num_imgs = counts.get(orig_cls, 0)
            percentage = (num_imgs / len(dataset)) * 100 if len(dataset) > 0 else 0
            
            print(f"  [{cls_name:<8}] (原始類別 {orig_cls} -> 訓練標籤 {mapped_target}): {num_imgs:>4} 張 ({percentage:.2f}%)")

    # 執行統計
    pprint_dataset_stats(train_dataset, "訓練集 (Train)")
    pprint_dataset_stats(val_dataset, "驗證集 (Val)")
    pprint_dataset_stats(test_dataset, "測試集 (Test)")

    # 建立 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    print("\n正在測試讀取第一個 Batch...")
    images, labels = next(iter(train_loader))
    print("--- 檢查第一個 Batch 的資料維度 ---")
    print(f"影像 Tensor 維度 (Batch, Channel, H, W): {images.shape}")
    print(f"訓練用標籤 Tensor (已轉為 0~5): {labels}")