import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter

# 引入統計指標與繪圖套件
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# 引入剛改好的外部分離核心
from cnn_model import train_model_with_early_stopping, evaluate_model

# =====================================================================
# 1. 完全體 Dataset (放在這裡最上面，確保下方的函數與 main 都能順利讀到)
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
        self.data_dir = data_dir
        self.transform = transform
        
        if label_path.endswith('.csv'):
            self.labels_df = pd.read_csv(label_path)
        elif label_path.endswith('.xlsx') or label_path.endswith('.xls'):
            self.labels_df = pd.read_excel(label_path)
        else:
            raise ValueError("標籤檔案格式必須是 CSV 或 Excel 檔案")

        self.image_indices = self.labels_df['index'].values
        self.raw_labels = self.labels_df['class'].values
        self.labels = [self.CLASS_MAP[x]["target"] for x in self.raw_labels]

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_idx = self.image_indices[idx]
        img_name = f"rgb_{int(img_idx):04d}.png"
        img_path = os.path.join(self.data_dir, img_name) 
        
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

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


# =====================================================================
# 2. Main 執行進入點
# =====================================================================
if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的訓練裝置: {device}")
    if torch.cuda.is_available():
        print(f"顯示卡型號: {torch.cuda.get_device_name(0)}")

    img_size = 256
    BATCH_SIZE = 32 
    NUM_WORKERS = 0  
    MAX_EPOCHS = 500  # 啟用早停，可以放心地把上限設大
    PATIENCE = 5     # 連續 5 輪驗證 Loss 沒下降就停
    IR=1e-4
    # --- Transforms & Datasets & DataLoaders ---
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

    print("\n正在初始化 Datasets 並統計動作分類...")
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

    # 執行數據集統計印出
    pprint_dataset_stats(train_dataset, "訓練集 (Train)")
    pprint_dataset_stats(val_dataset, "驗證集 (Val)")
    pprint_dataset_stats(test_dataset, "測試集 (Test)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # --- 模型初始化 ---
    print("\n正在載入預訓練 ResNet18 模型...")
    model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 6)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=IR)

    # --- 執行訓練與早停 ---
    print(f"\n開始執行訓練 (最高 {MAX_EPOCHS} 輪，早停容忍度: {PATIENCE})...")
    best_model, history = train_model_with_early_stopping(
        model=model_ft, 
        dataloaders=dataloaders_dict, 
        criterion=criterion, 
        optimizer=optimizer_ft, 
        num_epochs=MAX_EPOCHS, 
        patience=PATIENCE,
        device=device
    )

    # 儲存最佳模型權重
    torch.save(best_model.state_dict(), "best_posture_cnn.pth")

    # =====================================================================
    # 📈 功能一：繪製 Loss 與 Accuracy 對應 Epoch 曲線圖
    # =====================================================================
    epochs_range = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # 畫 Loss 曲線
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='royalblue', marker='o')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss', color='orange', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 畫 Accuracy 曲線
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc', color='royalblue', marker='o')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc', color='orange', marker='s')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)  
    print("\n📊 訓練曲線圖已儲存為 'learning_curves.png'")
    plt.show()

    # =====================================================================
    # 📊 功能二：在測試集 (Test Set) 上評估各項高階指標
    # =====================================================================
    print("\n正在對測試集 (Test Set) 進行最終指標評估...")
    true_labels, pred_labels = evaluate_model(best_model, test_loader, device=device)
    
    # 計算各項數據
    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
    
    print("\n==================== 測試集學術指標報告 ====================")
    print(f"精確度 (Accuracy) : {acc:.4f}")
    print(f"精準率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"F1 得分 (F1-Score) : {f1:.4f}")
    print("==========================================================")

    # =====================================================================
    # 🧱 功能三：計算並印出混淆矩陣 (Confusion Matrix)
    # =====================================================================
    class_names = [MedicalImageDataset.CLASS_MAP[i]["name"] for i in sorted(MedicalImageDataset.CLASS_MAP.keys())]
    cm = confusion_matrix(true_labels, pred_labels)
    
    print("\n[文字版混淆矩陣]")
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    print(cm_df)

    # 繪製漂亮圖像版混淆矩陣
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix on Test Dataset')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)  
    print("\n🧱 混淆矩陣圖形已儲存為 'confusion_matrix.png'")
    plt.show()