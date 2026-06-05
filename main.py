import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from collections import Counter
from pathlib import Path
from datetime import datetime  # 💡 引入時間套件

# 引入統計指標與繪圖套件
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

# 引入外部分離核心
from cnn_model import train_model_with_early_stopping, evaluate_model

# =====================================================================
# 1. 完全體 Dataset
# =====================================================================
class MedicalImageDataset(Dataset):
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
    counts = Counter(dataset.raw_labels)
    print(f"\n==================== {name} 姿勢分類統計 ====================")
    print(f"總張數: {len(dataset)}")
    
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
    BATCH_SIZE = 64 
    NUM_WORKERS = 4  
    MAX_EPOCHS = 500  
    PATIENCE = 10     
    IR = 1e-4

    # 💡 建立自訂成果儲存資料夾 
    BASE_DIR = Path(__file__).resolve().parent
    OUTPUT_DIR = BASE_DIR / "Train_Results"
    OUTPUT_DIR.mkdir(exist_ok=True)  # 在本機自動建立一個 Train_Results 資料夾
    
    # 💡 取得當前時間戳記（精確到毫秒），例如：20260604_143025_123
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

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

    print("正在初始化 Datasets 並統計動作分類...")

    # 做資料增強測試集
    train_dataset = MedicalImageDataset(
        data_dir=str(BASE_DIR / 'Posture_New_Split' / 'Posture_train' / 'rgb'), 
        label_path=str(BASE_DIR / 'Posture_New_Split' / 'Posture_train' / 'labels.csv'), 
        transform=train_transform
    )
    
    val_dataset = MedicalImageDataset(
        data_dir=str(BASE_DIR / 'Posture_New_Split' / 'Posture_valdidate' / 'rgb'), 
        label_path=str(BASE_DIR / 'Posture_New_Split' / 'Posture_valdidate' / 'labels.csv'), 
        transform=val_test_transform
    )
    
    test_dataset = MedicalImageDataset(
        data_dir=str(BASE_DIR / 'Posture_New_Split' / 'Posture_test' / 'rgb'), 
        label_path=str(BASE_DIR / 'Posture_New_Split' / 'Posture_test' / 'labels.csv'), 
        transform=val_test_transform
    )
    pprint_dataset_stats(train_dataset, "訓練集 (Train)")
    pprint_dataset_stats(val_dataset, "驗證集 (Val)")
    pprint_dataset_stats(test_dataset, "測試集 (Test)")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    dataloaders_dict = {'train': train_loader, 'val': val_loader}

    # =====================================================================
    # 3. 雙模型初始化設定 (EfficientNet-B0 & SqueezeNet)
    # =====================================================================
    models_to_train = {}

    # 模型一：EfficientNet-B0
    print("\n正在載入預訓練 EfficientNet-B0 模型...")
    effnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    #全模型訓練
    # num_ftrs_eff = effnet.classifier[1].in_features
    # effnet.classifier[1] = nn.Linear(num_ftrs_eff, 6)
    
    # 1. 先將所有參數全部凍結
    for param in effnet.parameters():
        param.requires_grad = False
        
    # 2. 重新定義最後一層分類器 (新定義的層 requires_grad 預設為 True)
    num_ftrs_eff = effnet.classifier[1].in_features
    effnet.classifier[1] = nn.Linear(num_ftrs_eff, 6)
    
    # 3. 手動解凍倒數第 2、第 3 層 (EfficientNet 的最後幾個模組)
    # 解凍特徵提取的最後一個卷積區塊以及 GAP 平均池化層
    for param in effnet.features[-1].parameters():  # 倒數第一個特徵模組
        param.requires_grad = True
    for param in effnet.avgpool.parameters():       # 全局平均池化層
        param.requires_grad = True
    # 註：classifier[1] 已經是解凍狀態
    models_to_train["EfficientNet-B0"] = effnet.to(device)

    # 模型二：SqueezeNet 1.1
    print("正在載入預訓練 SqueezeNet 1.1 模型...")
    #全模型訓練
    squeezenet = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    # squeezenet.classifier[1] = nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1))
    # squeezenet.num_classes = 6
    
    
    # 1. 先將所有參數全部凍結
    for param in squeezenet.parameters():
        param.requires_grad = False
        
    # 2. 重新定義最後一層分類卷積層 (會自動解凍這層)
    squeezenet.classifier[1] = nn.Conv2d(512, 6, kernel_size=(1, 1), stride=(1, 1))
    squeezenet.num_classes = 6
    
    # 3. 手動解凍特徵層的最後一個卷積模組 (features[-1] 是最後一個 Fire 模組)
    for param in squeezenet.features[-1].parameters():
        param.requires_grad = True
    models_to_train["SqueezeNet"] = squeezenet.to(device)

    # 準備字典收集成果
    all_histories = {}
    best_models = {}

    # =====================================================================
    # 4. 開始迴圈輪流訓練
    # =====================================================================
    for model_name, model in models_to_train.items():
        print("\n" + "="*60)
        print(f" 🚀 開始訓練模型: {model_name} ")
        print("="*60)
        
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.AdamW(model.parameters(), lr=IR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # 💡 核心修改：只將 requires_grad == True 的解凍參數送入優化器
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=IR, 
            weight_decay=1e-4
        )
        print(f"開始執行訓練 (最高 {MAX_EPOCHS} 輪，早停容忍度: {PATIENCE})...")
        best_mdl, history = train_model_with_early_stopping(
            model=model, 
            dataloaders=dataloaders_dict, 
            criterion=criterion, 
            optimizer=optimizer, 
            num_epochs=MAX_EPOCHS, 
            patience=PATIENCE,
            device=device
        )
        
        # 儲存結果與權重
        best_models[model_name] = best_mdl
        all_histories[model_name] = history
        
        # 💡 修改：將權重存進 Train_Results 資料夾，檔名加上時間戳記到毫秒
        weight_save_path = OUTPUT_DIR / f"{timestamp}_{model_name}_best.pth"
        torch.save(best_mdl.state_dict(), weight_save_path)
        print(f"🎉 {model_name} 訓練完成！最佳權重已儲存至: {weight_save_path}")

    # =====================================================================
    # 📈 功能一：繪製雙模型學習曲線比較圖
    # =====================================================================
    plt.figure(figsize=(14, 6))
    colors = {'EfficientNet-B0': 'royalblue', 'SqueezeNet': 'crimson'}
    markers = {'EfficientNet-B0': 'o', 'SqueezeNet': 's'}

    # 畫雙模型 Loss 比較
    plt.subplot(1, 2, 1)
    for model_name, history in all_histories.items():
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs_range, history['train_loss'], label=f'{model_name} Train Loss', color=colors[model_name], linestyle='--', marker=markers[model_name])
        plt.plot(epochs_range, history['val_loss'], label=f'{model_name} Val Loss', color=colors[model_name], linestyle='-', marker=markers[model_name])
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 畫雙模型 Accuracy 比較
    plt.subplot(1, 2, 2)
    for model_name, history in all_histories.items():
        epochs_range = range(1, len(history['train_acc']) + 1)
        plt.plot(epochs_range, history['train_acc'], label=f'{model_name} Train Acc', color=colors[model_name], linestyle='--', marker=markers[model_name])
        plt.plot(epochs_range, history['val_acc'], label=f'{model_name} Val Acc', color=colors[model_name], linestyle='-', marker=markers[model_name])
    plt.title('Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # 💡 修改：學習曲線圖存進資料夾，並移除 plt.show() 防止中途卡住
    curve_save_path = OUTPUT_DIR / f"{timestamp}_model_comparison_curves.png"
    plt.savefig(curve_save_path, dpi=300)  
    print(f"\n📊 雙模型對比曲線圖已儲存為: '{curve_save_path}'")
    # plt.show() 👈 移除阻斷式視窗

    # =====================================================================
    # 📊 功能二 & 三：在測試集上評估指標與繪製混淆矩陣
    # =====================================================================
    class_names = [MedicalImageDataset.CLASS_MAP[i]["name"] for i in sorted(MedicalImageDataset.CLASS_MAP.keys())]

    for model_name, best_mdl in best_models.items():
        print(f"\n正在對 {model_name} 的測試集 (Test Set) 進行最終指標評估...")
        true_labels, pred_labels = evaluate_model(best_mdl, test_loader, device=device)
        
        # 評估指標
        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
        
        print(f"\n==================== {model_name} 測試集指標 ====================")
        print(f"精確度 (Accuracy) : {acc:.4f}")
        print(f"精準率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall)   : {recall:.4f}")
        print(f"F1 得分 (F1-Score) : {f1:.4f}")
        print("==========================================================")

        # 混淆矩陣安全版
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
        
        print(f"\n[文字版混淆矩陣 - {model_name}]")
        print(cm_df)

        # 繪製圖像版混淆矩陣
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix: {model_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 💡 修改：混淆矩陣圖存進資料夾，且不使用 plt.show() 讓迴圈自動跑完
        cm_save_path = OUTPUT_DIR / f"{timestamp}_confusion_matrix_{model_name}.png"
        plt.savefig(cm_save_path, dpi=300)  
        print(f"🧱 {model_name} 混淆矩陣圖形已儲存為: '{cm_save_path}'")
        # plt.show() 👈 移除阻斷式視窗

    print("\n🎉 所有模型訓練與測試集評估已全部順暢完成！")