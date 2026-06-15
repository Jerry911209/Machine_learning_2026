# main.py
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix

# 引用自訂模組 (model_trainer.py 與 plotter.py 不需修改)
from model_trainer import train_model_with_early_stopping, evaluate_model
from plotter import plot_learning_curve, plot_confusion_matrix

# =====================================================================
# 1. 論文重現：SqueezeNet + EfficientNetB0 雙流融合網絡
# =====================================================================
class GaitMCCAStyleNet(nn.Module):
    def __init__(self, num_classes=6):
        super(GaitMCCAStyleNet, self).__init__()
        
        # 載入預訓練的 SqueezeNet 1_1
        self.squeeze_net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1)) # 512 維
        
        # 載入預訓練的 EfficientNet B0
        self.efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1)) # 1280 維
        
        # 融合層：將 512 + 1280 = 1792 維特徵進行融合線性對齊
        self.fusion_layer = nn.Sequential(
            nn.Linear(1792, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        f_squeeze = torch.flatten(self.squeeze_pool(self.squeeze_features(x)), 1) 
        f_efficient = torch.flatten(self.efficient_pool(self.efficient_features(x)), 1) 
        
        # 特徵拼接
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1) 
        f_fused = self.fusion_layer(f_combined)
        return self.classifier(f_fused)

# =====================================================================
# 2. 完全自適應對齊的 Dataset 類別定義
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
        
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"❌ 找不到標籤 CSV 檔案: {label_path}")
            
        # 讀取 CSV (自動將第一行視為欄位名稱 index,class)
        self.labels_df = pd.read_csv(label_path)
        
        col_file = self.labels_df.columns[0]  # 第一欄 (index)
        col_label = self.labels_df.columns[1] # 第二欄 (class)
        
        self.samples = []
        for _, row in self.labels_df.iterrows():
            val_file = str(row[col_file]).strip()
            val_label = str(row[col_label]).strip()
            
            # 安全防護：跳過可能重複讀取到的標頭文字
            if val_label.isalpha() or 'class' in val_label.lower() or 'label' in val_label.lower():
                continue
                
            try:
                l_id = int(float(val_label)) # 轉換成整數標籤
                
                # 如果 CSV 內是純數字 (如 1)，自動轉換為 rgb_0001.png
                if val_file.isdigit():
                    f_name = f"rgb_{int(val_file):04d}.png"
                else:
                    if 'rgb_' in val_file:
                        f_name = val_file if val_file.endswith('.png') else f"{val_file}.png"
                    else:
                        f_name = f"rgb_{val_file}.png" if not val_file.endswith('.png') else val_file
            except ValueError:
                continue
            
            if l_id in self.CLASS_MAP:
                # 圖片位於各自資料夾底下的 'rgb' 子目錄中
                img_path = os.path.join(data_dir, 'rgb', f_name)
                
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.CLASS_MAP[l_id]["target"]))

        # 如果載入失敗，印出完整的路徑配對狀況，方便一秒排查
        if len(self.samples) == 0:
            sample_val = self.labels_df.iloc[0][col_file] if len(self.labels_df) > 0 else "1"
            sample_name = f"rgb_{int(sample_val):04d}.png" if str(sample_val).isdigit() else "rgb_0001.png"
            expected_example = os.path.join(data_dir, 'rgb', sample_name)
            
            raise ValueError(f"❌ 找不到任何有效的影像檔案！\n"
                             f"   -> 影像主目錄: {data_dir}\n"
                             f"   -> 標籤檔路徑: {label_path}\n"
                             f"   -> 程式預期圖片應存在於: {expected_example}\n"
                             f"   -> 請確認該路徑與檔案是否存在。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
            return img, target
        except Exception as e:
            print(f"⚠️ 讀取影像失敗，自動跳過: {img_path} (錯誤: {e})")
            return self.__getitem__((idx + 1) % len(self.samples))

# =====================================================================
# 3. 主執行程序
# =====================================================================
if __name__ == "__main__":
    # --- 💡 模式控制開關 (Ablation Study) ---
    # "SqueezeNet_Only" | "EfficientNet_Only" | "GaitMCCA_Fusion"
    RUN_MODE = "GaitMCCA_Fusion" 
    
    # --- 參數設定區 ---
    BATCH_SIZE = 64
    NUM_CLASSES = 6
    EPOCHS = 300
    PATIENCE = 15  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 當前使用的硬體裝置: {DEVICE} | 當前模型模式: {RUN_MODE}")

    # 資料夾主路徑設定
    BASE_DIR = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\Posture_New_Split"
    
    # === 💡 核心修改區：完美對齊獨立的三大實體資料夾路徑 ===
    TRAIN_DIR = os.path.join(BASE_DIR, "Posture_train")
    VAL_DIR = os.path.join(BASE_DIR, "Posture_valdidate")             # 修正：指向獨立的驗證集資料夾
    TEST_DIR = os.path.join(BASE_DIR, "Posture_test")   # 修正：指向獨立的測試集資料夾 (Posture_test)
    
    # 自動偵測對齊各自資料夾下的 label.csv 或 labels.csv
    def get_valid_csv_path(folder_path):
        p1 = os.path.join(folder_path, "label.csv")
        p2 = os.path.join(folder_path, "labels.csv")
        return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else p1)

    TRAIN_LABEL = get_valid_csv_path(TRAIN_DIR)
    VAL_LABEL = get_valid_csv_path(VAL_DIR)
    TEST_LABEL = get_valid_csv_path(TEST_DIR)

    # --- 影像增強與預處理 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # --- 載入數據集 ---
    print("\n📦 正在載入數據集...")
    train_dataset = MedicalImageDataset(TRAIN_DIR, TRAIN_LABEL, transform=data_transforms['train'])
    val_dataset = MedicalImageDataset(VAL_DIR, VAL_LABEL, transform=data_transforms['val_test'])
    test_dataset = MedicalImageDataset(TEST_DIR, TEST_LABEL, transform=data_transforms['val_test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    }

    print(f"📊 數據載入完畢！(訓練集: {len(train_dataset)} 筆, 驗證集: {len(val_dataset)} 筆, 測試集: {len(test_dataset)} 筆)")
    class_names = ["Standing", "Sitting", "Lying", "Bending", "Crawling", "Empty"]

    # --- 動態初始化對應的模型架構 ---
    if RUN_MODE == "SqueezeNet_Only":
        model_name = "SqueezeNet_Baseline"
        print(f"   -> 正在建立單獨 Baseline: SqueezeNet 1_1...")
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1))
        model.num_classes = NUM_CLASSES
        
    elif RUN_MODE == "EfficientNet_Only":
        model_name = "EfficientNetB0_Baseline"
        print(f"   -> 正在建立單獨 Baseline: EfficientNet B0...")
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
        
    elif RUN_MODE == "GaitMCCA_Fusion":
        model_name = "GaitMCCA_Squeeze_Efficient_Fusion"
        print(f"   -> 正在重現論文：載入雙流 MCCA 風格融合網絡...")
        model = GaitMCCAStyleNet(num_classes=NUM_CLASSES)
        
    else:
        raise ValueError("❌ 未知的 RUN_MODE，請檢查設定！")

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # --- 1. 執行訓練 ---
    print(f"\n🏋️ 啟動模型訓練模型 [{model_name}]...")
    model, history = train_model_with_early_stopping(
        model, dataloaders, criterion, optimizer, 
        num_epochs=EPOCHS, patience=PATIENCE, device=DEVICE
    )

    # --- 2. 繪製學習曲線 ---
    plot_learning_curve(history, model_name)

    # --- 3. 測試集性能評估 ---
    print(f"\n🔍 正在使用測試集進行性能評估...")
    true_labels, pred_labels, metrics = evaluate_model(model, dataloaders['test'], device=DEVICE)
    acc, precision, recall, f1 = metrics
    
    print(f"\n==================== {model_name} 測試集指標 ====================")
    print(f"精確度 (Accuracy) : {acc:.4f}")
    print(f"精準率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall)   : {recall:.4f}")
    print(f"F1 得分 (F1-Score) : {f1:.4f}")
    print("==========================================================")

    # 印出文字版混淆矩陣
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    print(f"\n[文字版混淆矩陣 - {model_name}]\n", cm_df)

    # --- 4. 繪製混淆矩陣圖 ---
    plot_confusion_matrix(true_labels, pred_labels, class_names, model_name)
    
    print(f"\n🎉 模式 [{model_name}] 執行完畢！結果已獨立儲存。")