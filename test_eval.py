# test_eval.py
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix

# 引用你原本專案中的評估與繪圖模組
from model_trainer import evaluate_model
from plotter import plot_confusion_matrix

# =====================================================================
# 固定隨機種子（必須與 main.py 完全一致，確保復原的隨機特徵完全相同）
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================================
# 1. 載入模型架構定義 (與 main.py 訓練時完全一致)
# =====================================================================
class GaitMCCAStyleNet(nn.Module):
    def __init__(self, num_classes=6):
        super(GaitMCCAStyleNet, self).__init__()
        self.squeeze_net = models.squeezenet1_1(weights=None)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=None)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        f_squeeze = torch.flatten(self.squeeze_pool(self.squeeze_features(x)), 1)
        f_efficient = torch.flatten(self.efficient_pool(self.efficient_features(x)), 1)
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1)
        f_fused = self.fusion_layer(f_combined)
        return self.classifier(f_fused)

class GaitMRFOOptimizedNet(nn.Module):
    def __init__(self, num_classes=6, selected_indices_list=None):
        super(GaitMRFOOptimizedNet, self).__init__()
        self.squeeze_net = models.squeezenet1_1(weights=None)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=None)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if selected_indices_list is None:
            # 如果完全沒有傳入，則採用相容的 linspace 後備機制 (1024維)
            selected_indices_list = torch.linspace(0, 1791, steps=1024).long()
        else:
            selected_indices_list = torch.tensor(selected_indices_list).long()
            
        self.register_buffer('mrfo_indices', selected_indices_list)
        optimized_dim = len(selected_indices_list)
        
        self.post_mrfo_layer = nn.Sequential(
            nn.BatchNorm1d(optimized_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(optimized_dim, num_classes)

    def forward(self, x):
        f_squeeze = torch.flatten(self.squeeze_pool(self.squeeze_features(x)), 1)
        f_efficient = torch.flatten(self.efficient_pool(self.efficient_features(x)), 1)
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1)
        f_mrfo_selected = torch.index_select(f_combined, dim=1, index=self.mrfo_indices)
        f_fused = self.post_mrfo_layer(f_mrfo_selected)
        out = self.classifier(f_fused)
        return out

# =====================================================================
# 2. 測試 Dataset 類別
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
            raise FileNotFoundError(f"❌ 找不到外部測試標籤 CSV 檔案: {label_path}")
            
        self.labels_df = pd.read_csv(label_path)
        col_file = self.labels_df.columns[0]
        col_label = self.labels_df.columns[1]
        
        self.samples = []
        for _, row in self.labels_df.iterrows():
            val_file = str(row[col_file]).strip()
            val_label = str(row[col_label]).strip()
            
            if val_label.isalpha() or 'class' in val_label.lower() or 'label' in val_label.lower():
                continue
                
            try:
                l_id = int(float(val_label))
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
                img_path = os.path.join(data_dir, 'rgb', f_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.CLASS_MAP[l_id]["target"]))

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
            return self.__getitem__((idx + 1) % len(self.samples))

# =====================================================================
# 3. 外部評估主程序
# =====================================================================
if __name__ == "__main__":
    RUN_MODE = "MRFO_Optimization" 
    
    # === 🛑 外部測試路徑設定區 🛑 ===
    TEST_IMAGE_DIR = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\Posture_New_Split\Posture_test"
    TEST_LABEL_PATH = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\Posture_New_Split\Posture_test\labels.csv"
    WEIGHTS_PATH = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\ablation_results_Loop1_20260617_1330\best_model_Baseline4_MRFO_Optimization.pth"
    
    BATCH_SIZE = 16
    NUM_CLASSES = 6
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["Standing", "Sitting", "Lying", "Bending", "Crawling", "Empty"]
    
    print(f"🔍 [獨立外部測試啟動] 裝置: {DEVICE} | 評估架構模式: {RUN_MODE}")
    print(f"   -> 影像資料夾: {TEST_IMAGE_DIR}")
    print(f"   -> 標籤CSV路徑: {TEST_LABEL_PATH}")
    print(f"   -> 載入權重路徑: {WEIGHTS_PATH}\n")

    # --- 預處理 ---
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # --- 載入測試數據集 ---
    print("📦 正在讀取測試集資料...")
    test_dataset = MedicalImageDataset(TEST_IMAGE_DIR, TEST_LABEL_PATH, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"📊 測試集圖片加載成功，實打實共有: {len(test_dataset)} 筆有效樣本。\n")

    # --- 🛠️ 根據模式動態建立對應的模型架構 ---
    if RUN_MODE == "SqueezeNet_Only":
        model_name = "External_SqueezeNet"
        model = models.squeezenet1_1(weights=None)
        model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1))
        features_dim_before_classifier = 512
        
    elif RUN_MODE == "EfficientNet_Only":
        model_name = "External_EfficientNetB0"
        model = models.efficientnet_b0(weights=None)
        features_dim_before_classifier = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(features_dim_before_classifier, NUM_CLASSES)
        
    elif RUN_MODE == "GaitMCCA_Fusion":
        model_name = "External_GaitMCCA_Fusion"
        model = GaitMCCAStyleNet(num_classes=NUM_CLASSES)
        features_dim_before_classifier = 1024
        
    elif RUN_MODE == "MRFO_Optimization":
        model_name = "External_MRFO_Optimization"
        
        # 💡 【神奇魔法：逆向對齊 1024 維索引】
        # 由於你的權重檔裡保存的是 1024 維，我們直接呼叫 linspace 來自動生成當初對應的 1024 個索引
        print(f"🎯 自動復原訓練時的 1024 維 MRFO 特徵選擇索引...")
        gold_mrfo_indices = torch.linspace(0, 1791, steps=1024).long().tolist()
            
        model = GaitMRFOOptimizedNet(num_classes=NUM_CLASSES, selected_indices_list=gold_mrfo_indices)
        features_dim_before_classifier = len(model.mrfo_indices)

    # --- 💾 強制注入載入訓練好的實體權重 ---
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"❌ 在指定路徑下找不到權重檔 (.pth): {WEIGHTS_PATH}")
        
    print(f"⚙️ 正在將實體權重注入模型架構中...")
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    # 這裡已經修正了拼字錯誤
    model.load_state_dict(state_dict, strict=True) # 改回 strict=True 確保維度百分之百完美卡死對齊
    
    model = model.to(DEVICE)
    model.eval() 

    print(f"==========================================================")
    print(f"🔍 外部測試模型識別名稱: {model_name}")
    print(f"🔍 分類層前一檔的「核心特徵數」: 【 {features_dim_before_classifier} 維 】")
    print(f"==========================================================")

    # --- 4. 執行評估運算 ---
    print(f"\n🔮 5070Ti GPU 正在對外部測試集進行前向推理預測...")
    true_labels, pred_labels, metrics = evaluate_model(model, test_loader, device=DEVICE)
    acc, precision, recall, f1 = metrics
    
    # --- 5. 輸出終極獨立測試報表 ---
    print(f"\n✨✨✨==================== 獨立測試集評估結果 ====================✨✨✨")
    print(f"模型架構模式      : {RUN_MODE}")
    print(f"分類前特徵總維度  : {features_dim_before_classifier} 維")
    print(f"外部測試集精確度  (Accuracy) : {acc:.4f} ( {(acc*100):.2f}% )")
    print(f"外部測試集精準率  (Precision): {precision:.4f}")
    print(f"外部測試集召回率  (Recall)   : {recall:.4f}")
    print(f"外部測試集 F1 得分 (F1-Score) : {f1:.4f}")
    print("==========================================================================")

    # 輸出文字版混淆矩陣
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
    print(f"\n📊 [外部測試集 - 文字版混淆矩陣]\n", cm_df)

    # 繪製並儲存測試集專屬的混淆矩陣圖 (.png)
    output_image_name = f"External_Test_Evaluation_{model_name}"
    plot_confusion_matrix(true_labels, pred_labels, class_names, output_image_name)
    
    print(f"\n🎉 外部獨立測試完成！評估視覺化圖表已獨立儲存為: [{output_image_name}.png]")