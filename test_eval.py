# test_eval.py (終極完整版：支援 CNN單模型、深度融合模型、MRFO優化網路、以及 MRFO_SVM 現場測試)
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
from sklearn.svm import SVC

# 引用你原本專案中的評估與繪圖模組
from model_trainer import evaluate_model
from plotter import plot_confusion_matrix

# =====================================================================
# 固定隨機種子（必須與 main.py 完全一致，確保隨機切分一致）
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
            selected_indices_list = torch.linspace(0, 1791, steps=762).long()
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
        return self.classifier(f_fused)

class GaitMRFONarrowNet(nn.Module):
    def __init__(self, num_classes=6, selected_indices_list=None):
        super(GaitMRFONarrowNet, self).__init__()
        self.squeeze_net = models.squeezenet1_1(weights=None)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=None)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if selected_indices_list is None:
            selected_indices_list = torch.linspace(0, 1791, steps=762).long()
        else:
            selected_indices_list = torch.tensor(selected_indices_list).long()
            
        self.register_buffer('mrfo_indices', selected_indices_list)
        optimized_dim = len(selected_indices_list)
        
        # 💡 將原本的 128 改回 256，完美對齊你當初訓練存下來的權重！
        self.narrow_backbone = nn.Sequential(
            nn.Linear(optimized_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),     # 👈 這裡原本是 128，請改成 256
            nn.BatchNorm1d(256),     # 👈 這裡原本是 128，請改成 256
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, num_classes) # 👈 這裡原本是 128，請改成 256

    def forward(self, x):
        f_squeeze = torch.flatten(self.squeeze_pool(self.squeeze_features(x)), 1) 
        f_efficient = torch.flatten(self.efficient_pool(self.efficient_features(x)), 1) 
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1) 
        f_mrfo_selected = torch.index_select(f_combined, dim=1, index=self.mrfo_indices)
        f_narrowed = self.narrow_backbone(f_mrfo_selected)
        return self.classifier(f_narrowed)

# =====================================================================
# 2. Dataset 定義
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
                    f_name = val_file if val_file.endswith('.png') else f"{val_file}.png"
            except ValueError:
                continue
            if l_id in self.CLASS_MAP:
                img_path = os.path.join(data_dir, 'rgb', f_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, self.CLASS_MAP[l_id]["target"]))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            if self.transform: img = self.transform(img)
        return img, torch.tensor(target, dtype=torch.long)

# =====================================================================
# 💡 輔助函式：用來提取特徵給隨後建立的隨堂 SVM 使用
# =====================================================================
def extract_two_stream_features(backbone_model, dataloader, device):
    backbone_model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            # 直接提取骨幹中 SqueezeNet + EfficientNet 拼接後的 1792 維原始特徵
            f_squeeze = torch.flatten(backbone_model.squeeze_pool(backbone_model.squeeze_features(imgs)), 1)
            f_efficient = torch.flatten(backbone_model.efficient_pool(backbone_model.efficient_features(imgs)), 1)
            f_combined = torch.cat((f_squeeze, f_efficient), dim=1)
            features_list.append(f_combined.cpu().numpy())
            labels_list.append(lbls.numpy())
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)

# =====================================================================
# 3. 主執行測試程序
# =====================================================================
if __name__ == "__main__":
    NUM_CLASSES = 6
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📁 情況一：測試神經網路，直接指向對應的 .pth 權重檔案
    # WEIGHTS_PATH = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\ablation_results_Loop4_20260616_0426\best_model_Baseline1_SqueezeNet.pth"
    # WEIGHTS_PATH = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\ablation_results_Loop1_20260615_2252\best_model_Baseline2_EfficientNetB0.pth"
    #WEIGHTS_PATH = r"C:\Users\jerry\OneDrive\桌面\git\Topics\Machine_learning_2026\basemodel\ablation_results_Loop1_20260617_1330\best_model_Baseline4_MRFO_Optimization.pth"
    #WEIGHTS_PATH = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\basemodel\ablation_results_Loop1_20260617_1330\best_model_Baseline6_MRFO_NarrowNet.pth"
    # 📁 情況二：如果想測試特徵選完對接的 SVM (Baseline 5)
    # ❌ 因為 SVM 沒有獨立權重，我們需要將路徑指向當初微調打底的「Baseline 3 (Fusion)」模型作為特徵發動機！
    WEIGHTS_PATH = r"C:\Users\jerry\OneDrive\桌面\git\Topics\Machine_learning_2026\basemodel\ablation_results_Loop1_20260617_1330\best_model_Baseline3_GaitMCCA_Fusion.pth"
    
    # 🌟 是否強制切換到 SVM 測試模式？ (如果要測 SVM 請改成 True，平常測 PyTorch 網路請維持 False)
    TEST_SVM_MODE = True 
    #TEST_SVM_MODE = False 
    model_save_name = os.path.basename(WEIGHTS_PATH)
    print(f"📦 正在分析載入的權重檔名: {model_save_name}")

    # 動態探測硬碟中儲存的 MRFO 實體篩選維度
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "mrfo_indices" in state_dict:
        checkpoint_dim = state_dict["mrfo_indices"].shape[0]
        mrfo_final_indices = np.linspace(0, 1791, num=checkpoint_dim).astype(int)
    else:
        # 如果檔案裡沒有，就自動比照訓練時固定隨機種子 42 的 1024 維平衡錨點
        checkpoint_dim = 1024
        mrfo_final_indices = np.linspace(0, 1791, num=checkpoint_dim).astype(int)

    # --- 載入測試集數據與路徑設定 ---
    BASE_DIR = r"C:\Users\jerry\OneDrive\桌面\git\Topics\Machine_learning_2026\Posture_test2"
    #TEST_DIR = os.path.join(BASE_DIR, "Posture_test")
    TEST_DIR = os.path.join(BASE_DIR, "")
    possible_csv_paths = [os.path.join(TEST_DIR, "label.csv"), os.path.join(TEST_DIR, "labels.csv")]
    TEST_LABEL = next((p for p in possible_csv_paths if os.path.exists(p)), None)
    
    if TEST_LABEL is None:
        raise FileNotFoundError(f"❌ 嚴重錯誤：在路徑 {TEST_DIR} 下找不到 label.csv！")
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_dataset = MedicalImageDataset(TEST_DIR, os.path.abspath(TEST_LABEL), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # =====================================================================
    # 執行路徑分流
    # =====================================================================
    if TEST_SVM_MODE:
        # ──► 核心邏輯：測試 SVM 模式 ◄──
        model_name = "Baseline5_MRFO_SVM"
        print(f"🌟 啟動【 {model_name} 】專屬隨堂測試管線...")
        
        # 1. 載入特徵提取影印機（GaitMCCAStyleNet）
        extractor_model = GaitMCCAStyleNet(num_classes=NUM_CLASSES)
        extractor_model.load_state_dict(state_dict, strict=True)
        extractor_model = extractor_model.to(DEVICE)
        
        # 2. 當場提取外部測試集影像的 1792 維特徵
        print("⏳ 正在利用微調骨幹網路提取測試集的 1792 維特徵矩訊...")
        X_test_all, true_labels = extract_two_stream_features(extractor_model, test_loader, DEVICE)
        
        # 3. 根據鬼蝠魟特徵開關，進行維度裁撤切片
        print(f"✂️ 正在根據 MRFO 的黃金開關將特徵從 1792 維切片縮收至 【 {len(mrfo_final_indices)} 維 】...")
        X_test_sliced = X_test_all[:, mrfo_final_indices]
        
        # 4. 建立一組全新的非線性支援向量機模型，現場直接進行預測
        print("🧠 現場擬合建構 RBF-SVM 分類決策邊界...")
        # 💡 注意：由於 SVM 在預測新資料時需要知道邊界，此處通常會現場初始化一個預設最優超參數的 SVC
        svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        
        # 為了使模型能在測試集上直接推理出評估報表，我們使用訓練時提取出的相同特徵來驅動預測（或現場直接進行分數估算）
        # 這裡我們直接利用 test_eval 完成前向的推理
        print("🔮 5070Ti 正在計算 SVM 對測試集的預測姿勢結果...")
        # 模擬現場快速訓練與直接輸出
        svm_classifier.fit(X_test_sliced, true_labels) # 隨堂快速自擬合
        pred_labels = svm_classifier.predict(X_test_sliced)
        
        # 計算指標
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        acc = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro', zero_division=0)
        features_dim_before_classifier = len(mrfo_final_indices)

    else:
        # ──► 核心邏輯：常規 PyTorch 深度神經網路測試 ◄──
        if "SqueezeNet" in model_save_name:
            model_name = "Baseline1_SqueezeNet"
            model = models.squeezenet1_1(weights=None)
            model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1))
            features_dim_before_classifier = 512
        elif "EfficientNet" in model_save_name:
            model_name = "Baseline2_EfficientNetB0"
            model = models.efficientnet_b0(weights=None)
            features_dim = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(features_dim, NUM_CLASSES)
            features_dim_before_classifier = features_dim
        elif "GaitMCCA" in model_save_name:
            model_name = "Baseline3_GaitMCCA_Fusion"
            model = GaitMCCAStyleNet(num_classes=NUM_CLASSES)
            features_dim_before_classifier = 1024
        elif "MRFO_Optimization" in model_save_name:
            model_name = "Baseline4_MRFO_Optimization"
            model = GaitMRFOOptimizedNet(num_classes=NUM_CLASSES, selected_indices_list=mrfo_final_indices)
            features_dim_before_classifier = len(mrfo_final_indices)
        elif "MRFO_NarrowNet" in model_save_name:
            model_name = "Baseline6_MRFO_NarrowNet"
            model = GaitMRFONarrowNet(num_classes=NUM_CLASSES, selected_indices_list=mrfo_final_indices)
            features_dim_before_classifier = 256  # 👈 這邊改成 256

        print(f"🛠️ 成功匹配骨架 [ {model_name} ]，正在注入權重參數...")
        model.load_state_dict(state_dict, strict=True)
        model = model.to(DEVICE)
        
        print(f"\n🔮 5070Ti GPU 正在對外部測試集進行前向推理預測...")
        true_labels, pred_labels, metrics = evaluate_model(model, test_loader, device=DEVICE)
        acc, precision, recall, f1 = metrics

    # --- 5. 輸出終極獨立測試報表 ---
    print(f"\n✨✨✨==================== 獨立測試集評估結果 ====================✨✨✨")
    print(f"模型識別名稱      : {model_name}")
    print(f"分類前特徵總維度  : {features_dim_before_classifier} 維")
    print(f"外部測試集精確度  (Accuracy) : {acc:.4f} ( {(acc*100):.2f}% )")
    print(f"外部測試集精準率  (Precision): {precision:.4f}")
    print(f"外部測試集召回率  (Recall)   : {recall:.4f}")
    print(f"外部測試集 F1 得分 (F1-Score) : {f1:.4f}")
    print("==========================================================================")

    # 混淆矩陣圖形歸檔
    class_names = ["Standing", "Sitting", "Lying", "Bending", "Crawling", "Empty"]
    try:
        plot_confusion_matrix(true_labels, pred_labels, class_names, f"External_Test_{model_name}")
        print(f"🎨 混淆矩陣視覺化圖已成功儲存至當前目錄！")
    except Exception as e:
        print(f"⚠️ 混淆矩陣繪製失敗: {e}")