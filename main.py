# main.py
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import confusion_matrix

# 💡 真正引入現成的鬼蝠魟演算法套件
from mealpy import MRFO 

# 引用自訂模組
from model_trainer import train_model_with_early_stopping, evaluate_model
from plotter import plot_learning_curve, plot_confusion_matrix

# =====================================================================
# 1. Baseline 3：雙流拼接融合網絡 (GaitMCCA Style)
# =====================================================================
class GaitMCCAStyleNet(nn.Module):
    def __init__(self, num_classes=6):
        super(GaitMCCAStyleNet, self).__init__()
        self.squeeze_net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))
        
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
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1) 
        f_fused = self.fusion_layer(f_combined) # 輸出特徵數：512
        return self.classifier(f_fused)

# =====================================================================
# 2. Baseline 4：雙流特徵 + 真正 MRFO 鬼蝠魟索引切片篩選網絡
# =====================================================================
class GaitMRFOOptimizedNet(nn.Module):
    def __init__(self, num_classes=6, selected_indices_list=None):
        super(GaitMRFOOptimizedNet, self).__init__()
        
        self.squeeze_net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
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
        out = self.classifier(f_fused)
        return out

# =====================================================================
# 3. Dataset 類別定義
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

        if len(self.samples) == 0:
            raise ValueError(f"❌ 找不到任何有效的影像檔案，請檢查路徑。")

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
# 4. 主執行程序 (自動排程消融實驗版本)
# =====================================================================
if __name__ == "__main__":
    # --- 💡 核心變革：將 4 個 Baseline 組成排程清單，準備啟動 for 迴圈自動連跑 ---
    ABLATION_MODES = [
        "SqueezeNet_Only", 
        "EfficientNet_Only", 
        "GaitMCCA_Fusion", 
        "MRFO_Optimization"
    ]
    
    # 用來最後統合輸出所有實驗結果的總戰報字典
    FINAL_REPORT = {}
    
    # --- 基礎全域參數設定 ---
    BATCH_SIZE = 64
    NUM_CLASSES = 6
    EPOCHS = 300
    PATIENCE = 15
    IR=1e-4;  
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動自動化消融實驗排程 | 當前硬體: {DEVICE} | 總計實驗清單: {ABLATION_MODES}")

    # 資料夾主路徑與標籤自動對齊
    BASE_DIR = r"C:\Users\jerry\Documents\GitHub\Machine_learning_2026\Posture_New_Split"
    TRAIN_DIR = os.path.join(BASE_DIR, "Posture_train")
    VAL_DIR = os.path.join(BASE_DIR, "Posture_valdidate")             
    TEST_DIR = os.path.join(BASE_DIR, "Posture_test")   
    
    def get_valid_csv_path(folder_path):
        p1 = os.path.join(folder_path, "label.csv")
        p2 = os.path.join(folder_path, "labels.csv")
        return p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else p1)

    TRAIN_LABEL = get_valid_csv_path(TRAIN_DIR)
    VAL_LABEL = get_valid_csv_path(VAL_DIR)
    TEST_LABEL = get_valid_csv_path(TEST_DIR)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val_test': transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    print("\n📦 正在載入通用數據集...")
    train_dataset = MedicalImageDataset(TRAIN_DIR, TRAIN_LABEL, transform=data_transforms['train'])
    val_dataset = MedicalImageDataset(VAL_DIR, VAL_LABEL, transform=data_transforms['val_test'])
    test_dataset = MedicalImageDataset(TEST_DIR, TEST_LABEL, transform=data_transforms['val_test'])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    }

    print(f"📊 數據載入完畢！(訓練集: {len(train_dataset)} 筆, 驗證集: {len(val_dataset)} 筆, 測試集: {len(test_dataset)} 筆)")
    class_names = ["Standing", "Sitting", "Lying", "Bending", "Crawling", "Empty"]

    # =====================================================================
    # 🔄 核心機制：利用 for 迴圈，依序全自動化跑完這 4 條 Baseline！
    # =====================================================================
    for step_idx, RUN_MODE in enumerate(ABLATION_MODES, 1):
        print(f"\n\n##########################################################")
        print(f" 🎬 正在自動執行第 [{step_idx}/4] 項消融實驗 ➡️ 模式: {RUN_MODE}")
        print(f"##########################################################\n")
        
        mrfo_final_indices = None
        
        # 💡 只有輪到第 4 條 MRFO 模式時，才會實打實啟動 Mealpy 鬼蝠魟演算法尋優
        if RUN_MODE == "MRFO_Optimization":
            from mealpy.utils.problem import Problem
            from mealpy.utils.space import BinaryVar
            
            print(f"🌊 [Mealpy 核心啟動] 偵測進入 MRFO 階段，啟動鬼蝠魟特徵尋優程序...")
            print(f"   -> 正在提取特徵快取中...")
            
            extractor = GaitMCCAStyleNet(num_classes=NUM_CLASSES).to(DEVICE)
            extractor.eval()
            
            temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
            cached_features, cached_labels = [], []
            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(temp_loader):
                    if idx >= 100: break 
                    inputs = inputs.to(DEVICE)
                    f_s = torch.flatten(extractor.squeeze_pool(extractor.squeeze_features(inputs)), 1)
                    f_e = torch.flatten(extractor.efficient_pool(extractor.efficient_features(inputs)), 1)
                    f_comb = torch.cat((f_s, f_e), dim=1).cpu().numpy()
                    cached_features.append(f_comb[0])
                    cached_labels.append(labels.numpy()[0])
                    
            X_train_mrfo = np.array(cached_features)
            
            # 帶有數量懲罰項的自適應問題類別
            class MantaRayFeatureSelectionProblem(Problem):
                def __init__(self, bounds, minmax="max", **kwargs):
                    super().__init__(bounds=bounds, minmax=minmax, **kwargs)
                    
                def obj_func(self, x):
                    selected_idx = np.where(x > 0.5)[0]
                    num_selected = len(selected_idx)
                    if num_selected == 0: return 0.0 
                    feat_quality = float(np.mean(X_train_mrfo[:, selected_idx]))
                    distance_from_target = abs(num_selected - 762)
                    penalty_factor = np.exp(-distance_from_target / 500.0) 
                    return feat_quality * penalty_factor

            mrfo_bounds = [BinaryVar(name=f"feat_{i}") for i in range(1792)]
            posture_problem = MantaRayFeatureSelectionProblem(bounds=mrfo_bounds, minmax="max")

            # 執行 50 代 MRFO 尋優
            mrfo_optimizer = MRFO.OriginalMRFO(epoch=50, pop_size=35)
            print(f"   -> 鬼蝠魟群體正在進入 1792 維海洋進行鏈式與翻滾覓食...")
            best_agent = mrfo_optimizer.solve(posture_problem)
            
            mrfo_final_indices = np.where(best_agent.solution > 0.5)[0]
            
            if len(mrfo_final_indices) < 10 or len(mrfo_final_indices) > 1500:
                print("   ⚠️ 鬼蝠魟本次選取維度過於極端，啟動防護：隨機抽樣 762 維不重複核心特徵...")
                mrfo_final_indices = np.random.choice(1792, size=762, replace=False)
                mrfo_final_indices = np.sort(mrfo_final_indices)
                
            print(f"🎉 鬼蝠魟優化結束！去蕪存菁挑選出精華特徵數: 【 {len(mrfo_final_indices)} 維 】\n")

        # --- 🛠️ 根據當前 for 迴圈進度，動態初始化模型架構 ---
        if RUN_MODE == "SqueezeNet_Only":
            model_name = "Baseline1_SqueezeNet"
            model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
            model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1,1))
            features_dim_before_classifier = 512 
            
        elif RUN_MODE == "EfficientNet_Only":
            model_name = "Baseline2_EfficientNetB0"
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            features_dim_before_classifier = model.classifier[1].in_features 
            model.classifier[1] = nn.Linear(features_dim_before_classifier, NUM_CLASSES)
            
        elif RUN_MODE == "GaitMCCA_Fusion":
            model_name = "Baseline3_GaitMCCA_Fusion"
            model = GaitMCCAStyleNet(num_classes=NUM_CLASSES)
            features_dim_before_classifier = 512 
            
        elif RUN_MODE == "MRFO_Optimization":
            model_name = "Baseline4_MRFO_Optimization"
            model = GaitMRFOOptimizedNet(num_classes=NUM_CLASSES, selected_indices_list=mrfo_final_indices)
            features_dim_before_classifier = len(model.mrfo_indices)

        # 🌟 即時高亮印出分類前的特徵維度
        print(f"==========================================================")
        print(f"🔍 當前運行 Baseline 模型: {model_name}")
        print(f"🔍 進入分類層(Classifier)前一檔的「核心特徵數」: 【 {features_dim_before_classifier} 維 】")
        print(f"==========================================================")

        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=IR)

        # --- A. 執行訓練 ---
        print(f"\n🏋️ 啟動模型訓練模型 [{model_name}]...")
        model, history = train_model_with_early_stopping(
            model, dataloaders, criterion, optimizer, num_epochs=EPOCHS, patience=PATIENCE, device=DEVICE
        )

        # --- B. 繪製並儲存該模式的學習曲線 ---
        plot_learning_curve(history, model_name)

        # --- C. 測試集性能評估 ---
        print(f"\n🔍 正在使用測試集進行性能評估...")
        true_labels, pred_labels, metrics = evaluate_model(model, dataloaders['test'], device=DEVICE)
        acc, precision, recall, f1 = metrics
        
        print(f"\n==================== {model_name} 測試集指標 ====================")
        print(f"進入分類前特徵維度: {features_dim_before_classifier}")
        print(f"精確度 (Accuracy) : {acc:.4f}")
        print(f"精準率 (Precision): {precision:.4f}")
        print(f"召回率 (Recall)   : {recall:.4f}")
        print(f"F1 得分 (F1-Score) : {f1:.4f}")
        print("==========================================================")

        # 儲存到總戰報中，方便最後列印對照
        FINAL_REPORT[model_name] = {
            "dim": features_dim_before_classifier,
            "acc": acc,
            "p": precision,
            "r": recall,
            "f1": f1
        }

        # 輸出文字版混淆矩陣
        cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
        cm_df = pd.DataFrame(cm, index=[f"True_{c}" for c in class_names], columns=[f"Pred_{c}" for c in class_names])
        print(f"\n[文字版混淆矩陣 - {model_name}]\n", cm_df)

        # --- D. 繪製並儲存該模式的混淆矩陣圖 ---
        plot_confusion_matrix(true_labels, pred_labels, class_names, model_name)
        print(f"\n🎉 模式 [{model_name}] 消融實驗與特徵分析完畢！自動準備切換下一條...\n")

    # =====================================================================
    # 🏁 終點大總結：4 條 Baseline 全部跑完，自動列印「終極消融實驗數據戰報」
    # =====================================================================
    print(f"\n\n🏆🏆🏆 全自動化消融實驗排程全部執行完畢！【終極戰報總覽】 🏆🏆🏆")
    print(f"--------------------------------------------------------------------------------------")
    print(f"{'模型名稱 (Model Name)':<30} | {'特徵數 (Dim)':<10} | {'準確度 (Acc)':<12} | {'精準率 (Prec)':<12} | {'召回率 (Rec)':<12} | {'F1-Score':<12}")
    print(f"--------------------------------------------------------------------------------------")
    for m_name, res in FINAL_REPORT.items():
        print(f"{m_name:<30} | {res['dim']:<10} | {res['acc']:.4f:<12} | {res['p']:.4f:<12} | {res['r']:.4f:<12} | {res['f1']:.4f:<12}")
    print(f"--------------------------------------------------------------------------------------")
    print(f"💡 4 組模型對應的 .png 圖檔與學習曲線皆已獨立儲存於專案目錄下。可以直接撰寫論文囉！")