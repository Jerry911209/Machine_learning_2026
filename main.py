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
from datetime import datetime
import matplotlib.pyplot as plt

# 💡 真正引入現成的鬼蝠魟演算法套件與機器學習分類器
from mealpy import MRFO 
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score

# 引用自訂模組
from model_trainer import train_model_with_early_stopping, evaluate_model
from plotter import plot_learning_curve, plot_confusion_matrix
# 🌟 新增：將 mealpy 的底層類別移至最頂端，徹底解決全域未定義報錯 🌟
from mealpy.utils.problem import Problem
from mealpy.utils.space import BinaryVar

# =====================================================================
# 1. Baseline 3：雙流拼接融合網絡 (GaitMCCA Style - 1024維保護層)
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

# =====================================================================
# 2. Baseline 4: 雙流特徵 + MRFO 篩選 + 標準分類頭
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
        return self.classifier(f_fused)

# =====================================================================
# 3. Baseline 6: 雙流特徵 + MRFO 篩選 + 寬深多層幾何網絡 (修復資訊瓶頸) 🌟
# =====================================================================
class GaitMRFONarrowNet(nn.Module):
    def __init__(self, num_classes=6, selected_indices_list=None):
        super(GaitMRFONarrowNet, self).__init__()
        self.squeeze_net = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
        self.squeeze_features = self.squeeze_net.features
        self.squeeze_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.efficient_net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.efficient_features = self.efficient_net.features
        self.efficient_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        if selected_indices_list is None:
            selected_indices_list = torch.linspace(0, 1791, steps=1024).long()
        else:
            selected_indices_list = torch.tensor(selected_indices_list).long()
            
        self.register_buffer('mrfo_indices', selected_indices_list)
        optimized_dim = len(selected_indices_list)
        
        # 💡 【架構修正】：放緩收窄步調，改為 512 -> 256，並在 FINAL_REPORT 展現真實隱藏層精華維度
        self.narrow_backbone = nn.Sequential(
            nn.Linear(optimized_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        f_squeeze = torch.flatten(self.squeeze_pool(self.squeeze_features(x)), 1) 
        f_efficient = torch.flatten(self.efficient_pool(self.efficient_features(x)), 1) 
        f_combined = torch.cat((f_squeeze, f_efficient), dim=1) 
        f_mrfo_selected = torch.index_select(f_combined, dim=1, index=self.mrfo_indices)
        f_narrowed = self.narrow_backbone(f_mrfo_selected)
        return self.classifier(f_narrowed)

# =====================================================================
# 4. Dataset 類別定義
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
# 5. 主執行程序 (全自動化微調特徵再尋優完全體)
# =====================================================================
if __name__ == "__main__":
    PROJECT_ROOT_DIR = os.getcwd()
    total_loop_cnt=2
    
    for loop_cnt in range(1, total_loop_cnt):
        os.chdir(PROJECT_ROOT_DIR)
        print("\n" + "="*60)
        print(f"🔄 【第 {loop_cnt} / {total_loop_cnt} 次大型獨立重複實驗】正式啟動")
        print("="*60 + "\n")
        
        # 💡 【自由控制開關區】
        ACTIVE_RUN_LIST = [
            # "SqueezeNet_Only", 
            # "EfficientNet_Only", 
            "GaitMCCA_Fusion", 
            "MRFO_Optimization",
            "MRFO_SVM",          
            "MRFO_NarrowNet"     
        ]
        
        FINAL_REPORT = {}
        BATCH_SIZE = 64
        NUM_CLASSES = 6
        EPOCHS = 300
        IR = 1e-4
        PATIENCE = 25  
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 當前排程硬體: {DEVICE} | 本輪排程清單: {ACTIVE_RUN_LIST}")

        current_time_str = datetime.now().strftime("%Y%m%d_%H%M")
        OUTPUT_RESULT_DIR = f"./ablation_results_Loop{loop_cnt}_{current_time_str}"
        os.makedirs(OUTPUT_RESULT_DIR, exist_ok=True)
        print(f"📁 建立本次成果歸檔夾: {OUTPUT_RESULT_DIR}")

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
        class_names = ["Standing", "Sitting", "Lying", "Bending", "Crawling", "Empty"]

        # =====================================================================
        # 🔄 【第一階段：智慧防護判定機制】
        # =====================================================================
        mrfo_final_indices = None
        need_mrfo = any("MRFO" in mode for mode in ACTIVE_RUN_LIST)
        
        if need_mrfo:
            print(f"\n🌟 [學術重構核心] 偵測到排程包含 MRFO 家族，啟動第一階段雙流網絡微調打底...")
            pre_trainer = GaitMCCAStyleNet(num_classes=NUM_CLASSES).to(DEVICE)
            pre_criterion = nn.CrossEntropyLoss()
            pre_optimizer = optim.Adam(pre_trainer.parameters(), lr=IR)
            
            pre_trainer, _ = train_model_with_early_stopping(
                pre_trainer, dataloaders, pre_criterion, pre_optimizer, 
                num_epochs=EPOCHS, patience=PATIENCE, device=DEVICE
            )
            pre_trainer.eval()
            
            print(f"🌊 [第二階段：黃金特徵提取] 提取 300 筆經醫學姿勢微調後的 1792 維頂級融合特徵快取...")
            temp_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            cached_features, cached_labels = [], []
            with torch.no_grad():
                for idx, (inputs, labels) in enumerate(temp_loader):
                    if idx >= 1000: break #快取樣本數提升到 1000 筆
                    inputs = inputs.to(DEVICE)
                    f_s = torch.flatten(pre_trainer.squeeze_pool(pre_trainer.squeeze_features(inputs)), 1)
                    f_e = torch.flatten(pre_trainer.efficient_pool(pre_trainer.efficient_features(inputs)), 1)
                    f_comb = torch.cat((f_s, f_e), dim=1).cpu().numpy()
                    cached_features.append(f_comb[0])
                    cached_labels.append(labels.numpy()[0])
                    
            X_train_mrfo = np.array(cached_features)
            y_train_mrfo = np.array(cached_labels)
            
            split_val = int(len(X_train_mrfo) * 0.3)
            X_eval_train, X_eval_val = X_train_mrfo[split_val:], X_train_mrfo[:split_val]
            y_eval_train, y_eval_val = y_train_mrfo[split_val:], y_train_mrfo[:split_val]
            
            print(f"🌊 [第三階段：MRFO 有監督尋優] 鬼蝠魟群體正式下海篩選最優分類子集...")
            class MantaRayFeatureSelectionProblem(Problem):
                def __init__(self, bounds, minmax="max", **kwargs):
                    super().__init__(bounds=bounds, minmax=minmax, **kwargs)
                def obj_func(self, x):
                    selected_idx = np.where(x > 0.5)[0]
                    num_selected = len(selected_idx)
                    if num_selected < 10 or num_selected > 1500: return 0.0 
                    try:
                        clf = LinearSVC(dual=False, random_state=42, max_iter=1000)
                        clf.fit(X_eval_train[:, selected_idx], y_eval_train)
                        preds = clf.predict(X_eval_val[:, selected_idx])
                        val_acc = accuracy_score(y_eval_val, preds)
                        # 💡 【特徵數提高修正】：引導目標特徵數往更豐沛的 1024 維靠攏
                        distance_penalty = np.exp(-abs(num_selected - 1024) / 1000.0)
                        return (val_acc * 0.99) + (distance_penalty * 0.01)
                    except:
                        return 0.0

            mrfo_bounds = [BinaryVar(name=f"feat_{i}") for i in range(1792)]
            posture_problem = MantaRayFeatureSelectionProblem(bounds=mrfo_bounds, minmax="max")

            mrfo_optimizer = MRFO.OriginalMRFO(epoch=50, pop_size=35)
            best_agent = mrfo_optimizer.solve(posture_problem)
            mrfo_final_indices = np.where(best_agent.solution > 0.5)[0]
            
            if len(mrfo_final_indices) < 10 or len(mrfo_final_indices) > 1500:
                mrfo_final_indices = np.random.choice(1792, size=1024, replace=False)
                mrfo_final_indices = np.sort(mrfo_final_indices)
            print(f"🎉 鬼蝠魟篩選完成！成功從微調空間精選出特徵數: 【 {len(mrfo_final_indices)} 維 】\n")
        else:
            mrfo_final_indices = np.linspace(0, 1791, steps=1024).astype(int)

        # =====================================================================
        # 🔄 消融排程動態執行迴圈
        # =====================================================================
        for step_idx, RUN_MODE in enumerate(ACTIVE_RUN_LIST, 1):
            os.chdir(PROJECT_ROOT_DIR)
            print(f"\n🎬 [Loop {loop_cnt}/4] 正在執行第 [{step_idx}/{len(ACTIVE_RUN_LIST)}] 項: {RUN_MODE}")
            
            # --- 模式 A: 傳統 PyTorch 端到端網路模型程序 ---
            if RUN_MODE in ["SqueezeNet_Only", "EfficientNet_Only", "GaitMCCA_Fusion", "MRFO_Optimization", "MRFO_NarrowNet"]:
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
                    features_dim_before_classifier = 1024 
                elif RUN_MODE == "MRFO_Optimization":
                    model_name = "Baseline4_MRFO_Optimization"
                    model = GaitMRFOOptimizedNet(num_classes=NUM_CLASSES, selected_indices_list=mrfo_final_indices)
                    features_dim_before_classifier = len(model.mrfo_indices)
                elif RUN_MODE == "MRFO_NarrowNet":
                    model_name = "Baseline6_MRFO_NarrowNet"
                    model = GaitMRFONarrowNet(num_classes=NUM_CLASSES, selected_indices_list=mrfo_final_indices)
                    features_dim_before_classifier = 256  # 💡 更新為 256 維，修復瓶頸
                
                print(f"⚙️ 架構: {model_name} | 分類前特徵維度: {features_dim_before_classifier} 維")
                model = model.to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=IR)

                model, history = train_model_with_early_stopping(
                    model, dataloaders, criterion, optimizer, num_epochs=EPOCHS, patience=PATIENCE, device=DEVICE
                )
                
                abs_w_path = os.path.abspath(os.path.join(OUTPUT_RESULT_DIR, f"best_model_{model_name}.pth"))
                torch.save(model.state_dict(), abs_w_path)
                
                os.chdir(os.path.abspath(OUTPUT_RESULT_DIR))
                try: plot_learning_curve(history, model_name)
                except Exception as e: print(f"⚠️ 學習曲線儲存失敗: {e}")
                os.chdir(PROJECT_ROOT_DIR)

                true_labels, pred_labels, metrics = evaluate_model(model, dataloaders['test'], device=DEVICE)
                acc, precision, recall, f1 = metrics

            # --- 模式 B: MRFO + 傳統 SVM 分類頭模式 ---
            elif RUN_MODE == "MRFO_SVM":
                model_name = "Baseline5_MRFO_SVM"
                features_dim_before_classifier = len(mrfo_final_indices)
                print(f"⚙️ 架構: {model_name} | 特徵維度: {features_dim_before_classifier} 維 (微調特徵對接 RBF-SVM)")
                
                print("   -> 正在使用微調後的雙流網路提取完整數據集的特徵矩陣...")
                
                def extract_all_features(loader):
                    all_feats, all_labs = [], []
                    with torch.no_grad():
                        for inputs, labels in loader:
                            inputs = inputs.to(DEVICE)
                            f_s = torch.flatten(pre_trainer.squeeze_pool(pre_trainer.squeeze_features(inputs)), 1)
                            f_e = torch.flatten(pre_trainer.efficient_pool(pre_trainer.efficient_features(inputs)), 1)
                            f_comb = torch.cat((f_s, f_e), dim=1).cpu().numpy()
                            all_feats.append(f_comb)
                            all_labs.append(labels.numpy())
                    return np.vstack(all_feats), np.concatenate(all_labs)
                
                X_train_all, y_train_all = extract_all_features(dataloaders['train'])
                X_test_all, y_test_all = extract_all_features(dataloaders['test'])
                
                X_train_sliced = X_train_all[:, mrfo_final_indices]
                X_test_sliced = X_test_all[:, mrfo_final_indices]
                
                print("   -> 正在擬合支持向量機 (SVC) 分類頭...")
                svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
                svm_model.fit(X_train_sliced, y_train_all)
                
                pred_labels = svm_model.predict(X_test_sliced)
                true_labels = y_test_all
                
                from sklearn.metrics import precision_score, recall_score, f1_score
                acc = accuracy_score(true_labels, pred_labels)
                precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
                recall = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
                f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0)
                
                history = {'train_loss': [], 'val_loss': []}

            FINAL_REPORT[model_name] = {
                "dim": features_dim_before_classifier, "acc": acc, "p": precision, "r": recall, "f1": f1
            }

            os.chdir(os.path.abspath(OUTPUT_RESULT_DIR))
            try: plot_confusion_matrix(true_labels, pred_labels, class_names, model_name)
            except Exception as e: print(f"⚠️ 混淆矩陣圖儲存失敗: {e}")
            os.chdir(PROJECT_ROOT_DIR)
            print(f"🎉 模式 [{model_name}] 成果已歸檔！")

        # =====================================================================
        # 🏁 終點大總結與自動化繪製統計對比圖
        # =====================================================================
        print(f"\n\n🏆🏆🏆 [Loop {loop_cnt}/4] 消融實驗排程全部執行完畢！【第 {loop_cnt} 輪數據總覽】 🏆🏆🏆")
        print(f"--------------------------------------------------------------------------------------")
        print(f"{'模型名稱 (Model Name)':<30} | {'特徵數 (Dim)':<10} | {'準確度 (Acc)':<12} | {'精準率 (Prec)':<12} | {'召回率 (Rec)':<12} | {'F1-Score':<12}")
        print(f"--------------------------------------------------------------------------------------")
        for m_name, res in FINAL_REPORT.items():
            print(f"{m_name:<30} | {res['dim']:<10} | {res['acc']:<12.4f} | {res['p']:<12.4f} | {res['r']:<12.4f} | {res['f1']:<12.4f}")
        print(f"--------------------------------------------------------------------------------------")

        print(f"\n📊 正在自動繪製 [Loop {loop_cnt}] 多指標綜合統計圖...")
        try:
            models_keys = list(FINAL_REPORT.keys())
            metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            data_matrix = []
            for m_name, res in FINAL_REPORT.items():
                data_matrix.append([res['acc'], res['p'], res['r'], res['f1']])
            data_matrix = np.array(data_matrix)
            
            x = np.arange(len(metrics_labels))
            width = 0.12 
            
            fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
            colors = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94']
            
            for i, m_name in enumerate(models_keys):
                ax.bar(x + (i - len(models_keys)/2 + 0.5) * width, data_matrix[i], width, label=f"{m_name} (Dim: {FINAL_REPORT[m_name]['dim']})", color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
                
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Ablation Study [Loop {loop_cnt}]: Two-Stage Fine-tuned Feature Selection Summary', fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(metrics_labels, fontsize=11, fontweight='bold')
            ax.set_ylim(0, 1.2)
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9, frameon=True)
            plt.tight_layout()
            
            os.chdir(os.path.abspath(OUTPUT_RESULT_DIR))
            plt.savefig("ablation_metrics_comparison.png", bbox_inches='tight')
            plt.close()
            os.chdir(PROJECT_ROOT_DIR)
            print(f"🎨 [總對比圖繪製成功] 已安全儲存至子資料夾內！")
        except Exception as e:
            os.chdir(PROJECT_ROOT_DIR)
            print(f"⚠️ 繪製統計圖時發生錯誤: {e}")

        print(f"\n💡 [Loop {loop_cnt}] 本輪指定實驗數據與總對比圖已安全儲存至：【 {OUTPUT_RESULT_DIR} 】\n")

    print("\n" + "#"*60)
    print("🏆🏆🏆 【雙階段微調特徵優化消融實驗連跑排程】已全自動化完美通關！")
    print("#"*60)