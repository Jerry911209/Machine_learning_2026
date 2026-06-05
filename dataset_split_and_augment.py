import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import Counter
from torch.utils.data import Dataset

# =====================================================================
# 1. ⚙️ 使用者自訂：各分類的擴增「總倍數」設定
# =====================================================================
AUGMENT_MULTIPLIERS = {
    1: 1,  # Standing (原始類別 1) -> 保持 1 倍 (不增強)
    2: 1,  # Sitting  (原始類別 2) -> 保持 1 倍 (不增強)
    3: 1,  # Lying    (原始類別 3) -> 保持 1 倍 (不增強)
    4: 3,  # Bending  (原始類別 4) -> 擴增至 5 倍
    5: 3,  # Crawling (原始類別 5) -> 擴增至 8 倍
    6: 3  # Empty    (原始類別 6) -> 擴增至 10 倍
}

# =====================================================================
# 2. 📊 統計專用 Dataset 與 函數 (跟 main.py 保持完全一致)
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

    def __init__(self, label_path):
        if label_path.endswith('.csv'):
            self.labels_df = pd.read_csv(label_path)
        elif label_path.endswith('.xlsx') or label_path.endswith('.xls'):
            self.labels_df = pd.read_excel(label_path)
        else:
            raise ValueError("標籤檔案格式必須是 CSV 或 Excel 檔案")
        self.raw_labels = self.labels_df['class'].values

    def __len__(self):
        return len(self.labels_df)

def pprint_dataset_stats(dataset, name):
    """精美列印並統計數據集中的姿勢類別分布"""
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
# 3. 路徑設定 (全面採用相對路徑)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
OLD_DATA_DIR = BASE_DIR / 'Posture_train_valdidate_test'
NEW_DATA_DIR = BASE_DIR / 'Posture_New_Split'  # 新的資料夾名稱

# 定義要建立的新資料夾結構
for phase in ['Posture_train', 'Posture_valdidate', 'Posture_test']:
    (NEW_DATA_DIR / phase / 'rgb').mkdir(parents=True, exist_ok=True)

print("📁 新資料夾結構已建立。")

# =====================================================================
# 4. 讀取並合併舊的 Train 與 Val 資料
# =====================================================================
print("\n正在讀取並合併舊的 Train 與 Val 標籤...")
old_train_df = pd.read_csv(OLD_DATA_DIR / 'Posture_train' / 'labels.csv')
old_val_df = pd.read_csv(OLD_DATA_DIR / 'Posture_valdidate' / 'labels.csv')

old_train_df['source_phase'] = 'Posture_train'
old_val_df['source_phase'] = 'Posture_valdidate'

combined_df = pd.concat([old_train_df, old_val_df], ignore_index=True)

# =====================================================================
# 5. 重新拆分 6:4 (Train : Val)
# =====================================================================
print("\n正在依 6:4 比例重新劃分訓練集與驗證集...")
new_train_df, new_val_df = train_test_split(
    combined_df, 
    test_size=0.4, 
    random_state=42, 
    stratify=combined_df['class']
)

print(f"新 🚀 訓練集 (Train) 原始張數: {len(new_train_df)} (60%)")
print(f"新 🧪 驗證集 (Val) 原始張數: {len(new_val_df)} (40%)")

# =====================================================================
# 6. 定義指定的「影像資料增強 (Data Augmentation)」
# =====================================================================
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),                  # 隨機水平翻轉
    transforms.RandomRotation(degrees=10),                    # 10度以內隨機旋轉
    #transforms.RandomPerspective(distortion_scale=0.2, p=1.0) # 隨機透視變形
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)) # 💡 改用 5% 內的輕微平移，取代透視變形！
])

# =====================================================================
# 7. 開始搬移、複製與依倍數隨機資料增強
# =====================================================================
def process_phase(df, target_phase, is_train=False):
    """將圖片分配到新資料夾。如果是訓練集，會根據自訂倍數進行多次不同的隨機增強"""
    new_labels = []
    new_img_counter = 1  # 新資料夾的圖片從 rgb_0001.png 開始重新編號
    
    print(f"\n正在處理 {target_phase} 的圖片複製與自訂倍數擴增...")
    
    for _, row in df.iterrows():
        orig_idx = int(row['index'])
        orig_class = int(row['class'])
        source_phase = row['source_phase']
        
        old_img_name = f"rgb_{orig_idx:04d}.png"
        old_img_path = OLD_DATA_DIR / source_phase / 'rgb' / old_img_name
        
        if old_img_path.exists():
            img = Image.open(old_img_path)
            
            # --- 情況 A：如果是訓練集，根據自訂倍數處理 ---
            if is_train:
                multiplier = AUGMENT_MULTIPLIERS.get(orig_class, 1)
                
                # 1. 先存一張原圖
                new_img_name_orig = f"rgb_{new_img_counter:04d}.png"
                new_img_path_orig = NEW_DATA_DIR / target_phase / 'rgb' / new_img_name_orig
                img.save(new_img_path_orig)
                
                new_labels.append({'index': new_img_counter, 'class': orig_class})
                new_img_counter += 1
                
                # 2. 根據剩餘倍數，生成多次隨機增強圖
                for _ in range(multiplier - 1):
                    new_img_name_aug = f"rgb_{new_img_counter:04d}.png"
                    new_img_path_aug = NEW_DATA_DIR / target_phase / 'rgb' / new_img_name_aug
                    
                    img_augmented = augment_transform(img)
                    img_augmented.save(new_img_path_aug)
                    
                    new_labels.append({'index': new_img_counter, 'class': orig_class})
                    new_img_counter += 1
                    
            # --- 情況 B：如果是驗證集，不做增強，只存原圖 ---
            else:
                new_img_name = f"rgb_{new_img_counter:04d}.png"
                new_img_path = NEW_DATA_DIR / target_phase / 'rgb' / new_img_name
                img.save(new_img_path)
                
                new_labels.append({'index': new_img_counter, 'class': orig_class})
                new_img_counter += 1
        else:
            print(f"⚠️ 找不到檔案: {old_img_path}")

    new_df = pd.DataFrame(new_labels)
    new_df.to_csv(NEW_DATA_DIR / target_phase / 'labels.csv', index=False)
    print(f"✨ {target_phase} 實體圖片與標籤處理完畢！")

# 執行新 Train 與新 Val
process_phase(new_train_df, 'Posture_train', is_train=True)
process_phase(new_val_df, 'Posture_valdidate', is_train=False)

# --- 測試集（Test）完整複製過去 ---
print("\n正在將測試集 (Test) 完整複製到新資料夾（保持不動）...")
old_test_rgb_dir = OLD_DATA_DIR / 'Posture_test' / 'rgb'
new_test_rgb_dir = NEW_DATA_DIR / 'Posture_test' / 'rgb'

for img_name in os.listdir(old_test_rgb_dir):
    shutil.copy(old_test_rgb_dir / img_name, new_test_rgb_dir / img_name)

shutil.copy(OLD_DATA_DIR / 'Posture_test' / 'labels.csv', NEW_DATA_DIR / 'Posture_test' / 'labels.csv')
print("✨ 測試集複製完畢。")


# =====================================================================
# 8. 📊 [新增] 執行最終生成數據集的精美分類統計
# =====================================================================
print("\n" + "="*25 + " 最終新數據集結構統計 " + "="*25)

# 讀取剛剛產生的新 labels 進行統計

new_train_dataset = MedicalImageDataset(str(NEW_DATA_DIR / 'Posture_train' / 'labels.csv'))
new_val_dataset = MedicalImageDataset(str(NEW_DATA_DIR / 'Posture_valdidate' / 'labels.csv'))
new_test_dataset = MedicalImageDataset(str(NEW_DATA_DIR / 'Posture_test' / 'labels.csv'))

pprint_dataset_stats(new_train_dataset, "訓練集 (Train) 已完成擴增")
pprint_dataset_stats(new_val_dataset, "驗證集 (Val) 無擴增")
pprint_dataset_stats(new_test_dataset, "測試集 (Test) 原始狀態")

print("\n🎉 所有步驟與最終數據統計已成功執行完畢！")