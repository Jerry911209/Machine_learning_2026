import time
import copy
import torch

def train_model_with_early_stopping(model, dataloaders, criterion, optimizer, num_epochs=30, patience=5, device="cuda"):
    """
    支援早停機制 (Early Stopping) 的進階 PyTorch 訓練核心
    """
    since = time.time()
    
    # 用來記錄繪圖數據的歷史字典
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase:<5} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # 記錄歷史紀錄
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # 早停與最佳權重儲存判斷
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    patience_counter = 0  # 重置計數器
                    print(f"🌟 發現更低的 Val Loss! 最佳權重已更新。")
                else:
                    patience_counter += 1
                    print(f"⚠️ Val Loss 未改善。早停計數器: {patience_counter}/{patience}")

        # 檢查是否觸發早停
        if patience_counter >= patience:
            print(f"\n🛑 驗證集 Loss 連續 {patience} 個 Epoch 未改善，觸發早停機制！")
            break

    time_elapsed = time.time() - since
    print(f'\n訓練結束！總耗時: {time_elapsed // 60:.0f}分 {time_elapsed % 60:.0f}秒')
    print(f'最佳驗證集準確度 (Best Val Acc): {best_acc:.4f}')

    # 載入表現最好的模型權重
    model.load_state_dict(best_model_wts)
    return model, history


def evaluate_model(model, dataloader, device="cuda"):
    """
    專門用來收集預測結果與真實標籤的函數，用來計算測試集指標與混淆矩陣
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return all_labels, all_preds