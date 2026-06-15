# plotter.py
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curve(history, model_name):
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training and Verification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training and Verification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    chart_filename = f"learning_curve_{model_name}_{datetime.now().strftime('%m%d_%H%M')}.png"
    plt.savefig(chart_filename, dpi=300)
    print(f"💾 學習曲線圖已儲存為: {chart_filename}")
    plt.close()


def plot_confusion_matrix(true_labels, pred_labels, class_names, model_name):
    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    cm_filename = f"confusion_matrix_{model_name}_{datetime.now().strftime('%m%d_%H%M')}.png"
    plt.savefig(cm_filename, dpi=300)
    print(f"💾 混淆矩陣影像已儲存為: {cm_filename}")
    plt.close()