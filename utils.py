# utils.py  （仅需替换/新增下面这一段，其余函数保持不变）
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os, json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns  # 可选：更漂亮的可视化



# ---------- 1. 读取类别映射 ----------
LABEL_FILE = 'data/labels.txt'          # 按需修改
with open(LABEL_FILE, encoding='utf-8') as f:
    id2cls = [line.strip().split(maxsplit=1)[1] for line in f]

# ---------- 2. 自定义数据集 ----------
class TxtDataset(Dataset):
    def __init__(self, list_file, transform=None, root='.'):
        self.samples = []               # [(img_path, label), ...]
        self.transform = transform
        self.root = root
        with open(list_file, encoding='utf-8') as f:
            for ln in f:
                p, lbl = ln.strip().split()
                self.samples.append( (os.path.join(root, 'data', p), int(lbl)) )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 3. 生成 DataLoader ----------
def get_splits(batch_size=32):
    tf_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    tf_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 训练 / 验证数据集
    full_trainset = TxtDataset('data/train.txt', transform=tf_train, root='.')
    n = len(full_trainset)
    train_len = int(n * 0.875)          # 相当于 70% 总数据
    val_len   = n - train_len           # 相当于 10% 总数据
    train_ds, val_ds = random_split(full_trainset, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))
    # 验证集用 test transform
    val_ds.dataset.transform = tf_test

    # 测试数据集
    test_ds = TxtDataset('data/test.txt', transform=tf_test, root='.')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, id2cls

def get_data(batch_size=32):
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    full_dataset = datasets.ImageFolder('dataset', transform=transform_train)
    train_len = int(0.8 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])
    val_ds.dataset.transform = transform_test
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size), full_dataset.classes

# utils.py
def train_model(model, optimizer, name, device, train_loader, val_loader, extra_info=None, log_dir="logs"):
    import datetime
    criterion = torch.nn.CrossEntropyLoss()
    history = {'train': [], 'val': []}
    y_true, y_pred = [], []

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    with open(log_path, "w", encoding="utf-8") as log:
        # --------- 1. 写入超参数信息 ---------
        log.write(f"# Training Log for model: {name}\n")
        log.write(f"# Time: {datetime.datetime.now()}\n")
        log.write(f"# Device: {device}\n")
        log.write(f"# Optimizer: {type(optimizer).__name__}\n")
        log.write(f"# Initial LR: {optimizer.param_groups[0]['lr']}\n")
        log.write(f"# Batch size: {train_loader.batch_size}\n")
        if extra_info:
            log.write("# Extra Info:\n")
            for k, v in extra_info.items():
                log.write(f"#   {k}: {v}\n")
        log.write("epoch,train_acc,val_acc,avg_loss,lr\n")

        # --------- 2. 开始训练循环 ---------
        for epoch in range(extra_info["epochs"]):
            model.train()
            train_correct = 0
            running_loss = 0.0

            for x, y in tqdm(train_loader, desc=f"[{name}] Epoch {epoch+1} Training"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)

                n_classes = out.shape[1]
                if (y < 0).any() or (y >= n_classes).any():
                    raise ValueError("检测到非法标签")

                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * x.size(0)
                train_correct += (out.argmax(1) == y).sum().item()

            acc_train = train_correct / len(train_loader.dataset)
            avg_loss = running_loss / len(train_loader.dataset)
            history['train'].append(acc_train)

            # 验证
            model.eval()
            val_correct = 0
            y_true.clear()
            y_pred.clear()
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"[{name}] Epoch {epoch+1} Validation"):
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    pred = out.argmax(1)
                    val_correct += (pred == y).sum().item()
                    y_true += y.cpu().tolist()
                    y_pred += pred.cpu().tolist()

            acc_val = val_correct / len(val_loader.dataset)
            history['val'].append(acc_val)

            # 当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:02d} | Train Acc: {acc_train:.4f} | Val Acc: {acc_val:.4f} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            log.write(f"{epoch+1},{acc_train:.4f},{acc_val:.4f},{avg_loss:.4f},{current_lr:.6f}\n")

    return history, y_true, y_pred, log_path



def plot_curves(history, name):
    plt.plot(history['train'], label="Train")
    plt.plot(history['val'], label="Val")
    plt.title(f"{name} Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"results/{name}_curve.png")
    plt.close()

def plot_confusion(y_true, y_pred, name, class_names=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))  # 自定义大图尺寸

    sns.heatmap(
        cm,
        annot=False,           # 不标数字（可改成True）
        fmt='d',
        cmap='Blues',
        xticklabels=class_names if class_names and len(class_names) <= 50 else False,
        yticklabels=class_names if class_names and len(class_names) <= 50 else False
    )

    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"results/{name}_confusion_matrix.png", dpi=300)
    plt.close()