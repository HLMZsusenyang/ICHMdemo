import os, itertools, torch, torch.nn as nn
from torchvision import models
from nadamw import NAGAdamW
from utils import (get_splits, train_model,
                   plot_curves, plot_confusion)
import matplotlib.pyplot as plt

# ---------- 1. 优化器工厂 ----------
def make_optimizer(opt_name, params, lr):
    if opt_name == "SGD":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-2)
    if opt_name == "NAGAdamW":
        return NAGAdamW(params, lr=lr, weight_decay=1e-2)
    raise ValueError(f"Unknown optimizer {opt_name}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("search",      exist_ok=True)   # 日志目录
os.makedirs("best_model",  exist_ok=True)   # 权重目录
os.makedirs("results",     exist_ok=True)   # 曲线 & 混淆矩阵

# ---------- 2. 搜索空间 ----------
optimizers = ["SGD", "NAGAdamW"]
lrs        = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes= [16, 32, 64, 128]
EPOCHS     = 25

# ---------- 3. 网格搜索 ----------
for opt_name, lr, bs in itertools.product(optimizers, lrs, batch_sizes):
    tag   = f"ResNet18-{opt_name}_lr{lr}_bs{bs}"
    print(f"\n=== {tag} ===")

    # 3.1 数据
    train_loader, val_loader, test_loader, class_names = get_splits(batch_size=bs)

    # 3.2 模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 42)  # 修改类数
    model = model.to(device)

    # 3.3 优化器
    optimizer = make_optimizer(opt_name, model.parameters(), lr)

    # 3.4 训练
    history, y_true, y_pred, log_path = train_model(
        model, optimizer, name=tag, device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        extra_info={"epochs": EPOCHS,
                    "batch_size": bs,
                    "augmentation": "flip+rotate",
                    "notes": "grid search"},
        log_dir="search"
    )

    # 3.5 测试集评估
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            test_correct += (pred == y).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    print(f"[{tag}] Test Acc: {test_acc:.4f}")

    # 3.6 写入 log
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[TEST] Final Test Accuracy: {test_acc:.4f}\n")

    # 3.7 保存权重
    weight_path = f"best_model/{tag}.pth"
    torch.save(model.state_dict(), weight_path)

    # 3.8 可视化
    plot_curves(history, tag)
    plot_confusion(y_true, y_pred, tag, class_names=class_names)
