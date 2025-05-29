from torchvision import models
from utils import train_model, plot_curves, plot_confusion, get_splits
from nadamw import NAGAdamW
import torch.nn as nn
import torch
import os
import matplotlib.pyplot as plt


optimizers = {
    "Adam": lambda params, **kwargs: torch.optim.Adam(params, **kwargs),
    "AdamW": lambda params, **kwargs: torch.optim.AdamW(params, **kwargs),
    "SGD": lambda params, **kwargs: torch.optim.SGD(params, momentum=0.9, **kwargs),
    "NAGAdamW": lambda params, **kwargs: NAGAdamW(params, **kwargs),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)

batch_size = 32

train_loader, val_loader, test_loader, class_names = get_splits(batch_size = batch_size)

for name, opt_class in optimizers.items():
    print(f"\n===== Optimizer: {name} =====")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 42)  # 类别数
    model = model.to(device)

    optimizer = opt_class(model.parameters(), lr=1e-4, weight_decay=1e-2)
    history, y_true, y_pred, log_path = train_model(model, optimizer, name, device, train_loader, val_loader, extra_info={"epochs": 25, "augmentation": "flip+rotate", "notes": "baseline run", "batch_size": batch_size})


    print("开始测试集推理...", flush=True)
    print(f"测试集大小: {len(test_loader.dataset)}")

    # 测试集推理（完全不与训练/验证重合）
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(1)
            test_correct += (pred == y).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    print(f"[{name}] 测试集准确率: {test_acc:.4f}")
    
    if log_path:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[TEST] Final Test Accuracy: {test_acc:.4f}\n")

    plot_curves(history, name)
    plot_confusion(y_true, y_pred, name)