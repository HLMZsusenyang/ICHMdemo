import io, base64, os
from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import models, transforms

# ---------- 配置 ----------
# classes = [
#     'aiye', 'ajiao', 'baibiandou', 'baibu', 'baifan', 'baihe', 'baihuasheshecao', 'baikou', 'baimaogen', 'baishao',
#     'baitouweng', 'baizhu', 'baiziren', 'bajitian', 'banlangen', 'banxia', 'beishashenkuai', 'beishashentiao',
#     'biejia', 'cangzhu', 'caoguo', 'caokou', 'cebaiye', 'chaihu', 'chantui', 'chenpi', 'chenxiang', 'chishao',
#     'chishizhi', 'chongcao', 'chuanshanjia', 'chuanxinlian', 'cishi', 'dafupi', 'dangshen', 'danshen', 'daqingye',
#     'daxueteng', 'digupi', 'dilong', 'diyu', 'duzhong'
# ]

classes = [
    '艾叶', '阿胶', '白扁豆', '百部', '白矾', '百合', '白花蛇舌草', '白蔻', '白茅根', '白芍',
    '白头翁', '白术', '柏子仁', '巴戟天', '板蓝根', '半夏', '北沙参块', '北沙参条',
    '鳖甲', '苍术', '草果', '草蔻', '侧柏叶', '柴胡', '蝉蜕', '陈皮', '沉香', '赤芍',
    '赤石脂', '虫草', '穿山甲', '穿心莲', '磁石', '大腹皮', '党参', '丹参', '大青叶',
    '大血藤', '地骨皮', '地龙', '地榆', '杜仲'
]


device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 模型 ----------
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(torch.load("best_model/ResNet18-SGD_lr0.0001_bs16.pth", map_location=device))
model = model.to(device).eval()

# ---------- 预处理 ----------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict(img: Image.Image):
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0).to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]  # [C]
        top5_prob, top5_idx = torch.topk(probs, 5)
        top5_prob = top5_prob.cpu().numpy()
        top5_idx = top5_idx.cpu().numpy()
        top5 = [
            {"class": classes[i], "probability": float(f"{p:.4f}")}
            for i, p in zip(top5_idx, top5_prob)
        ]
        if top5[0]["probability"] < 0.5:
            return {"prediction": "无法识别", "top5": top5}
        else:
            return {"prediction": top5[0]["class"], "top5": top5}


# ---------- Flask ----------
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files["image"]
        print("收到文件:", file.filename)
        img = Image.open(file.stream).convert("RGB")
        result = predict(img)   # ← 原来是 label = predict(img)
        return jsonify(result)  # ← 原来是 jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": f"识别失败: {str(e)}"}), 500


# ---------- 主函数 ----------
if __name__ == "__main__":
    # 若打算线上部署，改成 app.run(host="0.0.0.0", port=80, debug=False)
    app.run(debug=True)
