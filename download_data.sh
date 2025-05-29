#!/bin/bash

# 设置下载链接
URL="https://github.com/HLMZsusenyang/ICHMdemo/releases/download/v1.0/data.zip"
ZIP_NAME="data.zip"

# 下载到当前目录
echo "📥 正在下载数据集..."
wget -O "$ZIP_NAME" "$URL"

# 解压到当前目录
echo "📂 正在解压..."
unzip -q "$ZIP_NAME" -d .

# 删除压缩包（可选）
echo "🧹 正在清理压缩包..."
rm "$ZIP_NAME"

echo "✅ 已完成：文件下载并解压到当前目录"
