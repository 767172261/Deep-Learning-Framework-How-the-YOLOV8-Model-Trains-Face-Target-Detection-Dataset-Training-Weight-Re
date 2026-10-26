人脸目标检测数据集
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/88072035176f47179802e53623a47bb1.png)
YOLO格式数据集]
名称：人脸目标检测数据集
数量：12900张图像
类别数量：
类别类型：
用途：目标检测、分类分割、多模态


标注格式：xml和txt格式 yolo可直接训练
标签：face
标注软件：labelimg
训练集train 12880张有标签txt和xml）
验证集val 3226张（有标签txt和xml）
测试集test 16097张（没有标签）
适用范围：绝大部分目标检测任务


以下是针对 **人脸目标检测 YOLO 数据集**（12,900 张图像，1 类 `face`，YOLO 格式）的 **完整、可直接运行的 YOLOv8 训练代码**，包含环境配置、训练脚本、评估命令、推理示例及优化建议。

---

## ✅ 一、环境准备

### 1. 安装依赖（Python ≥ 3.8）
```bash
# 安装 Ultralytics YOLOv8（官方推荐）
pip install ultralytics opencv-python numpy matplotlib tqdm

# 可选：升级 PyTorch（若需 GPU 加速）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
```

> ✅ 支持 CPU / GPU 自动切换

---

## ✅ 二、数据集目录结构（标准 YOLO 格式）

确保你的数据组织如下：

```
face_detection_dataset/
├── images/
│   ├── train/      # e.g., 10320 张
│   ├── val/        # e.g., 1290 张
│   └── test/       # e.g., 1290 张
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml       # 配置文件（见下方）
```

---

## ✅ 三、`data.yaml` 配置文件

创建 `face_detection_dataset/data.yaml`：

```yaml
# face_detection_dataset/data.yaml
train: ./images/train
val: ./images/val
test: ./images/test

nc: 1
names: ['face']
```

> ⚠️ 路径使用 **相对路径**，与训练脚本同级或指定绝对路径。

---

## ✅ 四、YOLOv8 完整训练脚本（Python）

保存为 `train_face_detection.py`：

```python
# train_face_detection.py
from ultralytics import YOLO
import torch

# 自动选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 使用设备: {device}")

# 加载预训练模型（推荐 yolov8n 或 yolov8s）
# yolov8n: 更快，适合边缘部署；yolov8s: 精度更高
model = YOLO('yolov8n.pt')  # 首次运行会自动下载权重

# 开始训练
results = model.train(
    data='face_detection_dataset/data.yaml',  # 数据配置路径
    epochs=100,                               # 建议 80~150 轮（人脸收敛快）
    imgsz=640,                                # 输入尺寸（可尝试 1280 提升小脸召回）
    batch=64,                                 # 根据 GPU 显存调整（RTX 3060 可设 64）
    name='face_yolov8n',                      # 实验名称
    device=device,
    
    # 数据增强（针对人脸特点优化）
    hsv_h=0.015,     # 色调扰动（模拟肤色变化）
    hsv_s=0.5,       # 饱和度
    hsv_v=0.4,       # 亮度（应对光照变化）
    degrees=10.0,    # 旋转 ±10°（模拟头部姿态）
    translate=0.1,
    scale=0.5,
    mosaic=1.0,      # ✅ 强烈推荐！提升小脸和密集人脸检测
    flipud=0.0,      # 不上下翻转（人脸通常不倒置）
    fliplr=0.5,      # 左右翻转（对称性合理）
    
    # 训练策略
    patience=20,     # 早停：20 轮无提升则停止
    save_period=10,  # 每 10 轮保存一次 checkpoint
    workers=8        # 数据加载线程数
)

print("✅ 训练完成！最佳模型路径：")
print(f"   runs/detect/face_yolov8n/weights/best.pt")
```

---

## ✅ 五、命令行训练（快速启动）

```bash
yolo detect train \
  data=face_detection_dataset/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=64 \
  name=face_yolov8n \
  device=0
```

---

## ✅ 六、模型评估

```bash
# 在测试集上评估性能
yolo detect val \
  model=runs/detect/face_yolov8n/weights/best.pt \
  data=face_detection_dataset/data.yaml
```

> 输出将包含 **mAP@0.5、Precision、Recall**，高质量人脸数据集通常可达：
> - mAP@0.5: **95%+**
> - Recall: **>97%**（得益于高密度标注）

---

## ✅ 七、推理示例（单图 + 视频）

### 单图检测
```python
from ultralytics import YOLO

model = YOLO('runs/detect/face_yolov8n/weights/best.pt')
results = model.predict(source='group_photo.jpg', conf=0.3, save=True)
```

### 实时摄像头检测
```python
model = YOLO('best.pt')
results = model.predict(source=0, show=True, conf=0.4)  # source=0 表示默认摄像头
```

---

## ✅ 八、针对人脸检测的优化建议

| 问题 | 解决方案 |
|------|--------|
| **小人脸漏检** | • 使用 `imgsz=1280`• 启用 `mosaic=1.0`• 尝试 YOLOv8 的 P2 头（需自定义 YAML） |
| **密集人脸重叠** | • 降低 NMS IoU 阈值：推理时加 `iou=0.45` |
| **侧脸/遮挡脸** | • 数据集中应包含此类样本• 增强中加入 `degrees=15` |
| **部署到手机/嵌入式** | • 导出 ONNX/TensorRT：`model.export(format='onnx')` |

---

## ✅ 九、预期效果

- **训练时间**：RTX 3060 上约 1~2 小时（100 轮）
- **推理速度**：
  - YOLOv8n @ 640x640：**~120 FPS**（GPU）
  - YOLOv8n @ 320x320：**>200 FPS**（适合移动端）
- **典型应用**：手机美颜、安防监控、会议系统人脸追踪

---


