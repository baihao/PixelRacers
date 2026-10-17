# HF-Net 权重（ONNX）

论文：[From Coarse to Fine: Robust Hierarchical Localization at Large Scale](https://arxiv.org/abs/1812.03506)  
官方代码：[ethz-asl/hfnet](https://github.com/ethz-asl/hfnet)（TensorFlow 1.x）

本目录期望文件：

```text
models/hfnet.onnx
```

## 如何得到 ONNX

1. 从论文项目页 / 作者提供的 mirror 下载 **HF-Net SavedModel**（ETH ASL 原链接曾迁移，需自行查找当前 dump）。
2. 用 [tf2onnx](https://github.com/onnx/tensorflow-onnx) 或 [Irwin-Liu/hfnet-tf2onnx](https://github.com/Irwin-Liu/hfnet-tf2onnx) 转换：

```bash
python -m tf2onnx.convert --opset 11 --fold_const \
  --saved-model /path/to/hfnet_saved_model \
  --output models/hfnet.onnx
```

3. 安装推理依赖：

```bash
pip install onnxruntime   # 或 onnxruntime-gpu
```

## Notebook 后端

| `BACKEND` | 含义 |
|---|---|
| `sp_nv_nn` | **默认可跑**：SuperPoint + NetVLAD + **mutual NN**（论文匹配方式，但非单体 HF-Net） |
| `hfnet_onnx` | 真正的 HF-Net ONNX（需本目录 `hfnet.onnx`） |

`sp_nv_nn` 仍可验证论文里「NN 替代 LightGlue」带来的加速；特征提取的 MobileNet 一体加速需 `hfnet_onnx`。
