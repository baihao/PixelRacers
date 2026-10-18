# HF-Net weights (ONNX)

Paper: [From Coarse to Fine: Robust Hierarchical Localization at Large Scale](https://arxiv.org/abs/1812.03506)  
Official code: [ethz-asl/hfnet](https://github.com/ethz-asl/hfnet) (TensorFlow 1.x)

Expected file in this directory:

```text
models/hfnet.onnx
```

## Obtaining ONNX

1. Download the **HF-Net SavedModel** from the paper project / author mirrors (ETH ASL links may move; find a current dump).
2. Convert with [tf2onnx](https://github.com/onnx/tensorflow-onnx) or [Irwin-Liu/hfnet-tf2onnx](https://github.com/Irwin-Liu/hfnet-tf2onnx):

```bash
python -m tf2onnx.convert --opset 11 --fold_const \
  --saved-model /path/to/hfnet_saved_model \
  --output models/hfnet.onnx
```

3. Install inference runtime:

```bash
pip install onnxruntime   # or onnxruntime-gpu
```

## Notebook / pipeline backends

| `BACKEND` | Meaning |
|-----------|---------|
| `sp_nv_nn` | **Default**: SuperPoint + NetVLAD + **mutual NN** (paper-style matching, not monolithic HF-Net) |
| `hfnet_onnx` | Full HF-Net ONNX (requires `hfnet.onnx` here) |

`sp_nv_nn` validates NN matching speedups from the paper; end-to-end MobileNet-style speed needs `hfnet_onnx` plus HF-Net-rebuilt maps.
