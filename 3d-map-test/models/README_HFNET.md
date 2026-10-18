# HF-Net weights (ONNX)

Paper: [From Coarse to Fine: Robust Hierarchical Localization at Large Scale](https://arxiv.org/abs/1812.03506)  
Official code: [ethz-asl/hfnet](https://github.com/ethz-asl/hfnet) (TensorFlow 1.x)

Expected file in this directory:

```text
models/hfnet.onnx
```

## Obtaining ONNX

1. Download the **HF-Net SavedModel** from the paper project page / author mirrors (ETH ASL links may have moved; locate a current dump).
2. Convert with [tf2onnx](https://github.com/onnx/tensorflow-onnx) or [Irwin-Liu/hfnet-tf2onnx](https://github.com/Irwin-Liu/hfnet-tf2onnx):

```bash
python -m tf2onnx.convert --opset 11 --fold_const \
  --saved-model /path/to/hfnet_saved_model \
  --output models/hfnet.onnx
```

3. Install the inference runtime:

```bash
pip install onnxruntime   # or onnxruntime-gpu
```

## Notebook backends

| `BACKEND` | Meaning |
|---|---|
| `sp_nv_nn` | **Default**: SuperPoint + NetVLAD + **mutual NN** (paper-style matching, not monolithic HF-Net) |
| `hfnet_onnx` | Full HF-Net ONNX (requires `hfnet.onnx` here) |

`sp_nv_nn` can still validate the paper’s “NN instead of LightGlue” speedup; end-to-end MobileNet-style feature extraction needs `hfnet_onnx`.
