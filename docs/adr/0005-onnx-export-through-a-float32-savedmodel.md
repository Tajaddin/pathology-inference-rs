# 0005. ONNX export through a float32 SavedModel

Date: 2026-10-18. Status: accepted.

## Context

The model trains in mixed precision (float16 activations) on the GPU and is served on CPU by ONNX Runtime from
Rust. The first export path, `tf2onnx.convert.from_keras`, and the next two tried (`from_function` on a traced
`tf.function`, and Keras 3's own `model.export(format="onnx")`) all produced an ONNX graph with two extra inputs:
EfficientNetV2's `Normalization` layer keeps its ImageNet mean and variance as captured tensors, and those
converters turned the captures into graph inputs. ONNX Runtime then refused to run on images alone.

## Decision

- Rebuild the network in float32 and copy the trained weights into it (float16 casts only cost time on CPU).
- Export that model as a TensorFlow SavedModel, then convert the SavedModel with the `tf2onnx` command line
  (opset 17). The constants are frozen into the graph and the ONNX model has one input, `image`.
- Check the result on real test patches every time: ONNX Runtime against the trained TensorFlow model, batched
  (a single 2,048-patch call ran the 8 GB GPU out of memory during the first full run).
- A unit test exports an untrained network and requires agreement within 1e-4 in probability.

## Consequences

- The export needs a temporary directory and a subprocess; it adds about 20 seconds.
- The parity numbers compare the float16-trained TensorFlow model with the float32 ONNX graph, so they include
  the precision difference, not only converter error.
