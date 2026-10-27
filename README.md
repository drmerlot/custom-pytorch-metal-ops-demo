# custom-pytorch-metal-ops-demo

Custom PyTorch operators for Apple silicon, written as Metal GPU kernels and wired into PyTorch through a C++
extension. A worked example of the whole path from a `.metal` shader to an `nn.Module` with a working backward pass.

## How it works

1. **Metal kernels** (`custom/metal/`): matrix multiply, matrix add (with row broadcasting), vector add, ReLU and
   leaky ReLU.
2. **Build** (`setup.py`): compiles each shader to AIR with `xcrun metal`, links them into one `.metallib`, and
   embeds that library in a C header with `xxd`, so the extension carries its own kernels.
3. **Objective-C++ dispatchers** (`custom/dispatch_*.mm`): load the embedded library, build a compute pipeline, and
   encode the kernel on PyTorch's own MPS command buffer and dispatch queue (`torch::mps::get_command_buffer`), so
   custom ops interleave correctly with PyTorch's GPU work and tensors never leave the device.
4. **C++ ops** (`custom/custom_ops.cpp`): argument checks (`TORCH_CHECK` on device, contiguity, dtype), output
   allocation, and pybind11 bindings through `torch.utils.cpp_extension`.
5. **Python layer** (`custom/custom_ops.py`): `torch.autograd.Function`s with explicit backward passes, and
   `nn.Module`s built on them: `CustomLinear`, `CustomReLU`, `CustomLeakyReLU`.

## Build and test

Needs an Apple silicon Mac with Xcode (for the Metal compiler) and PyTorch with MPS.

```bash
python -m venv .venv && .venv/bin/pip install torch "setuptools<70" pytest
.venv/bin/python setup.py build_ext --inplace
.venv/bin/python -m pytest tests          # every op checked against PyTorch, including CustomLinear's gradients
.venv/bin/python main.py                  # small demos and a timing comparison
```

`notebooks/training_example.ipynb` trains a small network built from the custom layers; `notebooks/benchmarks.ipynb`
times the ops.

## What to expect

These are teaching kernels, written for clarity: each thread computes one output element. They match PyTorch's
results exactly in the tests, but they are not tuned. The matrix multiply has no tiling or threadgroup memory and is
about 100 times slower than PyTorch's own MPS matmul on a 10,000 × 10,000 problem (M3 Max). The point is the
integration path, not speed.

## License

MIT
