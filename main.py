import time
import torch
import custom_ops_cpp
import torch.mps.profiler as mps_profiler


# test the relu activation function.
a = torch.tensor(
    [[-0.15, 0.88, -0.74],
     [0.001, -0.55, 0.12]]
).to('mps')
print(f"Input tensor a: {a}")

# run the function
result = custom_ops_cpp.relu(a)
print(f"relu result: {result} with dim {result.shape}")


# test the add_tensors function
a = torch.tensor([1.0, 2.0, 3.0]).to('mps')
b = torch.tensor([4.0, 5.0, 6.0]).to('mps')
print(f"Input tensor a: {a}")
print(f"Input tensor b: {b}")
print(f"Input device: {a.device}")

result = custom_ops_cpp.add_tensors(a, b)
print(f"Addition result: {result}")
print(f"Output device {result.device}")
assert result.device == torch.device('mps:0'), "Output tensor is (maybe?) not on the MPS device"


# test the matrix_multiply function
a = torch.tensor(
    [[1., 2., 3.],
     [2., 3., 4.]]
).to('mps')
b = torch.tensor(
    [[4., 5.],
     [6., 7.],
     [8., 9.]]
).to('mps')
print(f"Input tensor a: {a} with dim {a.shape}")
print(f"Input tensor b: {b} with dim {b.shape}")
print(f"Input device: {a.device}")

result = custom_ops_cpp.matrix_multiply(a, b)
print(f"Mat multi result: {result} with dim {result.shape}")
print(f"Output device {result.device}")
assert result.device == torch.device('mps:0'), "Output tensor is (maybe?) not on the MPS device"

# bigger test with comparison to just pytorch.

# two big ish matricies
a = torch.full((10000, 10000), 10.1).to('mps')
b = torch.full((10000, 10000), 30.3).to('mps')

st = time.time()
mps_profiler.start(mode='interval', wait_until_completed=True)
result = custom_ops_cpp.matrix_multiply(a, b)
mps_profiler.stop()
ed = time.time()
el = ed - st
print(f"custom metal kernel based op finished in {el} with test value {result[0,0]} and size {result.shape}")

# now just with pytorch @ operator
st = time.time()
result = a @ b
ed = time.time()
el = ed - st
print(f"pytorch @ standard op finished in {el} with test value {result[0,0]} and size {result.shape}")


# now just with pytorch @ operator but with cpu
a.to('cpu')
b.to('cpu')
st = time.time()
result = a @ b
ed = time.time()
el = ed - st
print(f"pytorch @ standard op on cpu finished in {el} with test value {result[0,0]} and size {result.shape}")
