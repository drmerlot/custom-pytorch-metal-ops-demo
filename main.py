import torch
import my_extension

# test the add_tensors function
a = torch.tensor([1.0, 2.0, 3.0]).to('mps')
b = torch.tensor([4.0, 5.0, 6.0]).to('mps')
print(f"Input tensor a: {a}")
print(f"Input tensor b: {b}")
print(f"Input device: {a.device}")

result = my_extension.add_tensors(a, b)
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

result = my_extension.matrix_multiply(a, b)
print(f"Mat multi result: {result} with dim {result.shape}")
print(f"Output device {result.device}")
assert result.device == torch.device('mps:0'), "Output tensor is (maybe?) not on the MPS device"

# bigger test
a = torch.full((20, 30), 10.1).to('mps')
b = torch.full((30, 20), 30.3).to('mps')

print(f"Input tensor a: {a} with dim {a.shape}")
print(f"Input tensor b: {b} with dim {b.shape}")
print(f"Input device: {a.device}")

result = my_extension.matrix_multiply(a, b)
print(f"Mat multi result: {result} with dim {result.shape}")
print(f"Output device {result.device}")
assert result.device == torch.device('mps:0'), "Output tensor is (maybe?) not on the MPS device"






