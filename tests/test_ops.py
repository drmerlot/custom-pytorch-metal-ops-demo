"""Each custom Metal op against PyTorch's own result. Needs an Apple silicon Mac (MPS) and the built extension
(python setup.py build_ext --inplace)."""
import pytest
import torch

if not torch.backends.mps.is_available():
    pytest.skip("needs the MPS device (Apple silicon)", allow_module_level=True)
custom_cpp = pytest.importorskip("custom_cpp")


@pytest.mark.parametrize("shape", [(2, 3), (64, 32), (257, 129)])
def test_elementwise(shape):
    x = torch.randn(*shape, device="mps")
    torch.testing.assert_close(custom_cpp.relu(x), torch.relu(x))
    torch.testing.assert_close(custom_cpp.leaky_relu(x), torch.nn.functional.leaky_relu(x))


def test_add_tensors():
    a, b = torch.randn(1000, device="mps"), torch.randn(1000, device="mps")
    torch.testing.assert_close(custom_cpp.add_tensors(a, b), a + b)


def test_matrix_multiply_and_add():
    a, b = torch.rand(64, 32, device="mps"), torch.rand(32, 16, device="mps")
    torch.testing.assert_close(custom_cpp.matrix_multiply(a, b), a @ b)
    bias = torch.rand(16, device="mps")
    torch.testing.assert_close(custom_cpp.matrix_add(a @ b, bias), a @ b + bias)


def test_custom_linear_gradients_match_torch():
    from custom import CustomLinear
    torch.manual_seed(0)
    layer = CustomLinear(8, 4).to("mps")
    ref = torch.nn.Linear(8, 4).to("mps")
    ref.load_state_dict({"weight": layer.weight.detach().clone(), "bias": layer.bias.detach().clone()})
    x = torch.randn(5, 8, device="mps")
    layer(x).sum().backward()
    ref(x).sum().backward()
    torch.testing.assert_close(layer.weight.grad, ref.weight.grad)
    torch.testing.assert_close(layer.bias.grad, ref.bias.grad)
