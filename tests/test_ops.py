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


def test_activation_gradients_match_torch():
    from custom import CustomLeakyReLU, CustomReLU
    for custom, ref in ((CustomReLU(), torch.nn.ReLU()), (CustomLeakyReLU(), torch.nn.LeakyReLU())):
        x = torch.randn(300, 70, device="mps", requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        g = torch.randn(300, 70, device="mps")
        custom(x).backward(g)
        ref(x2).backward(g)
        torch.testing.assert_close(x.grad, x2.grad)


def test_network_trains_like_torch():
    """The same small network built from the custom layers and from torch's layers, same weights and data,
    takes the same training steps."""
    from custom import CustomLeakyReLU, CustomLinear
    torch.manual_seed(0)
    custom = torch.nn.Sequential(CustomLinear(1, 32), CustomLeakyReLU(), CustomLinear(32, 1)).to("mps")
    ref = torch.nn.Sequential(torch.nn.Linear(1, 32), torch.nn.LeakyReLU(), torch.nn.Linear(32, 1)).to("mps")
    ref.load_state_dict({k: v.detach().clone() for k, v in custom.state_dict().items()})
    x = torch.rand(200, 1, device="mps") * 2 - 1
    y = torch.sin(3 * x)
    for model in (custom, ref):
        opt = torch.optim.SGD(model.parameters(), lr=0.05)
        for _ in range(20):
            loss = torch.nn.functional.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    for a, b in zip(custom.parameters(), ref.parameters()):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-5)
