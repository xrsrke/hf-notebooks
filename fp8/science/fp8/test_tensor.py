import torch
from copy import deepcopy
from tensor import FP8Tensor
import pytest
import numpy as np

def test_quantize_and_dequantize_tensor_in_fp8():
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cpu")
    ref_tensor = deepcopy(tensor)
    fp8_tensor = FP8Tensor(tensor)

    assert not np.array_equal(fp8_tensor.cpu().numpy(), ref_tensor.cpu().numpy())

    dequant_tensor = fp8_tensor.to(torch.float32)
    # NOTE: sometimes type(tensor) is FP8Tensor, but it still passes, so we directly check the class name
    # to make sure it's a torch.Tensor
    assert isinstance(dequant_tensor, torch.Tensor)
    assert dequant_tensor.__class__ == torch.Tensor
    assert dequant_tensor.dtype == torch.float32

    torch.testing.assert_close(tensor, ref_tensor)


@pytest.mark.parametrize("is_quantized", [True, False])
def test_setting_new_data_for_fp8_and_fp16_tensor(is_quantized):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cpu")
    fp8_tensor = FP8Tensor(tensor)

    new_data = torch.randn(fp8_tensor.shape, dtype=torch.float32, device="cpu") * 2
    ref_new_data = deepcopy(new_data)
    # expected_quantized_tensor = FP8Tensor(ref_new_data)

    new_data = FP8Tensor(new_data) if is_quantized else new_data
    fp8_tensor.data = new_data

    # assert torch.equal(fp8_tensor, expected_quantized_tensor)

    # dequantized_tensor = fp8_tensor.to(torch.float32)
    # assert torch.allclose(dequantized_tensor, ref_new_data)
    assert fp8_tensor.data.data_ptr() == new_data.data.data_ptr()


def test_zero_out_data_of_fp8_and_fp16_tensor():
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cpu")
    fp8_tensor = FP8Tensor(tensor)

    fp8_tensor.zero_()

    assert torch.equal(fp8_tensor, torch.zeros_like(fp8_tensor))

    dequantized_tensor = fp8_tensor.to(torch.float32)
    assert torch.equal(dequantized_tensor, torch.zeros_like(tensor))


def test_serialize_fp8_tensor():
    test_context = TestContext()
    store_folder = test_context.get_auto_remove_tmp_dir()
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cpu")

    fp8_tensor = FP8Tensor(tensor)

    torch.save(fp8_tensor, f"{store_folder}/fp8_tensor.pt")
    torch.load(f"{store_folder}/fp8_tensor.pt")


def test_fp8_and_fp16_tensor_attrs(tensor_cls, expected_dtype):
    tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
    ref_tensor = tensor.detach().clone()

    fp8_tensor = FP8Tensor(tensor)

    assert isinstance(fp8_tensor, tensor_cls)
    # assert isinstance(fp8_tensor.fp8_meta, FP8Meta)
    assert fp8_tensor.dtype == expected_dtype
    assert fp8_tensor.device == ref_tensor.device
    assert fp8_tensor.shape == ref_tensor.shape
    assert fp8_tensor.numel() == ref_tensor.numel()
    assert fp8_tensor.device == ref_tensor.device


# def test_clone_fp8_tensor():
#     tensor = torch.randn((64, 64), dtype=torch.float32, device="cuda:0")
#     fp8_tensor = FP8Tensor(deepcopy(tensor))

#     cloned_fp8_tensor = fp8_tensor.clone()

#     assert isinstance(cloned_fp8_tensor, FP8Tensor)
#     assert id(cloned_fp8_tensor) != id(fp8_tensor)
#     assert cloned_fp8_tensor.device == fp8_tensor.device

#     assert torch.equal(cloned_fp8_tensor, fp8_tensor)
#     assert cloned_fp8_tensor.data_ptr() != fp8_tensor.data_ptr()
#     assert cloned_fp8_tensor.data.data_ptr() != fp8_tensor.data.data_ptr()

#     # assert cloned_fp8_tensor.fp8_meta == fp8_tensor.fp8_meta
#     # assert id(cloned_fp8_tensor.fp8_meta) != id(fp8_tensor.fp8_meta)

# @pytest.mark.parametrize("interval", [1, 5])
def test_create_fp8_tensor_from_metadata(dtype):
    INTERVAL = 5
    TOTAL_STEPS, REMAINING_STEPS = 20, 16
    tensor = torch.randn((16, 16), dtype=torch.float32, device="cpu")
    fp8_tensor = FP8Tensor(tensor, dtype=dtype, interval=INTERVAL)

    new_values = [torch.randn((16, 16), dtype=torch.float32, device="cpu") for i in range(TOTAL_STEPS)]

    for i in range(TOTAL_STEPS):
        if TOTAL_STEPS - REMAINING_STEPS == i:
            current_tensor = fp8_tensor.orig_data
            fp8_meta = deepcopy(fp8_tensor.fp8_meta)

        fp8_tensor.data = new_values[i]

    resumed_fp8_tensor = FP8Tensor.from_metadata(current_tensor, fp8_meta)
    for i in range(TOTAL_STEPS - REMAINING_STEPS, TOTAL_STEPS):
        resumed_fp8_tensor.data = new_values[i]

    # NOTE: we expect a resume tensor to have the state trajectory of the original tensor
    assert resumed_fp8_tensor == fp8_tensor

def test_fp8_and_fp16_tensor_storage_memory(tensor_cls, dtype):
    tensor = torch.randn((4, 4), dtype=torch.float32, device="cpu")
    ref_tensor = deepcopy(tensor)

    fp8_tensor = FP8Tensor(tensor, dtype=dtype)

    assert id(fp8_tensor) != id(ref_tensor)

    assert isinstance(fp8_tensor.data, torch.Tensor)
    assert id(fp8_tensor.data) != id(ref_tensor.data)
    assert fp8_tensor.data_ptr() == fp8_tensor.data.data_ptr()
    assert fp8_tensor.data.data_ptr() != ref_tensor.data_ptr()
