import pytest
import torch
from von.device import (
    _detect_device,
    get_device_description,
    is_openvino_available,
    is_openvino_gpu_available,
    OpenVINODevice,
)


def test_openvino_device_class():
    dev = OpenVINODevice(target="GPU", device_name="Intel Graphics Xe")
    assert dev.type == "openvino"
    assert dev.target == "GPU"
    assert str(dev) == "openvino:gpu"
    assert "openvino" in repr(dev)
    assert dev == "openvino"
    assert dev == "openvino:gpu"
    assert dev == "intel"
    assert dev == OpenVINODevice("GPU")
    assert dev != OpenVINODevice("CPU")


def test_device_detection_explicit_cpu():
    dev = _detect_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type == "cpu"
    assert get_device_description(dev) == "CPU"


def test_device_detection_openvino():
    if is_openvino_available():
        dev = _detect_device("openvino")
        assert isinstance(dev, OpenVINODevice)
        assert dev.target == "GPU"
        desc = get_device_description(dev)
        assert "Intel GPU" in desc
        assert "OpenVINO" in desc

        dev_cpu = _detect_device("openvino:cpu")
        assert isinstance(dev_cpu, OpenVINODevice)
        assert dev_cpu.target == "CPU"


def test_auto_detection():
    dev = _detect_device("auto")
    desc = get_device_description(dev)
    assert isinstance(desc, str)
    if is_openvino_gpu_available() and not torch.cuda.is_available():
        assert isinstance(dev, OpenVINODevice)
        assert "Intel GPU" in desc
