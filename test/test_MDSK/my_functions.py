"""
存储了在调用模型的推理过程中的常用函数
"""


def downSample(data):
    """signal.decimate 函数基于抗混叠滤波器的方法，先进行低通滤波，然后降采样。"""
    from scipy import signal
    import torch

    data = data.numpy()
    q = 5  # 下采样的倍数
    data = signal.decimate(data, q).copy()
    data = torch.tensor(data)

    return data


