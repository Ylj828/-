import torch
from torch import Tensor


def add():
    x: Tensor = torch.tensor([1, 2])
    y: Tensor = torch.tensor([3, 4])

    z: object = x.add(y)
    print(z, x)
    x.add_(y)
    print(x)


if __name__ == '__main__':
    print(torch.cuda.is_available())

    x = torch.tensor([10.0])
    # x = x.cuda()
    print(x)

    y = torch.randn(2, 3)
    # y  = y.cuda()
    print(y)

    z = x + y
    print(z)

    from torch.backends import cudnn

    print(cudnn.is_acceptable(x))

    add()
