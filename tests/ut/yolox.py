import argparse

import cv2
import numpy as np

from mindspore import nn, ops, ParameterTuple, Tensor, context

from mindspore.nn import WithLossCell

from mindyolo import parse_args
from mindyolo.models.losses.yolox_loss import YOLOLossCell
from mindyolo.models.yolox import YOLOx

context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=4)


def preproc(img, input_size, swap=(2, 0, 1)):
    """ padding image and transpose dim """
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def ut_yolox_train():
    # data
    parser = argparse.ArgumentParser(description='Train', parents=[])
    config = parse_args(parser)

    network = YOLOx(config)
    # model = YOLOLossCell(network, config)
    model = network

    img = cv2.imread(config.img_path)
    padded_img, r = preproc(img, config.input_size)
    Tensor(padded_img[None])
    model()
    weight = network.trainable_params()
    optimizer = nn.SGD(weight, learning_rate=1e-3)
    grad_fn = ops.GradOperation(get_by_list=True)(model, ParameterTuple(optimizer.weight))
    for i in range(10):
        loss = model(img, target)
        gradients = grad_fn(img, target)
        print(f'loss of the {i} step', loss)
        print(f'gradients {gradients[0]}')

    # train loop


def ut_yolox_infer():
    # data
    parser = argparse.ArgumentParser(description='Train', parents=[])
    config = parse_args(parser)

    network = YOLOx(config)
    # model = YOLOLossCell(network, config)
    model = network

    img = cv2.imread(config.img_path)
    padded_img, r = preproc(img, config.input_size)
    img = Tensor(padded_img[None])
    model(img)


if __name__ == '__main__':
    ut_yolox_infer()
    exit()