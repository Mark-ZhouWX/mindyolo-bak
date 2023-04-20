import mindspore as ms
from mindspore import nn, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindyolo.models.layers.yolox_blocks import BaseConv
from mindyolo.models.necks.pafpn import YOLOPAFPN

__all__ = [
    'YOLOx',
    'yolox'
]

from mindyolo.models.registry import register_model


def _cfg(url='', **kwargs):
    return {
        'url': url,
        **kwargs
    }


default_cfgs = {
    'yolox': _cfg(url='')
}


class YOLOx(nn.Cell):
    """ connect yolox backbone and head """

    def __init__(self,
                 backbone="yolopafpn",
                 nc=80, # number of classes
                 depth_wise=False,
                 input_size=(640, 640),
                 depth=0.33,
                 width=0.50,
                 stride=(8, 16, 32),
                 **kwargs
                 ):
        super(YOLOx, self).__init__()
        self.num_classes = nc
        self.attr_num = self.num_classes + 5
        self.depthwise = depth_wise
        self.strides = Tensor(stride, ms.float32)
        self.input_size = input_size

        # network
        if backbone == "yolopafpn":
            self.depth = depth
            self.width = width
            self.backbone = YOLOPAFPN(depth=self.depth, width=self.width, input_w=self.input_size[1], input_h=self.input_size[0])
            self.head_inchannels = [1024, 512, 256]
            self.activation = "silu"

        else:
            self.backbone = YOLOFPN(input_w=self.input_size[1], input_h=self.input_size[0])
            self.head_inchannels = [512, 256, 128]
            self.activation = "lrelu"
            self.width = 1.0

        self.head_l = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='l',
                                      act=self.activation, width=self.width)
        self.head_m = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='m',
                                      act=self.activation, width=self.width)
        self.head_s = DetectionPerFPN(in_channels=self.head_inchannels, num_classes=self.num_classes, scale='s',
                                      act=self.activation, width=self.width)

    def construct(self, x):
        """ forward """
        outputs = []
        x_l, x_m, x_s = self.backbone(x)
        cls_output_l, reg_output_l, obj_output_l = self.head_l(x_l)  # (bs, 80, 80, 80)(bs, 4, 80, 80)(bs, 1, 80, 80)
        cls_output_m, reg_output_m, obj_output_m = self.head_m(x_m)  # (bs, 80, 40, 40)(bs, 4, 40, 40)(bs, 1, 40, 40)
        cls_output_s, reg_output_s, obj_output_s = self.head_s(x_s)  # (bs, 80, 20, 20)(bs, 4, 20, 20)(bs, 1, 20, 20)
        if self.training:
            output_l = P.Concat(axis=1)((reg_output_l, obj_output_l, cls_output_l))  # (bs, 85, 80, 80)
            output_m = P.Concat(axis=1)((reg_output_m, obj_output_m, cls_output_m))  # (bs, 85, 40, 40)
            output_s = P.Concat(axis=1)((reg_output_s, obj_output_s, cls_output_s))  # (bs, 85, 20, 20)

            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h

        else:

            output_l = P.Concat(axis=1)(
                (reg_output_l, P.Sigmoid()(obj_output_l), P.Sigmoid()(cls_output_l)))  # bs, 85, 80, 80

            output_m = P.Concat(axis=1)(
                (reg_output_m, P.Sigmoid()(obj_output_m), P.Sigmoid()(cls_output_m)))  # bs, 85, 40, 40

            output_s = P.Concat(axis=1)(
                (reg_output_s, P.Sigmoid()(obj_output_s), P.Sigmoid()(cls_output_s)))  # bs, 85, 20, 20
            output_l = self.mapping_to_img(output_l, stride=self.strides[0])  # (bs, 6400, 85)x_c, y_c, w, h
            output_m = self.mapping_to_img(output_m, stride=self.strides[1])  # (bs, 1600, 85)x_c, y_c, w, h
            output_s = self.mapping_to_img(output_s, stride=self.strides[2])  # (bs,  400, 85)x_c, y_c, w, h
        outputs.append(output_l)
        outputs.append(output_m)
        outputs.append(output_s)
        outputs_cat = P.Concat(axis=1)(outputs)
        return outputs_cat if self.training else (outputs_cat, 1)  # batch_size, 8400, 85

    def mapping_to_img(self, output, stride):
        """ map to origin image scale for each fpn """
        batch_size = P.Shape()(output)[0]
        n_ch = self.attr_num
        grid_size = P.Shape()(output)[2:4]
        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        stride = P.Cast()(stride, output.dtype)
        grid_x = P.Cast()(F.tuple_to_array(range_x), output.dtype)
        grid_y = P.Cast()(F.tuple_to_array(range_y), output.dtype)
        grid_y = P.ExpandDims()(grid_y, 1)
        grid_x = P.ExpandDims()(grid_x, 0)
        yv = P.Tile()(grid_y, (1, grid_size[1]))
        xv = P.Tile()(grid_x, (grid_size[0], 1))
        grid = P.Stack(axis=2)([xv, yv])  # (80, 80, 2)
        grid = P.Reshape()(grid, (1, 1, grid_size[0], grid_size[1], 2))  # (1,1,80,80,2)
        output = P.Reshape()(output,
                             (batch_size, n_ch, grid_size[0], grid_size[1]))  # bs, 6400, 85-->(bs,85,80,80)
        output = P.Transpose()(output, (0, 2, 1, 3))  # (bs,85,80,80)-->(bs,80,85,80)
        output = P.Transpose()(output, (0, 1, 3, 2))  # (bs,80,85,80)--->(bs, 80, 80, 85)
        output = P.Reshape()(output, (batch_size, 1 * grid_size[0] * grid_size[1], -1))  # bs, 6400, 85
        grid = P.Reshape()(grid, (1, -1, 2))  # grid(1, 6400, 2)

        # reconstruct
        output_xy = output[..., :2]
        output_xy = (output_xy + grid) * stride
        output_wh = output[..., 2:4]
        output_wh = P.Exp()(output_wh) * stride
        output_other = output[..., 4:]
        output_t = P.Concat(axis=-1)([output_xy, output_wh, output_other])
        return output_t  # bs, 6400, 85           grid(1, 6400, 2)


class DetectionPerFPN(nn.Cell):
    """ head  """
    # FIXME rename to YOLOxDecoupleHead
    def __init__(self, num_classes, scale, in_channels=None, act="silu", width=1.0):
        super(DetectionPerFPN, self).__init__()
        if in_channels is None:
            in_channels = [1024, 512, 256]
        self.scale = scale
        self.num_classes = num_classes
        Conv = BaseConv
        if scale == 's':
            self.stem = BaseConv(in_channels=int(in_channels[0] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'm':
            self.stem = BaseConv(in_channels=int(in_channels[1] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        elif scale == 'l':
            self.stem = BaseConv(in_channels=int(in_channels[2] * width), out_channels=int(256 * width), ksize=1,
                                 stride=1, act=act)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")

        self.cls_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.reg_convs = nn.SequentialCell(
            [
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
                Conv(
                    in_channels=int(256 * width),
                    out_channels=int(256 * width),
                    ksize=3,
                    stride=1,
                    act=act,
                ),
            ]
        )
        self.cls_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=self.num_classes, kernel_size=1, stride=1,
                                   pad_mode="pad", has_bias=True)

        self.reg_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=4, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

        self.obj_preds = nn.Conv2d(in_channels=int(256 * width), out_channels=1, kernel_size=1, stride=1,
                                   pad_mode="pad",
                                   has_bias=True)

    def construct(self, x):
        """ forward """
        x = self.stem(x)
        cls_x = x
        reg_x = x
        cls_feat = self.cls_convs(cls_x)

        cls_output = self.cls_preds(cls_feat)

        reg_feat = self.reg_convs(reg_x)
        reg_output = self.reg_preds(reg_feat)
        obj_output = self.obj_preds(reg_feat)

        return cls_output, reg_output, obj_output


@register_model
def yolox(cfg, in_channels=3, num_classes=None, **kwargs) -> YOLOx:
    """Get GoogLeNet model.
     Refer to the base class `models.GoogLeNet` for more details."""
    model = YOLOx(**cfg, **kwargs)
    return model