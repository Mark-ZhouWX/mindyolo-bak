import math

import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common import initializer as init
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindyolo.models.layers.conv import ConvNormAct


class YOLOxHead(nn.Cell):
    def __init__(self, nc=80, stride=(8, 16, 32), ch=(256, 512, 1024), act=True, sync_bn=False):
        super().__init__()
        assert isinstance(stride, (tuple, list)) and len(stride) > 0
        assert isinstance(ch, (tuple, list)) and len(ch) > 0

        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 4 + 1
        self.stride = Tensor(stride, ms.int32)

        self.stems = nn.CellList() # len = num_layer
        self.cls_convs = nn.CellList()
        self.reg_convs = nn.CellList()
        self.cls_preds = nn.CellList()
        self.reg_preds = nn.CellList()
        self.obj_preds = nn.CellList()

        hidden_ch = ch[2]//4
        for i in range(self.nl):  # three kind of resolution, 80, 40, 20
            self.stems.append(ConvNormAct(ch[i], hidden_ch, 3, act=act, sync_bn=sync_bn))
            self.cls_convs.append(nn.SequentialCell(
                [ConvNormAct(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                 ConvNormAct(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn)]
            ))
            self.reg_convs.append(nn.SequentialCell(
                [ConvNormAct(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn),
                 ConvNormAct(hidden_ch, hidden_ch, 3, act=act, sync_bn=sync_bn)]
            ))
            self.cls_preds.append(nn.Conv2d(hidden_ch, self.nc, 1, pad_mode='pad', has_bias=True))
            self.reg_preds.append(nn.Conv2d(hidden_ch, 4, 1, pad_mode='pad', has_bias=True))
            self.obj_preds.append(nn.Conv2d(hidden_ch, 1, 1, pad_mode='pad', has_bias=True))

    def construct(self, feat_list):
        assert isinstance(feat_list, (tuple, list)) and len(feat_list) == self.nl
        outputs = []
        for i in range(self.nl):  # 80, 40, 20
            # Get head features
            x = self.stems[i](feat_list[i])

            cls_feat = self.cls_convs[i](x)
            cls_output = self.cls_preds[i](cls_feat)

            reg_feat = self.reg_convs[i](x)
            reg_output = self.reg_preds[i](reg_feat)
            obj_output = self.obj_preds[i](reg_feat)

            # Convert to origin image scale (640)
            output = ops.concat([reg_output, obj_output, cls_output], 1) if self.training else \
                ops.concat([reg_output, ops.sigmoid(obj_output), ops.sigmoid(cls_output)], 1)
            output = self.convert_to_origin_scale(output, stride=self.stride[i])
            outputs.append(output)
        outputs_cat = ops.concat(outputs, 1)
        return outputs_cat if self.training else (outputs_cat, 1)

    def initialize_biases(self, prior_prob=1e-2):
        for i in range(self.nl):  # 80, 40, 20
            for cell in [self.cls_preds[i], self.obj_preds[i]]:
                cell.bias.set_data(init.initializer(-math.log((1 - prior_prob) / prior_prob), cell.bias.shape,
                                                    cell.bias.dtype))

    def convert_to_origin_scale(self, output, stride):
        """ map to origin image scale for each fpn """
        # TODO use ops, _make_grid, move stride tensor to init
        batch_size = P.Shape()(output)[0]
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
                             (batch_size, self.no, grid_size[0], grid_size[1]))  # bs, 6400, 85-->(bs,85,80,80)
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
        return output_t  # bs, 6400, 85