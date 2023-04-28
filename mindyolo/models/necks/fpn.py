from mindspore import nn

from mindyolo.models.backbones.darknet import Darknet
from mindyolo.models.layers.yolox_blocks import BaseConv
from mindspore.ops import operations as P


class YOLOFPN(nn.Cell):
    """
    YOLOFPN module, Darknet53 is the default backbone of this model
    """
    # FIXME decouple neck and backbone
    def __init__(self, input_w, input_h, depth=53, in_features=None):
        super(YOLOFPN, self).__init__()
        if in_features is None:
            in_features = ["dark3", "dark4", "dark5"]
        self.backbone = Darknet(depth)
        self.in_features = in_features

        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)
        # upsample
        self.upsample0 = P.ResizeNearestNeighbor((input_h // 16, input_w // 16))
        self.upsample1 = P.ResizeNearestNeighbor((input_h // 8, input_w // 8))

    def _make_cbl(self, _in, _out, ks):
        """ make cbl layer """
        return BaseConv(_in, _out, ks, stride=1, act="lrelu")

    def _make_embedding(self, filters_list, in_filters):
        """ make embedding """
        m = nn.SequentialCell(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def construct(self, inputs):
        """ forward """
        out_features = self.backbone(inputs)
        x2, x1, x0 = out_features

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample0(x1_in)
        x1_in = P.Concat(axis=1)([x1_in, x1])
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample1(x2_in)
        x2_in = P.Concat(axis=1)([x2_in, x2])
        out_dark3 = self.out2(x2_in)
        outputs = (out_dark3, out_dark4, x0)
        return outputs
