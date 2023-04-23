import numpy as np
import mindspore
import mindspore.numpy as mnp
from mindspore import Tensor, nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindyolo.models.losses.box_ops import batch_bboxes_iou
from mindyolo.models.registry import register_model
from mindyolo.models.utils.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, box_scale, box_clip

__all__ = [
    'YOLOxLoss',
]


@register_model
class YOLOxLoss(nn.Cell):
    """ yolox with loss cell """
    def __init__(self,
                 num_class=80,
                 input_size=(640, 640),
                 num_candidate_ota=10,
                 strides=(8, 16, 32),
                 use_l1=False,
                 use_summary=False,
                 **kwargs
                 ):
        super(YOLOxLoss, self).__init__()
        self.n_candidate_k = num_candidate_ota
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.num_class = num_class

        self.unsqueeze = P.ExpandDims()
        self.reshape = P.Reshape()
        self.one_hot = P.OneHot()
        self.zeros = P.ZerosLike()
        self.sort_ascending = P.Sort(descending=False)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")

        self.strides = strides
        self.input_size = input_size
        self.grids = [(input_size[0] // _stride) * (input_size[1] // _stride) for _stride in strides]
        self.num_total_anchor = sum(self.grids)
        self.anchor_center_pos, self.anchor_strides = self._get_anchor_center_and_stride(norm=False)

        self.use_l1 = use_l1
        self.use_summary = use_summary
        self.summary = ops.ScalarSummary()
        self.assign = ops.Assign()

        self.loss_item_name = ['loss', 'lbox', 'lobj', 'lcls', 'lboxl1']  # branch name returned by lossitem for print


    def _get_anchor_center_and_stride(self, norm=False):
        """
        creat a table for all layer of anchors(grids), the value is the pixel position of the grid center and its stride.
        The coordinate of the value is relative to the input img
        Returns:
            anchor_center_pos (Tensor[num_total_anchor, 2]): pixel position of the grid center
            anchor_strides (Tensor[num_total_anchor,]): anchor strides
        """

        anchor_strides_list = []
        for s, g in zip(self.strides, self.grids):
            layer_stride = ops.ones((g,), mindspore.float32) * float(s)
            anchor_strides_list.append(layer_stride)
        anchor_strides = ops.concat(anchor_strides_list)
        # (num_total_anchor, 2)
        anchor_strides = ops.stack([anchor_strides, anchor_strides], axis=1)

        anchor_center_pos_list = []
        for stride in self.strides:
            size_x = self.input_size[0] // stride
            size_y = self.input_size[1] // stride
            grid_x, grid_y = ops.meshgrid((mnp.arange(size_y), mnp.arange(size_x)))
            # TODO check meshgrid
            grids = ops.stack((grid_x, grid_y), 2).reshape(-1, 2)
            anchor_center_pos_list.append(grids)

        # (num_total_anchor, 2)
        anchor_center_pos = ops.concat(anchor_center_pos_list, 0)

        # to the scale of input img
        anchor_center_pos = (anchor_center_pos + 0.5) * anchor_strides

        if norm:
            anchor_center_pos[..., 0] /= self.input_size[0]
            anchor_center_pos[..., 1] /= self.input_size[1]

            anchor_strides [..., 0] /= self.input_size[0]
            anchor_strides [..., 1] /= self.input_size[1]

        return anchor_center_pos, anchor_strides

    def in_box(self, anchors, boxes):
        # splitted_diff = ops.split(ops.concat([anchors - boxes[..., :2], boxes[..., 2:] - anchors],axis=-1),
        #                           axis=-1, output_num=4)
        # temp1 = ops.logical_and(splitted_diff[0] > 0.0, splitted_diff[1] > 0.0)
        # temp2 = ops.logical_and(splitted_diff[2] > 0.0, splitted_diff[3] > 0.0)
        # in_mask = ops.logical_and(temp1, temp2).squeeze(-1)

        splitted_diff1 = anchors - boxes[..., :2]
        splitted_diff2 = boxes[..., 2:] - anchors
        temp1 = ops.logical_and(splitted_diff1[..., 0] > 0.0, splitted_diff1[..., 1] > 0.0)
        temp2 = ops.logical_and(splitted_diff2[..., 0] > 0.0, splitted_diff2[..., 1] > 0.0)
        in_mask = ops.logical_and(temp1, temp2)

        return in_mask

    def _get_foreground(self, gt_boxes, gt_valid_mask, center_radius=2.5):
        """
        get the mask of foreground anchor point,
        ref: simOTA, link
        Args:
             gt_boxes (Tensor[bs, num_gt_max, 4]): gt box in [x1,y1, x2, y2] format, normed
             gt_valid_mask (Tensor[bs, num_gt_max]) : gt box valid mask, indicates valid if true
             num_valid_gt (int): num of valid gt boxes
             center_radius (float): radius threshold to judge whether an anchor is an inlier of the gt center.
                The unit is pixel in the feature map scale.
        # TODO 根据megvii源码，fg_mask应当与in_center_box_mask对应，in_center_box_mask为落在核心框内，待验证
        Returns:
             fg_mask (Tensor(bs, num_total_anchor)): mask to indicate whether an anchor falls in any gt box
             in_center_box_mask (Tensor(bs, num_gt_max, num_total_anchor)): mask to indicate whether an anchor
                falls both in a specific gt box and the core box with radius center_radius

        """
        bs, num_gt_max, _ = gt_boxes.shape

        gt_box_xyxy = gt_boxes
        gt_box_center = 0.5 * (gt_box_xyxy[..., :2] + gt_box_xyxy[..., 2:])
        # 1. Gt box mask
        # (bs, num_gt_max, num_total_anchor, 4)
        in_box_mask = self.in_box(self.anchor_center_pos, gt_box_xyxy.expand_dims(2))
        # fg_mask = in_box_mask.any(1)

        # 2. Gt core box mask
        # (bs, num_gt_max, num_total_anchor, 4)
        gt_core_box_xyxy = ops.concat([gt_box_center[:, :, None, :] - 0.5 * center_radius * self.anchor_strides,
                                       gt_box_center[:, :, None, :] + 0.5 * center_radius * self.anchor_strides],
                                      axis=-1)
        # (bs, num_gt_max, num_total_anchor)
        in_center_mask = self.in_box(self.anchor_center_pos, gt_core_box_xyxy)
        in_center_box_mask = ops.logical_and(in_box_mask, in_center_mask)

        # 3. Fill padding pos with false (bs, num_gt_max, num_total_anchor)
        expanded_gt_valid_mask = ops.repeat_elements(gt_valid_mask[:, :, None].astype(mindspore.int32),
                                                     rep=self.num_total_anchor, axis=2).astype(mindspore.bool_)
        in_center_box_mask = ops.logical_and(expanded_gt_valid_mask, in_center_box_mask)
        pre_fg_mask = ops.logical_and(expanded_gt_valid_mask, in_box_mask.any(1, keep_dims=True))
        return in_center_box_mask, pre_fg_mask

    def construct(self, preds, targets, imgs=None):
        """
        forward with loss return
        Args:
            preds (Tensor[bs, num_total_anchor, 85]):
            targets (Tensor[bs, num_gt_max, 6]): 0: batch_id, 1: label, 2-6: box
        """
        # loss = (preds ** 2).mean()
        # return loss, ops.stop_gradient(ops.stack((loss, Tensor(0.0), Tensor(0.0), Tensor(0.0), Tensor(0.0))))
        #
        gt_valid_mask = targets[..., 1] >= 0  # defalut class column
        gt_box_xyxy = box_cxcywh_to_xyxy(targets[:, :, 2:]) # (batch_size, gt_max, 4) in [xyxy] format
        # reverse norm
        gt_box_xyxy_raw = box_clip(box_scale(gt_box_xyxy, self.input_size), self.input_size)
        # to cxcywh format
        bbox_true = box_xyxy_to_cxcywh(gt_box_xyxy_raw)
        is_inbox_and_incenter, pre_fg_mask = self._get_foreground(gt_box_xyxy_raw, gt_valid_mask)

        batch_size = P.Shape()(preds)[0]
        gt_max = P.Shape()(targets)[1]
        outputs = preds  # batch_size, 8400, 85
        total_num_anchors = P.Shape()(outputs)[1]
        bbox_preds = outputs[:, :, :4]  # batch_size, num_total_anchor, 4

        obj_preds = outputs[:, :, 4:5]  # batch_size, num_total_anchor, 1
        cls_preds = outputs[:, :, 5:]  # (batch_size, num_total_anchor, num_class)

        # process label
        gt_classes = F.cast(targets[:, :, 1:2].squeeze(-1), mindspore.int32)
        pair_wise_ious = batch_bboxes_iou(bbox_true, bbox_preds, xyxy=False)  # (batch_size, gt_max, 8400)
        pair_wise_ious = pair_wise_ious * pre_fg_mask
        pair_wise_iou_loss = -P.Log()(pair_wise_ious + 1e-8) * pre_fg_mask
        gt_classes_ = self.one_hot(gt_classes, self.num_class, self.on_value, self.off_value)
        # (bs, num_gt_max, num_class) -> (bs, num_gt_max, num_total_anchor, num_class)
        gt_classes_expaned = ops.repeat_elements(self.unsqueeze(gt_classes_, 2), rep=total_num_anchors, axis=2)
        gt_classes_expaned = F.stop_gradient(gt_classes_expaned)
        cls_preds_ = P.Sigmoid()(ops.repeat_elements(self.unsqueeze(cls_preds, 1), rep=gt_max, axis=1)) * \
                     P.Sigmoid()(
                         ops.repeat_elements(self.unsqueeze(obj_preds, 1), rep=gt_max, axis=1)
                     )
        # (bs, num_gt_max, num_total_anchor, num_class) -> (bs, num_gt_max, num_total_anchor)
        pair_wise_cls_loss = P.ReduceSum()(
            P.BinaryCrossEntropy(reduction="none")(P.Sqrt()(cls_preds_), gt_classes_expaned, None), -1)
        pair_wise_cls_loss = pair_wise_cls_loss * pre_fg_mask
        cost = pair_wise_cls_loss + 3.0 * pair_wise_iou_loss
        punishment_cost = 1000.0 * (1.0 - F.cast(is_inbox_and_incenter, mindspore.float32))
        cost = F.cast(cost + punishment_cost, mindspore.float16)
        # dynamic k matching
        ious_in_boxes_matrix = pair_wise_ious  # (batch_size, gt_max, 8400)
        ious_in_boxes_matrix = F.cast(pre_fg_mask * ious_in_boxes_matrix, mindspore.float16)
        topk_ious, _ = P.TopK(sorted=True)(ious_in_boxes_matrix, self.n_candidate_k)

        dynamic_ks = P.ReduceSum()(topk_ious, 2).astype(mindspore.int32).clip(xmin=1, xmax=total_num_anchors - 1,
                                                                              dtype=mindspore.int32)

        # (1, batch_size * gt_max, 2)
        batch_iter = Tensor(np.arange(0, batch_size * gt_max), mindspore.int32)
        dynamic_ks_indices = P.Stack(axis=1)((batch_iter, dynamic_ks.reshape((-1,))))

        dynamic_ks_indices = F.stop_gradient(dynamic_ks_indices)

        values, _ = P.TopK(sorted=True)(-cost, self.n_candidate_k)  # b_s , 50, 8400
        values = P.Reshape()(-values, (-1, self.n_candidate_k))
        max_neg_score = self.unsqueeze(P.GatherNd()(values, dynamic_ks_indices).reshape(batch_size, -1), 2)
        # positive sample for each gt
        pos_mask = F.cast(cost < max_neg_score, mindspore.float32)  # (batch_size, gt_num, 8400)
        pos_mask = pre_fg_mask * pos_mask
        # ----dynamic_k---- END-----------------------------------------------------------------------------------------

        # pick the one with the lower cost if a sample is positive for more than one gt
        cost_t = cost * pos_mask + (1.0 - pos_mask) * 2000.
        min_index, _ = P.ArgMinWithValue(axis=1)(cost_t)
        ret_posk = P.Transpose()(ops.one_hot(min_index, gt_max, Tensor(1.0), Tensor(0.0)), (0, 2, 1))
        pos_mask = pos_mask * ret_posk
        pos_mask = F.stop_gradient(pos_mask)
        # AA problem--------------END ----------------------------------------------------------------------------------

        # calculate target ---------------------------------------------------------------------------------------------
        # Cast precision
        pos_mask = F.cast(pos_mask, mindspore.float16)
        bbox_true = F.cast(bbox_true, mindspore.float16)
        gt_classes_ = F.cast(gt_classes_, mindspore.float16)

        reg_target = P.BatchMatMul(transpose_a=True)(pos_mask, bbox_true)  # (batch_size, 8400, 4)
        pred_ious_this_matching = self.unsqueeze(P.ReduceSum()((ious_in_boxes_matrix * pos_mask), 1), -1)
        cls_target = P.BatchMatMul(transpose_a=True)(pos_mask, gt_classes_)

        cls_target = cls_target * pred_ious_this_matching
        obj_target = P.ReduceMax()(pos_mask, 1)  # (batch_size, 8400)

        # calculate l1_target
        reg_target = F.stop_gradient(reg_target)
        cls_target = F.stop_gradient(cls_target)
        obj_target = F.stop_gradient(obj_target)
        bbox_preds = F.cast(bbox_preds, mindspore.float32)
        reg_target = F.cast(reg_target, mindspore.float32)
        obj_preds = F.cast(obj_preds, mindspore.float32)
        obj_target = F.cast(obj_target, mindspore.float32)
        cls_preds = F.cast(cls_preds, mindspore.float32)
        cls_target = F.cast(cls_target, mindspore.float32)
        loss_l1 = 0.0
        if self.use_l1:
            l1_target = self.get_l1_format(reg_target)
            l1_preds = self.get_l1_format(bbox_preds)
            l1_target = F.stop_gradient(l1_target)
            l1_target = F.cast(l1_target, mindspore.float32)
            l1_preds = F.cast(l1_preds, mindspore.float32)
            loss_l1 = P.ReduceSum()(self.l1_loss(l1_preds, l1_target), -1) * obj_target
            loss_l1 = P.ReduceSum()(loss_l1)
        # calculate target -----------END-------------------------------------------------------------------------------
        loss_iou = IOUloss()(P.Reshape()(bbox_preds, (-1, 4)), reg_target).reshape(batch_size, -1) * obj_target
        loss_iou = P.ReduceSum()(loss_iou)
        loss_obj = self.bce_loss(P.Reshape()(obj_preds, (-1, 1)), P.Reshape()(obj_target, (-1, 1)))
        loss_obj = P.ReduceSum()(loss_obj)

        loss_cls = P.ReduceSum()(self.bce_loss(cls_preds, cls_target), -1) * obj_target
        loss_cls = P.ReduceSum()(loss_cls)

        num_fg_mask = P.ReduceSum()(obj_target) == 0
        num_fg = (num_fg_mask == 0) * P.ReduceSum()(obj_target) + 1.0 * num_fg_mask

        loss_iou = 5 * loss_iou / num_fg
        loss_cls = loss_cls / num_fg
        loss_obj = loss_obj / num_fg
        loss_l1 = loss_l1/ num_fg
        loss_all = loss_iou + loss_cls + loss_obj + loss_l1

        if self.use_summary:
            self.summary('loss', loss_all)
            self.summary('num_fg', num_fg)
            self.summary('loss_iou', loss_iou)
            self.summary('loss_cls', loss_cls)
            self.summary('loss_obj', loss_obj)
            self.summary('loss_l1', loss_l1)

        return loss_all, ops.stop_gradient(ops.stack((loss_all, loss_iou, loss_obj, loss_cls, loss_l1)))

    def get_l1_format_single(self, reg_target, stride, eps):
        """ calculate L1 loss related """
        reg_target = reg_target / stride
        reg_target_xy = reg_target[:, :, :2]
        reg_target_wh = reg_target[:, :, 2:]
        reg_target_wh = P.Log()(reg_target_wh + eps)
        return P.Concat(-1)((reg_target_xy, reg_target_wh))

    def get_l1_format(self, reg_target, eps=1e-8):
        """ calculate L1 loss related """
        reg_target_l = reg_target[:, 0:self.grids[0], :]  # (bs, 6400, 4)
        reg_target_m = reg_target[:, self.grids[0]:self.grids[1] + self.grids[0], :]  # (bs, 1600, 4)
        reg_target_s = reg_target[:, -self.grids[2]:, :]  # (bs, 400, 4)

        reg_target_l = self.get_l1_format_single(reg_target_l, self.strides[0], eps)
        reg_target_m = self.get_l1_format_single(reg_target_m, self.strides[1], eps)
        reg_target_s = self.get_l1_format_single(reg_target_s, self.strides[2], eps)

        l1_target = P.Concat(axis=1)([reg_target_l, reg_target_m, reg_target_s])
        return l1_target


class IOUloss(nn.Cell):
    """ Iou loss """
    # FIXME use iou from iou_loss.py
    def __init__(self, reduction="none"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.reshape = P.Reshape()

    def construct(self, pred, target):
        """ forward """
        pred = self.reshape(pred, (-1, 4))
        target = self.reshape(target, (-1, 4))
        tl = P.Maximum()(pred[:, :2] - pred[:, 2:] / 2, target[:, :2] - target[:, 2:] / 2)
        br = P.Minimum()(pred[:, :2] + pred[:, 2:] / 2, target[:, :2] + target[:, 2:] / 2)
        area_p = (pred[:, 2:3] * pred[:, 3:4]).squeeze(-1)
        area_g = (target[:, 2:3] * target[:, 3:4]).squeeze(-1)
        en = F.cast((tl < br), tl.dtype)
        en = (en[:, 0:1] * en[:, 1:2]).squeeze(-1)
        area_i = br - tl
        area_i = (area_i[:, 0:1] * area_i[:, 1:2]).squeeze(-1) * en
        area_u = area_p + area_g - area_i

        iou = area_i / (area_u + 1e-16)
        loss = 1 - iou * iou
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
