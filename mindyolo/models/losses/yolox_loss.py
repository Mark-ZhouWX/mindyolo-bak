import numpy as np
import mindspore
from mindspore import Tensor, nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.ops import operations as P

from mindyolo.models.losses.box_ops import batch_bboxes_iou


class YOLOLossCell(nn.Cell):
    """ yolox with loss cell """
    # FIXME rename to YOLOxLoss
    def __init__(self, network=None, config=None):
        super(YOLOLossCell, self).__init__()
        self.network = network
        self.n_candidate_k = config.n_candidate_k
        self.on_value = Tensor(1.0, mindspore.float32)
        self.off_value = Tensor(0.0, mindspore.float32)
        self.depth = config.num_classes

        self.unsqueeze = P.ExpandDims()
        self.reshape = P.Reshape()
        self.one_hot = P.OneHot()
        self.zeros = P.ZerosLike()
        self.sort_ascending = P.Sort(descending=False)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.l1_loss = nn.L1Loss(reduction="none")
        self.batch_iter = Tensor(np.arange(0, config.per_batch_size * config.max_gt), mindspore.int32)
        self.strides = config.fpn_strides
        self.grids = [(config.input_size[0] // _stride) * (config.input_size[1] // _stride) for _stride in
                      config.fpn_strides]
        self.use_l1 = config.use_l1
        self.use_summary = config.use_summary
        self.summary = ops.ScalarSummary()
        self.assign = ops.Assign()

    def construct(self, img, labels=None, pre_fg_mask=None, is_inbox_and_incenter=None):
        """ forward with loss return """
        batch_size = P.Shape()(img)[0]
        gt_max = P.Shape()(labels)[1]
        outputs = self.network(img)  # batch_size, 8400, 85
        total_num_anchors = P.Shape()(outputs)[1]
        bbox_preds = outputs[:, :, :4]  # batch_size, 8400, 4

        obj_preds = outputs[:, :, 4:5]  # batch_size, 8400, 1
        cls_preds = outputs[:, :, 5:]  # (batch_size, 8400, 80)

        # process label
        bbox_true = labels[:, :, 1:]  # (batch_size, gt_max, 4)

        gt_classes = F.cast(labels[:, :, 0:1].squeeze(-1), mindspore.int32)
        pair_wise_ious = batch_bboxes_iou(bbox_true, bbox_preds, xyxy=False)
        pair_wise_ious = pair_wise_ious * pre_fg_mask
        pair_wise_iou_loss = -P.Log()(pair_wise_ious + 1e-8) * pre_fg_mask
        gt_classes_ = self.one_hot(gt_classes, self.depth, self.on_value, self.off_value)
        gt_classes_expaned = ops.repeat_elements(self.unsqueeze(gt_classes_, 2), rep=total_num_anchors, axis=2)
        gt_classes_expaned = F.stop_gradient(gt_classes_expaned)
        cls_preds_ = P.Sigmoid()(ops.repeat_elements(self.unsqueeze(cls_preds, 1), rep=gt_max, axis=1)) * \
                     P.Sigmoid()(
                         ops.repeat_elements(self.unsqueeze(obj_preds, 1), rep=gt_max, axis=1)
                     )

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
        dynamic_ks_indices = P.Stack(axis=1)((self.batch_iter, dynamic_ks.reshape((-1,))))

        dynamic_ks_indices = F.stop_gradient(dynamic_ks_indices)

        values, _ = P.TopK(sorted=True)(-cost, self.n_candidate_k)  # b_s , 50, 8400
        values = P.Reshape()(-values, (-1, self.n_candidate_k))
        max_neg_score = self.unsqueeze(P.GatherNd()(values, dynamic_ks_indices).reshape(batch_size, -1), 2)
        pos_mask = F.cast(cost < max_neg_score, mindspore.float32)  # (batch_size, gt_num, 8400)
        pos_mask = pre_fg_mask * pos_mask
        # ----dynamic_k---- END-----------------------------------------------------------------------------------------
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
        loss_all = (5 * loss_iou + loss_cls + loss_obj + loss_l1) / num_fg

        if self.use_summary:
            self.summary('num_fg', num_fg)
            self.summary('loss_iou', loss_iou * 5 / num_fg)
            self.summary('loss_cls', loss_cls / num_fg)
            self.summary('loss_obj', loss_obj / num_fg)
            self.summary('loss_l1', loss_l1 / num_fg)

        return loss_all

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
