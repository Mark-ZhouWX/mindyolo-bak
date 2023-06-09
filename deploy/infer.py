import os
import time
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from infer_engine import *
from mindyolo.data import COCO80_TO_COCO91_CLASS
from mindyolo.data import COCODataset, create_loader
from mindyolo.utils import logger
from mindyolo.utils.config import parse_args
from mindyolo.utils.metrics import non_max_suppression, scale_coords, xyxy2xywh


def detect(nc=80, anchor=(), stride=()):
    no = nc + 5
    nl = len(anchor)
    na = len(anchor[0]) // 2
    anchor_grid = np.array(anchor).reshape(nl, 1, -1, 1, 1, 2)

    def forward(x):
        z = ()
        outs = ()
        for i in range(len(x)):
            out = x[i]
            bs, _, ny, nx = out.shape
            out = out.reshape(bs, na, no, ny, nx).transpose(0, 1, 3, 4, 2)
            outs += (out,)

            xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
            grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2))
            y = 1 / (1 + np.exp(-out))
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride[i]
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid[i]
            z += (y.reshape(bs, -1, no),)
        return np.concatenate(z, 1), outs

    return forward


def infer(cfg):
    # Create Network
    network = MindXModel("./models/yolov5s.om")
    de = detect(nc=80, anchor=cfg.network.anchors, stride=cfg.network.stride)

    dataset = COCODataset(
        dataset_path=cfg.data.val_set,
        img_size=cfg.img_size,
        transforms_dict=cfg.data.test_transforms,
        is_training=False, augment=False, rect=cfg.rect, single_cls=cfg.single_cls,
        batch_size=cfg.per_batch_size, stride=max(cfg.network.stride),
    )
    dataloader = create_loader(
        dataset=dataset,
        batch_collate_fn=dataset.test_collate_fn,
        dataset_column_names=dataset.dataset_column_names,
        batch_size=cfg.per_batch_size,
        epoch_size=1, rank=0, rank_size=1, shuffle=False, drop_remainder=False,
        num_parallel_workers=cfg.data.num_parallel_workers,
        python_multiprocessing=True
    )

    loader = dataloader.create_dict_iterator(output_numpy=True, num_epochs=1)
    # anno_json_path = os.path.join(self.cfg.data.dataset_dir, self.cfg.data.val_anno_path)
    dataset_dir = cfg.data.val_set[:-len(cfg.data.val_set.split('/')[-1])]
    anno_json_path = os.path.join(dataset_dir, 'annotations/instances_val2017.json')
    coco91class = COCO80_TO_COCO91_CLASS
    is_coco_dataset = ('coco' in cfg.data.dataset_name)

    step_num = dataloader.get_dataset_size()
    sample_num = 0
    infer_times = 0.
    nms_times = 0.
    result_dicts = []
    for i, data in enumerate(loader):
        imgs, _, paths, ori_shape, pad, hw_scale = data['image'], data['labels'], data['img_files'], \
            data['hw_ori'], data['pad'], data['hw_scale']
        nb, _, height, width = imgs.shape

        # Run infer
        _t = time.time()
        out = network.infer(imgs)  # inference and training outputs
        out, _ = de(out)
        infer_times += time.time() - _t

        # Run NMS
        t = time.time()
        out = non_max_suppression(out, conf_thres=cfg.conf_thres, iou_thres=cfg.iou_thres,
                                  multi_label=True, time_limit=cfg.nms_time_limit)
        nms_times += time.time() - t

        # Statistics pred
        for si, pred in enumerate(out):
            path = Path(str(paths[si]))
            sample_num += 1
            if len(pred) == 0:
                continue

            # Predictions
            predn = np.copy(pred)
            scale_coords(imgs[si].shape[1:], predn[:, :4], ori_shape[si], ratio=hw_scale[si],
                         pad=pad[si])  # native-space pred

            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            box = xyxy2xywh(predn[:, :4])  # xywh
            box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
            for p, b in zip(pred.tolist(), box.tolist()):
                result_dicts.append({'image_id': image_id,
                                     'category_id': coco91class[int(p[5])] if is_coco_dataset else int(p[5]),
                                     'bbox': [round(x, 3) for x in b],
                                     'score': round(p[4], 5)})
        print(f"Sample {step_num}/{i + 1}, time cost: {(time.time() - _t) * 1000:.2f} ms.")

        # Compute mAP
    try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        anno = COCO(anno_json_path)  # init annotations api
        pred = anno.loadRes(result_dicts)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        if is_coco_dataset:
            eval.params.imgIds = [int(Path(im_file).stem) for im_file in dataset.img_files]
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
    except Exception as e:
        logger.warning(f'pycocotools unable to run: {e}')
        raise e

    t = tuple(x / sample_num * 1E3 for x in (infer_times, nms_times, infer_times + nms_times)) + \
        (height, width, cfg.per_batch_size)  # tuple
    logger.info(f'Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g;' % t)

    return map, map50


if __name__ == '__main__':
    args = parse_args('val')
    infer(args)
