#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from util.io import store_gz_json
from util.video import get_metadata
from util.coco import COCO_NAMES


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('video_file')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--chunk_len', type=int, default=1000)
    parser.add_argument('-o', '--out_dir')
    return parser.parse_args()


SELECTED_CLASS_IDS = {COCO_NAMES['person'], COCO_NAMES['sports ball']}


class VideoAsFrames(Dataset):

    def __init__(self, video_file):
        self.meta = get_metadata(video_file)

        self.vc = None
        assert os.path.exists(video_file)
        self.video_file = video_file

    def __getitem__(self, i):
        if self.vc is None:
            self.vc = cv2.VideoCapture(self.video_file)
            self.frame_num = 0

        ret, frame = self.vc.read()

        frame_num = self.frame_num
        self.frame_num += 1

        if not ret:
            self.vc.release()
            print(self.frame_num, self.video_file)
            frame = np.zeros((2, 2, 3), np.uint8)
        return frame_num, ret, frame

    def __len__(self):
        return self.meta.num_frames


def show(img, dets):
    for det in dets:
        x, y, w, h = det['box']
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 0, 255), 4)
        cv2.fillPoly(img, [np.array(det['seg']).astype(int)], (0, 255, 0))
    cv2.imshow('img', img)
    cv2.waitKey(1000)


def infer(model, img):
    # Not batched, not quite efficient
    result = model([img.numpy()], verbose=False)[0]

    dets = []
    count = 0
    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls)
        if not cls_id in SELECTED_CLASS_IDS:
            continue

        x, y, x2, y2 = box.xyxy[0].tolist()
        w, h = x2 - x, y2 -y
        conf = box.conf.item()

        dets.append({
            'id': count,
            'box': [x, y, w, h],
            'conf': conf,
            'cls': cls_id,
            'seg': result.masks[i].xy[0].tolist()
        })
        count += 1
    return dets


def main(args):
    yolo_model = YOLO('yolov8x-seg')
    yolo_model.info()

    dataset = VideoAsFrames(args.video_file)
    print('Num frames:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, num_workers=1)

    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    buffer = []
    for frame_num, ret, frame in tqdm(dataloader):
        B, C, H, W = frame.shape
        assert B == 1
        frame = frame[0]

        if frame_num % args.chunk_len == 0:
            if len(buffer) > 0 and args.out_dir is not None:
                store_gz_json(os.path.join(
                    args.out_dir, 'detect_{:8d}.json.gz'.format(
                        frame_num - args.chunk_len)),
                    buffer)
            buffer = []

        if not ret:
            print('Failed to load frame:', frame_num)
            break

        frame_pred = infer(yolo_model, frame)
        if args.out_dir is None:
            show(frame.numpy(), frame_pred)

        buffer.append(frame_pred)

    if len(buffer) > 0 and args.out_dir is not None:
        store_gz_json(os.path.join(
            args.out_dir, 'detect_{:8d}.json.gz'.format(
                frame_num - args.chunk_len)),
            buffer)
    print('Done!')


if __name__ == '__main__':
    main(get_args())
