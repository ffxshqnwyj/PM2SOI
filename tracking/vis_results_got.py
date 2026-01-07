import argparse
import os
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
from tqdm import tqdm

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from lib.test.evaluation import get_dataset  # noqa: E402
from lib.test.utils.load_text import load_text  # noqa: E402


def draw_box(image, box, color, thickness=2):
    x, y, w, h = box
    pt1 = (int(round(x)), int(round(y)))
    pt2 = (int(round(x + w)), int(round(y + h)))
    cv.rectangle(image, pt1, pt2, color, thickness, lineType=cv.LINE_AA)


def visualize(dataset_name: str,
              result_dir: Path,
              save_root: Path,
              seq_filter=None,
              max_frames: int = -1):
    dataset = get_dataset(dataset_name)
    save_root.mkdir(parents=True, exist_ok=True)

    seq_filter = set(seq_filter) if seq_filter else None

    for seq in tqdm(dataset, desc=f'Visualizing {dataset_name}'):
        if seq_filter and seq.name not in seq_filter:
            continue

        result_file = result_dir / f'{seq.name}.txt'
        if not result_file.is_file():
            print(f'[WARN] Result not found for {seq.name}: {result_file}')
            continue

        pred_bboxes = load_text(str(result_file), delimiter=('\t', ','), dtype=np.float64)
        pred_bboxes = np.array(pred_bboxes, dtype=np.float32)

        gt_bboxes = np.array(seq.ground_truth_rect, dtype=np.float32) if seq.ground_truth_rect is not None else None

        seq_save_dir = save_root / seq.name
        seq_save_dir.mkdir(parents=True, exist_ok=True)

        frame_count = len(seq.frames)
        for frame_id in range(frame_count):
            frame_path = seq.frames[frame_id]
            image = cv.imread(frame_path, cv.IMREAD_COLOR)
            if image is None:
                print(f'[WARN] Unable to read frame: {frame_path}')
                continue

            if gt_bboxes is not None and frame_id < len(gt_bboxes):
                draw_box(image, gt_bboxes[frame_id], (0, 255, 0), thickness=2)  # green for GT

            if frame_id < len(pred_bboxes):
                draw_box(image, pred_bboxes[frame_id], (0, 0, 255), thickness=2)  # red for prediction

            save_path = seq_save_dir / f'{frame_id:05d}.jpg'
            cv.imwrite(str(save_path), image)

            if 0 <= max_frames <= frame_id + 1:
                break


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tracking results by overlaying bounding boxes.')
    parser.add_argument('--dataset', required=True, help='Dataset name, e.g. got10k_test')
    parser.add_argument('--result_path', required=True,
                        help='Directory containing tracking results (*.txt per sequence)')
    parser.add_argument('--save_root', required=True, help='Directory to save visualized frames')
    parser.add_argument('--seq', nargs='+', default=None,
                        help='Specific sequence names to visualize (default: all sequences)')
    parser.add_argument('--max_frames', type=int, default=-1,
                        help='Visualize at most this many frames per sequence (-1 for all)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize(
        dataset_name=args.dataset,
        result_dir=Path(args.result_path),
        save_root=Path(args.save_root),
        seq_filter=args.seq,
        max_frames=args.max_frames
    )
