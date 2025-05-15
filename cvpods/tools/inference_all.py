#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import time
import torch
import numpy as np
from loguru import logger
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from cvpods.engine import DefaultPredictor
from cvpods.utils.visualizer import Visualizer
from cvpods.data.datasets.builtin_meta import _get_builtin_metadata
from config import config  # 기존 학습 config

def process_image(img_path, output_dir, model_path, dataset_name):
    # 각 프로세스에서 모델 개별 로딩
    cfg = config
    cfg.MODEL.WEIGHTS = model_path
    predictor = DefaultPredictor(cfg)

    image = cv2.imread(img_path)
    if image is None:
        logger.warning(f"[{img_path}] 이미지 읽기 실패")
        return

    # 🔍 순수 추론 시간 측정
    start_infer = time.time()
    outputs = predictor(image)
    end_infer = time.time()
    infer_time = end_infer - start_infer
    logger.info(f"⏱️ {os.path.basename(img_path)} - 순수 추론 시간: {infer_time:.4f}초")

    try:
        thing_classes = _get_builtin_metadata(dataset_name)["thing_classes"]
    except KeyError:
        thing_classes = ["class_0", "class_1", "class_2"]

    visualizer = Visualizer(image[:, :, ::-1], scale=1.0)
    instances = outputs["instances"].to("cpu")

    # confidence 필터링
    keep = instances.scores >= 0.5
    instances = instances[keep]
    instances.pred_classes = instances.pred_classes.int()
    instances._fields["scores"] = instances.scores

    vis_output = visualizer.draw_instance_predictions(instances)
    result_image = vis_output.get_image()[:, :, ::-1]

    save_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(save_path, result_image)
    logger.info(f"✅ 저장 완료: {save_path}")



def main_parallel(model_path, image_dir, output_dir, num_workers=11):
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = config.DATASETS.TEST[0]

    image_exts = [".jpg", ".jpeg", ".png"]
    image_files = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in image_exts
    ]

    logger.info(f"총 이미지 수: {len(image_files)}")

    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        func = partial(process_image,
                       output_dir=output_dir,
                       model_path=model_path,
                       dataset_name=dataset_name)
        list(executor.map(func, image_files))

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"🔥 전체 추론 시간: {total_time:.2f}초")
    logger.info(f"🔥 이미지당 평균 추론 시간: {total_time / len(image_files):.2f}초")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="D3T 병렬 추론 + 시간 측정")
    parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
    parser.add_argument("--image-dir", type=str, required=True, help="이미지 폴더 경로")
    parser.add_argument("--output-dir", type=str, required=True, help="결과 저장 경로")
    parser.add_argument("--num-workers", type=int, default=8, help="병렬 처리할 프로세스 수")
    args = parser.parse_args()

    main_parallel(args.model_path, args.image_dir, args.output_dir, args.num_workers)
