#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import argparse
import numpy as np
from loguru import logger

from cvpods.checkpoint import DefaultCheckpointer
from cvpods.engine import DefaultPredictor
from cvpods.utils.visualizer import Visualizer
from cvpods.data.datasets.builtin_meta import _get_builtin_metadata  # ✅ MetadataCatalog 대신 사용
import time  # 🔹 추가

from config import config  # 기존 학습 시 사용한 config 파일

from net import build_model


def load_model(cfg, model_path):
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cuda:1"  
    predictor = DefaultPredictor(cfg)
    return predictor



# def visualize_results(image, outputs, dataset_name, save_path):
#     try:
#         metadata = _get_builtin_metadata(dataset_name)
#     except KeyError:
#         metadata = {"thing_classes": ["unknown"]}

#     visualizer = Visualizer(image[:, :, ::-1], scale=1.0)
#     v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
#     v.metadata = metadata
#     result_image = v.get_image()[:, :, ::-1]
#     cv2.imwrite(save_path, result_image)
#     logger.info(f"결과 저장: {save_path}")



###원하는 클래스만 시각화###
# def visualize_results(image, outputs, dataset_name, save_path):
#     try:
#         metadata = _get_builtin_metadata(dataset_name)
#     except KeyError:
#         metadata = {"thing_classes": ["unknown"]}

#     instances = outputs["instances"].to("cpu")

#     # 🔹 원하는 class ID만 필터링 (예: class_id == 0 만)
#     target_class_id = 0  # person
#     keep = instances.pred_classes == target_class_id
#     filtered_instances = instances[keep]

#     visualizer = Visualizer(image[:, :, ::-1], scale=1.0)
#     v = visualizer.draw_instance_predictions(filtered_instances)
#     v.metadata = metadata
#     result_image = v.get_image()[:, :, ::-1]
#     cv2.imwrite(save_path, result_image)
#     logger.info(f"결과 저장: {save_path}")


###unkown 변경 시각화###
def visualize_results(image, outputs, dataset_name, save_path):
    try:
        metadata = _get_builtin_metadata(dataset_name)
    except KeyError:
        metadata = {"thing_classes": ["person", "car", "bicycle"]}

    visualizer = Visualizer(image[:, :, ::-1], scale=1.0)
    visualizer.metadata = metadata

    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy().astype(int)
    classes = instances.pred_classes.numpy().astype(int)
    scores = instances.scores.numpy()

    vis_img = image.copy()
    for box, cls, score in zip(boxes, classes, scores):
        if score < 0.6:
            continue

        # 시각화용 라벨만 unknown으로 설정 (원래 클래스는 그대로 둠)
        if score < 0.7:
            label = f"unknown: {score:.2f}"
            color = (0, 255, 255)  # 노란색
        else:
            if cls == 0:
                label = f"person: {score:.2f}"
                color = (0, 0, 255)
            elif cls == 1:
                label = f"car: {score:.2f}"
                color = (0, 255, 0)
            elif cls == 2:
                label = f"bicycle: {score:.2f}"
                color = (255, 0, 0)
            else:
                label = f"class{cls}: {score:.2f}"
                color = (255, 255, 255)

        # draw box and label
        cv2.rectangle(vis_img, (box[0], box[1]), (box[2], box[3]), color, 2)
        cv2.putText(vis_img, label, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.imwrite(save_path, vis_img)
    logger.info(f"결과 저장: {save_path}")



def main(args):
    
    cfg = config
    predictor = load_model(cfg, args.model_path)
    logger.info(f"모델 {args.model_path} 로드 완료")

    dataset_name = cfg.DATASETS.TEST[0]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_exts = [".jpg", ".jpeg", ".png"]
    image_files = [f for f in os.listdir(args.image_dir)
                   if os.path.splitext(f)[1].lower() in image_exts]

    for img_name in image_files:
        img_path = os.path.join(args.image_dir, img_name)
        image = cv2.imread(img_path)



        if image is None:
            logger.warning(f"[{img_name}] 이미지 읽기 실패")
            continue

        start_time = time.time()  # 🔹 추론 시작 시간
        outputs = predictor(image)
        end_time = time.time()  # 🔹 추론 종료 시간
        infer_time = end_time - start_time
        logger.info(f"[{img_name}] 추론 시간: {infer_time:.4f}초")

        save_path = os.path.join(args.output_dir, img_name)
        visualize_results(image, outputs, dataset_name, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3T Batch Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
    parser.add_argument("--image-dir", type=str, required=True, help="테스트할 이미지 폴더 경로")
    parser.add_argument("--output-dir", type=str, required=True, help="결과 이미지 저장 폴더")

    args = parser.parse_args()
    main(args)
