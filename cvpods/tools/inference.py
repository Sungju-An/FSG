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

from config import config  # 기존 학습 시 사용한 config 파일


def load_model(cfg, model_path):
    """ 모델을 불러오는 함수 """
    cfg.MODEL.WEIGHTS = model_path
    predictor = DefaultPredictor(cfg)  # Inference 모드로 변환
    return predictor


def visualize_results(image, outputs, dataset_name, save_path):
    """ 객체 탐지 결과를 시각화하여 저장하는 함수 """
    try:
        metadata = _get_builtin_metadata(dataset_name)  # ✅ Metadata 가져오기
    except KeyError:
        metadata = {"thing_classes": ["unknown"]}  # ✅ 기본 클래스 설정

    visualizer = Visualizer(image[:, :, ::-1], scale=1.0)  # ✅ metadata 제거
    v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))

    # ✅ metadata는 draw_instance_predictions()에서만 사용해야 함
    v.metadata = metadata  

    result_image = v.get_image()[:, :, ::-1]  # BGR -> RGB 변환
    cv2.imwrite(save_path, result_image)
    logger.info(f"결과 저장: {save_path}")




def main(args):
    cfg = config
    predictor = load_model(cfg, args.model_path)
    logger.info(f"모델 {args.model_path} 로드 완료")

    dataset_name = cfg.DATASETS.TEST[0]  # ✅ 데이터셋 이름 가져오기

    image = cv2.imread(args.image_path)
    if image is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {args.image_path}")

    outputs = predictor(image)
    visualize_results(image, outputs, dataset_name, args.output_path)  # ✅ dataset_name을 전달



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3T Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
    parser.add_argument("--image-path", type=str, required=True, help="테스트할 이미지 경로")
    parser.add_argument("--output-path", type=str, required=True, help="결과 이미지 저장 경로")

    args = parser.parse_args()
    main(args)
