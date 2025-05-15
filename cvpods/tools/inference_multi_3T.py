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
from cvpods.data.datasets.builtin_meta import _get_builtin_metadata

from config import config
from net import build_model


import copy

def load_model(cfg, model_path, teacher_type=None):
    cfg = copy.deepcopy(cfg)
    cfg.MODEL.WEIGHTS = model_path
    model = build_model(cfg)
    checkpointer = DefaultCheckpointer(model)
    
    if teacher_type is None:
        # student는 name 없이 기본 로딩
        checkpointer.resume_or_load(model_path, resume=False)
    else:
        # ema 또는 ema_ir teacher 로딩
        checkpointer.resume_or_load(model_path, resume=False, name=teacher_type)
    
    model.eval()
    return model


def run_inference(model, image):
    with torch.no_grad():
        inputs = [{"image": torch.as_tensor(image.transpose(2, 0, 1)).float().cuda()}]
        outputs = model(inputs)[0]
        return outputs


def visualize_results(image, outputs, dataset_name, save_path):
    try:
        metadata = _get_builtin_metadata(dataset_name)
    except KeyError:
        metadata = {"thing_classes": ["unknown"]}

    visualizer = Visualizer(image[:, :, ::-1], scale=1.0)
    v = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
    v.metadata = metadata
    result_image = v.get_image()[:, :, ::-1]
    cv2.imwrite(save_path, result_image)
    logger.info(f"결과 저장: {save_path}")


def main(args):
    cfg = config
    dataset_name = cfg.DATASETS.TEST[0]

    logger.info("💡 Student 모델 로드")
    model_student = load_model(cfg, args.model_path, teacher_type=None)

    logger.info("💡 RGB Teacher (ema) 모델 로드")
    model_rgb_teacher = load_model(cfg, args.model_path, teacher_type="ema")

    logger.info("💡 Thermal Teacher (ema_ir) 모델 로드")
    model_ir_teacher = load_model(cfg, args.model_path, teacher_type="ema_ir")

    models = {
        "student": model_student,
        "rgb_teacher": model_rgb_teacher,
        "thermal_teacher": model_ir_teacher
    }

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    image_exts = [".jpg", ".jpeg", ".png"]
    image_files = [f for f in os.listdir(args.image_dir)
                   if os.path.splitext(f)[1].lower() in image_exts]

    for model_name, model in models.items():
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        for img_name in image_files:
            img_path = os.path.join(args.image_dir, img_name)
            image = cv2.imread(img_path)

            if image is None:
                logger.warning(f"[{img_name}] 이미지 읽기 실패")
                continue

            outputs = run_inference(model, image)
            save_path = os.path.join(model_output_dir, img_name)
            visualize_results(image, outputs, dataset_name, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3T All-Model Batch Inference")
    parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
    parser.add_argument("--image-dir", type=str, required=True, help="테스트할 이미지 폴더 경로")
    parser.add_argument("--output-dir", type=str, required=True, help="결과 이미지 저장 폴더")

    args = parser.parse_args()
    main(args)
