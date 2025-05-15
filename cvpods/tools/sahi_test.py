# #!/usr/bin/python3
# # -*- coding: utf-8 -*-
# import os
# import cv2
# import torch
# import argparse
# import numpy as np
# from loguru import logger

# from cvpods.engine import DefaultPredictor
# from cvpods.utils.visualizer import Visualizer
# from cvpods.data.datasets.builtin_meta import _get_builtin_metadata
# from config import config  # 기존 학습 시 사용한 config 파일

# from sahi.predict import get_sliced_prediction
# from sahi.utils.cv import visualize_object_predictions
# from sahi import AutoDetectionModel
# from sahi.prediction import ObjectPrediction, PredictionResult

# class CustomAutoDetectionModel(AutoDetectionModel):
#     def __init__(self, predictor, confidence_threshold=0.5):
#         self.predictor = predictor
#         self.confidence_threshold = confidence_threshold
#         self.object_prediction_list = []
#         self.category_mapping = {0: "person", 1: "vehicle", 2: "bicycle"}  # ✅ D3T 클래스 매핑 추가

#     def perform_inference(self, image: np.ndarray):
#         """SAHI가 기대하는 `perform_inference()`를 구현"""
#         outputs = self.predictor(image)
#         instances = outputs["instances"].to("cpu")

#         predictions = []
#         if instances.has("pred_boxes"):
#             boxes = instances.pred_boxes.tensor.numpy()
#             scores = instances.scores.numpy()
#             labels = instances.pred_classes.numpy()

#             for box, score, label in zip(boxes, scores, labels):
#                 if score >= self.confidence_threshold:
#                     predictions.append({
#                         "bbox": box.tolist(),
#                         "score": float(score),
#                         "category_id": int(label)
#                     })

#         self.object_prediction_list = predictions
#         return predictions

#     def convert_original_predictions(self, shift_amount=[0, 0], full_shape=[0, 0]):
#         """좌표 변환을 적용하여 박스가 원본 위치에 맞도록 수정"""
#         self.object_prediction_list = [
#             ObjectPrediction(
#                 bbox=[
#                     pred["bbox"][0] + shift_amount[0],  # X 좌표 변환
#                     pred["bbox"][1] + shift_amount[1],  # Y 좌표 변환
#                     pred["bbox"][2] + shift_amount[0],  # X2 좌표 변환
#                     pred["bbox"][3] + shift_amount[1],  # Y2 좌표 변환
#                 ],
#                 score=pred["score"],
#                 category_id=pred["category_id"],
#                 full_shape=full_shape
#             )
#             for pred in self.object_prediction_list
#         ]


# def load_model(cfg, model_path):
#     """ SAHI가 요구하는 모델 반환 """
#     cfg.MODEL.WEIGHTS = model_path
#     predictor = DefaultPredictor(cfg)
#     return CustomAutoDetectionModel(predictor)

# def main(args):
#     cfg = config
#     detection_model = load_model(cfg, args.model_path)
#     logger.info(f"모델 {args.model_path} 로드 완료")

#     image = cv2.imread(args.image_path)
#     if image is None:
#         raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {args.image_path}")

#     # ✅ SAHI 기반 슬라이싱 추론 적용
#     results = get_sliced_prediction(
#         args.image_path,
#         detection_model,
#         slice_height=300,
#         slice_width=300,
#         overlap_height_ratio=0.2,
#         overlap_width_ratio=0.2,
#     )

#     # ✅ SAHI 결과 시각화
#     visualize_object_predictions(
#         image=np.array(results.image),
#         object_prediction_list=results.object_prediction_list,
#         output_dir=os.path.dirname(args.output_path),
#         file_name=os.path.basename(args.output_path).split('.')[0],
#         hide_labels=True,
#         hide_conf=True
#     )
#     logger.info(f"SAHI 기반 결과 저장: {args.output_path}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="D3T SAHI Inference Script")
#     parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
#     parser.add_argument("--image-path", type=str, required=True, help="테스트할 이미지 경로")
#     parser.add_argument("--output-path", type=str, required=True, help="결과 이미지 저장 경로")

#     args = parser.parse_args()
#     main(args)





#########폴더 내에 여러 이미지 추론############

import glob
#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import argparse
import numpy as np
from loguru import logger

from cvpods.engine import DefaultPredictor
from cvpods.utils.visualizer import Visualizer
from cvpods.data.datasets.builtin_meta import _get_builtin_metadata
from config import config  # 기존 학습 시 사용한 config 파일

from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from sahi import AutoDetectionModel
from sahi.prediction import ObjectPrediction, PredictionResult

import time

class CustomAutoDetectionModel(AutoDetectionModel):
    def __init__(self, predictor, confidence_threshold=0.7):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold
        self.object_prediction_list = []
        self.category_mapping = {0: "person", 1: "vehicle", 2: "bicycle"}  # ✅ D3T 클래스 매핑 추가

    ###오리지널##
    # def perform_inference(self, image: np.ndarray):
    #     """SAHI가 기대하는 `perform_inference()`를 구현"""
    #     outputs = self.predictor(image)
    #     instances = outputs["instances"].to("cpu")

    #     predictions = []
    #     if instances.has("pred_boxes"):
    #         boxes = instances.pred_boxes.tensor.numpy()
    #         scores = instances.scores.numpy()
    #         labels = instances.pred_classes.numpy()

    #         for box, score, label in zip(boxes, scores, labels):
    #             if score >= self.confidence_threshold:
    #                 predictions.append({
    #                     "bbox": box.tolist(),
    #                     "score": float(score),
    #                     "category_id": int(label)
    #                 })

    #     self.object_prediction_list = predictions
    #     return predictions
    

    ###unknown시각화

    def perform_inference(self, image: np.ndarray):
        """SAHI가 기대하는 `perform_inference()`를 구현"""
        outputs = self.predictor(image)
        instances = outputs["instances"].to("cpu")

        predictions = []
        if instances.has("pred_boxes"):
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            labels = instances.pred_classes.numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score >= 0.68:
                    if score < 0.7:
                        category_id = 99  # 🔸 unknown 클래스 ID (SAHI 시각화에서 별도 처리 가능)
                    else:
                        category_id = int(label)
                    predictions.append({
                        "bbox": box.tolist(),
                        "score": float(score),
                        "category_id": category_id
                    })

        self.object_prediction_list = predictions
        return predictions


    def convert_original_predictions(self, shift_amount=[0, 0], full_shape=[0, 0]):
        """좌표 변환을 적용하여 박스가 원본 위치에 맞도록 수정"""
        self.object_prediction_list = [
            ObjectPrediction(
                bbox=[
                    pred["bbox"][0] + shift_amount[0],  # X 좌표 변환
                    pred["bbox"][1] + shift_amount[1],  # Y 좌표 변환
                    pred["bbox"][2] + shift_amount[0],  # X2 좌표 변환
                    pred["bbox"][3] + shift_amount[1],  # Y2 좌표 변환
                ],
                score=pred["score"],
                category_id=pred["category_id"],
                full_shape=full_shape
            )
            for pred in self.object_prediction_list
        ]


def load_model(cfg, model_path):
    """ SAHI가 요구하는 모델 반환 """
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = "cuda:1" 
    predictor = DefaultPredictor(cfg)
    return CustomAutoDetectionModel(predictor)

def main(args):
    cfg = config
    detection_model = load_model(cfg, args.model_path)
    logger.info(f"모델 {args.model_path} 로드 완료")

    image_paths = []

    if os.path.isdir(args.image_path):  # 폴더 입력 시
        image_paths = sorted(glob.glob(os.path.join(args.image_path, "*.[jp][pn]g")))
        if not image_paths:
            raise FileNotFoundError(f"{args.image_path} 경로에 이미지 파일이 없습니다.")
        os.makedirs(args.output_path, exist_ok=True)
    elif os.path.isfile(args.image_path):  # 단일 이미지
        image_paths = [args.image_path]
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    else:
        raise ValueError(f"유효하지 않은 경로입니다: {args.image_path}")

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            logger.warning(f"이미지를 열 수 없습니다: {img_path}")
            continue


        start_time = time.time()  # 🔹 추론 시작 시간
        # SAHI 기반 슬라이싱 추론 적용
        results = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=300,
            slice_width=300,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
        )
        end_time = time.time()  # 🔹 추론 종료 시간
        infer_time = end_time - start_time
        logger.info(f" 추론 시간: {infer_time:.4f}초")

        # 결과 저장 경로 지정
        if os.path.isdir(args.image_path):
            filename = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(args.output_path, f"{filename}_sahi.png")
        else:
            save_path = args.output_path


            

        # 결과 시각화 및 저장
        visualize_object_predictions(
            image=np.array(results.image),
            object_prediction_list=results.object_prediction_list,
            output_dir=os.path.dirname(save_path),
            file_name=os.path.basename(save_path).split('.')[0],
            hide_labels=False,              #True
            hide_conf=False,                 #True
            rect_th=1,          # 🔽 박스 테두리 두께 (기본값은 2 또는 3)
            text_size=0.5,      # 🔽 텍스트 크기 (기본값은 1.0)
            text_th=1           # 🔽 텍스트 테두리 두께 (기본값은 2)
        )
        logger.info(f"SAHI 기반 결과 저장: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="D3T SAHI Inference Script")
    parser.add_argument("--model-path", type=str, required=True, help="pth 모델 파일 경로")
    parser.add_argument("--image-path", type=str, required=True, help="테스트할 이미지 경로")
    parser.add_argument("--output-path", type=str, required=True, help="결과 이미지 저장 경로")

    args = parser.parse_args()
    main(args)