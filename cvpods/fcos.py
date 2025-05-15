import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from dataset import build_train_loader, build_val_loader  # ⚠️ validation loader 필요
from loguru import logger
from losses import HMfocalLoss

from cvpods.engine.runner import (RUNNERS, DefaultCheckpointer, DefaultRunner,
                                  DistributedDataParallel, Infinite,
                                  auto_scale_config, comm, get_bn_modules,
                                  hooks, maybe_convert_module)
from cvpods.layers import cat
from cvpods.modeling.losses import iou_loss
from cvpods.modeling.meta_arch.retinanet import permute_to_N_HWA_K
from cvpods.structures import Boxes, pairwise_iou
import copy

@RUNNERS.register()
class SemiRunner(DefaultRunner):
        def __init__(self, cfg, build_model):
        if cfg.WANDB and comm.is_main_process():
            wandb.init(project='HT', entity='xxx')
            wandb.config.update(cfg)
            wandb.run.name = Path(cfg.OUTPUT_DIR).stem
            wandb.define_metric("AP50_tea", summary="max")
            wandb.define_metric("AP50_stu", summary="max")

        self.step2_start = cfg.TRAINER.STEP2
        self.iter_min = cfg.TRAINER.STEP2
        self.rgb_ir_change_total = cfg.TRAINER.RGB_IR_CHANGE_TOTAL
        self.rgb_ir_change_rgb = cfg.TRAINER.RGB_IR_CHANGE_RGB
        self.rgb_ir_change_ir = self.rgb_ir_change_total - self.rgb_ir_change_rgb
        self.rgb_ir_change_increase = cfg.TRAINER.RGB_IR_CHANGE_INCREASE

        self._hooks = []
        self.data_loader = build_train_loader(cfg)
        self.val_loader = build_val_loader(cfg)
        self.best_ap50 = 0.

        model = build_model(cfg)
        self.model = maybe_convert_module(model)
        logger.info(f"Model: \n{self.model}")

        self.optimizer = self.build_optimizer(cfg, self.model)

        if cfg.TRAINER.FP16.ENABLED:
            self.mixed_precision = True
            if cfg.TRAINER.FP16.TYPE == "APEX":
                from apex import amp
                self.model, self.optimizer = amp.initialize(
                    self.model, self.optimizer, opt_level=cfg.TRAINER.FP16.OPTS.OPT_LEVEL
                )
        else:
            self.mixed_precision = False

        if comm.get_world_size() > 1:
            torch.cuda.set_device(comm.get_local_rank())
            if cfg.MODEL.DDP_BACKEND == "torch":
                self.model = DistributedDataParallel(
                    self.model,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=False,
                    find_unused_parameters=True
                )
            elif cfg.MODEL.DDP_BACKEND == "apex":
                from apex.parallel import DistributedDataParallel as ApexDistributedDataParallel
                self.model = ApexDistributedDataParallel(self.model)
            else:
                raise ValueError("non-supported DDP backend: {}".format(cfg.MODEL.DDP_BACKEND))

        if not cfg.SOLVER.LR_SCHEDULER.get("EPOCH_WISE", False):
            epoch_iters = -1
        else:
            epoch_iters = cfg.SOLVER.LR_SCHEDULER.get("EPOCH_ITERS")
            logger.warning(f"Setup LR Scheduler in EPOCH mode: {epoch_iters}")

        auto_scale_config(cfg, self.data_loader)

        self.scheduler = self.build_lr_scheduler(cfg, self.optimizer, epoch_iters=epoch_iters)
        self.model.train()
        self._data_loader_iter = iter(self.data_loader)
        self._val_loader_iter = iter(self.val_loader)

        self.start_iter = 0
        self.start_epoch = 0
        self.max_iter = cfg.SOLVER.LR_SCHEDULER.MAX_ITER
        self.max_epoch = cfg.SOLVER.LR_SCHEDULER.MAX_EPOCH
        self.window_size = cfg.TRAINER.get("WINDOW_SIZE", None)

        self.cfg = cfg
        self.burn_in_steps = cfg.TRAINER.SSL.BURN_IN_STEPS

        # ✅ Meta Pseudo Labels용 teacher 모델 및 optimizer 정의 (RGB & IR)
        self.teacher_model = copy.deepcopy(self.model)
        self.teacher_model.train()
        self.teacher_optimizer = torch.optim.SGD(
            self.teacher_model.parameters(),
            lr=0.003,
            momentum=0.9,
            weight_decay=5e-4
        )

        self.teacher_model_ir = copy.deepcopy(self.model)
        self.teacher_model_ir.train()
        self.teacher_optimizer_ir = torch.optim.SGD(
            self.teacher_model_ir.parameters(),
            lr=0.003,
            momentum=0.9,
            weight_decay=5e-4
        )

        self.checkpointer = DefaultCheckpointer(
            self.model,
            cfg.OUTPUT_DIR,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

        self.alpha = cfg.TRAINER.DISTILL.HM.ALPHA
        self.beta = cfg.TRAINER.DISTILL.HM.BETA

        self.vfl_loss = HMfocalLoss(
            use_sigmoid=cfg.MODEL.FCOS.VFL.USE_SIGMOID,
            alpha=cfg.MODEL.FCOS.VFL.ALPHA,
            gamma=cfg.MODEL.FCOS.VFL.GAMMA,
            weight_type=cfg.MODEL.FCOS.VFL.WEIGHT_TYPE,
            loss_weight=cfg.MODEL.FCOS.VFL.LOSS_WEIGHT
        ).cuda()

        self.register_hooks(self.build_hooks())


    def run_step(self):
        assert self.model.training
        start = time.perf_counter()

        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        unsup_weak, unsup_strong, _, _ = zip(*data)
        unsup_weak = list(unsup_weak)
        unsup_strong = list(unsup_strong)

        # 🔁 지그재그 학습: RGB ↔ IR 번갈아 학습
        use_ir = (self.iter // self.rgb_ir_change_total) % 2 == 1
        teacher_model = self.teacher_model_ir if use_ir else self.teacher_model
        teacher_optimizer = self.teacher_optimizer_ir if use_ir else self.teacher_optimizer

        with torch.no_grad():
            teacher_logits, teacher_deltas, teacher_quality, teacher_boxes = teacher_model(unsup_weak, get_data=True)

        student_logits, student_deltas, student_quality, student_boxes = self.model(unsup_strong, get_data=True)

        loss_dict = self.get_distill_loss(
            student_logits, student_deltas, student_quality,
            teacher_logits, teacher_deltas, teacher_quality,
            student_boxes, teacher_boxes,
            name="_mpl"
        )
        losses = sum([v for v in loss_dict.values() if v.requires_grad])

        self.optimizer.zero_grad()
        losses.backward(create_graph=True)
        grads = [p.grad for p in self.model.parameters()]

        student_fast = copy.deepcopy(self.model)
        for p, g in zip(student_fast.parameters(), grads):
            if g is not None:
                p.data = p.data - self.optimizer.param_groups[0]['lr'] * g

        try:
            val_data = next(self._val_loader_iter)
        except StopIteration:
            self._val_loader_iter = iter(self.val_loader)
            val_data = next(self._val_loader_iter)

        val_inputs = [x["image"].to(self.model.device) for x in val_data]
        val_targets = [x["instances"].to(self.model.device) for x in val_data]

        val_logits, val_deltas, val_quality, _ = student_fast(val_data, get_data=True)
        val_gt_classes, val_gt_deltas, val_gt_quality = student_fast.get_ground_truth(val_data, val_targets)

        val_loss_dict = student_fast.losses(
            val_gt_classes, val_gt_deltas, val_gt_quality,
            val_logits, val_deltas, val_quality,
            name="_val"
        )
        val_loss = sum(val_loss_dict.values())

        teacher_optimizer.zero_grad()
        val_loss.backward()
        teacher_optimizer.step()

        self.optimizer.step()
        self.optimizer.zero_grad()

        data_time = time.perf_counter() - start
        self._write_metrics(loss_dict, data_time)
        self.step_outputs = {"loss_for_backward": losses}
        self.inner_iter += 1

