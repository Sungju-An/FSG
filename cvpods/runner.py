import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import wandb
from dataset import build_train_loader, build_val_loader
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


    def build_hooks(self):
        cfg = self.cfg

        ret = [
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.IterationTimer(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            ) if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model) else None,
        ]

        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_iter=self.max_iter,
                max_epoch=self.max_epoch
            ))

            def test_and_save_results():
                logger.info('####################Evaluating: Student####################')
                self._last_eval_results = self.test(self.cfg, self.model)

                if self.cfg.WANDB:
                    metric = {key + '_stu': value for key, value in self._last_eval_results['bbox'].items()}
                    wandb.log(metric, step=self.iter)
                    wandb.run.summary.update(metric)

                return self._last_eval_results

            def test_and_save_teacher_results():
                logger.info("####################Evaluating: Teacher (RGB) ####################")
                results = self.test(self.cfg, self.teacher_model)
                if self.cfg.WANDB:
                    metric = {key + '_tea_rgb': value for key, value in results['bbox'].items()}
                    wandb.log(metric, step=self.iter)
                return results

            def test_and_save_teacher_ir_results():
                logger.info("####################Evaluating: Teacher (IR) ####################")
                results = self.test(self.cfg, self.teacher_model_ir)
                if self.cfg.WANDB:
                    metric = {key + '_tea_ir': value for key, value in results['bbox'].items()}
                    wandb.log(metric, step=self.iter)
                return results

            ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))
            ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_teacher_results))
            ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_teacher_ir_results))

            ret.append(hooks.PeriodicWriter(
                self.build_writers(), period=self.cfg.GLOBAL.LOG_INTERVAL
            ))

        return [hook for hook in ret if hook is not None]


    def resume_or_load(self, resume=True):
        self.checkpointer.resume = resume

        if resume:
            self.start_iter = (
                self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=resume
                ).get("iteration", -1) + 1
            )
        else:
            self.start_iter = (
                self.checkpointer.resume_or_load(
                    self.cfg.MODEL.WEIGHTS, resume=False
                ).get("iteration", -1) + 1
            )

        if self.max_epoch is not None:
            if isinstance(self.data_loader.sampler, Infinite):
                length = len(self.data_loader.sampler.sampler)
            else:
                length = len(self.data_loader)
            self.start_epoch = self.start_iter // length

        self.scheduler.last_epoch = self.start_iter - 1





    def run_step(self):                                   ########초기 burn in stage O
        assert self.model.training
        start = time.perf_counter()

        try:
            data = next(self._data_loader_iter)
        except StopIteration:
            self._data_loader_iter = iter(self.data_loader)
            data = next(self._data_loader_iter)

        unsup_weak, unsup_strong, sup_weak, sup_strong = zip(*data)
        unsup_weak = list(unsup_weak)
        unsup_strong = list(unsup_strong)
        sup_weak = list(sup_weak)
        sup_strong = list(sup_strong)

        if self.iter <= self.step2_start:
            # 🔥 Burn-in: RGB labeled data로 student supervised 학습
            inputs = sup_weak + sup_strong
            loss_dict = self.model(inputs)
            loss_dict = {k: v * self.cfg.TRAINER.DISTILL.SUP_WEIGHT for k, v in loss_dict.items()}
            losses = sum([v for v in loss_dict.values() if v.requires_grad])

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self._write_metrics(loss_dict, time.perf_counter() - start)
            self.step_outputs = {"loss_for_backward": losses}
            self.inner_iter += 1
            return

        # 🔁 지그재그 반복: iter에 따라 IR ↔ RGB 반복
        use_ir = (self.iter // self.rgb_ir_change_total) % 2 == 1

        # 🔔 도메인 전환 로그 출력
        if self.iter % self.rgb_ir_change_total == 0:
            logger.info(f"[Iter {self.iter}] 🔁 Switching to {'IR' if use_ir else 'RGB'} teacher phase.")
        teacher_model = self.teacher_model_ir if use_ir else self.teacher_model
        teacher_optimizer = self.teacher_optimizer_ir if use_ir else self.teacher_optimizer

        # MPL: teacher → pseudo-label → student 학습
        with torch.no_grad():
            tea_logits, tea_deltas, tea_quality, tea_boxes = teacher_model(unsup_weak, get_data=True)

        stu_logits, stu_deltas, stu_quality, stu_boxes = self.model(unsup_strong, get_data=True)

        loss_dict = self.get_distill_loss(
            stu_logits, stu_deltas, stu_quality,
            tea_logits, tea_deltas, tea_quality,
            stu_boxes, tea_boxes,
            name="_mpl"
        )
        losses = sum([v for v in loss_dict.values() if v.requires_grad])

        # student 1-step 업데이트 (create_graph 유지)
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

        # 지그재그 반복 업데이트
        if self.iter == 20000 or self.iter == 30000:
            self.rgb_ir_change_ir += self.rgb_ir_change_increase
            self.rgb_ir_change_rgb -= self.rgb_ir_change_increase

        if self.iter == self.iter_min + self.rgb_ir_change_total:
            self.iter_min += self.rgb_ir_change_total

        data_time = time.perf_counter() - start
        self._write_metrics(loss_dict, data_time)
        self.step_outputs = {"loss_for_backward": losses}
        self.inner_iter += 1











    def get_distill_loss(self,
                         student_logits, student_deltas, student_quality,
                         teacher_logits, teacher_deltas, teacher_quality,
                         box_xyxy=None, tea_box_xyxy=None, name=""):
        num_classes = self.cfg.MODEL.FCOS.NUM_CLASSES

        student_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in student_logits
        ], dim=1).view(-1, num_classes)
        teacher_logits = torch.cat([
            permute_to_N_HWA_K(x, num_classes) for x in teacher_logits
        ], dim=1).view(-1, num_classes)

        student_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in student_deltas
        ], dim=1).view(-1, 4)
        teacher_deltas = torch.cat([
            permute_to_N_HWA_K(x, 4) for x in teacher_deltas
        ], dim=1).view(-1, 4)

        student_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in student_quality
        ], dim=1).view(-1, 1)
        teacher_quality = torch.cat([
            permute_to_N_HWA_K(x, 1) for x in teacher_quality
        ], dim=1).view(-1, 1)

        with torch.no_grad():
            ratio = self.cfg.TRAINER.DISTILL.RATIO
            count_num = int(teacher_logits.size(0) * ratio)
            teacher_probs = teacher_logits.sigmoid()
            cls_prob = torch.max(teacher_probs, 1)[0]
            iou = teacher_quality.sigmoid().squeeze()
            hm = (cls_prob ** self.alpha) * (iou ** self.beta)
            sorted_vals, sorted_inds = torch.topk(hm, teacher_logits.size(0))
            mask = torch.zeros_like(hm)
            mask[sorted_inds[:count_num]] = 1.
            fg_num = sorted_vals[:count_num].sum()
            b_mask = mask > 0.

        delta_weights = torch.ones_like(teacher_quality).squeeze()
        loss_uncertainty = (1 - hm) / self.cfg.TRAINER.DISTILL.UN_REGULAR_ALPHA
        loss_weight = torch.exp(-loss_uncertainty.detach())

        loss_logits = F.binary_cross_entropy(
            student_logits.sigmoid(),
            teacher_probs,
            weight=loss_weight.unsqueeze(1),
            reduction="sum",
        ) / fg_num

        loss_deltas = (iou_loss(
            student_deltas,
            teacher_deltas,
            box_mode="ltrb",
            loss_type='giou',
            reduction="none",
        ) * delta_weights * loss_weight.unsqueeze(1)).mean()

        loss_quality = F.binary_cross_entropy(
            student_quality.sigmoid(),
            teacher_quality.sigmoid(),
            weight=loss_weight.unsqueeze(1),
            reduction='mean'
        )

        return {
            "distill_loss_logits" + name: loss_logits,
            "distill_loss_deltas" + name: loss_deltas,
            "distill_loss_quality" + name: loss_quality,
            "fore_ground_sum" + name: fg_num,
        }

    