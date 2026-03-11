"""
DriveVLA Projector-Only Training
=================================
只训练 3 个 multimodal projector (mm_projector_scene / track / map),
冻结 LLM (Qwen2.5) 和 Vision Tower (UniAD)。

使用预计算的 UniAD 特征 (.pth) 进行训练, 避免在线运行 UniAD 前向,
大幅减少显存占用和训练时间。

启动方式:
  bash scripts/train_projector.sh
"""

import os
import sys
import json
import time
import copy
import math
import logging
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from transformers import Trainer, TrainingArguments
from torch.utils.data import DataLoader

from mmengine import Config

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init, rank0_print
from llava.train.train import DataArguments
from llava.constants import IGNORE_INDEX

from drivevla.data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from drivevla.data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset


# ============================================================
# 1. 参数定义
# ============================================================
@dataclass
class ModelArguments:
    model_path: str = field(default="checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct")
    attn_implementation: str = field(default="sdpa")


@dataclass
class ProjectorTrainingArguments(TrainingArguments):
    # 覆盖部分默认值, 使其适合 projector-only 训练
    output_dir: str = field(default="output/train_projector")
    num_train_epochs: float = field(default=3.0)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_steps: int = field(default=500)
    save_total_limit: int = field(default=3)
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    remove_unused_columns: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    ddp_find_unused_parameters: bool = field(default=False)


@dataclass
class DataTrainingArguments(DataArguments):
    data_path: str = field(default=None)
    lazy_preprocess: bool = field(default=True)
    frames_upbound: int = field(default=32)
    use_uniad_pth: bool = field(default=True)
    in_nuscenes_order: bool = field(default=True)


# ============================================================
# 2. 冻结/解冻工具
# ============================================================
def freeze_model(model):
    """冻结模型所有参数"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_projectors(model):
    """只解冻 3 个 projector"""
    inner = model.get_model()
    projector_names = ['mm_projector_scene', 'mm_projector_track', 'mm_projector_map']
    unfrozen_count = 0

    for name in projector_names:
        proj = getattr(inner, name, None)
        if proj is not None:
            for param in proj.parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
            rank0_print(f"  Unfrozen {name}: {sum(p.numel() for p in proj.parameters()):,} params")
        else:
            rank0_print(f"  WARNING: {name} not found in model")

    return unfrozen_count


def print_trainable_summary(model):
    """打印可训练参数摘要"""
    total_params = 0
    trainable_params = 0
    trainable_modules = {}

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # 提取 top-level module name
            module_key = '.'.join(name.split('.')[:4])
            if module_key not in trainable_modules:
                trainable_modules[module_key] = 0
            trainable_modules[module_key] += param.numel()

    rank0_print(f"\n{'='*60}")
    rank0_print(f"  Parameter Summary")
    rank0_print(f"{'='*60}")
    rank0_print(f"  Total params:      {total_params:>12,} ({total_params/1e6:.1f}M)")
    rank0_print(f"  Trainable params:  {trainable_params:>12,} ({trainable_params/1e6:.1f}M)")
    rank0_print(f"  Frozen params:     {total_params - trainable_params:>12,} ({(total_params - trainable_params)/1e6:.1f}M)")
    rank0_print(f"  Trainable ratio:   {trainable_params/total_params*100:.4f}%")
    rank0_print(f"\n  Trainable modules:")
    for name, count in sorted(trainable_modules.items()):
        rank0_print(f"    {name}: {count:,}")
    rank0_print(f"{'='*60}\n")


# ============================================================
# 3. 自定义 Trainer (适配 DriveVLA 的数据格式)
# ============================================================
class ProjectorTrainer(Trainer):
    """
    自定义 Trainer, 处理 DriveVLA 特殊的数据格式:
    - uniad_pth (预计算感知特征)
    - uniad_data (在线 UniAD 数据)
    - qa_instance_ind (查询物体索引)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        前向计算 loss。
        DriveVLA 的 forward 会在内部调用 prepare_inputs_labels_for_multimodal_uniad_vlm,
        将 uniad 特征注入到 inputs_embeds 中, 然后走标准的 CausalLM forward 计算 loss。
        """
        # 从 inputs 中提取 DriveVLA 需要的特殊字段
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        labels = inputs.get("labels")
        uniad_data = inputs.get("uniad_data", None)
        uniad_pth = inputs.get("uniad_pth", None)
        qa_instance_ind = inputs.get("qa_instance_ind", None)

        # 调用 LlavaQwenForCausalLM.forward()
        # forward 内部会:
        #   1. 从 kwargs 中提取 uniad_pth/uniad_data
        #   2. 调用 prepare_inputs_labels_for_multimodal_uniad_vlm
        #   3. 替换特殊 token (-201/-202/-203) 为感知特征 embedding
        #   4. 走标准 Qwen2ForCausalLM forward 计算 cross-entropy loss
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            uniad_pth=uniad_pth,
            uniad_data=uniad_data,
            qa_instance_ind=qa_instance_ind,
        )

        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """
        只保存 projector 权重, 而不是整个模型。
        大幅减小 checkpoint 体积。
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model = self.model
        inner = model.get_model() if hasattr(model, 'get_model') else model.module.get_model()

        # 保存 3 个 projector 的权重
        projector_state = {}
        for name in ['mm_projector_scene', 'mm_projector_track', 'mm_projector_map']:
            proj = getattr(inner, name, None)
            if proj is not None:
                for k, v in proj.state_dict().items():
                    projector_state[f"{name}.{k}"] = v.cpu()

        save_path = os.path.join(output_dir, "projector_weights.bin")
        torch.save(projector_state, save_path)
        rank0_print(f"Saved projector weights to {save_path} ({len(projector_state)} tensors)")

        # 同时保存完整模型 (可选, 方便直接用 from_pretrained 加载)
        if self.args.should_save:
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.config.save_pretrained(output_dir)
            model_to_save.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            rank0_print(f"Saved full model checkpoint to {output_dir}")


# ============================================================
# 4. 主训练函数
# ============================================================
def train():
    # 4.1 解析参数
    parser = transformers.HfArgumentParser(
        (ModelArguments, ProjectorTrainingArguments, DataTrainingArguments)
    )
    model_args, training_args, data_args = parser.parse_args_into_dataclasses()

    rank0_print("=" * 60)
    rank0_print("  DriveVLA Projector-Only Training")
    rank0_print("=" * 60)
    rank0_print(f"  model_path         = {model_args.model_path}")
    rank0_print(f"  output_dir         = {training_args.output_dir}")
    rank0_print(f"  data_path          = {data_args.data_path}")
    rank0_print(f"  use_uniad_pth      = {data_args.use_uniad_pth}")
    rank0_print(f"  num_train_epochs   = {training_args.num_train_epochs}")
    rank0_print(f"  batch_size         = {training_args.per_device_train_batch_size}")
    rank0_print(f"  grad_accum_steps   = {training_args.gradient_accumulation_steps}")
    rank0_print(f"  learning_rate      = {training_args.learning_rate}")
    rank0_print(f"  bf16               = {training_args.bf16}")

    # 4.2 加载模型
    rank0_print("\n>>> Loading model...")
    disable_torch_init()

    overwrite_config = {}
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_args.model_path,
        model_base=None,
        model_name="llava_qwen",
        device_map="auto",
        multimodal=True,
        attn_implementation=model_args.attn_implementation,
        overwrite_config=overwrite_config,
    )

    rank0_print(f"  Model loaded: {type(model).__name__}")
    rank0_print(f"  Config hidden_size = {model.config.hidden_size}")
    rank0_print(f"  Config mm_hidden_size = {getattr(model.config, 'mm_hidden_size', 'N/A')}")
    rank0_print(f"  Config mm_projector_type = {getattr(model.config, 'mm_projector_type', 'N/A')}")

    # 4.3 冻结所有参数, 然后只解冻 projector
    rank0_print("\n>>> Freezing all parameters...")
    freeze_model(model)

    rank0_print(">>> Unfreezing projectors only...")
    unfrozen = unfreeze_projectors(model)
    rank0_print(f"  Total unfrozen params: {unfrozen:,}")

    print_trainable_summary(model)

    # 验证: 确保 LLM 和 Vision Tower 确实被冻结
    inner = model.get_model()
    llm_trainable = sum(p.numel() for n, p in model.named_parameters()
                        if p.requires_grad and 'mm_projector' not in n)
    assert llm_trainable == 0, f"LLM/VisionTower should be frozen but has {llm_trainable} trainable params!"
    rank0_print("  Verified: LLM and Vision Tower are fully frozen.")

    # 4.4 加载数据集
    rank0_print("\n>>> Loading dataset...")
    uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")

    if data_args.use_uniad_pth:
        # 使用预计算特征, 不需要在线运行 UniAD
        dataset_cfg = uniad_cfg.data.test
    else:
        # 需要在线运行 UniAD
        dataset_cfg = uniad_cfg.data.train_llava_and_vision_tower

    train_dataset = LLaVANuScenesDataset(
        tokenizer=tokenizer,
        data_args=data_args,
        NuScenesE2EDataset_config=copy.deepcopy(dataset_cfg),
        llava_train_mode=True,
        llava_test_mode=False,
        use_uniad_pth=data_args.use_uniad_pth,
        in_nuscenes_order=data_args.in_nuscenes_order,
    )
    rank0_print(f"  Dataset size: {len(train_dataset)}")

    # 4.5 Data Collator
    data_collator = DataCollatorForLLaVANuScenesDataset(
        tokenizer=tokenizer,
        llava_train_mode=True,
    )

    # 4.6 启动训练
    rank0_print("\n>>> Starting training...")

    trainer = ProjectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # 训练
    if list((training_args.output_dir / "checkpoint-*").parent.glob("checkpoint-*")) if hasattr(training_args.output_dir, 'parent') else False:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 4.7 保存最终模型
    rank0_print("\n>>> Saving final model...")
    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    rank0_print(">>> Training complete!")


if __name__ == "__main__":
    train()
