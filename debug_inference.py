"""
DriveVLA 完整推理流程 Debug 脚本
=================================
直接运行: PYTHONPATH=$(pwd):$PYTHONPATH python debug_inference.py --model-path checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct
可选参数:
  --use-uniad-pth    使用预计算的 UniAD 特征 (.pth)
  --sample-idx 0     选择第几个样本进行 debug (默认 0)
  --max-new-tokens 512
  --model-path       模型路径
"""

import argparse
import os
import sys
import json
import time
import copy
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 工具函数
# ============================================================
def sep(title, char="=", width=100):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")

def print_tensor(name, t, max_elements=8):
    """打印 tensor 的详细信息"""
    if t is None:
        print(f"  {name}: None")
        return
    if isinstance(t, (list, tuple)):
        print(f"  {name}: {type(t).__name__}, len={len(t)}")
        for i, item in enumerate(t[:3]):
            if isinstance(item, torch.Tensor):
                print(f"    [{i}]: shape={item.shape}, dtype={item.dtype}, device={item.device}")
                print(f"         min={item.min().item():.6f}, max={item.max().item():.6f}, mean={item.float().mean().item():.6f}")
            else:
                print(f"    [{i}]: {type(item).__name__} = {str(item)[:200]}")
        if len(t) > 3:
            print(f"    ... ({len(t) - 3} more items)")
        return
    if isinstance(t, torch.Tensor):
        print(f"  {name}: shape={t.shape}, dtype={t.dtype}, device={t.device}")
        print(f"         min={t.min().item():.6f}, max={t.max().item():.6f}, mean={t.float().mean().item():.6f}")
        flat = t.flatten()
        n = min(max_elements, flat.shape[0])
        print(f"         first {n} values: {flat[:n].tolist()}")
    elif isinstance(t, dict):
        print(f"  {name}: dict with {len(t)} keys: {list(t.keys())[:10]}")
    else:
        print(f"  {name}: {type(t).__name__} = {str(t)[:200]}")

def print_model_structure(model, prefix="", max_depth=3, current_depth=0):
    """打印模型结构（带参数量）"""
    if current_depth >= max_depth:
        return
    for name, child in model.named_children():
        num_params = sum(p.numel() for p in child.parameters())
        trainable_params = sum(p.numel() for p in child.parameters() if p.requires_grad)
        print(f"  {prefix}{name}: {child.__class__.__name__} "
              f"(params={num_params:,}, trainable={trainable_params:,})")
        print_model_structure(child, prefix=prefix + "  ", max_depth=max_depth, current_depth=current_depth + 1)


# ============================================================
# STEP 0: 参数解析
# ============================================================
sep("STEP 0: 参数解析")

parser = argparse.ArgumentParser(description="DriveVLA Debug Inference Script")
parser.add_argument("--model-path", type=str, default="checkpoints/DriveVLA-Qwen2.5-0.5B-Instruct")
parser.add_argument("--use-uniad-pth", action="store_true", help="Use precomputed UniAD features (.pth)")
parser.add_argument("--sample-idx", type=int, default=0, help="Which sample to debug")
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--attn-implementation", type=str, default="sdpa")
parser.add_argument("--data-path", type=str, default=None, help="Path to conversation data JSON")
args = parser.parse_args()

print(f"  model_path       = {args.model_path}")
print(f"  use_uniad_pth    = {args.use_uniad_pth}")
print(f"  sample_idx       = {args.sample_idx}")
print(f"  max_new_tokens   = {args.max_new_tokens}")
print(f"  attn_impl        = {args.attn_implementation}")
print(f"  data_path        = {args.data_path}")
print(f"  CUDA available   = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU              = {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory       = {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# STEP 1: 加载模型
# ============================================================
sep("STEP 1: 加载模型 — load_pretrained_model()")

from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

print(">>> 1.1 disable_torch_init() — 禁用默认权重初始化以加速加载")
disable_torch_init()

print(">>> 1.2 调用 load_pretrained_model()")
print(f"    model_path = {args.model_path}")
print(f"    model_name = 'llava_qwen'")
print(f"    multimodal = True")
print(f"    overwrite_config = {{image_aspect_ratio: 'pad', vision_tower_test_mode: True}}")

llava_model_args = {
    "multimodal": True,
    "attn_implementation": args.attn_implementation,
}
overwrite_config = {"image_aspect_ratio": "pad", "vision_tower_test_mode": True}
llava_model_args["overwrite_config"] = overwrite_config

t0 = time.time()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    args.model_path,
    model_base=None,
    model_name="llava_qwen",
    device_map=device,
    **llava_model_args
)
t_load = time.time() - t0
print(f"\n>>> 模型加载完成, 耗时 {t_load:.2f}s")

# ----- Debug: 模型基本信息 -----
sep("STEP 1 DEBUG: 模型详细信息")

print(f">>> 1.3 Tokenizer 信息:")
print(f"    type            = {type(tokenizer).__name__}")
print(f"    vocab_size      = {tokenizer.vocab_size}")
print(f"    model_max_length= {tokenizer.model_max_length}")
print(f"    padding_side    = {tokenizer.padding_side}")
print(f"    pad_token_id    = {tokenizer.pad_token_id}")
print(f"    eos_token_id    = {tokenizer.eos_token_id}")

# 检查特殊 token
from llava.constants import (
    IMAGE_TOKEN_INDEX, SCENE_TOKEN_INDEX, TRACK_TOKEN_INDEX,
    MAP_TOKEN_INDEX, OBJECT_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_SCENE_TOKEN, DEFAULT_TRACK_TOKEN,
    DEFAULT_MAP_TOKEN, DEFAULT_OBJECT_TOKEN,
    DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_END_TOKEN,
    DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_END_TOKEN,
)

special_tokens_map = {
    DEFAULT_IMAGE_TOKEN: IMAGE_TOKEN_INDEX,
    DEFAULT_SCENE_TOKEN: SCENE_TOKEN_INDEX,
    DEFAULT_TRACK_TOKEN: TRACK_TOKEN_INDEX,
    DEFAULT_MAP_TOKEN: MAP_TOKEN_INDEX,
    DEFAULT_OBJECT_TOKEN: OBJECT_TOKEN_INDEX,
}
print(f"\n>>> 1.4 特殊 Token 映射 (文本 → 负数 index):")
for tok, idx in special_tokens_map.items():
    tok_id = tokenizer.convert_tokens_to_ids(tok) if tok in tokenizer.get_vocab() else "NOT_IN_VOCAB"
    print(f"    '{tok}' → index={idx}, vocab_id={tok_id}")

print(f"\n>>> 1.5 控制 Token:")
for tok_name in [DEFAULT_TRAJ_START_TOKEN, DEFAULT_TRAJ_END_TOKEN,
                 DEFAULT_SCENE_START_TOKEN, DEFAULT_SCENE_END_TOKEN,
                 DEFAULT_TRACK_START_TOKEN, DEFAULT_TRACK_END_TOKEN,
                 DEFAULT_MAP_START_TOKEN, DEFAULT_MAP_END_TOKEN]:
    tok_id = tokenizer.convert_tokens_to_ids(tok_name) if tok_name in tokenizer.get_vocab() else "NOT_IN_VOCAB"
    print(f"    '{tok_name}' → vocab_id={tok_id}")

print(f"\n>>> 1.6 Model 类型: {type(model).__name__}")
print(f"    config.hidden_size      = {model.config.hidden_size}")
print(f"    config.num_hidden_layers = {model.config.num_hidden_layers}")
print(f"    config.vocab_size        = {model.config.vocab_size}")
print(f"    config.mm_hidden_size    = {getattr(model.config, 'mm_hidden_size', 'N/A')}")
print(f"    config.mm_projector_type = {getattr(model.config, 'mm_projector_type', 'N/A')}")
print(f"    config.mm_vision_tower   = {getattr(model.config, 'mm_vision_tower', 'N/A')}")
print(f"    image_processor          = {image_processor}")
print(f"    context_len              = {context_len}")

print(f"\n>>> 1.7 模型结构 (top-3 layers):")
print_model_structure(model, max_depth=3)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n>>> 1.8 参数统计:")
print(f"    总参数量      = {total_params:,} ({total_params / 1e6:.1f}M)")
print(f"    可训练参数量  = {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

# ----- Debug: Vision Tower -----
sep("STEP 1 DEBUG: Vision Tower (UniAD) 详细信息")

vision_tower = model.get_vision_tower()
print(f">>> 1.9 Vision Tower 类型: {type(vision_tower).__name__}")
print(f"    is_loaded              = {vision_tower.is_loaded}")
print(f"    vision_tower_name      = {vision_tower.vision_tower_name}")
print(f"    vision_tower_test_mode = {getattr(vision_tower, 'vision_tower_test_mode', 'N/A')}")

if hasattr(vision_tower, 'vision_tower') and vision_tower.vision_tower is not None:
    uniad_model = vision_tower.vision_tower
    print(f"    UniadTrackMapModel type = {type(uniad_model).__name__}")
    print(f"    vision_model type       = {type(uniad_model.vision_model).__name__}")
    vt_params = sum(p.numel() for p in uniad_model.parameters())
    print(f"    UniAD 参数量            = {vt_params:,} ({vt_params / 1e6:.1f}M)")

    # UniAD 子模块
    if hasattr(uniad_model.vision_model, 'img_backbone'):
        bb = uniad_model.vision_model.img_backbone
        print(f"    img_backbone type       = {type(bb).__name__} (ResNet-{getattr(bb, 'depth', '?')})")
        bb_params = sum(p.numel() for p in bb.parameters())
        print(f"    img_backbone params      = {bb_params:,} ({bb_params / 1e6:.1f}M)")

    if hasattr(uniad_model.vision_model, 'img_neck'):
        neck = uniad_model.vision_model.img_neck
        neck_params = sum(p.numel() for p in neck.parameters())
        print(f"    img_neck type           = {type(neck).__name__}, params={neck_params:,}")

    if hasattr(uniad_model.vision_model, 'pts_bbox_head'):
        head = uniad_model.vision_model.pts_bbox_head
        head_params = sum(p.numel() for p in head.parameters())
        print(f"    pts_bbox_head type      = {type(head).__name__}, params={head_params:,}")

    if hasattr(uniad_model.vision_model, 'seg_head'):
        seg = uniad_model.vision_model.seg_head
        seg_params = sum(p.numel() for p in seg.parameters())
        print(f"    seg_head type           = {type(seg).__name__}, params={seg_params:,}")

# ----- Debug: Projectors -----
sep("STEP 1 DEBUG: Multimodal Projectors 详细信息")

inner_model = model.get_model()
for proj_name in ['mm_projector_scene', 'mm_projector_track', 'mm_projector_map']:
    proj = getattr(inner_model, proj_name, None)
    if proj is not None:
        proj_params = sum(p.numel() for p in proj.parameters())
        print(f">>> {proj_name}:")
        print(f"    type   = {type(proj).__name__}")
        print(f"    params = {proj_params:,}")
        print(f"    structure:")
        for i, layer in enumerate(proj):
            print(f"      [{i}] {layer}")
    else:
        print(f">>> {proj_name}: NOT FOUND")


# ============================================================
# STEP 2: 加载数据集 & 获取单个样本
# ============================================================
sep("STEP 2: 加载数据集 & 获取单个样本")

from mmengine import Config
from llava.train.train import DataArguments
from drivevla.data_utils.nuscenes_llava_dataset import LLaVANuScenesDataset
from drivevla.data_utils.nuscenes_llava_datacollector import DataCollatorForLLaVANuScenesDataset

print(">>> 2.1 加载 UniAD 配置")
uniad_cfg = Config.fromfile("projects/configs/stage1_track_map/base_track_map.py")
print(f"    data_root = {uniad_cfg.data_root}")
print(f"    ann_file_test = {uniad_cfg.data.test.ann_file}")
print(f"    test pipeline steps = {len(uniad_cfg.data.test.pipeline)}")

print(">>> 2.2 创建 DataArguments")
data_args = DataArguments(
    data_path=args.data_path,
    lazy_preprocess=True,
    frames_upbound=32,
)
print(f"    data_path = {data_args.data_path}")

print(">>> 2.3 创建 LLaVANuScenesDataset (test mode)")
t0 = time.time()
test_dataset = LLaVANuScenesDataset(
    tokenizer, data_args, uniad_cfg.data.test,
    llava_test_mode=True,
    use_uniad_pth=args.use_uniad_pth,
)
t_dataset = time.time() - t0
print(f"    数据集创建完成, 耗时 {t_dataset:.2f}s")
print(f"    数据集大小 = {len(test_dataset)}")
print(f"    in_nuscenes_order = {test_dataset.in_nuscenes_order}")
print(f"    use_uniad_pth     = {test_dataset.use_uniad_pth}")
print(f"    llava_test_mode   = {test_dataset.llava_test_mode}")

# ----- 获取单个样本 -----
sep("STEP 2 DEBUG: 获取单个样本 __getitem__()")

idx = args.sample_idx
print(f">>> 2.4 获取样本 idx={idx}")

t0 = time.time()
sample = test_dataset[idx]
t_sample = time.time() - t0
print(f"    获取样本耗时 {t_sample:.2f}s")
print(f"    样本 keys = {list(sample.keys())}")

print(f"\n>>> 2.5 样本内容详情:")
print(f"    id       = {sample.get('id', 'N/A')}")
print(f"    question (前 300 字符):")
question_text = sample.get('question', 'N/A')
print(f"    ---")
print(f"    {question_text[:300]}")
if len(question_text) > 300:
    print(f"    ... (total {len(question_text)} chars)")
print(f"    ---")

print_tensor("input_ids", sample.get("input_ids"))
if sample.get("qa_instance_ind") is not None:
    print(f"    qa_instance_ind = {sample['qa_instance_ind']}")

# 解码 input_ids 看看里面有哪些特殊 token
input_ids_flat = sample["input_ids"].flatten().tolist()
special_count = {
    "SCENE (-201)": input_ids_flat.count(SCENE_TOKEN_INDEX),
    "TRACK (-202)": input_ids_flat.count(TRACK_TOKEN_INDEX),
    "MAP (-203)": input_ids_flat.count(MAP_TOKEN_INDEX),
    "OBJECT (-204)": input_ids_flat.count(OBJECT_TOKEN_INDEX),
    "IMAGE (-200)": input_ids_flat.count(IMAGE_TOKEN_INDEX),
}
print(f"\n>>> 2.6 input_ids 中的特殊 Token 计数:")
for k, v in special_count.items():
    print(f"    {k}: {v}")

print(f"\n>>> 2.7 input_ids 总 token 数 = {len(input_ids_flat)}")

# Debug uniad_data
if "uniad_data" in sample:
    sep("STEP 2 DEBUG: UniAD 在线数据 (uniad_data)")
    ud = sample["uniad_data"]
    print(f"    uniad_data keys = {list(ud.keys())}")
    for k, v in ud.items():
        if isinstance(v, torch.Tensor):
            print_tensor(f"uniad_data['{k}']", v)
        elif isinstance(v, np.ndarray):
            print(f"  uniad_data['{k}']: ndarray shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  uniad_data['{k}']: {type(v).__name__}, len={len(v)}")
        elif isinstance(v, dict):
            print(f"  uniad_data['{k}']: dict, keys={list(v.keys())[:8]}")
        else:
            print(f"  uniad_data['{k}']: {type(v).__name__} = {str(v)[:100]}")

if "uniad_pth" in sample:
    sep("STEP 2 DEBUG: UniAD 预计算特征 (uniad_pth)")
    up = sample["uniad_pth"]
    print(f"    uniad_pth type = {type(up).__name__}")
    if isinstance(up, dict):
        print(f"    uniad_pth keys = {list(up.keys())}")
        for k, v in up.items():
            if isinstance(v, dict):
                print(f"    uniad_pth['{k}']: dict, keys={list(v.keys())[:10]}")
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        print_tensor(f"  uniad_pth['{k}']['{kk}']", vv)
                    else:
                        print(f"    uniad_pth['{k}']['{kk}']: {type(vv).__name__} = {str(vv)[:100]}")
            elif isinstance(v, torch.Tensor):
                print_tensor(f"uniad_pth['{k}']", v)
            else:
                print(f"    uniad_pth['{k}']: {type(v).__name__} = {str(v)[:100]}")


# ============================================================
# STEP 3: Data Collator 处理
# ============================================================
sep("STEP 3: DataCollator 处理")

print(">>> 3.1 创建 DataCollatorForLLaVANuScenesDataset (test mode)")
data_collator = DataCollatorForLLaVANuScenesDataset(tokenizer=tokenizer, llava_test_mode=True)

print(">>> 3.2 调用 data_collator([sample]) — _test_call()")
batch = data_collator([sample])

print(f"    batch keys = {list(batch.keys())}")
print(f"\n>>> 3.3 Batch 内容详情:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print_tensor(f"batch['{k}']", v)
    elif isinstance(v, dict):
        print(f"  batch['{k}']: dict, keys={list(v.keys())[:10]}")
    elif isinstance(v, str):
        print(f"  batch['{k}']: '{v[:100]}'")
    else:
        print(f"  batch['{k}']: {type(v).__name__} = {str(v)[:200]}")


# ============================================================
# STEP 4: 将数据移到 GPU
# ============================================================
sep("STEP 4: 将数据移到 GPU")

from drivevla.utils.tensor_utils import move_data_to_device

print(f">>> 4.1 move_data_to_device(batch, {device})")
batch = move_data_to_device(batch, device)
print(f"    完成, 验证 input_ids device = {batch['input_ids'].device}")


# ============================================================
# STEP 5: 模型推理 — 详细拆解 generate() 调用
# ============================================================
sep("STEP 5: 模型推理 — 拆解 generate() 内部流程")

model.eval()
input_ids = batch["input_ids"]
uniad_data = batch.get("uniad_data", None)
uniad_pth = batch.get("uniad_pth", None)
qa_instance_ind = batch.get("qa_instance_ind", None)

print(f">>> 5.1 输入参数:")
print_tensor("input_ids", input_ids)
print(f"  uniad_data = {'dict with keys: ' + str(list(uniad_data.keys())) if isinstance(uniad_data, dict) else type(uniad_data).__name__ if uniad_data else 'None'}")
print(f"  uniad_pth  = {'dict with keys: ' + str(list(uniad_pth.keys())) if isinstance(uniad_pth, dict) else type(uniad_pth).__name__ if uniad_pth else 'None'}")
print(f"  qa_instance_ind = {qa_instance_ind}")

# ============================================================
# STEP 5a: UniAD 感知特征提取
# ============================================================
sep("STEP 5a: UniAD 感知特征提取 (Vision Tower)")

with torch.inference_mode():
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):

        if uniad_pth is not None:
            print(">>> 5a.1 使用预计算特征 (uniad_pth), 跳过在线推理")
            vision_tower_result = uniad_pth
        elif uniad_data is not None:
            print(">>> 5a.1 在线运行 UniAD Vision Tower")
            print(f"    调用 vision_tower(uniad_data)")

            if 'img' in uniad_data:
                print_tensor("    输入图像 uniad_data['img']", uniad_data['img'])
            if 'img_metas' in uniad_data:
                meta = uniad_data['img_metas']
                if isinstance(meta, list) and len(meta) > 0:
                    m = meta[0] if isinstance(meta[0], dict) else meta[0][0] if isinstance(meta[0], list) else meta
                    if isinstance(m, dict):
                        print(f"    img_metas[0] keys = {list(m.keys())[:10]}")
                        if 'filename' in m:
                            print(f"    img_metas[0]['filename'] = {m['filename'][:3]}...")

            t0 = time.time()
            vision_tower_result = vision_tower(uniad_data)
            t_vt = time.time() - t0
            print(f"    Vision Tower 推理完成, 耗时 {t_vt:.2f}s")
        else:
            print(">>> 5a.1 无 UniAD 数据, vision_tower_result = None")
            vision_tower_result = None

        if vision_tower_result is not None:
            sep("STEP 5a DEBUG: Vision Tower 输出详情")
            print(f"    vision_tower_result keys = {list(vision_tower_result.keys())}")

            # result_track
            if 'result_track' in vision_tower_result:
                rt = vision_tower_result['result_track']
                print(f"\n  >>> result_track keys = {list(rt.keys())}")
                for k, v in rt.items():
                    if isinstance(v, torch.Tensor):
                        print_tensor(f"  result_track['{k}']", v)
                    elif isinstance(v, dict):
                        print(f"    result_track['{k}']: dict, keys={list(v.keys())[:8]}")
                    elif isinstance(v, (list, tuple)):
                        print(f"    result_track['{k}']: {type(v).__name__}, len={len(v)}")
                    else:
                        print(f"    result_track['{k}']: {type(v).__name__}")

                # 特别打印关键特征
                if 'track_query_embeddings' in rt:
                    print_tensor("\n  ★ track_query_embeddings (核心: 物体跟踪特征)", rt['track_query_embeddings'])
                if 'img_feat_2D' in rt:
                    print_tensor("  ★ img_feat_2D (2D 图像特征)", rt['img_feat_2D'])
                if 'bev_embed' in rt:
                    print_tensor("  ★ bev_embed (BEV 特征)", rt['bev_embed'])
                if 'track_gt_inds_to_embed_idx' in rt:
                    print(f"    ★ track_gt_inds_to_embed_idx = {rt['track_gt_inds_to_embed_idx']}")

            # result_seg
            if 'result_seg' in vision_tower_result:
                rs = vision_tower_result['result_seg']
                print(f"\n  >>> result_seg keys = {list(rs.keys())}")
                for k, v in rs.items():
                    if isinstance(v, torch.Tensor):
                        print_tensor(f"  result_seg['{k}']", v)
                    else:
                        print(f"    result_seg['{k}']: {type(v).__name__}")

            # planning_gt
            if 'planning_gt' in vision_tower_result:
                pg = vision_tower_result['planning_gt']
                print(f"\n  >>> planning_gt keys = {list(pg.keys())}")
                for k, v in pg.items():
                    if isinstance(v, torch.Tensor):
                        print_tensor(f"  planning_gt['{k}']", v)
                    else:
                        print(f"    planning_gt['{k}']: {type(v).__name__} = {str(v)[:100]}")


        # ============================================================
        # STEP 5b: encode_vision_tower_result — 投影到 LLM 空间
        # ============================================================
        sep("STEP 5b: encode_vision_tower_result() — 投影感知特征到 LLM 空间")

        if vision_tower_result is not None:
            print(">>> 5b.1 调用 model.encode_vision_tower_result()")

            # 手动模拟以打印中间结果
            result_track = vision_tower_result["result_track"]
            result_seg = vision_tower_result["result_seg"]

            track_query_embeddings = result_track["track_query_embeddings"]
            chosen_output_query_things = result_seg["chosen_output_query_things"]
            output_query_stuff = result_seg['output_query_stuff']
            map_seg_query_embeddings = torch.cat([chosen_output_query_things, output_query_stuff], dim=0)

            print(f"\n  >>> 投影前的原始特征 (256 维):")
            print_tensor("  track_query_embeddings", track_query_embeddings)
            print_tensor("  chosen_output_query_things", chosen_output_query_things)
            print_tensor("  output_query_stuff", output_query_stuff)
            print_tensor("  map_seg_query_embeddings (concat)", map_seg_query_embeddings)

            # Scene feature: img_feat_2D 处理
            img_feat_2D = result_track["img_feat_2D"]
            print_tensor("\n  img_feat_2D (原始)", img_feat_2D)
            img_feat_2D_proc = img_feat_2D.squeeze(0)
            print_tensor("  img_feat_2D (squeeze)", img_feat_2D_proc)
            img_feat_2D_proc = F.adaptive_max_pool2d(img_feat_2D_proc, (3, 5))
            print_tensor("  img_feat_2D (pool2d 3x5)", img_feat_2D_proc)
            img_feat_2D_proc = img_feat_2D_proc.flatten(2).transpose(1, 2).reshape(-1, 256)
            print_tensor("  img_feat_2D (flatten → reshape)", img_feat_2D_proc)

            # 调用 projectors
            scene_feature = inner_model.mm_projector_scene(img_feat_2D_proc)
            print(f"\n  >>> 投影后的特征 (→ {model.config.hidden_size} 维):")
            print_tensor("  ★ scene_feature (mm_projector_scene)", scene_feature)

            if track_query_embeddings is not None:
                track_feature = inner_model.mm_projector_track(track_query_embeddings.to(dtype=img_feat_2D_proc.dtype))
                print_tensor("  ★ track_feature (mm_projector_track)", track_feature)
            else:
                track_feature = None
                print("  ★ track_feature = None (无检测到的物体)")

            map_feature = inner_model.mm_projector_map(map_seg_query_embeddings.to(dtype=img_feat_2D_proc.dtype))
            print_tensor("  ★ map_feature (mm_projector_map)", map_feature)

            # Object feature
            if qa_instance_ind is not None and 'track_gt_inds_to_embed_idx' in result_track:
                mapping = result_track['track_gt_inds_to_embed_idx']
                embed_idx = mapping.get(qa_instance_ind, None)
                if embed_idx is not None and track_feature is not None:
                    qa_obj_feature = track_feature[embed_idx].unsqueeze(0)
                    print_tensor("  ★ qa_instance_track_feature (查询物体)", qa_obj_feature)
                else:
                    print(f"  ★ qa_instance_ind={qa_instance_ind} 未在跟踪结果中找到")

        else:
            scene_feature = track_feature = map_feature = None
            print(">>> vision_tower_result 为 None, 跳过特征投影")


        # ============================================================
        # STEP 5c: prepare_inputs_labels_for_multimodal_uniad_vlm
        # ============================================================
        sep("STEP 5c: Token 替换 & Embedding 拼接")

        print(">>> 5c.1 调用 prepare_inputs_labels_for_multimodal_uniad_vlm()")
        print("    此函数将 input_ids 中的特殊 token 替换为感知特征 embedding")

        (new_input_ids, position_ids, attention_mask, _, inputs_embeds, _) = \
            model.prepare_inputs_labels_for_multimodal_uniad_vlm(
                input_ids, None, None, None, None, None,
                ["image"], None,
                uniad_data=uniad_data if uniad_pth is None else None,
                uniad_pth=uniad_pth,
                qa_instance_ind=qa_instance_ind,
            )

        print(f"\n>>> 5c.2 输出:")
        print_tensor("  new_input_ids (should be None)", new_input_ids)
        print_tensor("  position_ids", position_ids)
        print_tensor("  attention_mask", attention_mask)
        print_tensor("  ★ inputs_embeds (最终输入 LLM)", inputs_embeds)

        if inputs_embeds is not None:
            print(f"\n>>> 5c.3 inputs_embeds 形状分析:")
            print(f"    batch_size = {inputs_embeds.shape[0]}")
            print(f"    seq_len    = {inputs_embeds.shape[1]}")
            print(f"    hidden_dim = {inputs_embeds.shape[2]}")

            original_text_tokens = len(input_ids_flat) - sum(special_count.values())
            scene_tokens = scene_feature.shape[0] if scene_feature is not None else 0
            track_tokens = track_feature.shape[0] if track_feature is not None else 0
            map_tokens = map_feature.shape[0] if map_feature is not None else 0
            print(f"\n    Token 组成估算:")
            print(f"      原始文本 tokens = {original_text_tokens}")
            print(f"      scene tokens   = {scene_tokens}  (from <SCENE>)")
            print(f"      track tokens   = {track_tokens}  (from <TRACK>)")
            print(f"      map tokens     = {map_tokens}    (from <MAP>)")
            print(f"      总计 ≈ {original_text_tokens + scene_tokens + track_tokens + map_tokens}")
            print(f"      实际 seq_len   = {inputs_embeds.shape[1]} (含 padding)")


        # ============================================================
        # STEP 6: LLM 自回归生成
        # ============================================================
        sep("STEP 6: Qwen2 LLM 自回归生成")

        print(f">>> 6.1 调用 Qwen2ForCausalLM.generate()")
        print(f"    inputs_embeds shape = {inputs_embeds.shape}")
        print(f"    do_sample          = False")
        print(f"    temperature        = 0")
        print(f"    max_new_tokens     = {args.max_new_tokens}")
        print(f"    num_beams          = 1")

        t0 = time.time()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_ids = model.generate(
                input_ids,
                uniad_data=uniad_data if uniad_pth is None else None,
                uniad_pth=uniad_pth,
                qa_instance_ind=qa_instance_ind,
                do_sample=False,
                temperature=0,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
        t_gen = time.time() - t0

        print(f"\n>>> 6.2 生成完成, 耗时 {t_gen:.2f}s")
        print_tensor("  generated_ids", generated_ids)
        print(f"    生成的 token 数 = {generated_ids.shape[1]}")
        print(f"    生成速度 ≈ {generated_ids.shape[1] / t_gen:.1f} tokens/s")


# ============================================================
# STEP 7: 解码输出
# ============================================================
sep("STEP 7: 解码输出")

print(">>> 7.1 tokenizer.batch_decode()")
raw_answers = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"    解码结果数量 = {len(raw_answers)}")
for i, ans in enumerate(raw_answers):
    print(f"\n    --- Answer [{i}] (len={len(ans)}) ---")
    print(f"    {ans}")
    print(f"    --- End Answer [{i}] ---")


# ============================================================
# STEP 8: 轨迹提取
# ============================================================
sep("STEP 8: 轨迹提取 — retrieve_traj()")

from drivevla.utils.trajectory_utils import retrieve_traj, trajectory_is_valid

for i, ans in enumerate(raw_answers):
    print(f"\n>>> 8.{i+1} 处理 Answer [{i}]:")
    print(f"    原始文本: {ans[:200]}...")

    try:
        traj = retrieve_traj(ans)
        print(f"    提取的轨迹 ({len(traj)} 个点):")
        for j, (x, y) in enumerate(traj):
            t_sec = (j + 1) * 0.5
            print(f"      [{j}] t={t_sec:.1f}s: ({x:.4f}, {y:.4f})")

        is_valid = trajectory_is_valid(traj)
        print(f"    轨迹有效性: {is_valid}")

        if is_valid:
            # 计算轨迹统计
            distances = [np.sqrt(x**2 + y**2) for x, y in traj]
            print(f"    累积距离 (距原点):")
            for j, d in enumerate(distances):
                print(f"      t={0.5*(j+1):.1f}s: {d:.4f}m")

            # 相邻点间距
            seg_dists = []
            for j in range(1, len(traj)):
                dx = traj[j][0] - traj[j-1][0]
                dy = traj[j][1] - traj[j-1][1]
                seg_dists.append(np.sqrt(dx**2 + dy**2))
            print(f"    相邻点间距:")
            for j, d in enumerate(seg_dists):
                speed = d / 0.5  # m/s
                print(f"      段{j}→{j+1}: {d:.4f}m (≈{speed:.2f}m/s ≈ {speed*3.6:.1f}km/h)")

    except Exception as e:
        print(f"    ❌ 轨迹提取失败: {e}")

# ============================================================
# STEP 9: 构造最终输出
# ============================================================
sep("STEP 9: 构造最终输出")

result = {
    'id': batch.get('id', 'unknown'),
    'question': batch.get('question', ''),
    'answer': raw_answers,
}

print(f">>> 最终输出:")
print(json.dumps(result, indent=2, ensure_ascii=False)[:2000])

# ============================================================
# 显存统计
# ============================================================
if torch.cuda.is_available():
    sep("GPU 显存统计")
    print(f"  已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  最大分配: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

sep("DEBUG 完成", "★")
print(f"  所有步骤执行完毕!")
print(f"  模型: {args.model_path}")
print(f"  样本 idx: {args.sample_idx}")
print(f"  生成 token 数: {generated_ids.shape[1]}")
print(f"  轨迹提取成功: {all(trajectory_is_valid(retrieve_traj(a)) for a in raw_answers)}")
