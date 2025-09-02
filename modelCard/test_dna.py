#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.fx as fx
from typing import List, Optional, Dict, Any

def get_shape(node: fx.Node) -> Optional[torch.Size]:
    meta = node.meta.get('tensor_meta')
    return meta.shape if meta else None

def get_module_by_path(model: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    try:
        obj = model
        for p in path.split('.'):
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj
    except:
        return None

def has_fp16_params(model: torch.nn.Module) -> bool:
    return any(p.dtype == torch.float16 for p in model.parameters())

def has_bf16_params(model: torch.nn.Module) -> bool:
    return any(p.dtype == torch.bfloat16 for p in model.parameters())

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def detect_model_dna(model: torch.nn.Module, traced_graph: fx.GraphModule) -> Dict[str, Any]:
    """
    从模型和其 FX Graph 中提取所有架构与训练机制
    """
    graph = traced_graph.graph
    result = {
        "PRE_LN": False,
        "POST_LN": False,
        "RESIDUAL_SCALING": None,  # float 或 None
        "INITIALIZATION_SCALE": None,
        "EMBEDDING_SCALING": None,
        "DEEPNORM": False,
        "OPTIMIZER_CHOICE": None,
        "WEIGHT_DECAY": None,
        "LEARNING_RATE_SCHEDULE": None,
        "GRADIENT_CLIPPING": None,
        "MIXED_PRECISION": None,
    }

    # ==================== 1. PRE_LN vs POST_LN ====================
    # 原理：看 LayerNorm 在 residual 前还是后
    # Pre-LN: LN → Attn/FFN → Add
    # Post-LN: Attn/FFN → LN → Add

    # 找所有 add（残差连接）
    residual_adds = [n for n in graph.nodes if n.target == torch.add]

    for add_node in residual_adds:
        users = list(add_node.users)
        if not users:
            continue

        # 检查 add 的输出是否进入 LN
        for user in users:
            if user.op == 'call_module':
                mod = get_module_by_path(model, str(user.target))
                if isinstance(mod, torch.nn.LayerNorm):
                    result["POST_LN"] = True
                    break
        else:
            # 如果 add 后没有 LN，则检查输入是否有 LN
            for arg in add_node.args:
                if isinstance(arg, fx.Node) and arg.op == 'call_module':
                    mod = get_module_by_path(model, str(arg.target))
                    if isinstance(mod, torch.nn.LayerNorm):
                        result["PRE_LN"] = True
                        break

    # ==================== 2. RESIDUAL_SCALING ====================
    # 原理：看 add 前是否有 scaling（如 x * 0.5）
    for add_node in residual_adds:
        for arg in add_node.args:
            if isinstance(arg, fx.Node):
                # 检查是否有乘法 scaling
                for producer in arg.all_input_nodes:
                    if producer.target == torch.mul:
                        for const in producer.args:
                            if isinstance(const, (int, float)) and 0 < const < 1:
                                result["RESIDUAL_SCALING"] = const
                                break

    # ==================== 3. INITIALIZATION_SCALE ====================
    # 原理：检查线性层权重初始化是否缩放过
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            std = module.weight.data.std().item()
            # 标准 Xavier 初始化 std ≈ sqrt(2 / (in + out))
            expected_std = (2.0 / (module.in_features + module.out_features)) ** 0.5
            ratio = std / expected_std
            if ratio < 0.8 or ratio > 1.2:
                result["INITIALIZATION_SCALE"] = ratio
                break

    # ==================== 4. EMBEDDING_SCALING ====================
    # 原理：Embedding 输出是否被 scale（如 * sqrt(d_model)）
    embedding_nodes = [n for n in graph.nodes if n.op == 'call_module']
    for node in embedding_nodes:
        mod = get_module_by_path(model, str(node.target))
        if isinstance(mod, torch.nn.Embedding):
            for user in node.users:
                if user.target == torch.mul:
                    for arg in user.args:
                        if isinstance(arg, (int, float)) and arg > 1:
                            result["EMBEDDING_SCALING"] = arg
                            break

    # ==================== 5. DEEPNORM ====================
    # 原理：DeepNorm 要求：
    # - 每层有特定的 scaling：FFN 和 Attn 的输入被缩放
    # - 通常伴随特定初始化
    # 这里我们检测是否有全局 scaling 模式
    deepnorm_candidates = 0
    total_layers = 0

    for node in graph.nodes:
        if node.target == torch.matmul:
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    for producer in arg.all_input_nodes:
                        if producer.target == torch.mul:
                            scale = None
                            for const in producer.args:
                                if isinstance(const, (int, float)):
                                    scale = const
                            if scale and 0.1 < scale < 0.7:  # DeepNorm 常见 scale
                                deepnorm_candidates += 1
    # 简化判断：如果有多个 sub-layer 被 scale，则可能是 DeepNorm
    if deepnorm_candidates >= 3:
        result["DEEPNORM"] = True

    # ==================== 6. OPTIMIZER_CHOICE ====================
    # 注意：optimizer 无法从 model 推断，需外部输入
    # 但可以提示用户
    # result["OPTIMIZER_CHOICE"] = "AdamW"  # 需外部提供

    # ==================== 7. WEIGHT_DECAY ====================
    # 同样需外部输入，但可检测是否使用 decay
    # 这里留空，由训练配置注入

    # ==================== 8. LEARNING_RATE_SCHEDULE ====================
    # 无法从 model 推断，需外部输入

    # ==================== 9. GRADIENT_CLIPPING ====================
    # 无法从 model 推断，但可检测是否在训练中使用
    # 留空

    # ==================== 10. MIXED_PRECISION ====================
    if has_fp16_params(model):
        result["MIXED_PRECISION"] = "FP16"
    elif has_bf16_params(model):
        result["MIXED_PRECISION"] = "BF16"
    else:
        result["MIXED_PRECISION"] = "FP32"

    return result


import torch
import torch.fx as fx
from typing import Dict, Any, Optional

def get_module_by_path(model: torch.nn.Module, path: str) -> Optional[torch.nn.Module]:
    try:
        obj = model
        for p in path.split('.'):
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
        return obj
    except:
        return None

def is_same_tensor_or_weight(t1, t2) -> bool:
    """判断两个参数是否共享权重"""
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return t1.data_ptr() == t2.data_ptr()
    return False



def detect_advanced_arch_features(model: torch.nn.Module, traced_graph: fx.GraphModule) -> Dict[str, Any]:
    """
    检测高级架构特征：词嵌射绑定、位置编码、上下文窗口等
    """
    result = {
        "TIED_WEIGHTS": False,
        "SUBWORD_REGULARIZATION": False,
        "NTK_AWARE_ROPE": False,
        "POSITION_INTERPOLATION": False,
        "MAX_POSITION_EMBEDDINGS": None,
        "CONTEXT_WINDOW": None,
    }

    # ==================== 1. TIED_WEIGHTS ====================
    # 原理：检查 lm_head.weight 是否与 embedding.weight 共享
    embedding = None
    lm_head = None

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            embedding = module
        elif isinstance(module, torch.nn.Linear) and module.out_features == embedding.weight.shape[0] if embedding else False:
            # 粗略判断是 lm_head（输出维度 = vocab_size）
            if hasattr(module, 'weight') and module.weight.shape[0] == model.config.vocab_size:
                lm_head = module

    if embedding and lm_head:
        if is_same_tensor_or_weight(embedding.weight, lm_head.weight):
            result["TIED_WEIGHTS"] = True

    # ==================== 2. SUBWORD_REGULARIZATION ====================
    # 原理：Subword Regularization 是训练时策略（如 BPE dropout）
    # 无法从静态模型判断，但可检测是否使用可训练 tokenizer
    # 或是否存在 dropout 在 token embedding 路径上
    # ❌ 无法 100% 确定，需训练配置
    # 但可提示：如果 tokenizer 支持 dropout，则可能启用
    # result["SUBWORD_REGULARIZATION"] = False  # 需外部输入

    # ==================== 3. NTK_AWARE_ROPE ====================
    # 原理：NTK-Aware RoPE 修改了 freq_cis 的计算方式
    # freq = base ** (2i/d) * (i / 8 + 1)  # NTK-aware 扩展
    # 我们检查 RotaryEmbedding 中的 freq 缓冲区是否非常规

    for name, module in model.named_modules():
        if hasattr(module, 'freq_cis'):  # 常见于 LLaMA 风格 RoPE
            freq_cis = getattr(module, 'freq_cis', None)
            if freq_cis is not None and freq_cis.numel() > 0:
                # 检查 freq 是否超出常规范围
                base = 10000.0
                dim = freq_cis.shape[-1] // 2
                # 计算理论 freq
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
                # 如果实际 freq_cis 的长度 > 常规 max_pos，说明是插值或 NTK-aware
                if freq_cis.shape[0] > 2048:  # 常见 max_pos=2048
                    # 进一步检查是否为 NTK-aware（freq 更密集）
                    if inv_freq[-1] > 1e-5 and (inv_freq / inv_freq[-1]).min() < 0.1:
                        result["NTK_AWARE_ROPE"] = True
                        break

    # ==================== 4. POSITION_INTERPOLATION ====================
    # 原理：Position Interpolation 会修改 RoPE 的 freq_cis
    # 或在推理时对 position_ids 进行缩放
    # 检测：freq_cis 长度 < max_position_embeddings

    config = getattr(model, 'config', None)
    if config and hasattr(config, 'max_position_embeddings'):
        max_pos_emb = config.max_position_embeddings
        result["MAX_POSITION_EMBEDDINGS"] = max_pos_emb

        # 检查实际使用的 freq_cis 长度
        for name, module in model.named_modules():
            if hasattr(module, 'freq_cis'):
                freq_len = getattr(module, 'freq_cis').shape[0]
                if freq_len < max_pos_emb:
                    result["POSITION_INTERPOLATION"] = True
                    break

    # ==================== 5. CONTEXT_WINDOW ====================
    # 原理：Context Window 通常等于 max_position_embeddings
    # 但有时更大（如通过 RoPE extrapolation）
    # 我们取 max_position_embeddings 作为基础值

    if result["MAX_POSITION_EMBEDDINGS"]:
        result["CONTEXT_WINDOW"] = result["MAX_POSITION_EMBEDDINGS"]

    # 更精确：检查模型实际支持的最大 seq_len
    # 可通过 trace 时传入长序列，观察是否报错
    # 这里简化为直接使用 max_position_embeddings

    return result


def detect_model_dna_complete(model: torch.nn.Module, traced_graph: fx.GraphModule) -> Dict[str, Any]:
    from collections import ChainMap

    # 基础架构检测（LN, Scaling, Mixed Precision 等）
    basic = detect_model_dna(model, traced_graph)
    # 高级特征检测（Tied Weights, RoPE, Context 等）
    advanced = detect_advanced_arch_features(model, traced_graph)

    # 合并
    return dict(ChainMap(basic, advanced))

# 假设你有一个模型
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="meta")
graph_module = fx.symbolic_trace(model)

# 分析
mechs = detect_model_dna_complete(model, graph_module.graph)
print(mechs)  # 输出: ['GQA']

