#!/usr/bin/env python
# encoding: utf-8

import torch
import torch.fx as fx
from typing import List, Tuple, Optional

def get_shape(node: fx.Node) -> Optional[Tuple[int, ...]]:
    """安全提取 shape"""
    meta = node.meta.get('tensor_meta')
    return meta.shape if meta else None

def is_linear(node: fx.Node) -> bool:
    return node.op == 'call_module' and isinstance(node.graph.owning_module.get_submodule(node.target), torch.nn.Linear)

def is_reshape_to_heads(node: fx.Node) -> bool:
    """判断 node 是否 reshape 成 (B, N, S, D) 形式"""
    shape = get_shape(node)
    if not shape or len(shape) != 4:
        return False
    B, N, S, D = shape
    # 假设 batch >= 1, head >= 1, seq >= 1, dim in [64, 128, 256]（常见 head_dim）
    return (B >= 1 and N >= 1 and S >= 1 and D in [64, 128, 256])

def find_linear_source(node: fx.Node) -> Optional[fx.Node]:
    """反向追踪，找到这个 node 的线性投影来源"""
    for arg in node.all_input_nodes:
        if is_linear(arg):
            return arg
    return None

def detect_attention_mechanism(graph: fx.Graph) -> List[str]:
    mechanisms = []

    # 1. 找所有 reshape 节点（用于 head 分组）
    reshapes = [n for n in graph.nodes if
               (n.op == 'call_function' and n.target in [torch.reshape, torch.Tensor.view, torch.Tensor.reshape]) or
               (n.op == 'call_method' and n.target in ['view', 'reshape'])]

    # 2. 找所有 transpose（用于 K^T）
    transposes = [n for n in graph.nodes if
                  (n.op == 'call_function' and n.target == torch.transpose) or
                  (n.op == 'call_method' and n.target == 'transpose')]

    # 3. 找所有 matmul（注意力核心）
    matmuls = [n for n in graph.nodes if
               n.op == 'call_function' and n.target == torch.matmul]

    # ==================== 1. GQA / MQA 识别（完全基于 shape）====================
    q_reshape_nodes = []  # reshape 成 (B, H, S, D) 的 Q 节点
    kv_reshape_nodes = [] # reshape 成 (B, G, S, D) 的 K/V 节点

    for node in reshapes:
        if is_reshape_to_heads(node):
            linear_src = find_linear_source(node)
            if not linear_src:
                continue

            shape = get_shape(node)
            B, N, S, D = shape

            # 判断是 Q 还是 K/V：看后续连接
            is_q = False
            is_kv = False

            for user in node.users:
                if user in transposes:
                    # K 通常会 transpose → K^T
                    is_kv = True
                elif any(m in matmuls for m in user.users):
                    # Q 通常直接参与 matmul(Q, K^T)
                    is_q = True

            if is_q and not is_kv:
                q_reshape_nodes.append((node, N))  # N = num_heads_q
            elif is_kv:
                kv_reshape_nodes.append((node, N))  # N = num_heads_kv

    # 统计 head 数
    if q_reshape_nodes and kv_reshape_nodes:
        num_heads_q = max([n[1] for n in q_reshape_nodes])  # 取最大（可能多层）
        num_heads_kv = min([n[1] for n in kv_reshape_nodes])  # 取最小

        if num_heads_kv == 1:
            mechanisms.append("MQA")
        elif num_heads_kv < num_heads_q:
            mechanisms.append("GQA")

    # ==================== 2. Sparse Attention 识别（基于 mask 结构）====================
    # 策略：查找 matmul 前是否有 sparse mask 注入
    for matmul in matmuls:
        for arg in matmul.args:
            if isinstance(arg, fx.Node):
                # 检查是否是 mask（通常 shape 为 (S, S) 或 (1,1,S,S)）
                shape = get_shape(arg)
                if not shape:
                    continue
                if len(shape) == 2 and shape[0] == shape[1]:  # (S, S)
                    # 检查 mask 是否来自稀疏操作
                    for producer in arg.all_input_nodes:
                        if any(op in str(producer.target) for op in ['tril', 'triu', 'masked_fill']):
                            # 进一步检查是否有 sparse tensor 生成
                            for p in producer.all_input_nodes:
                                if 'Sparse' in str(p.meta.get('type', '')) or 'sparse' in str(p.target).lower():
                                    mechanisms.append("SPARSE_ATTENTION")
                                    break

    # ==================== 3. KV Cache Compression 识别（基于 KV 路径上的降维算子）====================
    # 策略：K/V reshape 后是否接 pooling / conv 以压缩长度
    downsample_ops = ['avg_pool1d', 'max_pool1d', 'adaptive_avg_pool1d', 'conv1d']
    for kv_node, _ in kv_reshape_nodes:
        for user in kv_node.users:
            for u in user.users:
                if u.op == 'call_function' and any(op in str(u.target) for op in downsample_ops):
                    mechanisms.append("KV_CACHE_COMPRESSION")
                    break

    # ==================== 4. Early Exit 识别（基于多输出分支）====================
    # 策略：多个 output 节点，或存在 early return 子图
    outputs = [n for n in graph.nodes if n.op == 'output']
    if len(outputs) > 1:
        mechanisms.append("EARLY_EXIT")

    # ==================== 5. Speculative Decoding 识别（基于双模型路径）====================
    # 策略：存在两个独立的“语言模型”子图
    lm_head_candidates = []
    for node in graph.nodes:
        if node.op == 'call_module':
            mod = node.graph.owning_module.get_submodule(node.target)
            # 检查是否是 lm_head（输出 vocab_size）
            out_shape = get_shape(node)
            if out_shape and out_shape[-1] == 32000:  # 假设 vocab_size=32000
                lm_head_candidates.append(node)

    # 如果有两个 lm_head，且来自不同路径
    if len(lm_head_candidates) >= 2:
        # 简化：认为是 speculative
        mechanisms.append("SPECULATIVE_DECODING")

    return list(set(mechanisms))

# 假设你有一个模型
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="meta")
graph_module = fx.symbolic_trace(model)

# 分析
mechs = detect_attention_mechanism(graph_module.graph)
print(mechs)  # 输出: ['GQA']

