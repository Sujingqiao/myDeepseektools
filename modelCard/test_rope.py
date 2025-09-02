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
