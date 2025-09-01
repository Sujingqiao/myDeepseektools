// ===================================================================
// LLM 理论与工程驱动型指标结构体（用于知识归档）
// 每个字段对应一个理论/工程大类，值为该类下的具体指标（英文原名）
// ===================================================================

#ifndef LLM_INDICATORS_H
#define LLM_INDICATORS_H

// ===================================================================
// 原始结构体（保留您提供的部分）
// ===================================================================

typedef enum {
    PRE_LN,                     // Pre-Layer Normalization
    POST_LN,                    // Post-Layer Normalization
    RESIDUAL_SCALING,           // Residual Connection Scaling
    INITIALIZATION_SCALE,       // Weight Initialization Scale
    EMBEDDING_SCALING,          // Embedding Output Scaling
    DEEPNORM,                   // DeepNorm
    INFO_FLOW_COUNT
} InformationFlow;

typedef enum {
    OPTIMIZER_CHOICE,           // Optimizer Choice (e.g., AdamW, Lion)
    WEIGHT_DECAY,               // Weight Decay Rate
    LEARNING_RATE_SCHEDULE,     // Learning Rate Schedule (Warmup + Decay)
    GRADIENT_CLIPPING,          // Gradient Clipping (max norm)
    MIXED_PRECISION,            // Mixed Precision Training (FP16/BF16)
    OPTIM_COUNT
} OptimizationDynamics;

typedef enum {
    TOKENIZER_TYPE,             // Tokenizer Type (BPE, SentencePiece, etc.)
    VOCAB_SIZE,                 // Vocabulary Size
    SPECIAL_TOKENS,             // Special Tokens (e.g., <|start|>, <|end|>)
    TIED_WEIGHTS,               // Tied Input-Output Embedding Weights
    SUBWORD_REGULARIZATION,     // Subword Regularization
    REPR_LEARN_COUNT
} RepresentationLearning;

typedef enum {
    ROPE_BASE,                  // RoPE Base Frequency (θ)
    NTK_AWARE_ROPE,             // NTK-Aware RoPE
    POSITION_INTERPOLATION,     // Position Interpolation
    MAX_POSITION_EMBEDDINGS,    // Max Position Embeddings
    CONTEXT_WINDOW,             // Context Window Length
    GEN_EXTRAP_COUNT
} GeneralizationExtrapolation;

typedef enum {
    GQA,                        // Grouped Query Attention
    MQA,                        // Multi-Query Attention
    SPARSE_ATTENTION,           // Sparse Attention Pattern
    KV_CACHE_COMPRESSION,       // KV Cache Compression
    EARLY_EXIT,                 // Early Exit
    SPECULATIVE_DECODING,       // Speculative Decoding
    FLASH_ATTENTION,            // FlashAttention
    PAGED_ATTENTION,            // PagedAttention (vLLM)
    COMP_EFF_COUNT
} ComputationalEfficiency;


// ===================================================================
// 新增维度 1: 数据驱动型指标 (Data-Driven Indicators)
// ===================================================================

typedef enum {
    DATA_MIX,                   // Training Data Mix (e.g., ratio of code/text)
    DATA_QUALITY_SCORE,         // Data Quality Score (e.g., filtering heuristics)
    DEDUPLICATION_LEVEL,        // Data Deduplication Level (exact/near)
    TEXT_CLEANING_PIPELINE,     // Text Cleaning Pipeline (e.g., boilerplate removal)
    DOMAIN_COVERAGE,            // Domain Coverage (e.g., scientific, legal, code)
    DATA_AUGMENTATION,          // Data Augmentation Techniques
    DATA_DRIVEN_COUNT
} DataDrivenIndicators;


// ===================================================================
// 新增维度 2: 模型架构型指标 (Model Architecture Indicators)
// ===================================================================

typedef enum {
    NUM_LAYERS,                 // Number of Transformer Layers
    HIDDEN_SIZE,                // Hidden Dimension Size
    NUM_HEADS,                  // Number of Attention Heads
    FFN_INTERMEDIATE_SIZE,      // Feed-Forward Network Intermediate Size
    ATTENTION_HEAD_DIM,         // Attention Head Dimension
    MOE_ROUTING_ALGORITHM,      // Mixture-of-Experts Routing Algorithm
    MOE_NUM_EXPERTS,            // Number of Experts in MoE
    ARCHITECTURE_VARIANT,       // Architecture Variant (e.g., Decoder-only, Encoder-decoder)
    MODEL_ARCH_COUNT
} ModelArchitectureIndicators;


// ===================================================================
// 新增维度 3: 训练策略型指标 (Training Strategy Indicators)
// ===================================================================

typedef enum {
    BATCH_SIZE,                 // Global Batch Size
    GRADIENT_ACCUMULATION_STEPS, // Gradient Accumulation Steps
    SEQUENCE_LENGTH,            // Training Sequence Length
    ZERO_STAGE,                 // ZeRO Stage (in DeepSpeed)
    TENSOR_PARALLEL_SIZE,       // Tensor Parallelism Degree
    PIPELINE_PARALLEL_SIZE,     // Pipeline Parallelism Degree
    FULLY_SHARDED_DATA_PARALLEL, // Fully Sharded Data Parallel (FSDP)
    TRAINING_STRATEGY_COUNT
} TrainingStrategyIndicators;


// ===================================================================
// 新增维度 4: 推理优化型指标 (Inference Optimization Indicators)
// ===================================================================

typedef enum {
    QUANTIZATION_METHOD,        // Quantization Method (e.g., GPTQ, AWQ, GGUF)
    QUANTIZATION_BITS,          // Quantization Bit Precision (e.g., 4-bit, 8-bit)
    KERNEL_FUSION,              // Kernel Fusion (e.g., fused attention, fused MLP)
    MODEL_OFFLOADING,           // Model Offloading (to CPU/disk)
    BATCHED_INFERENCE,          // Batched Inference Support
    DYNAMIC_BATCHING,           // Dynamic Batching
    INFERENCE_ENGINE,           // Inference Engine (e.g., TensorRT-LLM, vLLM, TGI)
    INFERENCE_OPTIM_COUNT
} InferenceOptimizationIndicators;


// ===================================================================
// 新增维度 5: 评估基准型指标 (Evaluation Benchmark Indicators)
// ===================================================================

typedef enum {
    PERPLEXITY,                 // Language Modeling Perplexity
    MMLU_SCORE,                 // MMLU (Massive Multitask Language Understanding)
    GSM8K_SCORE,                // GSM8K (Grade School Math)
    HUMAN_EVAL,                 // HumanEval (Code Generation)
    BBH_SCORE,                  // BIG-Bench Hard
    TRUTHFULQA,                 // TruthfulQA
    TOXICITY_SCORE,             // Toxicity Score (e.g., Perspective API)
    BENCHMARK_COUNT
} EvaluationBenchmarkIndicators;


// ===================================================================
// 新增维度 6: 伦理与安全型指标 (Ethics & Safety Indicators)
// ===================================================================

typedef enum {
    ALIGNMENT_METHOD,           // Alignment Method (e.g., RLHF, DPO, RLAIF)
    SAFETY_CLASSIFIER,          // Safety Classifier (e.g., for content moderation)
    BIAS_METRIC,                // Bias Metric (e.g., gender, racial bias)
    PRIVACY_PRESERVING,         // Privacy-Preserving Training (e.g., differential privacy)
    RED_TEAMING,                // Red Teaming Results
    CONTENT_MODERATION,         // Content Moderation Rules
    ETHICS_SAFETY_COUNT
} EthicsSafetyIndicators;


// ===================================================================
// 新增维度 7: 硬件协同型指标 (Hardware Co-Design Indicators)
// ===================================================================

typedef enum {
    GPU_ARCHITECTURE,           // Target GPU Architecture (e.g., Ampere, Hopper)
    GPU_MEMORY_BANDWIDTH,       // GPU Memory Bandwidth Utilization
    TPU_COMPATIBILITY,          // TPU Compatibility and Optimization
    MEMORY_FOOTPRINT,           // Model Memory Footprint (GB)
    FLOPS_UTILIZATION,          // FLOPs Utilization Rate
    INTERCONNECT_BANDWIDTH,     // Inter-GPU Interconnect Bandwidth (e.g., NVLink)
    HARDWARE_CO_DESIGN_COUNT
} HardwareCoDesignIndicators;


// ===================================================================
// 综合结构体：LLM 理论与工程指标总集
// ===================================================================

typedef struct {
    InformationFlow info_flow[INFO_FLOW_COUNT];          // Information Flow Theory
    OptimizationDynamics optim[OPTIM_COUNT];             // Optimization Dynamics
    RepresentationLearning repr_learn[REPR_LEARN_COUNT]; // Representation Learning
    GeneralizationExtrapolation extrap[GEN_EXTRAP_COUNT]; // Generalization & Extrapolation
    ComputationalEfficiency comp_eff[COMP_EFF_COUNT];    // Computational Efficiency
    DataDrivenIndicators data_driven[DATA_DRIVEN_COUNT]; // Data-Driven Indicators
    ModelArchitectureIndicators model_arch[MODEL_ARCH_COUNT]; // Model Architecture
    TrainingStrategyIndicators train_strat[TRAINING_STRATEGY_COUNT]; // Training Strategy
    InferenceOptimizationIndicators infer_optim[INFERENCE_OPTIM_COUNT]; // Inference Optimization
    EvaluationBenchmarkIndicators benchmark[EVALUATION_BENCHMARK_COUNT]; // Evaluation
    EthicsSafetyIndicators ethics_safety[ETHICS_SAFETY_COUNT]; // Ethics & Safety
    HardwareCoDesignIndicators hw_co_design[HARDWARE_CO_DESIGN_COUNT]; // Hardware Co-Design
} LLMTheoryAndEngineeringIndicators;

#endif // LLM_INDICATORS_H
