// ===================================================================
// LLM 理论驱动型指标结构体（用于知识归档）
// 每个字段对应一个理论大类，值为该类下的具体指标（英文原名）
// ===================================================================

typedef struct {
    InformationFlow info_flow[INFO_FLOW_COUNT];          // Information Flow Theory
    OptimizationDynamics optim[OPTIM_COUNT];             // Optimization Dynamics
    RepresentationLearning repr_learn[REPR_LEARN_COUNT]; // Representation Learning
    GeneralizationExtrapolation extrap[GEN_EXTRAP_COUNT]; // Generalization & Extrapolation
    ComputationalEfficiency comp_eff[COMP_EFF_COUNT];    // Computational Efficiency
} LLMTheoryDrivenIndicators;

typedef enum {
    PRE_LN                     // Pre-Layer Normalization
    POST_LN                    // Post-Layer Normalization
    RESIDUAL_SCALING           // Residual Connection Scaling
    INITIALIZATION_SCALE       // Weight Initialization Scale
    EMBEDDING_SCALING          // Embedding Output Scaling
    DEEPNORM                   // DeepNorm
    INFO_FLOW_COUNT
} InformationFlow;

typedef enum {
    OPTIMIZER_CHOICE           // Optimizer Choice (e.g., AdamW, Lion)
    WEIGHT_DECAY               // Weight Decay Rate
    LEARNING_RATE_SCHEDULE     // Learning Rate Schedule (Warmup + Decay)
    GRADIENT_CLIPPING          // Gradient Clipping (max norm)
    MIXED_PRECISION            // Mixed Precision Training (FP16/BF16)
    OPTIM_COUNT
} OptimizationDynamics;

typedef enum {
    TOKENIZER_TYPE             // Tokenizer Type (BPE, SentencePiece, etc.)
    VOCAB_SIZE                 // Vocabulary Size
    SPECIAL_TOKENS             // Special Tokens (e.g., <|start|>, <|end|>)
    TIED_WEIGHTS               // Tied Input-Output Embedding Weights
    SUBWORD_REGULARIZATION     // Subword Regularization
    REPR_LEARN_COUNT
} RepresentationLearning;

typedef enum {
    ROPE_BASE                  // RoPE Base Frequency (θ)
    NTK_AWARE_ROPE             // NTK-Aware RoPE
    POSITION_INTERPOLATION     // Position Interpolation
    MAX_POSITION_EMBEDDINGS    // Max Position Embeddings
    CONTEXT_WINDOW             // Context Window Length
    GEN_EXTRAP_COUNT
} GeneralizationExtrapolation;

typedef enum {
    GQA                        // Grouped Query Attention
    MQA                        // Multi-Query Attention
    SPARSE_ATTENTION           // Sparse Attention Pattern
    KV_CACHE_COMPRESSION       // KV Cache Compression
    EARLY_EXIT                 // Early Exit
    SPECULATIVE_DECODING       // Speculative Decoding
    COMP_EFF_COUNT
} ComputationalEfficiency;
