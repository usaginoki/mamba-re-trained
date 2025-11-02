"""MAMBA Model Initialization and Configuration"""

import logging
from typing import Optional

import torch
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM

from .config import MambaModelConfig

logger = logging.getLogger(__name__)


def create_mamba_config(model_config: MambaModelConfig) -> MambaConfig:
    """Create MambaConfig from our configuration

    Args:
        model_config: Our MambaModelConfig instance

    Returns:
        HuggingFace MambaConfig instance
    """
    logger.info("Creating MAMBA configuration...")

    config = MambaConfig(
        # Core architecture
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.d_model,
        num_hidden_layers=model_config.n_layer,

        # SSM parameters
        state_size=model_config.d_state,
        conv_kernel=model_config.d_conv,
        expand=model_config.expand,

        # Time step
        time_step_rank=model_config.dt_rank,

        # Normalization and stability
        use_bias=model_config.bias,
        use_conv_bias=model_config.conv_bias,
        residual_in_fp32=model_config.residual_in_fp32,

        # Special tokens
        pad_token_id=model_config.pad_token_id,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,

        # Initialization
        initializer_range=model_config.initializer_range,
        tie_word_embeddings=model_config.tie_word_embeddings,
    )

    logger.info(f"MAMBA config created: {config}")

    return config


def initialize_model_from_scratch(
    model_config: MambaModelConfig,
    device: Optional[str] = None
) -> MambaForCausalLM:
    """Initialize MAMBA model with random weights

    Args:
        model_config: Model configuration
        device: Device to place model on (None = auto)

    Returns:
        Initialized MambaForCausalLM model
    """
    logger.info("Initializing MAMBA-130M from scratch (random weights)...")

    # Create config
    config = create_mamba_config(model_config)

    # Initialize model with random weights
    model = MambaForCausalLM(config)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model initialized successfully!")
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # Move to device if specified
    if device is not None:
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)

    return model


def load_pretrained_model(
    model_name: str = "state-spaces/mamba-130m-hf",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None
) -> MambaForCausalLM:
    """Load pretrained MAMBA model

    Args:
        model_name: HuggingFace model name or path
        device: Device to place model on (None = auto)
        torch_dtype: Data type for model weights (None = auto)

    Returns:
        Pretrained MambaForCausalLM model
    """
    logger.info(f"Loading pretrained MAMBA model: {model_name}")

    # Load model
    model = MambaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device else "auto",
    )

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded successfully!")
    logger.info(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    return model


def get_model_and_tokenizer(
    model_config: MambaModelConfig,
    tokenizer_name: str = "state-spaces/mamba-130m-hf",
    from_scratch: bool = True,
    device: Optional[str] = None
) -> tuple[MambaForCausalLM, AutoTokenizer]:
    """Get MAMBA model and tokenizer

    Args:
        model_config: Model configuration
        tokenizer_name: Tokenizer name or path
        from_scratch: If True, initialize with random weights; else load pretrained
        device: Device to place model on

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

    # Initialize or load model
    if from_scratch:
        model = initialize_model_from_scratch(model_config, device)
    else:
        model = load_pretrained_model(tokenizer_name, device)

    # Enable gradient checkpointing if needed (configured in training)
    # This is done in the training script

    return model, tokenizer


def print_model_architecture(model: MambaForCausalLM):
    """Print detailed model architecture information

    Args:
        model: MAMBA model to analyze
    """
    logger.info("=" * 70)
    logger.info("MAMBA Model Architecture")
    logger.info("=" * 70)

    # Model config
    config = model.config
    logger.info(f"\nConfiguration:")
    logger.info(f"  Vocabulary size: {config.vocab_size:,}")
    logger.info(f"  Hidden size (d_model): {config.hidden_size}")
    logger.info(f"  Number of layers: {config.num_hidden_layers}")
    logger.info(f"  State size (d_state): {config.state_size}")
    logger.info(f"  Convolution kernel: {config.conv_kernel}")
    logger.info(f"  Expand factor: {config.expand}")
    logger.info(f"  Time step rank: {config.time_step_rank}")

    # Parameters by type
    logger.info(f"\nParameter Statistics:")
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")

    # Memory estimation
    param_memory = total_params * 4 / (1024**3)  # FP32 in GB
    logger.info(f"\nMemory Estimation:")
    logger.info(f"  Parameters (FP32): {param_memory:.2f} GB")
    logger.info(f"  Parameters (FP16/BF16): {param_memory/2:.2f} GB")

    # Layer breakdown
    logger.info(f"\nLayer Breakdown:")
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embeddings' in n)
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if 'layers' in n)
    lm_head_params = sum(p.numel() for n, p in model.named_parameters() if 'lm_head' in n)

    logger.info(f"  Embeddings: {embedding_params:,} ({embedding_params/1e6:.1f}M)")
    logger.info(f"  Backbone (layers): {backbone_params:,} ({backbone_params/1e6:.1f}M)")
    logger.info(f"  LM Head: {lm_head_params:,} ({lm_head_params/1e6:.1f}M)")

    logger.info("=" * 70)


def verify_model_config(model: MambaForCausalLM, expected_config: MambaModelConfig):
    """Verify that model configuration matches expected configuration

    Args:
        model: MAMBA model to verify
        expected_config: Expected configuration

    Raises:
        ValueError if configuration mismatch detected
    """
    config = model.config

    mismatches = []

    if config.hidden_size != expected_config.d_model:
        mismatches.append(f"hidden_size: {config.hidden_size} != {expected_config.d_model}")

    if config.num_hidden_layers != expected_config.n_layer:
        mismatches.append(f"num_hidden_layers: {config.num_hidden_layers} != {expected_config.n_layer}")

    if config.vocab_size != expected_config.vocab_size:
        mismatches.append(f"vocab_size: {config.vocab_size} != {expected_config.vocab_size}")

    if config.state_size != expected_config.d_state:
        mismatches.append(f"state_size: {config.state_size} != {expected_config.d_state}")

    if mismatches:
        error_msg = "Model configuration mismatch:\n" + "\n".join(f"  - {m}" for m in mismatches)
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info("Model configuration verified successfully!")


def count_parameters(model: MambaForCausalLM) -> dict:
    """Count model parameters by category

    Args:
        model: MAMBA model

    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count by component
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embeddings' in n)
    backbone_params = sum(p.numel() for n, p in model.named_parameters() if 'layers' in n)
    lm_head_params = sum(p.numel() for n, p in model.named_parameters() if 'lm_head' in n)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
        "embeddings": embedding_params,
        "backbone": backbone_params,
        "lm_head": lm_head_params,
    }
