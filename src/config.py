"""Configuration for MAMBA-130M WikiText-103 Training"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MambaModelConfig:
    """MAMBA-130M Model Architecture Configuration

    Based on state-spaces/mamba-130m-hf exact specifications.
    """
    # Core architecture
    d_model: int = 768  # hidden_size
    n_layer: int = 24
    vocab_size: int = 50280

    # State Space Model (SSM) parameters
    d_state: int = 16  # SSM state expansion factor
    d_conv: int = 4  # Local convolution width
    expand: int = 2  # Block expansion factor (d_inner = expand * d_model)

    # Time step parameters
    dt_rank: str = "auto"  # "auto" = math.ceil(d_model / 16) = 48

    # Initialization and numerical stability
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True  # Use optimized CUDA kernels if available

    # Critical for training stability
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    rms_norm: bool = True

    # Special tokens
    pad_token_id: int = 0
    bos_token_id: int = 0
    eos_token_id: int = 0

    # Training
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False


@dataclass
class DataConfig:
    """WikiText-103 Dataset Configuration"""
    dataset_name: str = "Salesforce/wikitext"
    dataset_config: str = "wikitext-103-raw-v1"

    # Preprocessing
    max_seq_length: int = 2048
    preprocessing_num_workers: int = 4

    # Data loading
    train_batch_size: int = 4
    eval_batch_size: int = 4
    dataloader_num_workers: int = 4

    # Tokenization
    tokenizer_name: str = "state-spaces/mamba-130m-hf"

    # Caching
    cache_dir: Optional[str] = None
    overwrite_cache: bool = False


@dataclass
class TrainingConfig:
    """Training Hyperparameters

    Based on MAMBA paper recommendations and WikiText-103 fine-tuning best practices.
    """
    # Output and logging
    output_dir: str = "./outputs/mamba-130m-wikitext103"
    run_name: str = "mamba-130m-wikitext103"
    logging_dir: str = "./logs"

    # Training duration
    num_train_epochs: int = 5
    max_steps: int = -1  # -1 means use num_train_epochs

    # Batch size and accumulation
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32

    # Optimization
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0

    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    warmup_ratio: float = 0.0  # Use warmup_steps instead

    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # Preferred if GPU supports it
    fp16_full_eval: bool = False

    # Memory optimization
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"

    # Evaluation and saving
    eval_strategy: str = "steps"
    eval_steps: int = 1000
    save_strategy: str = "steps"
    save_steps: int = 2000
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Logging
    logging_steps: int = 100
    logging_first_step: bool = True
    report_to: list = field(default_factory=lambda: ["tensorboard", "wandb"])

    # Distributed training
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False

    # Reproducibility
    seed: int = 42
    data_seed: int = 42

    # Other
    remove_unused_columns: bool = False
    label_names: list = field(default_factory=lambda: ["labels"])
    include_inputs_for_metrics: bool = False

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.0


@dataclass
class WandbConfig:
    """Weights & Biases Configuration"""
    project: str = "mamba-wikitext103"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: list = field(default_factory=lambda: ["mamba", "wikitext103", "language-modeling"])
    notes: str = "Training MAMBA-130M from scratch on WikiText-103"

    # Logging
    log_model: bool = True
    watch_model: bool = False  # Can be expensive for large models


@dataclass
class ExperimentConfig:
    """Complete Experiment Configuration"""
    model: MambaModelConfig = field(default_factory=MambaModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Flags
    use_wandb: bool = True
    train_from_scratch: bool = True  # True = random init, False = load pretrained

    def __post_init__(self):
        """Validate configuration"""
        # Ensure output directories are set
        if self.training.logging_dir is None:
            self.training.logging_dir = f"{self.training.output_dir}/logs"

        # Set wandb run name if not specified
        if self.use_wandb and self.wandb.name is None:
            self.wandb.name = self.training.run_name


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration"""
    return ExperimentConfig()


def load_config_from_yaml(yaml_path: str) -> ExperimentConfig:
    """Load configuration from YAML file

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        ExperimentConfig instance
    """
    import yaml

    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Parse nested configs
    model_config = MambaModelConfig(**config_dict.get('model', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    wandb_config = WandbConfig(**config_dict.get('wandb', {}))

    return ExperimentConfig(
        model=model_config,
        data=data_config,
        training=training_config,
        wandb=wandb_config,
        use_wandb=config_dict.get('use_wandb', True),
        train_from_scratch=config_dict.get('train_from_scratch', True)
    )


def save_config_to_yaml(config: ExperimentConfig, yaml_path: str):
    """Save configuration to YAML file

    Args:
        config: ExperimentConfig instance
        yaml_path: Path to save YAML file
    """
    import yaml
    from dataclasses import asdict

    config_dict = {
        'model': asdict(config.model),
        'data': asdict(config.data),
        'training': asdict(config.training),
        'wandb': asdict(config.wandb),
        'use_wandb': config.use_wandb,
        'train_from_scratch': config.train_from_scratch
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
