from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
import transformers

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="THUDM/chatglm2-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    ),
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co/models."}
    ),
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    ),
    val_size: float = field(
        default=0.01, metadata={"help": "The proportion of the validation set."}
    ),
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    ),
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    use_lora: bool = field(
        default=True, metadata={"help": "Whether to use LoRA."}
    )
    num_layer_trainable: Optional[int] = field(
        default=3,
        metadata={"help": "Number of trainable layers for Freeze fine-tuning."}
    )
    name_module_trainable: Optional[Literal["mlp", "qkv"]] = field(
        default="mlp",
        metadata={"help": "Name of trainable modules for Freeze fine-tuning."}
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."}
    )
    lora_alpha: Optional[float] = field(
        default=32.0,
        metadata={"help": "The scale factor for LoRA fine-tuning. (similar with the learning rate)"}
    )
    lora_dropout: Optional[float] = field(
        default=0.1,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."}
    )
    lora_target: Optional[str] = field(
        default="query_key_value, mlp.dense",
        metadata={"help": "Name(s) of target modules to apply LoRA. Use commas to separate multiple modules."}
    )