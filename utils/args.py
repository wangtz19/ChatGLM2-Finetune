from dataclasses import dataclass, field
from typing import Optional, Dict, Literal
from transformers import Seq2SeqTrainingArguments
import torch

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="THUDM/chatglm2-6b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co/models."}
    )
    plot_loss: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to plot the training loss after fine-tuning or not."}
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model."}
    )
    quantization_type: Optional[Literal["fp4", "nf4"]] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."}
    )
    double_quantization: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to use double quantization in int4 training or not."}
    )

    def __post_init__(self):
        if self.quantization_bit is not None:
            assert self.quantization_bit in [4, 8], "We only accept 4-bit or 8-bit quantization."


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    val_size: float = field(
        default=0.01, metadata={"help": "The proportion of the validation set."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total input sequence length after tokenization."}
    )
    max_target_length: Optional[int] = field(
        default=1024,
        metadata={"help": "The maximum total output sequence length after tokenization."}
    )


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    stage: str = field(
        default="dpo",
        metadata={"help": "Stage of fine-tuning. `SFT, DPO` are supported."}
    )
    finetuning_method: str = field(
        default="freeze",
        metadata={"help": "Fine-tuning method. `Freeze, LoRA` are supported."}
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
    dpo_beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."}
    )


def postprocess_training_args(
        training_args: TrainingArguments
) -> TrainingArguments:
    if isinstance(training_args.lora_target, str):
        training_args.lora_target = [target.strip() for target in training_args.lora_target.split(",")] # support custom target modules of LoRA

    if training_args.num_layer_trainable > 0: # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = [27 - k for k in range(training_args.num_layer_trainable)]
    else: # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = [k for k in range(-training_args.num_layer_trainable)]

    if training_args.name_module_trainable == "mlp":
        training_args.trainable_layers = ["{:d}.mlp".format(idx) for idx in trainable_layer_ids]
    elif training_args.name_module_trainable == "qkv":
        training_args.trainable_layers = ["{:d}.attention.query_key_value".format(idx) for idx in trainable_layer_ids]

    return training_args


def postprocess_model_args(
        model_args: ModelArguments,
        training_args: TrainingArguments
) -> ModelArguments:
    if training_args.bf16:
        if not torch.cuda.is_bf16_supported():
            raise ValueError("Current device does not support bf16 training.")
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16
    else:
        model_args.compute_dtype = torch.float32

    return model_args