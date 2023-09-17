import torch
import transformers
from trl import DPOTrainer
from peft import LoraConfig, TaskType, get_peft_model
import sys
from utils import (
    PairwiseDataCollatorForChatGLM, 
    ModelArguments, 
    DataArguments, 
    TrainingArguments,
    preprocess_dataset,
    plot_loss,
    get_logger,
    LogCallback,
    count_parameters,
)
from pprint import pprint

logger = get_logger(__name__)

def postprocess_training_args(training_args):
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

                                                            
def train():
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args = postprocess_training_args(training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=data_args.model_max_length,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.padding_side = "left"

    dataset = preprocess_dataset(
        data_args.data_path, tokenizer, data_args.model_max_length
    )
    # collator = PairwiseDataCollatorForChatGLM(
    #     tokenizer=tokenizer
    # )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )

    if training_args.use_lora:
        # turn off adapters when serving as ref, no auxiliary ref model is needed
        ref_model = None 

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_target,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()

    else: # freeze fine-tuning
        ref_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
        )
        for param in ref_model.parameters():
            param.requires_grad = False
        ref_model.eval()

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in training_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)
    
    # print training arguments
    training_args_dict = training_args.to_dict()
    pprint(training_args_dict, indent=2)

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    callbacks = [LogCallback()]

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        beta=0.1,
        # data_collator=collator, # use default data_collator
        # max_length=data_args.model_max_length,
        # max_prompt_length=data_args.model_max_length,
        max_length=1024,
        max_prompt_length=512,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])


if __name__ == "__main__":
    train()
