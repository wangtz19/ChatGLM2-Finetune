import torch
import transformers
from transformers import Seq2SeqTrainer
from peft import LoraConfig, TaskType, get_peft_model
from utils import (
    DataCollatorForChatGLM, 
    ModelArguments, 
    DataArguments, 
    TrainingArguments,
    preprocess_dataset,
    plot_loss,
    get_logger,
    LogCallback,
    count_parameters,
    postprocess_training_args,
    prepare_model_for_training,
    postprocess_model_args,
)
from pprint import pprint

logger = get_logger(__name__)

                                                            
def train():
    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args = postprocess_training_args(training_args)
    model_args = postprocess_model_args(model_args, training_args)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        model_max_length=data_args.model_max_length,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.padding_side = "left"

    dataset = preprocess_dataset(
        tokenizer, training_args, data_args
    )
    collator = DataCollatorForChatGLM(
        tokenizer=tokenizer
    )

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        config=config,
        torch_dtype=model_args.compute_dtype, # change precision if needed
    )

    model.lm_head = model.transformer.output_layer
    output_embedding_base_layer = model.transformer
    output_embedding_layer_name = "output_layer"
    model = prepare_model_for_training(
        model, 
        output_embedding_base_layer, 
        output_embedding_layer_name
    )

    if training_args.finetuning_method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_target,
        )
        # model.enable_input_require_grads() # done in prepare_model_for_training
        model = get_peft_model(model, peft_config)
    elif training_args.finetuning_method == "freeze": # freeze fine-tuning
        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in training_args.trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)
    else:
        raise NotImplementedError(f"Unsupported fine-tuning method: {training_args.finetuning_method}")
    
    # print training arguments
    training_args_dict = training_args.to_dict()
    pprint(training_args_dict, indent=2)

    trainable_params, all_param = count_parameters(model)
    logger.info("trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
        trainable_params, all_param, 100 * trainable_params / all_param
    ))

    callbacks = [LogCallback()]

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=collator,
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
