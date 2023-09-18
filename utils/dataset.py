# import json
# from torch.utils.data import Dataset
from datasets import Dataset, load_dataset
import torch
from typing import Optional, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
from .args import TrainingArguments, DataArguments
from .constant import IGNORE_INDEX

# class PairwiseDataset(Dataset):
#     """Dataset for dpo fine-tuning."""

#     def __init__(
#         self,
#         data_path,
#         tokenizer: PreTrainedTokenizer,
#         model_max_length: int, # source & target max length
#     ):
#         super(PairwiseDataset, self).__init__()
#         self.data = json.load(open(data_path))
#         self.tokenizer = tokenizer
#         self.model_max_length = model_max_length
#         self.ignore_index = -100

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
#         return self.data[idx]

def preprocess_dataset(
    tokenizer: PreTrainedTokenizer,
    training_args: TrainingArguments,
    data_args: DataArguments,
) -> Dataset:
    
    def format_example(examples): # support question with a single answer or multiple answers
        for i in range(len(examples["instruction"])):
            if examples["instruction"][i] and examples["output"][i]:
                query, answer = examples["instruction"][i], examples["output"][i]
                prompt = ""
                if examples.get("history", None) is not None:
                    query = query + examples["input"][i] if examples["input"][i] else query
                    history = examples["history"][i] if examples["history"][i] else []
                    for j, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(j+1, old_query, response)
                    round_num = len(history) + 1
                else:
                    round_num = 1
                prompt += "[Round {}]\n\n问：{}\n\n答：".format(round_num, query)
                # prompt = prefix + prompt
                yield prompt, answer
    
    def preprocess_sft(examples):
        # v1: build inputs with format `X [gMASK] <sop> Y <eop>` and labels with format `[IGNORE] ... [IGNORE] Y <eop>`
        # v2: build inputs with format `[gMASK] sop X Y </s>` and labels with format `[IGNORE] ... [IGNORE] Y </s>`
        model_inputs = {"input_ids": [], "labels": []}
        for prompt, answer in format_example(examples):
            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            target_ids = tokenizer.encode(text=answer, add_special_tokens=False)

            if len(source_ids) > data_args.max_source_length - 2: # gmask and sop tokens
                source_ids = source_ids[:data_args.max_source_length - 2]
            if len(target_ids) > data_args.max_target_length - 1: # eos token
                target_ids = target_ids[:data_args.max_target_length - 1]

            context_length = len(source_ids) + 2 # gmask and sop tokens
            input_ids = tokenizer.build_inputs_with_special_tokens(source_ids, target_ids)
            labels = [IGNORE_INDEX] * context_length + input_ids[context_length:]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["labels"].append(labels)
        return model_inputs
    
    def preprocess_dpo_v1(examples): # for customized data collator
        # examples shpaed as {
        #   "instruction": [], # batch_size * instruction 
        #   "input": [], # batch_size * input
        #   "output": [] # batch_size * (output, rejected_outputs)
        # } 

        # v1: build input pairs with format `X [gMASK] <sop> Y1 <eop>` and `X [gMASK] <sop> Y2 <eop>`
        # v2: build input pairs with format `[gMASK] sop X Y1 </s>` and `[gMASK] sop X Y2 </s>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for i in range(len(examples["instruction"])):
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(1, 
                        examples["instruction"][i] + examples["input"][i]) # assume no history
            answers = examples["output"][i] # the chosen answer first, then the rejected answers 

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answers[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answers[1], add_special_tokens=False)

            if len(source_ids) > data_args.model_max_length - 2: # gmask and sop tokens
                source_ids = source_ids[: data_args.model_max_length - 2]
            if len(accept_ids) > data_args.model_max_length - 1: # eos tokens
                accept_ids = accept_ids[: data_args.model_max_length - 1]
            if len(reject_ids) > data_args.model_max_length - 1: # eos tokens
                reject_ids = reject_ids[: data_args.model_max_length - 1]
            
            accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids, accept_ids) # avoid copying error
            reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids, reject_ids)
            
            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs # leave padding to the collator
    
    def show_example_sft(example):
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(
            tokenizer.decode([d if d != IGNORE_INDEX else tokenizer.pad_token_id for d in example["labels"]],
                             skip_special_tokens=False)
        ))

    def show_example_dpo_v1(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"], skip_special_tokens=False)))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"], skip_special_tokens=False)))

    def preprocess_dpo_v2(examples): # for DPOTrainer default data collator
        model_inputs = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(examples["instruction"])):
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(0, 
                        examples["instruction"][i] + examples["input"][i]) # assume no history
            answers = examples["output"][i] # the chosen answer first, then the rejected answers
            model_inputs["prompt"].append(prompt)
            model_inputs["chosen"].append(answers[0])
            model_inputs["rejected"].append(answers[1])
        return model_inputs
    
    def show_example_dpo_v2(example):
        print("prompt:\n{}".format(example["prompt"]))
        print("prompt ids:\n{}".format(tokenizer.encode(example["prompt"])))
        print("chosen:\n{}".format(example["chosen"]))
        print("chosen ids:\n{}".format(tokenizer.encode(example["chosen"])))
        print("rejected:\n{}".format(example["rejected"]))
        print("rejected ids:\n{}".format(tokenizer.encode(example["rejected"])))

    if training_args.stage == "sft":
        preprocess_fn = preprocess_sft
        show_example_fn = show_example_sft
    elif training_args.stage == "dpo":
        preprocess_fn = preprocess_dpo_v2
        show_example_fn = show_example_dpo_v2
    else:
        raise NotImplementedError("finetuning stage {} not implemented".format(training_args.stage))
    # assume json data only
    dataset: Dataset = load_dataset("json", data_files=data_args.data_path)["train"]
    column_names = dataset.column_names
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=column_names,
    )
    dataset = dataset.train_test_split(test_size=0.01,
                                        seed=42, shuffle=True)
    show_example_fn(dataset["train"][0])
    return dataset