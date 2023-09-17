# import json
# from torch.utils.data import Dataset
from datasets import Dataset, load_dataset
import torch
from typing import Optional, Dict
from transformers.tokenization_utils import PreTrainedTokenizer


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
    data_files: str,
    tokenizer: PreTrainedTokenizer,
    model_max_length: int,
) -> Dataset:
    
    def preprocessing_v1(examples): # for customized data collator
        # examples shpaed as {
        #   "instruction": [], # batch_size * instruction 
        #   "input": [], # batch_size * input
        #   "output": [] # batch_size * (output, rejected_outputs)
        # } 

        # v1: build input pairs with format `X [gMASK] <sop> Y1 <eop>` and `X [gMASK] <sop> Y2 <eop>`
        # v2: build input pairs with format `[gMASK] sop X Y1 </s>` and `[gMASK] sop X Y2 </s>`
        model_inputs = {"accept_ids": [], "reject_ids": []}
        for i in range(len(examples["instruction"])):
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(0, 
                        examples["instruction"][i] + examples["input"][i]) # assume no history
            answers = examples["output"][i] # the chosen answer first, then the rejected answers 

            source_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
            accept_ids = tokenizer.encode(text=answers[0], add_special_tokens=False)
            reject_ids = tokenizer.encode(text=answers[1], add_special_tokens=False)

            if len(source_ids) > model_max_length - 2: # gmask and sop tokens
                source_ids = source_ids[: model_max_length - 2]
            if len(accept_ids) > model_max_length - 1: # eos tokens
                accept_ids = accept_ids[: model_max_length - 1]
            if len(reject_ids) > model_max_length - 1: # eos tokens
                reject_ids = reject_ids[: model_max_length - 1]
            
            accept_ids = tokenizer.build_inputs_with_special_tokens(source_ids, accept_ids) # avoid copying error
            reject_ids = tokenizer.build_inputs_with_special_tokens(source_ids, reject_ids)
            
            model_inputs["accept_ids"].append(accept_ids)
            model_inputs["reject_ids"].append(reject_ids)
        return model_inputs # leave padding to the collator

    def show_example_v1(example):
        print("accept_ids:\n{}".format(example["accept_ids"]))
        print("accepts:\n{}".format(tokenizer.decode(example["accept_ids"], skip_special_tokens=False)))
        print("reject_ids:\n{}".format(example["reject_ids"]))
        print("rejects:\n{}".format(tokenizer.decode(example["reject_ids"], skip_special_tokens=False)))

    def preprocessing_v2(examples): # for DPOTrainer default data collator
        model_inputs = {"prompt": [], "chosen": [], "rejected": []}
        for i in range(len(examples["instruction"])):
            prompt = "[Round {}]\n\n问：{}\n\n答：".format(0, 
                        examples["instruction"][i] + examples["input"][i]) # assume no history
            answers = examples["output"][i] # the chosen answer first, then the rejected answers
            model_inputs["prompt"].append(prompt)
            model_inputs["chosen"].append(answers[0])
            model_inputs["rejected"].append(answers[1])
        return model_inputs
    
    def show_example_v2(example):
        print("prompt:\n{}".format(example["prompt"]))
        print("prompt ids:\n{}".format(tokenizer.encode(example["prompt"])))
        print("chosen:\n{}".format(example["chosen"]))
        print("chosen ids:\n{}".format(tokenizer.encode(example["chosen"])))
        print("rejected:\n{}".format(example["rejected"]))
        print("rejected ids:\n{}".format(tokenizer.encode(example["rejected"])))

    # assume json data only
    dataset: Dataset = load_dataset("json", data_files=data_files)["train"]
    column_names = dataset.column_names
    dataset = dataset.map(
        preprocessing_v2,
        batched=True,
        remove_columns=column_names,
    )
    dataset = dataset.train_test_split(test_size=0.01,
                                        seed=42, shuffle=True)
    show_example_v2(dataset["train"][0])
    return dataset