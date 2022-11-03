import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

def create_dataloader(path, tokenizer, opt, big=False):
    data = []
    with open(path, "r", encoding='utf8') as f:
        file = f.readlines()
    for line in file:
        if not opt.istest:
            x, y, gold = line.split("\001")[0], line.strip().split("\001")[1], line.strip().split("\001")[2]
        else:
            x, y, gold = line.split("\001")[0], line.strip().split("\001")[1], 0
        data.append([x, y, gold])
    df = pd.DataFrame(data, columns=["input_text", "target_text", "label"])
    if not big:
        train_dataset = BartDataset(df, tokenizer, opt)
        return DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)
    if big:
        train_df, test_df = train_test_split(df, test_size=0.2)
        train_ds = BartDataset(train_df, tokenizer, opt)
        test_ds = BartDataset(test_df, tokenizer, opt)
        return DataLoader(train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0)

def get_inputs_dict(batch, tokenizer, device):
    source_ids, source_mask, y, y_mask, label = batch["source_ids"], batch["source_mask"], batch["target_ids"], batch['target_mask'], batch['labels']
    inputs = {
        "input_ids": source_ids.to(device),
        "attention_mask": source_mask.to(device),
        "decoder_input_ids": y.to(device),
        "decoder_attention_mask": y_mask.to(device),
        "labels": label.to(device),
    }
    return inputs
class BartDataset(Dataset):
    def __init__(self, data, tokenizer, opt, task='ACSA'):
        self.tokenizer = tokenizer
        if task == 'ACD':
            data = [
                ('<s>'+input_text+'</s>', target_text+'</s>', minor_name_to_id[label], tokenizer, opt)
                for input_text, target_text, label in zip(
                    data['input_text'], data['target_text'], data['label']
                )
            ]
        elif task == 'ACSA':
            data = [
                ('<s>'+input_text+'</s>', target_text+'</s>', polarity_to_id[label], tokenizer, opt)
                for input_text, target_text, label in zip(
                    data['input_text'], data['target_text'], data['label']
                )
            ]
        preprocess_fn = (self.preprocess_data_bart)

        self.examples = [preprocess_fn(d) for d in tqdm(data, disable=True)]

    def preprocess_data_bart(self, data):
        input_text, target_text, label, tokenizer, opt = data

        input_ids = tokenizer.batch_encode_plus(
            [input_text],
            max_length=opt.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        target_ids = tokenizer.batch_encode_plus(
            [target_text],
            max_length=opt.max_len,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        return {
            "source_ids": input_ids["input_ids"].squeeze(),
            "source_mask": input_ids["attention_mask"].squeeze(),
            "target_ids": target_ids["input_ids"].squeeze(),
            "target_mask": target_ids["attention_mask"].squeeze(),
            "labels" : torch.tensor(label)
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]