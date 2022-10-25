import tokenizers
from transformers import AutoTokenizer, BartTokenizer
import torch
from torch import tensor
import torch.nn.functional as F
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


major_id_to_name = ['전체', '패키지', '본품', '브랜드']
major_name_to_id = { major_id_to_name[i]: i for i in range(len(major_id_to_name)) }
minor_id_to_name = ['일반', '디자인', '가격', '품질', '인지도', '편의성', '다양성']
minor_name_to_id = { minor_id_to_name[i]: i for i in range(len(minor_id_to_name)) }
polarity_id_to_name = ['긍정적', '부정적', '중립적']
polarity_en_to_ko ={
    'positive' : '긍정적',
    'negative' : '부정적',
    'neutral' : '중립적'
}
polarity_to_id ={
    '긍정적':0,
    '부정적':1,
    '중립적':2
}
entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']
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
    def __init__(self, data, tokenizer, opt):
        self.tokenizer = tokenizer
        data = [
            #TODO : fix polarity_to_id accoding to task
            # ('<s>'+input_text+'</s>', target_text+'</s>', minor_name_to_id[label], tokenizer, opt)
            ('<s>'+input_text+'</s>', target_text+'</s>', polarity_to_id[label], tokenizer, opt)
            for input_text, target_text, label in zip(
                data['input_text'], data['target_text'], data['label']
            )
        ]
        preprocess_fn = (
            self.preprocess_data_bart
        )

        self.examples = [
            preprocess_fn(d) for d in tqdm(data, disable=True)
        ]
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