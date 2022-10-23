import tokenizers
from transformers import AutoTokenizer, BartTokenizer
import torch
from torch import tensor
import torch.nn.functional as F
from utils import jsonlload
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

def create_dataloader(path, tokenizer, opt):
    data = []
    with open(path, "r", encoding='utf8') as f:
        file = f.readlines()
    for line in file:
        x, y = line.split("\001")[0], line.strip().split("\001")[1]
        data.append([x, y])
    df = pd.DataFrame(data, columns=["input_text", "target_text"])
    train_dataset = BartDataset(df, tokenizer, opt)
    return DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0)

polarity_en_to_ko ={
    'positive' : '긍정적',
    'negative' : '부정적',
    'neutral' : '중립적'
}
entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']
class BartDataset(Dataset):
    def __init__(self, data, tokenizer, opt):
        self.tokenizer = tokenizer
        data = [
            (input_text, '<s>'+target_text+'</s>', tokenizer, opt)
            for input_text, target_text in zip(
                data['input_text'], data['target_text']
            )
        ]
        preprocess_fn = (
            self.preprocess_data_bart
        )

        self.examples = [
            preprocess_fn(d) for d in tqdm(data, disable=True)
        ]
    def preprocess_data_bart(self, data):
        input_text, target_text, tokenizer, opt = data

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
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]