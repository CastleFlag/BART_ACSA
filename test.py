import argparse
import torch
from transformers import AutoModel, AutoTokenizer, BartForSequenceClassification
from data import *
special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
    }

def test(opt, device):
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    tokenizer.add_special_tokens(special_tokens_dict)

    print('loading model')
    model = BartForSequenceClassification.from_pretrained(opt.base_model, num_labels=opt.num_labels)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print('end loading')

    dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument( "--train_data", type=str, default="data/acd_sample.jsonl", help="train file")
    parser.add_argument( "--train_data", type=str, default="data/acd_train.jsonl", help="train file")
    # parser.add_argument( "--train_data", type=str, default="data/acd_big.jsonl", help="train file")
    # parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-test.jsonl", help="test file")
    parser.add_argument( "--dev_data", type=str, default="data/acd_dev.jsonl", help="dev file")
    # parser.add_argument( "--dev_data", type=str, default="data/acd_dev_sample.jsonl", help="train file")
    parser.add_argument( "--batch_size", type=int, default=16) 
    parser.add_argument( "--base_model", type=str, default="hyunwoongko/kobart")
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(opt, device)