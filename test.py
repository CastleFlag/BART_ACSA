import argparse
import torch
from transformers import AutoModel, AutoTokenizer, BartForSequenceClassification
from data import *
from utils import clean_text
special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
    }

def test(opt, device):

    print(opt.base_model)
    bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_base_model)
    bart_tokenizer = AutoTokenizer.from_pretrained(opt.bart_base_model)
    bert_tokenizer.add_special_tokens(special_tokens_dict)
    bart_tokenizer.add_special_tokens(special_tokens_dict)
    bert_dev_dataloader = create_dataloader(opt.dev_data, bert_tokenizer, opt)
    bart_dev_dataloader = create_dataloader(opt.dev_data, bart_tokenizer, opt)

    print('loading model')
    ce4_model, ce7_model, pc_model = load_models(opt)
    ce4_model.resize_token_embeddings(len(bert_tokenizer))
    ce7_model.resize_token_embeddings(len(bart_tokenizer))
    pc_model.resize_token_embeddings(len(bart_tokenizer))
    ce4_model.to(device)
    ce7_model.eval()
    ce7_model.to(device)
    pc_model.to(device)
    pc_model.eval()
    print('end loading')

    for bert_batch, bart_batch in zip(bert_dev_dataloader, bart_dev_dataloader):
        bert_inputs = get_inputs_dict(bert_batch, bert_tokenizer, device)
        bart_inputs = get_inputs_dict(bart_batch, bart_tokenizer, device)
        
        with torch.no_grad():
            _, ce4_logits = ce4_model(**bert_inputs)
        print(ce4_logits)
        major_prediction = major_id_to_name[torch.argmax(ce4_logits, dim = -1)]
        print(major_prediction)
        with torch.no_grad():
            _, ce7_logits = ce7_model(bart_inputs['input_ids'], bart_inputs['attention_mask'], major_prediction)
        minor_prediction = minor_id_to_name[torch.argmax(ce7_logits, dim = -1)]

        with torch.no_grad():
            _, pc_logits = pc_model(input_ids, input_mask, target_id)
        pc_prediction = polarity_id_to_name[torch.argmax(ce7_logits, dim = -1)]
        print(inputs)
        # with torch.no_grad():
        #     _, ce4_logits = ce4_model(input_ids, attention_mask)

    # now = datetime.now()
    # current_day = now.strftime('%m%d')

    # jsondump(pred_data, opt.output_dir + opt.base_model +'_'+ current_day + '.json')
    # print(opt.output_dir + opt.base_model +'_'+ current_day + '.json')

def load_models(opt):
    model4 = BartForSequenceClassification.from_pretrained(opt.bert_base_model, num_labels=4)
    model4.load_state_dict(torch.load(opt.entity4_model_path, map_location=device))

    model7 = BartForSequenceClassification.from_pretrained(opt.bart_base_model, num_labels=7)
    model7.load_state_dict(torch.load(opt.entity7_model_path, map_location=device))

    pola_model = BartForSequenceClassification.from_pretrained(opt.bart_base_model, num_labels=3)
    pola_model.load_state_dict(torch.load(opt.polarity_model_path, map_location=device))

    return model4, model7, pola_model

def inference(tokenizer, ce4_model, ce7_model, pc_model, data):
    count = 0
    for line in data:
        sentence = line['sentence_form']
        # sentence['annotation'] = []
        count += 1
        # if type(sentence) != str:
        #     print("form type is arong: ", sentence)
        #     continue
        sentence = clean_text(sentence)

        tokenized_data = tokenizer(sentence, padding='max_length', max_length=256, truncation=True)
        input_ids = torch.tensor([tokenized_data['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_data['attention_mask']]).to(device)
        with torch.no_grad():
            _, ce4_logits = ce4_model(input_ids, attention_mask)
        major_prediction = major_id_to_name[torch.argmax(ce4_logits, dim = -1)]

        tokenized_major = tokenizer(major_prediction, padding='max_length', max_length=256, truncation=True)
        major_target_ids = torch.tensor([tokenized_major['input_ids']]).to(device)
        major_target_mask = torch.tensor([tokenized_major['attention_mask']]).to(device)
        with torch.no_grad():
            _, ce7_logits = ce7_model(input_ids, attention_mask, major_target_ids)
        minor_prediction = minor_id_to_name[torch.argmax(ce7_logits, dim = -1)]

        tokenized_minor = tokenizer(minor_prediction, padding='max_length', max_length=256, truncation=True)
        minor_target_ids = torch.tensor([tokenized_minor['input_ids']]).to(device)
        minor_target_mask = torch.tensor([tokenized_minor['attention_mask']]).to(device)
        with torch.no_grad():
            _, pc_logits = pc_model(input_ids, attention_mask, minor_target_ids)
        pc_prediction = polarity_id_to_name[torch.argmax(ce7_logits, dim = -1)]
        print()
        # sentence['annotation'].append([ce4_result+ce7_result, pc_result])
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--dev_data", type=str, default="data/acd_dev.jsonl", help="dev file")
    parser.add_argument( "--batch_size", type=int, default=8) 
    parser.add_argument( "--bert_base_model", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument( "--bart_base_model", type=str, default="digit82/kobart-summarization")
    # parser.add_argument( "--bart_base_model", type=str, default="hyunwoongko/kobart")
    parser.add_argument( "--entity4_model_path", type=str, default="./saved_models/model4.pt")
    parser.add_argument( "--entity7_model_path", type=str, default="./saved_models/model7.pt")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/pc_model.pt")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test(opt, device)