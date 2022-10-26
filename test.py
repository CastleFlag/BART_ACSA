import argparse
import torch
from transformers import AutoModel, AutoTokenizer, BartForSequenceClassification, BertForSequenceClassification 
from data import *
from utils import clean_text, unsimple_major, jsondump
special_tokens_dict = {
    'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
    }

def test(opt, device):

    print(opt.bert_base_model)
    print(opt.bart_base_model)
    bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_base_model)
    bart_tokenizer = AutoTokenizer.from_pretrained(opt.bart_base_model)
    bert_tokenizer.add_special_tokens(special_tokens_dict)
    bart_tokenizer.add_special_tokens(special_tokens_dict)
    bert_dev_dataloader = create_bert_dataloader(opt.dev_data, bert_tokenizer, opt, 'ACD')
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
            _, pc_logits = pc_model(bart_inputs['input_ids'], bart_inputs['input_mask'], minor_prediction)
        pc_prediction = polarity_id_to_name[torch.argmax(pc_logits, dim = -1)]
        print(pc_prediction)
        # with torch.no_grad():
        #     _, ce4_logits = ce4_model(input_ids, attention_mask)

    # now = datetime.now()
    # current_day = now.strftime('%m%d')

    # jsondump(pred_data, opt.output_dir + opt.base_model +'_'+ current_day + '.json')
    # print(opt.output_dir + opt.base_model +'_'+ current_day + '.json')

def load_models(opt, bert_tokenizer_len, bart_tokenizer_len):
    model4 = BertForSequenceClassification.from_pretrained(opt.bert_base_model, num_labels=4)
    model7 = BartForSequenceClassification.from_pretrained(opt.bart_base_model, num_labels=7)
    pola_model = BartForSequenceClassification.from_pretrained(opt.bart_base_model, num_labels=3)
    model4.resize_token_embeddings(bert_tokenizer_len)
    model7.resize_token_embeddings(bart_tokenizer_len)
    pola_model.resize_token_embeddings(bart_tokenizer_len)
    model4.load_state_dict(torch.load(opt.entity4_model_path, map_location=device))
    model7.load_state_dict(torch.load(opt.entity7_model_path, map_location=device))
    pola_model.load_state_dict(torch.load(opt.polarity_model_path, map_location=device))

    return model4, model7, pola_model

def inference(opt, device):
    print(opt.bert_base_model)
    print(opt.bart_base_model)
    bert_tokenizer = AutoTokenizer.from_pretrained(opt.bert_base_model)
    bart_tokenizer = AutoTokenizer.from_pretrained(opt.bart_base_model)
    bert_tokenizer.add_special_tokens(special_tokens_dict)
    bart_tokenizer.add_special_tokens(special_tokens_dict)
    print('loading model')
    ce4_model, ce7_model, pc_model = load_models(opt,len(bert_tokenizer), len(bart_tokenizer))
    ce4_model.to(device)
    ce7_model.eval()
    ce7_model.to(device)
    pc_model.to(device)
    pc_model.eval()
    print('end loading')
    count = 0
    hit = 0
    majorhit = 0
    minorhit = 0
    pchit = 0
    data = jsonlload(opt.dev_data)
    for line in data:
        sentence = line['sentence_form']
        # sentence['annotation'] = []
        # if type(sentence) != str:
        #     print("form type is arong: ", sentence)
        #     continue
        sentence = clean_text(sentence)

        bert_tokenized_data = bert_tokenizer(['<CLS>'+sentence+'<SEP>'], padding='max_length', max_length=256, truncation=True)
        bert_input_ids = torch.tensor(bert_tokenized_data['input_ids']).to(device)
        bert_attention_mask = torch.tensor([bert_tokenized_data['attention_mask']]).to(device)

        bart_tokenized_data = bart_tokenizer(['<s>'+sentence+'</s>'], padding='max_length', max_length=256, truncation=True, return_tensors='pt')
        bart_input_ids = bart_tokenized_data['input_ids'].to(device)
        bart_attention_mask = bart_tokenized_data['attention_mask'].to(device)

        with torch.no_grad():
            ce4_logits = ce4_model(bert_input_ids, bert_attention_mask).logits
        major_prediction = major_id_to_name[torch.argmax(ce4_logits, dim = -1)]
        tokenized_major = bart_tokenizer([major_prediction+'</s>'], padding='max_length', max_length=256, truncation=True, return_tensors='pt')
        tokenized_major_id = tokenized_major['input_ids'].to(device)
        tokenized_major_mask = tokenized_major['attention_mask'].to(device)

        with torch.no_grad():
            ce7_logits = ce7_model(bart_input_ids, bart_attention_mask, tokenized_major_id, tokenized_major_mask).logits
        minor_prediction = minor_id_to_name[torch.argmax(ce7_logits, dim = -1)]
        tokenized_minor = bart_tokenizer([minor_prediction+'</s>'], padding='max_length', max_length=256, truncation=True, return_tensors='pt')
        tokenized_minor_id = tokenized_minor['input_ids'].to(device)
        tokenized_minor_mask = tokenized_minor['attention_mask'].to(device)

        with torch.no_grad():
            pc_logits = pc_model(bart_input_ids, bart_attention_mask, tokenized_minor_id, tokenized_minor_mask).logits
        pc_prediction = polarity_id_to_name[torch.argmax(pc_logits, dim = -1)]

        # annotation = line['annotation'][0]
        # entity = annotation[0]
        # major, minor = entity.split('#')
        # major = simple_major(major)
        # polarity = polarity_en_to_ko[annotation[2]]
        # print(f'pred 4{major_prediction}=7{minor_prediction}=pc{pc_prediction}=')
        # print(f'gold 4{major}=7{minor}=pc{polarity}=')
        # if major==major_prediction:
        #     majorhit += 1
        # if minor==minor_prediction:
        #     minorhit += 1
        # if polarity==pc_prediction:
        #     pchit += 1
        # if major == major_prediction and minor==minor_prediction and polarity==pc_prediction:
        #     hit += 1
        # count +=1 
        # data.append([sentence, major_name_to_id[major]])
        

        line['annotation'].append([unsimple_major(major_prediction)+'#'+minor_prediction, pc_prediction])
    # print(f'accuracy : {hit/count} ma {majorhit/count} mi {minorhit/count} pc {pchit/count}')
    jsondump(data, 'output.json')
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument( "--dev_data", type=str, default="data/nikluge-sa-2022-dev.jsonl", help="dev file")
    parser.add_argument( "--dev_data", type=str, default="data/nikluge-sa-2022-test.jsonl", help="dev file")
    parser.add_argument( "--batch_size", type=int, default=8) 
    # parser.add_argument( "--bert_base_model", type=str, default="bert-base-multilingual-uncased")
    parser.add_argument( "--bert_base_model", type=str, default="xlm-roberta-base")
    parser.add_argument( "--bart_base_model", type=str, default="digit82/kobart-summarization")
    # parser.add_argument( "--bart_base_model", type=str, default="hyunwoongko/kobart")
    parser.add_argument( "--entity4_model_path", type=str, default="./saved_models/model4.pt")
    parser.add_argument( "--entity7_model_path", type=str, default="./saved_models/model7.pt")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/pc_model.pt")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--istest", type=bool, default=True)
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference(opt, device)