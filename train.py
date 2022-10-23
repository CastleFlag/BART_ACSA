# from models import Seq2SeqModel
# model = Seq2SeqModel()
# print(model)
# ARTICLE_TO_SUMMARIZE = (
#     "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
#     "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
#     "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
# )
# input_ids = model.encoder_tokenizer(ARTICLE_TO_SUMMARIZE, return_tensors="pt").input_ids

# # autoregressively generate summary (uses greedy decoding by default)
# generated_ids = model.generate(input_ids)
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)
import argparse
from transformers import AutoModel, AutoTokenizer
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from data import *
from models import BartForConditionalGeneration
from evalutation import evaluation, predict_val
from tqdm import tqdm
from shutil import copyfile
import os
from sklearn.model_selection import train_test_split

special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}
def get_inputs_dict(batch, tokenizer, device, test=False):
    pad_token_id = tokenizer.pad_token_id
    # pad_token_id =1 
    source_ids, source_mask, y = batch["source_ids"], batch["source_mask"], batch["target_ids"]
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone()
    if not test:
        lm_labels[y[:, 1:] == pad_token_id] = -100

    inputs = {
        "input_ids": source_ids.to(device),
        "attention_mask": source_mask.to(device),
        "decoder_input_ids": y_ids.to(device),
        "labels": lm_labels.to(device),
    }
    return inputs

def train(opt, device):
    entity_model_path = opt.entity_model_path + opt.base_model + '/' + str(opt.num_labels) + '/'
    polarity_model_path = opt.polarity_model_path + opt.base_model + '/'
    best_model_path = '../saved_model/best_model/'
    if not os.path.exists(entity_model_path):
        os.makedirs(entity_model_path)
    if not os.path.exists(polarity_model_path):
        os.makedirs(polarity_model_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    print(opt.base_model)
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    tokenizer.add_special_tokens(special_tokens_dict)

    train_dataloader = create_dataloader(opt.train_data, tokenizer, opt)
    dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt)

    print('loading model')
    model = BartForConditionalGeneration.from_pretrained(opt.base_model)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print('end loading')

    optimizer = AdamW(model.parameters(), lr=opt.learning_rate, eps=opt.eps)
    epochs = opt.num_train_epochs

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs = get_inputs_dict(batch, tokenizer, device)
            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()
            # print('batch_loss: ', loss.item())
            optimizer.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'{opt.train_target} Property_Epoch: {epoch+1}')
        print(f'Average train loss: {avg_train_loss}')

        # print(f'predict_val acc : {predict_val(opt.dev_data, model, tokenizer, device, opt)}')
        pred_list = []
        label_list = []
        if opt.do_eval:
            for batch in dev_dataloader:
                inputs = get_inputs_dict(batch, tokenizer, device, test=True)
                with torch.no_grad():
                    logits = model.generate(inputs['input_ids'])
                    y_true = tokenizer.batch_decode(inputs['labels'],skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    y_pred = tokenizer.batch_decode(logits ,skip_special_tokens=True, clean_up_tokenization_spaces=True, num_beams=3)
                pred_list.extend(y_pred)
                label_list.extend(y_true)
            f1score = evaluation(label_list, pred_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--train_target", type=str, default="Entity", help="train entity or polarity")
    # parser.add_argument( "--train_data", type=str, default="data/acd_sample.jsonl", help="train file")
    parser.add_argument( "--train_data", type=str, default="data/acd_train.jsonl", help="train file")
    # parser.add_argument( "--train_data", type=str, default="data/acd_big.jsonl", help="train file")
    # parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-test.jsonl", help="test file")
    parser.add_argument( "--dev_data", type=str, default="data/acd_dev.jsonl", help="dev file")
    # parser.add_argument( "--dev_data", type=str, default="data/acd_dev_sample.jsonl", help="train file")
    parser.add_argument( "--batch_size", type=int, default=16) 
    parser.add_argument( "--learning_rate", type=float, default=1e-5) 
    parser.add_argument( "--eps", type=float, default=1e-8)
    parser.add_argument( "--do_eval", type=bool, default=True)
    parser.add_argument( "--num_train_epochs", type=int, default=10)
    parser.add_argument( "--base_model", type=str, default="gogamza/kobart-summarization")
    # parser.add_argument( "--base_model", type=str, default="hyunwoongko/kobart")
    parser.add_argument( "--num_labels", type=int, default=25)
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--classifier_hidden_size", type=int, default=768)
    parser.add_argument( "--classifier_dropout_prob", type=float, default=0.1, help="dropout in classifier")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(opt, device)
