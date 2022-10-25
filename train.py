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
from transformers import AutoModel, AutoTokenizer, BartForSequenceClassification
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
def train(opt, device):
    entity_model_path = opt.entity_model_path +  str(opt.num_labels) + '/'
    best_model_path = '../saved_model/best_model/'
    if not os.path.exists(entity_model_path):
        os.makedirs(entity_model_path)
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)

    print(opt.base_model)
    tokenizer = AutoTokenizer.from_pretrained(opt.base_model)
    tokenizer.add_special_tokens(special_tokens_dict)

    # train_dataloader, dev_dataloader = create_dataloader(opt.train_data, tokenizer, opt, big=True)
    train_dataloader = create_dataloader(opt.train_data, tokenizer, opt)
    dev_dataloader = create_dataloader(opt.dev_data, tokenizer, opt)

    print('loading model')
    model = BartForSequenceClassification.from_pretrained(opt.base_model, num_labels=opt.num_labels)
    # model = BartForSequenceClassification.from_pretrained("valhalla/bart-large-sst2", problem_type="multi_label_classification")
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    print('end loading')

    optimizer = AdamW(model.parameters(), lr=opt.learning_rate, eps=opt.eps)
    epochs = opt.num_train_epochs

    total_steps = epochs * len(train_dataloader)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=total_steps
    # )

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            inputs = get_inputs_dict(batch, tokenizer, device)
            model.zero_grad()
            outputs = model(**inputs)
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            loss.backward()

            total_loss += loss.item()
            # print('batch_loss: ', loss.item())
            optimizer.step()
            # scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f'{opt.train_target} Property_Epoch: {epoch+1}')
        print(f'Average train loss: {avg_train_loss}')

        # print(f'predict_val acc : {predict_val(opt.dev_data, model, tokenizer, device, opt)}')
        if opt.do_eval:
            model.eval()
            pred_list = []
            label_list = []
            
            for batch in dev_dataloader:
                inputs = get_inputs_dict(batch, tokenizer, device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_class_id = logits.argmax(dim=1)
                pred_list.extend(predicted_class_id.cpu())
                label_list.extend(inputs['labels'].cpu())
            f1score = evaluation(label_list, pred_list)
        model_saved_path = entity_model_path + str(opt.num_labels) +'saved_model_epoch_' + str(epoch+1) + '.pt'
        torch.save(model.state_dict(), model_saved_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--train_target", type=str, default="Entity", help="train entity or polarity")
    parser.add_argument( "--train_data", type=str, default="data/ACSA_train.jsonl", help="train file")
    # parser.add_argument( "--test_data", type=str, default="../data/nikluge-sa-2022-test.jsonl", help="test file")
    parser.add_argument( "--dev_data", type=str, default="data/ACSA_dev.jsonl", help="dev file")
    # parser.add_argument( "--dev_data", type=str, default="data/acd_dev_sample.jsonl", help="train file")
    parser.add_argument( "--batch_size", type=int, default=16) 
    parser.add_argument( "--learning_rate", type=float, default=1e-5) 
    parser.add_argument( "--eps", type=float, default=1e-8)
    parser.add_argument( "--do_eval", type=bool, default=True)
    parser.add_argument( "--num_train_epochs", type=int, default=20)
    # parser.add_argument( "--base_model", type=str, default="gogamza/kobart-summarization")
    parser.add_argument( "--base_model", type=str, default="digit82/kobart-summarization")
    parser.add_argument( "--num_labels", type=int, default=3)
    parser.add_argument( "--entity_model_path", type=str, default="./saved_models/entity_model/")
    parser.add_argument( "--polarity_model_path", type=str, default="./saved_models/polarity_model/")
    parser.add_argument( "--output_dir", type=str, default="../output/")
    parser.add_argument( "--max_len", type=int, default=256)
    parser.add_argument( "--istest", type=bool, default=False, help="train/dev or test(no label)")
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(opt, device)
