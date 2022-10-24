from utils import *
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from sklearn.metrics import f1_score

def evaluation(y_true, y_pred):
    hit = 0
    total = 0
    for gold, pred in zip(y_true, y_pred):
        total += 1
        # print(f'g {gold} p {pred}')
        if gold == pred:
            hit += 1
    print(f'hit {hit} total {total}')
    print('accuracy: ', hit/total)

    # y_true = list(map(int, y_true))
    # y_pred = list(map(int, y_pred))
    
    f1 = f1_score(y_true, y_pred, average='macro')
    # print('f1_score: ', f1_score(y_true, y_pred, average=None))
    print('f1_score_micro: ', f1_score(y_true, y_pred, average='micro'))
    print('f1_score_macro: ', f1_score(y_true, y_pred, average='macro'))
    return f1
def evaluation_f1(true_data, pred_data):

    true_data_list = true_data
    pred_data_list = pred_data

    ce_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    pipeline_eval = {
        'TP': 0,
        'FP': 0,
        'FN': 0,
        'TN': 0
    }

    for i in range(len(true_data_list)):
        # TP, FN checking
        is_ce_found = False
        is_pipeline_found = False
        for y_ano  in true_data_list[i]['annotation']:
            y_category = y_ano[0]
            y_polarity = y_ano[2]

            for p_ano in pred_data_list[i]['annotation']:
                p_category = p_ano[0]
                p_polarity = p_ano[1]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is True:
                ce_eval['TP'] += 1
            else:
                ce_eval['FN'] += 1

            if is_pipeline_found is True:
                pipeline_eval['TP'] += 1
            else:
                pipeline_eval['FN'] += 1

            is_ce_found = False
            is_pipeline_found = False

        # FP checking
        for p_ano in pred_data_list[i]['annotation']:
            p_category = p_ano[0]
            p_polarity = p_ano[1]

            for y_ano  in true_data_list[i]['annotation']:
                y_category = y_ano[0]
                y_polarity = y_ano[2]

                if y_category == p_category:
                    is_ce_found = True
                    if y_polarity == p_polarity:
                        is_pipeline_found = True

                    break

            if is_ce_found is False:
                ce_eval['FP'] += 1

            if is_pipeline_found is False:
                pipeline_eval['FP'] += 1

    ce_precision = ce_eval['TP']/(ce_eval['TP']+ce_eval['FP']+1)
    ce_recall = ce_eval['TP']/(ce_eval['TP']+ce_eval['FN']+1)

    ce_result = {
        'Precision': ce_precision,
        'Recall': ce_recall,
        'F1': 2*ce_recall*ce_precision/(ce_recall+ce_precision)
    }

    pipeline_precision = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FP']+1)
    pipeline_recall = pipeline_eval['TP']/(pipeline_eval['TP']+pipeline_eval['FN']+1)

    pipeline_result = {
        'Precision': pipeline_precision,
        'Recall': pipeline_recall,
        'F1': 2*pipeline_recall*pipeline_precision/(pipeline_recall+pipeline_precision)
    }

def predict_val(dev_path, model, tokenizer, device, opt):
    major_candidate_list = ["전체", "패키지", "본품", "브랜드"]
    minor_candidate_list = ["일반", "디자인", "가격", '품질', '인지도', '편의성', '다양성']
    target_list = []

    for major in major_candidate_list:
        target_list.append('<s>'+ ACDTemplate(major, '') + '</s>')
        # for minor in minor_entity_candidate_list:
            # target_list.append(ACDTemplate(major, minor))
    with open(dev_path, "r", encoding='utf8') as f:
        file = f.readlines()

    count = 0
    total = 0
    for line in file:
        total += 1
        line = line.strip()
        # x, major, gold = line.split("\001")[0], line.split("\001")[1], line.split("\001")[2]
        x, gold = line.split("\001")[0], line.split("\001")[1]
        input_ids = tokenizer([x] * 4, return_tensors='pt')['input_ids']
        output_ids = tokenizer(target_list, return_tensors='pt', max_length=opt.max_len, padding=True, truncation=True)['input_ids']

        with torch.no_grad():
            output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids.to(device))[0]
            logits = output.softmax(dim=-1).to('cpu').numpy()        
        score_list = []
        for i in range(4):
            score = 1
            for j in range(logits[i].shape[0] - 2):
                score *= logits[i][j][output_ids[i][j + 1]]
            score_list.append(score)
        predict = major_candidate_list[np.argmax(score_list)]
        # print(f'pred {predict} gold {gold}')
        if predict == gold:
            count += 1
    return count/total