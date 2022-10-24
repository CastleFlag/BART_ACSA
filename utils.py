import json
import re
from responses import target

def jsonload(fname, encoding="utf-8"):
    with open(fname, encoding=encoding) as f:
        j = json.load(f)
    return j
# json 개체를 파일이름으로 깔끔하게 저장
def jsondump(j, fname):
    with open(fname, "w", encoding="UTF8") as f:
        json.dump(j, f, ensure_ascii=False)
# jsonl 파일 읽어서 list에 저장
def jsonlload(fname, encoding="utf-8"):
    json_list = []
    with open(fname, encoding=encoding) as f:
        for line in f.readlines():
            json_list.append(json.loads(line))
    return json_list
def simple_major(major):
    if major== '제품 전체':
        return '제품'
    elif major== '패키지/구성품':
        return '패키지'
    return major 

polarity_en_to_ko ={
    'positive' : '긍정적',
    'negative' : '부정적',
    'neutral' : '중립적'
}
def ACDACD_template(major_category, minor_category):
    return major_category+ '의 '+ minor_category + ' 항목이다.'
def ASCA_template(minor_category, polarity):
    return minor_category + '에 대해 ' + polarity_en_to_ko(polarity) + '이다.'
def ACD_template(minor_category, positive=True):
    if positive:
        return minor_category + '에 대한 평가이다.'
    else:
        return minor_category + '에 대한 평가가 아니다.'
def reverse_ACD(sentence):
    return sentence.split()[0][:-1]
def reverse_ACDACD(sentence):
    return sentence.split()[-2]
def reverse_ASCA(sentence):
    return sentence.split()[-1][:3]

def make_json(src_path, target_path, task, valid=False):
    raw_data = jsonlload(src_path)
    write_buffer = []
    for utterance in raw_data:
        sentence = utterance['sentence_form']
        sentence = re.compile('[^ 0-9A-Za-z가-힣]').sub('',sentence).strip()
        annotations = utterance['annotation']
        for annotation in annotations:
            entity = annotation[0]
            major, minor = entity.split('#')
            major = simple_major(major)
            polarity = polarity_en_to_ko[annotation[2]]
            if task == 'ACD':
                write_buffer.append(sentence + '\001' + major + '\001' + minor)
            elif task == 'ACSA':
                write_buffer.append(sentence + '\001' + minor + '\001' + polarity)

    with open(target_path, 'w', encoding='utf8') as f:
        for line in write_buffer:
            f.write(line+'\n')

if __name__ == '__main__':
    sources = ['data/sample.jsonl', 'data/nikluge-sa-2022-train.jsonl', 'data/nikluge-sa-2022-dev.jsonl','data/nikluge-sa-2022-test.jsonl', 'data/big_train.jsonl']
    targets = ['sample', 'train', 'dev', 'test', 'big']
    task = 'ACSA'
    for src, trg in zip(sources, targets):
        path = f'data/{task}_{trg}.jsonl'
        print(path)
        make_json(src, path, task)