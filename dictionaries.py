polarity_en_to_ko ={
    'positive' : '긍정적',
    'negative' : '부정적',
    'neutral' : '중립적'
}
polarity_ko_to_en = {
    '긍정적' : 'positive',
    '부정적' : 'negative',
    '중립적' : 'neutral'
}
polarity_id_to_name = ['긍정적', '부정적', '중립적']
polarity_to_id ={
    '긍정적':0,
    '부정적':1,
    '중립적':2
}

major_id_to_name = ['제품', '패키지', '본품', '브랜드']
major_name_to_id = { major_id_to_name[i]: i for i in range(len(major_id_to_name)) }
minor_id_to_name = ['일반', '디자인', '가격', '품질', '인지도', '편의성', '다양성']
minor_name_to_id = { minor_id_to_name[i]: i for i in range(len(minor_id_to_name)) }

entity_property_pair= [   
        '제품 전체#일반', '제품 전체#디자인','제품 전체#가격','제품 전체#품질','제품 전체#인지도', '제품 전체#편의성','제품 전체#다양성',
        '패키지/구성품#일반', '패키지/구성품#디자인','패키지/구성품#가격','패키지/구성품#품질''패키지/구성품#다양성', '패키지/구성품#편의성',
        '본품#일반', '본품#디자인','본품#가격', '본품#품질','본품#다양성','본품#인지도','본품#편의성',  
        '브랜드#일반', '브랜드#디자인', '브랜드#가격', '브랜드#품질', '브랜드#인지도']
special_tokens_dict = {
'additional_special_tokens': ['&name&', '&affiliation&', '&social-security-num&', '&tel-num&', '&card-num&', '&bank-account&', '&num&', '&online-account&']
}