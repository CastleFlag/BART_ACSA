from transformers import BartConfig, BartForConditionalGeneration, BartTokenizer, BertConfig, BertForMaskedLM, BertModel, BertTokenizer, EncoderDecoderConfig, EncoderDecoderModel
bartplm = 'digit82/kobart-summarization'
bertplm = 'bert-base-multilingual-uncased'
BART = True
    # "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
    # "bert": (BertConfig, BertModel, BertTokenizer),
class Seq2SeqModel:
    def __init__(self):
        if BART:
            self.model = BartForConditionalGeneration.from_pretrained(bartplm)
            self.encoder_tokenizer = BartTokenizer.from_pretrained(bartplm)
            self.decoder_tokenizer = BartTokenizer.from_pretrained(bartplm)
            # self.config = self.model.config
        else:
            self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(bertplm, bertplm)
            self.model.encoder = BertModel.from_pretrained(bertplm)
            self.model.decoder = BertForMaskedLM.from_pretrained(bertplm)
            self.encoder_tokenizer = BertTokenizer.from_pretrained(bertplm)
            self.decoder_tokenizer = BertTokenizer.from_pretrained(bertplm)
            # self.encoder_config = self.model.config.encoder
            # self.decoder_config = self.model.config.decoder