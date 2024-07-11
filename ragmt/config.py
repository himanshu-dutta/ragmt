MASKED_LM_LANG_CODE = "msk_LANG"
MASK_TOKEN = "<mask>"
NER_BATCH_SIZE = 32
N_DOCS = 2

LANG_CONFIG = {
    "english": {
        "dataset_ext": "en",
        "nllb_id": "eng_Latn",
        "ner_id": "en",
        "ner_model": "en_core_web_sm",
    },
    "german": {
        "dataset_ext": "de",
        "nllb_id": "deu_Latn",
        "ner_id": "de",
        "ner_model": "de_core_news_sm",
    },
}


##############################
#  RETRIEVER CONFIG
##############################
DOCUMENT_ENCODER = "facebook/dpr-ctx_encoder-multiset-base"
RETRIEVER_MAX_LENGTH = 512

##############################
#   GENERATOR CONFIG
##############################
GENERATOR_MODEL = "/home/artifacts/model/pretrained/nllb200-600M"
GENERATOR_INPUT_MAX_LENGTH = 1024  # has document and input
GENERATOR_OUTPUT_MAX_LENGTH = 512  # has only output

##############################
#   FAISS CONFIG
##############################
FAISS_HNSW_D = (
    768  # this should be set according to the output dimension of the document encoder
)
FAISS_HNSW_M = 128

##############################
#   TRAINING CONFIG
##############################
LEARNING_RATE = 2e-5
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8
