import torch
from transformers import BertModel, BertTokenizer
import numpy as np
import nltk
import warnings 
from .src.BertClassifier import BertClassifier
from .src.utils import preprocess
nltk.download('punkt')

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

MODEL_PATH ="./v_osint_topic_sentiment/models/bert_best_model.pt"
BERT_NAME ="NlpHUST/vibert4news-base-cased"
MAX_SENT_LENGTH = 100
MAX_WORD_LENGTH = 100

LABEL_MAPPING = {
    0: "tieu_cuc",
    1: "trung_tinh",
    2: "tich_cuc"
                }

bert_tokenizer = BertTokenizer.from_pretrained(BERT_NAME)
bert_model = BertModel.from_pretrained(BERT_NAME)
sentiment_model = BertClassifier(bert_model,num_classes=3)
sentiment_model.load_state_dict(torch.load(MODEL_PATH,map_location=device))
sentiment_model.to(device)
sentiment_model.eval()

def topic_sentiment_classification(title="",description="",content=""):
    text = title+'\n'+description+'\n'+content
    with torch.no_grad():           
        sentences_ids ,sentences_mask, num_sent = preprocess(bert_tokenizer,text,MAX_WORD_LENGTH, MAX_SENT_LENGTH)
        logits = sentiment_model(sentences_ids,sentences_mask,num_sent)
        logits = logits.cpu().detach().numpy()[0]
    index_pred = np.argmax(logits, -1)
    label_pred = LABEL_MAPPING[index_pred]
    result = {}
    result['sentiment_label'] = label_pred
    return result

    