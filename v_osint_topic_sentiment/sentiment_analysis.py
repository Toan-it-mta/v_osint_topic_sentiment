import torch
from transformers import BertModel, BertTokenizer
from src.BertClassifier import BertClassifier
import numpy as np
import nltk
import warnings 
from src.utils import preprocess
nltk.download('punkt')

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

MODEL_PATH ="./models/bert_best_model.pt"
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
sentiment_model.eval()

def topic_sentiment_classification(text:str):
    with torch.no_grad():           
        sentences_ids ,sentences_mask, num_sent = preprocess(bert_tokenizer,text,MAX_WORD_LENGTH, MAX_SENT_LENGTH)
        logits = sentiment_model(sentences_ids,sentences_mask,num_sent)
        logits = logits.cpu().detach().numpy()[0]
    index_pred = np.argmax(logits, -1)
    label_pred = LABEL_MAPPING[index_pred]
    result = {}
    result['sentiment_label'] = label_pred
    # result['topic_label'] = "unknow"
    return result

if __name__ == "__main__":
    text = """Người đứng đầu Bộ Quốc phòng tuyên bố rằng một thoả thuận hợp tác về mua sắm quốc phòng sẽ được ký với Bộ trưởng Quốc phòng Hoa Kỳ Lloyd Austin trong cuộc họp của họ vào thứ Sáu."""
    predict_out = topic_sentiment_classification(text)
    print(predict_out)

