from fastapi import FastAPI
from pydantic import BaseModel
from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import logging
import json

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(filename='./v_osint_sentiment.log',
                    level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                    )
class Item(BaseModel):
    title: str
    description: str
    content: str

def on_success(data=None, message="success"):
    if data is not None:
        return {
            "message": message,
            "result": data
        }
    return {
        "message": message,
    }


def on_fail(message="fail"):
    return {
        "message": message,
    }

@app.post("/topic_sentiment_analysis")
async def topic_sentiment_analysis(obj_input: Item):
    try:
        title = obj_input.title
        description = obj_input.description
        content = obj_input.content
        input_fnc = title+'\n'+description+'\n'+content
        result = topic_sentiment_classification(input_fnc)
        str_log = obj_input.__dict__
        str_log["label"] = result
        str_log = json.dumps(str_log)
        logging.info(str_log)
        return on_success(result['sentiment_label'])
    except Exception as err:
        return on_fail(err)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
