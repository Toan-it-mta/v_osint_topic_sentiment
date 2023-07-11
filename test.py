from v_osint_topic_sentiment.sentiment_analysis import topic_sentiment_classification

face_text = """ĐTQG nữ Việt Nam lên đường sang New Zealand tham dự World Cup 2023.
Chúc may mắn các cô gái Việt Nam"""

predict_out = topic_sentiment_classification(title="",description="",content=face_text)

print(predict_out)

