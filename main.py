from transformers import pipeline
from transformers import AutoTokenizer, AutoModel

# classifier = pipeline(task='sentiment-analysis')
# res = classifier('I have been waiting for Huggin Face course for my whole life.')
# print(res)
#
# question_answerer = pipeline(task='question-answering')
# respond = question_answerer(
#     question='How old is John?',
#     context='John was born in 2003.',
# )
# print(respond)


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
token = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
pretrained_model = AutoModel.from_pretrained(model_name)
output = pretrained_model(**token)

print(token)
