import json
from pipelines import pipeline

with open('CMORE/C-MORE_QuestionAnswer-10000-2.json', 'r') as f:
    data = json.load(f)

nlp = pipeline('question-generation')
l = []
for idx, d in enumerate(data):
    if idx % 100 == 0:
        print(idx)
    try:
        res = nlp(f"{d['answers'][0]} [SEP] {d['question']}")
        l.append(res[0]['question'])
    except:
        break

QG = "\n".join(l)
with open('QG.txt', 'w') as f:
    f.write(QG)