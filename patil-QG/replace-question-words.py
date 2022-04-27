import json
import regex as re

with open('CMORE/C-MORE_QuestionAnswerContext.json', 'r') as f:
    data = json.load(f)
with open('CMORE/C-MORE_QuestionsFromQuestionContext.json', 'r') as f:
    cardinal_question = set(json.load(f))

'''
question_word = {"CARDINAL": ["what"], "DATE": ["when", "what time", "what date"],
                 "EVENT": ["what event","what","which event"], "FAC": ["where","what buildings"],
                 "GPE": ["where", "what country"], "LANGUAGE": ["what language","which language"],
                 "LAW": ["which law","what law"], "LOC": ["where", "what location", "which place", "what place"],
                 "MONEY": ["how much money","how much"], "NORP": ["what", "what groups", "where"],
                 "ORDINAL": ["what rank","what"], "ORG": ["which organization","what organization", "what"],
                 "PERCENT": ["what percent", "what percentage"], "PERSON": ["who", "which person"],
                 "PRODUCT": ["what", "what product"], "QUANTITY": ["how many", "how much"],
                 "TIME": ["when", "what time"], "WORK_OF_ART": ["what", "what title"]}
'''
question_words = ["what", "when", "what time", "what date", "what event", "what", "which event",
            "where", "what buildings", "where", "what country", "what language",
            "which language", "which law", "what law", "where", "what location",
            "which place", "what place", "how much money","how much", "what", "what groups",
            "where", "what rank", "what", "which organization", "what organization", "what",
            "what percent", "what percentage", "who", "which person", "what", "what product",
            "how many", "how much", "when", "what time", "what", "what title"]

output = []

for idx in range(len(data)):
    question, answer = data[idx]["question"], data[idx]["answers"][0]
    found = False
    for q in question_words:
        if found:
            break
        for m in re.finditer(q, question):
            y = question[:m.start()] + answer + question[m.start() + len(q):]
            if y in cardinal_question:
                data[idx]["question"] = y
                data[idx]["question_word"] = q
                data[idx]["answer_start"] = [m.start()]
                found = True
                break
    if not found:
        print(f"not found: idx = {idx}, question = {question}, answer = {answer}")
    if len(data[idx]["question"].split()) > 5:
        output.append(data[idx]) 

with open('CMORE/C-MORE_replaced_QAC_clean.json', 'w') as f:
    json.dump(output, f)
