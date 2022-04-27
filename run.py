import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead, pipeline

moose_tokenizer = AutoTokenizer.from_pretrained("iarfmoose/t5-base-question-generator")
moose_model = AutoModelForSeq2SeqLM.from_pretrained("iarfmoose/t5-base-question-generator")
mrm_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
mrm_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def read_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)
  
def write_json(filename, data):
  with open(filename, 'w') as outfile:
    json.dump(data, outfile)

def qg_moose(answer, question):
  generator = pipeline(task="text2text-generation", model=moose_model, tokenizer=moose_tokenizer)
  return generator(f"<answer> {answer} <context> {question}")[0]['generated_text']

def qg_mrm(answer, question, max_length=256):
  input_text = "answer: %s  context: %s </s>" % (answer, question)
  features = mrm_tokenizer([input_text], return_tensors='pt')
  output = mrm_model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)
  raw_output = mrm_tokenizer.decode(output[0])
  tokens = raw_output.split('question: ')
  return tokens[-1][:-4]

qa = read_json('C-MORE_QuestionAnswer_clean.json')
result = []
for entry in qa:
    question = entry['question']
    answer = entry['answers']
    
    moose = qg_moose(answer, question)
    mrm = qg_mrm(answer, question)

    entry = {'question': question, 'answer': answer, 'moose': moose, 'mrm': mrm}
    result.append(entry)
write_json('output', result)