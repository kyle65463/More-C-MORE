import argparse
import json
from re import I

input_path = '/tmp2/C-MORE_replaced_QAC_clean_200k.json'
output_path = '/tmp2/b08902046/More-C-MORE/mrm_qa.json'

from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

def read_json(filename):
    with open(filename) as json_file:
        return json.load(json_file)
  
def write_json(filename, data):
  with open(filename, 'w') as outfile:
    json.dump(data, outfile)

def get_question(answer, context, max_length=256):
  input_text = "answer: %s  context: %s </s>" % (answer, context)
  features = tokenizer([input_text], return_tensors='pt')

  output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)
  q = tokenizer.decode(output[0])
  q = q.split(":")[1]
  q = q.split("</s>")[0]
  #print(q)
  return q


def main(): 
  qa = read_json(input_path)
  result = []
  for i, entry in enumerate(qa):
    if i % 10 == 0:
        print(i)
    id = entry["id"]    
    question = entry['question']
    answers = entry['answers']
    try:
      if len(answers) > 0:
        answer = answers[0]    
        mrm = get_question(answer, question)
        entry = {'id': id, 'question': mrm}
        result.append(entry)
    except:
      print(i, "is too long")
      continue
  write_json(output_path, result)

if __name__ == "__main__":
    main()
