import json
import argparse
from pipelines import pipeline

def main(args):
    with open(args.file, 'r') as f:
        data = json.load(f)
    
    nlp = pipeline('question-generation')
    l = []
    for idx, d in enumerate(data):
        if idx % 100 == 0:
            print(idx)
        try:
            res = nlp(f"{d['answers'][0]} [SEP] {d['question']}")
            l.append({'id': d['id'], 'question': res[0]['question']})
        except:
            break
    
    with open('result.json', 'w') as f:
        json.dump(l, f)

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_argument()
    main(args)
