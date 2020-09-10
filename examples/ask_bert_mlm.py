""" Script para hacer classification de temas preguntandole directamente a Roberta.

Método de uso:
    (Mediante un fichero como input) python3 get_topics.py topics.txt input_file.txt
    (Mediante el input standard)     python3 get_topics.py topics.txt 
                                     cat input_file.txt | python3 get_topics.py topics.txt

El topics.txt debe contener los diferentes topics que se van a usar y deberan estar separados por saltos de linea.
"""
import sys
from collections import defaultdict
from pprint import pprint

import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

if len(sys.argv) < 2:
    print('Usage:\tpython3 get_topics.py topics.txt input_file.txt\n\tpython3 get_topics.py topics.txt < input_file.txt')
    exit(1)

model = AutoModelWithLMHead.from_pretrained('bert-large-uncased-whole-word-masking')
tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')

model.eval()
"""
def generate(sentence, model=model, tokenizer=tokenizer): 
    ...:     sentence = torch.tensor([tokenizer.encode(sentence)]) 
    ...:     row, col = torch.where(sentence == tokenizer.mask_token_id) 
    ...:      
    ...:     output = model(sentence)[0] 
    ...:     topwords = output[row[0], col[0]].topk(5) 
    ...:     topwords = topwords.indices.squeeze().tolist() 
    ...:      
    ...:     return tokenizer.decode(topwords)
"""

def get_topic(context, topics, tokenizer=tokenizer, model=model):
    with torch.no_grad():
        sentences = [f"{context} {tokenizer.sep_token} Topic or domain about \"{topic}\"." for topic in topics]
        input_ids = tokenizer.batch_encode_plus(sentences, pad_to_max_length=True)
        outputs = model(torch.tensor(input_ids['input_ids']))[0]
        output_probs = torch.softmax(model(torch.tensor(input_ids['input_ids']))[0][:,0], 0).numpy()
        output = sorted(list(zip(output_probs, topics)), reverse=True)
    return output

with open(sys.argv[1], 'rt') as f:
    topics = [topic.rstrip() for topic in f]
    topic_by_tokens = defaultdict(dict)
    for topic in topics:
        tokens = tokenizer.encode(topic, add_special_tokens=False)
        topic_by_tokens[len(tokens)][topic] = tokens

input_stream = open(sys.argv[2], 'rt') if len(sys.argv) == 3 else sys.stdin

for line in input_stream:
    line = line.rstrip()
    topic_dist = get_topic(line, topics)
    print(line)
    pprint(topic_dist)
    print()