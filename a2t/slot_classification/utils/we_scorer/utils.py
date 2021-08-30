import json 
import spacy 
from spacy.tokens import Doc
# PRONOUN_FILE='pronoun_list.txt'
# pronoun_set = set() 
# with open(PRONOUN_FILE, 'r') as f:
#     for line in f:
#         pronoun_set.add(line.strip())
    

# def check_pronoun(text):
#     if text.lower() in pronoun_set:
#         return True 
#     else:
#         return False 
    

def clean_mention(text):
    '''
    Clean up a mention by removing 'a', 'an', 'the' prefixes.
    '''
    prefixes = ['the ', 'The ', 'an ', 'An ', 'a ', 'A ']
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text 


def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1

class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

def find_head(arg_start, arg_end, doc):
    cur_i = arg_start
    while doc[cur_i].head.i >= arg_start and doc[cur_i].head.i <=arg_end:
        if doc[cur_i].head.i == cur_i:
            # self is the head 
            break 
        else:
            cur_i = doc[cur_i].head.i
        
    arg_head = cur_i
    
    return (arg_head, arg_head)


def load_ontology(dataset, ontology_file=None):
    '''
    Read ontology file for event to argument mapping.
    ''' 
    ontology_dict ={} 
    if not ontology_file: # use the default file path 
        if not dataset:
            raise ValueError
        with open('event_role_{}.json'.format(dataset),'r') as f:
            ontology_dict = json.load(f)
    else:
        with open(ontology_file,'r') as f:
            ontology_dict = json.load(f)

    for evt_name, evt_dict in ontology_dict.items():
        for i, argname in enumerate(evt_dict['roles']):
            evt_dict['arg{}'.format(i+1)] = argname
            # argname -> role is not a one-to-one mapping 
            if argname in evt_dict:
                evt_dict[argname].append('arg{}'.format(i+1))
            else:
                evt_dict[argname] = ['arg{}'.format(i+1)]
    
    return ontology_dict

def find_arg_span(arg, context_words, trigger_start, trigger_end, head_only=False, doc=None):
    match = None 
    arg_len = len(arg)
    min_dis = len(context_words) # minimum distance to trigger 
    for i, w in enumerate(context_words):
        if context_words[i:i+arg_len] == arg:
            if i < trigger_start:
                dis = abs(trigger_start-i-arg_len)
            else:
                dis = abs(i-trigger_end)
            if dis< min_dis:
                match = (i, i+arg_len-1)
                min_dis = dis 
    
    if match and head_only:
        assert(doc!=None)
        match = find_head(match[0], match[1], doc)
    return match 

def get_entity_span(ex, entity_id):
    for ent in ex['entity_mentions']:
        if ent['id'] == entity_id:
            return (ent['start'], ent['end'])