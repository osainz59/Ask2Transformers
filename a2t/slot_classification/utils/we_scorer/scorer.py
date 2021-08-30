import os, sys
import json 
import argparse 
import re 
from copy import deepcopy
from collections import defaultdict 
from tqdm import tqdm
import spacy 
from pprint import pprint


from utils import load_ontology,find_arg_span, compute_f1, get_entity_span, find_head, WhitespaceTokenizer

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

'''
Scorer for argument extraction on ACE & KAIROS.
For the RAMS dataset, the official scorer is used. 

Outputs: 
Head F1 
Coref F1 
'''
def clean_span(ex, span):
    tokens = ex['tokens']
    if tokens[span[0]].lower() in {'the', 'an', 'a'}:
        if span[0]!=span[1]:
            return (span[0]+1, span[1])
    return span 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen-file',type=str,default='data/wikievents/dev.jsonl' )
    parser.add_argument('--test-file', type=str,default='data/wikievents/dev.jsonl')
    parser.add_argument('--coref-file', type=str)
    parser.add_argument('--head-only', action='store_true')
    parser.add_argument('--coref', action='store_true')
    parser.add_argument('--dataset',type=str, default='KAIROS', choices=['ACE', 'KAIROS','AIDA'])
    args = parser.parse_args() 
    
    coref_mapping = defaultdict(dict) # span to canonical entity_id mapping for each doc 
    if args.coref:
        if args.dataset == 'KAIROS' and args.coref_file:
            with open(args.coref_file, 'r') as f, open(args.test_file, 'r') as test_reader:
                for line, test_line  in zip(f, test_reader):
                    coref_ex = json.loads(line)
                    ex = json.loads(test_line)
                    doc_id = coref_ex['doc_key']
                    
                    for cluster, name in zip(coref_ex['clusters'], coref_ex['informative_mentions']):
                        canonical = cluster[0]
                        for ent_id in cluster:
                            ent_span = get_entity_span(ex, ent_id) 
                            ent_span = (ent_span[0], ent_span[1]-1) 
                            coref_mapping[doc_id][ent_span] = canonical
                    # this does not include singleton clusters 
        else:
            # for the ACE dataset 
            with open(args.test_file) as f:
                for line in f:
                    doc=json.loads(line.strip())
                    doc_id = doc['sent_id']
                    for entity in doc['entity_mentions']:
                        mention_id = entity['id']
                        ent_id = '-'.join(mention_id.split('-')[:-1]) 
                        coref_mapping[doc_id][(entity['start'], entity['end']-1)] = ent_id # all indexes are inclusive 

    

    pred_arg_num =0 
    gold_arg_num =0
    arg_idn_num =0 
    arg_class_num =0 

    arg_idn_coref_num =0
    arg_class_coref_num =0

    with open(args.gen_file, 'r') as pred_reader, open(args.test_file, 'r') as test_reader:
        for pred, gold in zip(pred_reader, test_reader):

            pred = json.loads(pred)
            gold = json.loads(gold)

            doc = None
            if args.head_only:
                doc = nlp(' '.join(pred['tokens']))

            for pred_event, gold_event in zip(pred['event_mentions'], gold['event_mentions']):
                assert pred_event['id'] == gold_event['id'], f"Predicted event and gold events must match. {pred_event['id']} - {gold_event['id']}"

                predicted_set = set()
                for arg in pred_event['arguments']:
                    # Get the entity span 
                    arg_span = get_entity_span(pred, arg['entity_id'])
                    arg_span = (arg_span[0], arg_span[1]-1)
                    arg_span = clean_span(pred, arg_span)
                    if args.head_only and arg_span[0]!=arg_span[1]:
                       arg_span = find_head(arg_span[0], arg_span[1], doc=doc)


                    predicted_set.add( (arg_span[0], arg_span[1], pred_event['event_type'], arg['role']) )
                        
                    
                # get gold spans         
                gold_set = set() 
                gold_canonical_set = set() # set of canonical mention ids, singleton mentions will not be here 
                for arg in gold_event['arguments']:
                    argname = arg['role']
                    entity_id = arg['entity_id']
                    span = get_entity_span(gold, entity_id)
                    span = (span[0], span[1]-1)
                    span = clean_span(gold, span)
                    # clean up span by removing `a` `the`
                    if args.head_only and span[0]!=span[1]:
                        span = find_head(span[0], span[1], doc=doc) 
                    
                    gold_set.add((span[0], span[1], gold_event['event_type'], argname))
                    if args.coref:
                        if span in coref_mapping[doc_id]:
                            canonical_id = coref_mapping[doc_id][span]
                            gold_canonical_set.add((canonical_id, gold_event['event_type'], argname))
                

                pred_arg_num += len(predicted_set)
                gold_arg_num += len(gold_set)
                # check matches 
                for pred_arg in predicted_set:
                    arg_start, arg_end, event_type, role = pred_arg
                    gold_idn = {item for item in gold_set
                                if item[0] == arg_start and item[1] == arg_end
                                and item[2] == event_type}
                    if gold_idn:
                        arg_idn_num += 1
                        gold_class = {item for item in gold_idn if item[-1] == role}
                        if gold_class:
                            arg_class_num += 1
                    elif args.coref:# check coref matches 
                        arg_start, arg_end, event_type, role = pred_arg
                        span = (arg_start, arg_end)
                        if span in coref_mapping[doc_id]:
                            canonical_id = coref_mapping[doc_id][span]
                            gold_idn_coref = {item for item in gold_canonical_set 
                                if item[0] == canonical_id and item[1] == event_type}
                            if gold_idn_coref:
                                arg_idn_coref_num +=1 
                                gold_class_coref = {item for item in gold_idn_coref
                                if item[2] == role}
                                if gold_class_coref:
                                    arg_class_coref_num +=1 
            

        
    if args.head_only:
        print('Evaluation by matching head words only....')
    
    
    role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num)
    role_prec, role_rec, role_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_class_num)

    
    print('Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
    print('Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
        role_prec * 100.0, role_rec * 100.0, role_f * 100.0))

    if args.coref:
        role_id_prec, role_id_rec, role_id_f = compute_f1(
        pred_arg_num, gold_arg_num, arg_idn_num + arg_idn_coref_num)
        role_prec, role_rec, role_f = compute_f1(
            pred_arg_num, gold_arg_num, arg_class_num + arg_class_coref_num)

        
        print('Coref Role identification: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_id_prec * 100.0, role_id_rec * 100.0, role_id_f * 100.0))
        print('Coref Role: P: {:.2f}, R: {:.2f}, F: {:.2f}'.format(
            role_prec * 100.0, role_rec * 100.0, role_f * 100.0))



                    




