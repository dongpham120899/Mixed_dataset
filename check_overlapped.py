import os
import numpy as np
import json
import spacy
from difflib import SequenceMatcher
from visualize_html import Visualize_HTML
from convert_json2brat import ConvertJson2Brat
import shutil
nlp = spacy.load("en_core_sci_sm")

visualizator = ConvertJson2Brat()


def check_similarity(sentence1, sentence2):
    doc_1 = nlp(sentence1)
    doc_2 = nlp(sentence2)

    return doc_1.similarity(doc_2)

def check_sequence_matcher(sentence1, sentence2):
    return SequenceMatcher(None, sentence1, sentence2).ratio()


def load_json(path):
    with open(path, "r") as file:
        data = json.load(file)

    return data

def entity_overlapped(entity_1, entity_2):
    range_1 = np.arange(entity_1['start'],entity_1['end'])
    range_2 = np.arange(entity_2['start'],entity_2['end'])

    return all(np.isin(range_1,range_2))

def relation_overlapped(relation_1, relation_2, entities_1, entities_2):
    ent_1_h = entities_1[relation_1['head']]
    ent_1_t = entities_1[relation_1['tail']]
    rel_1_type = relation_1['type']

    ent_2_h = entities_2[relation_2['head']]
    ent_2_t = entities_2[relation_2['tail']]
    rel_2_type = relation_2['type']

    # check overlapped relations
    conflicing = False
    if entity_overlapped(ent_1_h, ent_2_h) and entity_overlapped(ent_1_t, ent_2_t):
        # print(ent_1_h, ent_1_t)
        # print(ent_2_h, ent_2_t)
        # print(rel_1_type)
        # print(rel_2_type)
        # print("************")

        # check conflicing
        if rel_1_type!=rel_2_type:
            conflicing = True

    return conflicing

        
    # pass


def check_overlapped_relation_sample(sample_1, sample_2):
    relations_1 = sample_1['relations']
    relations_2 = sample_2['relations']
    entities_1 = sample_1['entities']
    entities_2 = sample_2['entities']

    conflicing = False
    for rel_1 in relations_1:
        for rel_2 in relations_2:
            conflicing = relation_overlapped(rel_1, rel_2, entities_1, entities_2)
            if conflicing is True:
                break
            # break
        # break

    return conflicing

def check_overlapped_entity_sample(sample_1, sample_2):
    # print(sample_1['eni'])
    entities_1 = sample_1['entities']
    entities_2 = sample_2['entities']

    print(entities_1)
    
    for idx_1, entity_1 in enumerate(entities_1):
        for idx_2, entity_2 in enumerate(entities_2):
            if  entity_overlapped(entity_1, entity_2):
                if entity_1['start']==entity_2['start']:
                    print("True")
                else:
                    print(entity_1, entity_2)
                    print("False")
            # if entity_1['start']==entity_2['start']:
                # print("*****")

def processing_text(text):
    text = text.replace("-LRB-", "(")
    text = text.replace("-RRB-", ")")
    text = text.replace("-LSB-", "[")
    text = text.replace("-RSB-", "]")
    # text = text.replace("``", "\"")
    return text

def check_overlaped(path_1, path_2, saved_path):
    data_1 = load_json(path_1)
    data_2 = load_json(path_2)

    count_overlapping = 0

    for sample_1 in data_1:
        for sample_2 in data_2:
            abs_1 = " ".join(sample_1["tokens"])

            abs_2 = " ".join(sample_2["tokens"])

            sim_core = check_sequence_matcher(processing_text(abs_1), processing_text(abs_2))
            sim_core = check_sequence_matcher(abs_1, abs_2)
            if sim_core > 0.6:
                # check_overlapped_entity_sample(sample_1, sample_2)
                # if sim_core < 0.99:
                #     prnt()
                conflicing = check_overlapped_relation_sample(sample_1, sample_2)
                # if sim_core < 0.99:
                if True:
                    count_overlapping += 1
                    sub_name = "SemEval-"+sample_1['orig_id'] + "_and_" + "SciERC-"+sample_2['orig_id'] + "score={}".format(sim_core)
                    print(sub_name)
                    print("*"*100)
                    sub_path = os.path.join(saved_path, sub_name)
                    if os.path.exists(sub_path) is False:
                        os.mkdir(sub_path)

                    # sample 1
                    # print(sample_1)
                    convertor.convert_each_sample(sample_1, sub_path)
                    convertor.convert_each_sample(sample_2, sub_path)
                # break
            # break
                # visualizator.visualize_sentence(sentence=sample_1['tokens'],
                #                                 entities=sample_1['entities'],
                #                                 relations=sample_1['relations'],
                #                                 saved_name=os.path.join(sub_path,sample_1['orig_id'])
                #                                 )

                # visualizator.visualize_sentence(sentence=sample_2['tokens'],
                #                                 entities=sample_2['entities'],
                                                # relations=sample_2['relations'],
                                                # saved_name=os.path.join(sub_path,sample_2['orig_id'])
                                                # )
        # break

    print("number of Overlapped sample:", count_overlapping)



if __name__ == "__main__":
    semeval_path = "train/semeval-2018_train.json"
    scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/train_data_1000_sentence.json"    
    # saved_path = ""
    saved_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/CHECK_overlapped_train_and_train_0.9"
    # saved_path = "/Users/phamdong/Documents/NII_internship/repo/brat-1.3p1/data/examples/overlapped_train_and_train"


    if os.path.exists(saved_path) is False:
        os.mkdir(saved_path)
    
    convertor = ConvertJson2Brat()
    check_overlaped(semeval_path, scierc_path, saved_path)

    # bart_path = "/Users/phamdong/Documents/NII_internship/repo/brat-1.3p1/data/examples/overlapped_train_and_test"
    # shutil.copy(saved_path, bart_path)

