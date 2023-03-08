import os
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import re
import spacy
import pandas as pd
from tqdm import tqdm
from spacy.tokens import Doc
import json

nlp = spacy.load("en_core_sci_sm")
# def custom_tokenizer(text):
#     tokens = text.split(" ")
#     try:
#         doc = Doc(nlp.vocab, tokens)
#     except:
#         print("ERROR: ",tokens)
#     return doc

# nlp.tokenizer = custom_tokenizer


RELATION_MAPPING = {"COMPARE": "Compare", "USAGE":"Used-for", "PART_WHOLE":"Part-of", "MODEL-FEATURE":"Feature-of","RESULT":"Evaluate-for", "TOPIC":"Topic"}

def load_relation(path):
    with open(path, "r") as f:
        data = f.readlines()

    # print(data)
    relation_df = pd.DataFrame(columns=["doc_id", "entity_1", "entity_2", "relation_type", "reverse"])
    for idx, rel in enumerate(data):
        rel = rel.replace("\n","")
        matched = re.search(r"\(.*\)", rel)
        if len(matched[0][1:-1].split(","))==3:
            id_1, id_2, _ = matched[0][1:-1].split(",")
            reverse = True
        else:
            id_1, id_2 = matched[0][1:-1].split(",")
            reverse = False
        doc_id = id_1.split(".")[0]
        rel_type = rel.replace(matched[0], "")

        # print(doc_id, id_1, id_2, rel_type, reverse)

        relation_df.loc[idx] = [doc_id, id_1, id_2, rel_type, reverse]

        # break
    return relation_df

def matching_entity(tokens, entities, id_sample):
    matched_tokens = []
    entity_list = []
    start_p_list = []
    end_p_list = []

    df = pd.DataFrame(columns=['text', 'entity_type', 'start', 'end'])
    entity_dict = {}
    count = 0
    count_entity = 0
    for idx, token in enumerate(tokens):
        if token[:4]=="ENT-":
            # try:
            entity_text = entities[id_sample+"."+str(token[4:])]
            # except:
            #     print(tokens)
            matched_tokens.append(entity_text)
            entity_list.append(id_sample+"."+str(token[4:]))
            start_p_list.append(count)
            end_p_list.append(count+len(entity_text.split()))

            entity_dict[id_sample+"."+str(token[4:])] = [entity_text, count, count+len(entity_text.split()), count_entity]
            count_entity += 1
            count = count + len(entity_text.split())
        else:
            matched_tokens.append(token)
            entity_list.append("NO-ENTITY")
            start_p_list.append(count)
            end_p_list.append(count+len(token.split()))
            # print(count, token)

            count = count + len(token.split())




    # print(matched_toke
    # print(matched_tokens, entity_list, start_p_list, end_p_list)
    df['text'] = matched_tokens
    df['entity_type'] = entity_list
    df['start'] = start_p_list
    df['end'] = end_p_list

    return df, entity_dict


def find_entity(text):
    text = text.replace("<abstract>", "")
    text = text.replace("</abstract>", "")
    cmd = r"(?<=<entity id=)(.*?)(?=</entity>)"
    match = re.findall(cmd, text)
    for i, en in enumerate(match):
        entity_match = "<entity id=" + en + "</entity>"
        # print(entity_match)

        entity_id = re.findall(r"\"(.*?)\"", en)[0].split(".")[1]
        text = text.replace(entity_match, "ENT-{}".format(entity_id))

    return text.strip()
        

def convert_xml_to_json(path, relation_df, save_path=None):
    with open(path, 'r') as f:
        data = f.read()

    Bs_data = BeautifulSoup(data, "xml")
    texts = Bs_data.find_all("text")
    documents = []
    for sample in tqdm(texts):
        doc_id = sample.get('id')
        # print(doc_id)
        doc_rel_df = relation_df[relation_df["doc_id"]==doc_id]
        abstract = sample.find('abstract')
        abstract_text = abstract.text.strip()
        entities = sample.find_all("entity")

        entities_dic = {}
        for entity in entities:
            id_entity = entity.get("id")
            text_entity = entity.text
            entities_dic[id_entity] = text_entity

        unseen_abstract = find_entity(str(abstract))
        unseen_tokens = nlp(unseen_abstract)
        original_tokens = nlp(abstract_text)
        unseen_tokens = [token.text for token in unseen_tokens]
        original_tokens = [token.text for token in original_tokens]

        # print(unseen_abstract)
        # print(original_tokens)

        sample_entity_df, sample_entity_dic = matching_entity(unseen_tokens, entities_dic, doc_id)

        sample_dic = {}
        sample_dic["tokens"] = original_tokens
        entity_list = []
        for key, value in sample_entity_dic.items():
            entity_list.append(dict(type="OtherScientificTerm", start=value[1], end=value[2]))

        relation_list = []
        for i in range(len(doc_rel_df)):
            entity_id_1 = doc_rel_df.iloc[i]['entity_1']
            entity_id_2 = doc_rel_df.iloc[i]['entity_2']
            rel_type = doc_rel_df.iloc[i]['relation_type']
            reverse = doc_rel_df.iloc[i]['reverse']

            if reverse:
                start = sample_entity_dic[entity_id_2][3]
                end = sample_entity_dic[entity_id_1][3]
            else:
                # print(sample_entity_dic)
                start = sample_entity_dic[entity_id_1][3]
                end = sample_entity_dic[entity_id_2][3]

            relation_list.append(dict(type=RELATION_MAPPING[rel_type], head=start, tail=end))

        sample_dic["entities"] = entity_list
        sample_dic["relations"] = relation_list
        sample_dic['orig_id'] = doc_id

        # print(sample_dic)
        documents.append(sample_dic)
        # break
    # print(root)

    if save_path:
        with open(save_path, "w") as file:
            json.dump(documents, file)
        
            




if __name__ == "__main__":
    xml_path = "train/training_1.1.text.xml"
    relation_path = "train/training_1.1.relations.txt"
    save_path = "train/semeval-2018_train.json"
    relation_df = load_relation(relation_path)
    convert_xml_to_json(xml_path, relation_df, save_path)