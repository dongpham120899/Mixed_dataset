import os
import json
from difflib import SequenceMatcher
from visualize_html import Visualize_HTML
from convert_json2brat import ConvertJson2Brat
from analysis_json import AnalysisData
import numpy as np

def saving_data(data, saved_path):
    with open(saved_path, "w") as file:
        json.dump(data, file)


class CreateMixedDataset():
    def __init__(self):
        self.visualizator = ConvertJson2Brat()

    def _create_mixed_(self, sem_path, sci_path, visualized_path, saved_path):
        sem_data = self.load_json(sem_path)
        sci_data = self.load_json(sci_path)

        count_overlapping = 0

        new_data = []
        overlapped_sci_list = []
        overlapped_sem_list = []
        overlapped_sci_data = []
        overlapped_sem_data = []
        half_set = []
        for sample_1 in sci_data:
            for sample_2 in sem_data:
                abs_1 = " ".join(sample_1["tokens"])
                abs_2 = " ".join(sample_2["tokens"])

                sub_name = "SemEval-"+sample_2['orig_id'] + "_and_" + "SciERC-"+sample_1['orig_id']
                # if sub_name!="SemEval-300_0_and_SciERC-P84-1078":
                #     continue
                # calculate similarity score (matcher char)
                sim_core = self.check_sequence_matcher(abs_2, abs_1)

                if sim_core > 0.6: # if two sample is overlapped
                    count_overlapping += 1

                    mixed_entities = self.check_overlapped_entity_sample(sample_1, sample_2) # mixed enities 
                    mixed_relations = self.get_mixed_relations(sample_1, sample_2, mixed_entities)
                    # standardize entity
                    mixed_entities = [dict(type=entity['type'], start=entity['start'], end=entity['end'])for entity in mixed_entities]
                    new_sample = dict(tokens=sample_2['tokens'], entities=mixed_entities, relations=mixed_relations, orig_id=sub_name)
                    new_data.append(new_sample)

                    # ****************************************************************
                    # visualize with brat
                    sub_path = os.path.join(visualized_path, sub_name)
                    if os.path.exists(sub_path) is False:
                        os.mkdir(sub_path)
                    self.visualizator.convert_each_sample(sample_1, sub_path)
                    self.visualizator.convert_each_sample(sample_2, sub_path)
                    self.visualizator.convert_each_sample(new_sample, sub_path)
                    # ****************************************************************
                    # print("*"*100)

                    overlapped_sci_list.append(sample_1['orig_id'])
                    overlapped_sem_list.append(sample_2['orig_id'])
                    overlapped_sci_data.append(sample_1)
                    overlapped_sem_data.append(sample_2)
                    if count_overlapping%2==0:
                        half_set.append(sample_1)
                    else: 
                        half_set.append(sample_2)
                    break


        un_overlapped_sci = []
        for sample_1 in sci_data:
            if sample_1['orig_id'] not in overlapped_sci_list:
                un_overlapped_sci.append(sample_1)

        # un_overlapped_sem = []
        # for sample_2 in sem_data:
        #     if sample_2['orig_id'] not in overlapped_sem_list:
        #         un_overlapped_sem.append(sample_2)

            # break
        # saving new data
        if saved_path:
            with open(saved_path, "w") as file:
                json.dump(new_data, file)

        print("number of Overlapped sample:", count_overlapping)

        return new_data, un_overlapped_sci, overlapped_sci_data, overlapped_sem_data, half_set
        
    def load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data
    
    def check_sequence_matcher(self, sentence1, sentence2):
        return SequenceMatcher(None, sentence1, sentence2).ratio()
    
    def entity_overlapped(self, entity_1, entity_2):
        range_1 = np.arange(entity_1['start'],entity_1['end'])
        range_2 = np.arange(entity_2['start'],entity_2['end'])

        if entity_1['start'] > entity_2['start']:
            start = entity_2['start']
        else:
            start = entity_1['start']

        if entity_1['end'] > entity_2['end']:
            end = entity_1['end']
        else:
            end = entity_2['end']

        return all(np.isin(range_1,range_2)), start, end

    def relation_overlapped(self, relation_1, relation_2, entities_1, entities_2):
        ent_1_h = entities_1[relation_1['head']]
        ent_1_t = entities_1[relation_1['tail']]
        rel_1_type = relation_1['type']

        ent_2_h = entities_2[relation_2['head']]
        ent_2_t = entities_2[relation_2['tail']]
        rel_2_type = relation_2['type']

        # check overlapped relations
        conflicing = False
        if self.entity_overlapped(ent_1_h, ent_2_h) and self.entity_overlapped(ent_1_t, ent_2_t):
            print(ent_1_h, ent_1_t)
            print(ent_2_h, ent_2_t)
            print(rel_1_type)
            print(rel_2_type)
            print("************")

            # check conflicing
            if rel_1_type!=rel_2_type:
                conflicing = True

        return conflicing
    
    def check_overlapped_relation_sample(self, sample_1, sample_2):
        relations_1 = sample_1['relations']
        relations_2 = sample_2['relations']
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']

        conflicing = False
        for rel_1 in relations_1:
            for rel_2 in relations_2:
                conflicing = self.relation_overlapped(rel_1, rel_2, entities_1, entities_2)
                if conflicing is True:
                    break
                # break
            # break

        return conflicing
    
    def mapping_relation_with_new_entities(self, relations, mixed_entities, prefix_rel):
        mapped_relations = []
        for rel_1 in relations:
            for idx, entity in enumerate(mixed_entities):
                if entity[prefix_rel]==rel_1['head']:
                    en_head = entity
                    idx_en_head = idx
                if entity[prefix_rel]==rel_1['tail']:
                    en_tail = entity
                    idx_en_tail = idx
            try:
                new_rel = dict(type=prefix_rel+"_"+rel_1['type'], head=idx_en_head, tail=idx_en_tail)
            except:
                print(prefix_rel)
                print("relations", relations)
                print(rel_1['head'])
                print("mixed_entities", mixed_entities)
            mapped_relations.append(new_rel)
        
        return mapped_relations

    def get_mixed_relations(self, sample_1, sample_2, mixed_entities):
        relations_1 = sample_1['relations']
        relations_2 = sample_2['relations']
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']

        # Sci Relations
        new_relations_1 = self.mapping_relation_with_new_entities(relations=relations_1, mixed_entities=mixed_entities, prefix_rel="sci")
        # SemEval Relations
        new_relations_2 = self.mapping_relation_with_new_entities(relations=relations_2, mixed_entities=mixed_entities, prefix_rel="sem")

        # Merge
        new_relations = new_relations_1 + new_relations_2

        return new_relations

    def mix_entities(self, entities_1, entities_2, entities_3):
        mixed_entities = entities_1 + entities_2 + entities_3

        return sorted(mixed_entities, key=lambda d: d['start'])

    def check_overlapped_entity_sample(self, sample_1, sample_2):
        entities_1 = sample_1['entities']
        entities_2 = sample_2['entities']
        tokens_1 = sample_1['tokens']
        tokens_2 = sample_2['tokens']

        # # print(entities_1)
        # print("entities_1", len(entities_1))
        # print("entities_2", len(entities_2))
        
        overlapped = []
        un_overlapped_SemEval = []
        un_overlapped_Sci = []
        for idx_1, entity_1 in enumerate(entities_1):
            for idx_2, entity_2 in enumerate(entities_2):
                is_entity_overlapped, start_e, end_e = self.entity_overlapped(entity_1, entity_2)
                if is_entity_overlapped:
                    # if entity_1['start']==entity_2['start'] and entity_1['end']==entity_2['end']:
                    #     print("Boundary Matcher", idx_1, tokens_1[entity_1['start']:entity_1['end']], idx_2, tokens_2[entity_2['start']:entity_2['end']])
                    # else:
                    #     print("Boundary Overlapped", idx_1, tokens_1[entity_1['start']:entity_1['end']], idx_2, tokens_2[entity_2['start']:entity_2['end']])


                    new_enity = dict(type=entity_1['type'], sci=idx_1, sem=idx_2, start=start_e, end=end_e)
                    overlapped.append(new_enity)

                # else:
                #     print("Un-overlapped", entity_1, entity_2)

        for i, entity_2 in enumerate(entities_2):
            if i not in [i['sem'] for i in overlapped]:
                new_enity = dict(type=entity_2['type'] + "_2", sci=None, sem=i, start=entity_2['start'], end=entity_2['end'])
                un_overlapped_SemEval.append(new_enity)

        for i, entity_1 in enumerate(entities_1):
            if i not in [i['sci'] for i in overlapped]:
                new_enity = dict(type=entity_1['type'] , sci=i, sem=None, start=entity_1['start'], end=entity_1['end'])
                un_overlapped_Sci.append(new_enity)


        # print("list of overlapped in SemEval", overlapped)
        # print("*"*20)
        # print("list of un-overlapped in SemEval", un_overlapped_SemEval)
        # print("*"*20)
        # print("list of un-overlapped in Sci", un_overlapped_Sci)

        entire_entities = self.mix_entities(overlapped, un_overlapped_SemEval, un_overlapped_Sci)

        return entire_entities
    


if __name__ == "__main__":
    semeval_path = "train/semeval-2018_train.json"
    train_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/train_data_1000_sentence.json"
    dev_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/dev_data_1000_sentence.json"
    test_scierc_path = "/Users/dongpham/Documents/NII_internship/datasets/SciERC_dataset/spert_dataset/test_data_1000_sentence.json"
    # saved_path = "overlapped_train_and_test" 
    visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/overlapped_train_and_train"

    if os.path.exists(visualized_path) is False:
        os.mkdir(visualized_path)

    creator = CreateMixedDataset()
    ov_train_train_data, train_un_overlapped_sci, train_overlapped_sci_data, train_overlapped_sem_data, train_half_set = creator._create_mixed_(sem_path=semeval_path, sci_path=train_scierc_path, visualized_path=visualized_path, saved_path=None)
    ov_train_dev_data, dev_un_overlapped_sci, dev_overlapped_sci_data, dev_overlapped_sem_data, dev_half_set = creator._create_mixed_(sem_path=semeval_path, sci_path=dev_scierc_path, visualized_path=visualized_path, saved_path=None)
    ov_train_test_data, test_un_overlapped_sci, test_overlapped_sci_data, test_overlapped_sem_data, test_half_set = creator._create_mixed_(sem_path=semeval_path, sci_path=test_scierc_path, visualized_path=visualized_path, saved_path=None)


    ov_full_data = ov_train_train_data + ov_train_dev_data + ov_train_test_data
    saving_data(ov_full_data, "experimental_ovp_dataset/mixed_set/mixed_set.json")

    half_set = train_half_set + dev_half_set + test_half_set
    saving_data(half_set, "experimental_ovp_dataset/half_sem_half_sci/half_set.json")

    test_sci_ovp = train_un_overlapped_sci + dev_un_overlapped_sci + test_un_overlapped_sci
    saving_data(test_sci_ovp, "experimental_ovp_dataset/sci_set/test_sci.json")

    train_sci_ovp = train_overlapped_sci_data + dev_overlapped_sci_data + test_overlapped_sci_data
    saving_data(train_sci_ovp, "experimental_ovp_dataset/sci_set/train_sci.json")

    train_sem_ovp = train_overlapped_sem_data + dev_overlapped_sem_data + test_overlapped_sem_data
    saving_data(train_sem_ovp, "experimental_ovp_dataset/sem_set/train_sem.json")


    train_sem_data = creator.load_json("train/semeval-2018_train.json")
    train_ovp_sem_list = [a['orig_id'] for a in train_sem_ovp]
    the_others_train_sem_data = []
    for sample in train_sem_data:
        if sample['orig_id'] not in train_ovp_sem_list:
            the_others_train_sem_data.append(sample)
    test_sem_data = creator.load_json("test/semeval-2018_test.json")
    test_sem_ovp = the_others_train_sem_data + test_sem_data
    saving_data(test_sem_ovp, "experimental_ovp_dataset/sem_set/test_sem.json")

    print("mixed set", len(ov_full_data))
    print("half set", len(half_set))
    print("train sci set", len(train_sci_ovp))
    print("test sci set", len(test_sci_ovp))
    print("train sem set", len(train_sem_ovp))
    print("test sem set", len(test_sem_ovp))






    # print("Overlapped", len(ov_full_data))
    # print("Sem", len(overlapped_sem))
    # print("Sci", len(train_un_overlapped_sci+dev_un_overlapped_sci+test_un_overlapped_sci))

