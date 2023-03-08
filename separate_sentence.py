import os
import numpy
import json
import spacy
nlp = spacy.load("en_core_sci_sm")

from convert_json2brat import ConvertJson2Brat


class SeparateSentence():
    def __init__(self, num_sentence, padding, visualized_path) -> None:
        self.num_sents = num_sentence
        self.padding = padding
        self.visualize = ConvertJson2Brat()
        if os.path.exists(visualized_path) is False:
            os.mkdir(visualized_path)
        self.visualized_path = visualized_path
        

    def _run_separte_(self, json_path, saved_path):
        data = self.load_json(json_path)

        new_data = []
        for sample in data:
            abt = " ".join(sample['tokens'])
            entities = sample['entities']
            relations = sample['relations']
            orig_id = sample['orig_id']
            doc = nlp(self.processing_text(abt))
            new_sentences = []
            new_entities = []
            new_relations = []
            for sent in doc.sents:
                sentence_text = [token.text for token in sent]
                sent_entities, idx_sent_entities = self._transfer_entity(sent=sent, entities=entities)
                sent_relations = self._trasfer_relation(sent=sent, relations=relations, idx_sent_entities=idx_sent_entities)
                # print("*"*100)
                new_sentences.append(sentence_text)
                new_entities.append(sent_entities)
                new_relations.append(sent_relations)

            # if self.num_sents==1:
            #     merge_data = []
            #     for idx in range(len(new_sentences)):
            #         new_sample = dict(tokens=new_sentences[idx], entities=new_entities[idx], relations=new_relations[idx], orig_id=str(idx)+"_"+orig_id)
            #         self.visualize.convert_each_sample(new_sample, self.visualized_path)
            #         merge_data.append(merge_data)
            # elif self.num_sents > 1:
            merge_data = self.merge_sentence(new_sentences, new_entities, new_relations, self.padding, orig_id)

            new_data.extend(merge_data)
            # break

        
        if saved_path:
            with open(saved_path, "w") as file:
                json.dump(new_data, file)

        print("Number of data:", len(new_data))

    def processing_text(self, text):
        text = text.replace("-LRB-", "(")
        text = text.replace("-RRB-", ")")
        text = text.replace("-LSB-", "[")
        text = text.replace("-RSB-", "]")

        return text
    
    def merge_sentence(self, sentence_list, entities_list, relations_list, padding=1, orig_id=None): # sliding window 
        assert padding < self.num_sents, "Padding should be smaller than num of sentences"
        # adding padding
        for idx in range(padding):
            sentence_list.insert(0, [])
            entities_list.insert(0, [])
            relations_list.insert(0, [])
            sentence_list.append([])
            entities_list.append([])
            relations_list.append([])

        if len(sentence_list) < self.num_sents:
            num_sents = len(sentence_list)
        else:
            num_sents = self.num_sents

        merge_data = []
        num_token_pre_sent = 0
        num_entities_pre = 0

        for idx in range(len(sentence_list)-(num_sents-1)):
            concat_sent = []
            concat_entity = []
            concat_relation = []
            if idx!=0:
                num_token_pre_sent += len(sentence_list[idx-1])
                num_entities_pre += len(entities_list[idx-1])
            for j in range(0, num_sents):
                convert_pos_entity = [dict(type=e['type'], start=e['start']-num_token_pre_sent, end=e['end']-num_token_pre_sent) for e in entities_list[idx+j]]
                convert_pos_rel = [dict(type=r['type'], head=r['head']-num_entities_pre, tail=r['tail']-num_entities_pre) for r in relations_list[idx+j]]
                concat_sent.extend(sentence_list[idx+j])
                concat_entity.extend(convert_pos_entity)
                concat_relation.extend(convert_pos_rel)

            # print("num_token_pre_sent", num_token_pre_sent)
            # print("num_entities_pre", num_entities_pre)
            # print(len(concat_sent))
            # print("*"*100)


            new_sample = dict(tokens=concat_sent, entities=concat_entity, relations=concat_relation, orig_id="from_"+str(idx-padding)+"_to_"+str(idx-padding+num_sents)+"_"+orig_id)
            try:
                self.visualize.convert_each_sample(new_sample, self.visualized_path)
            except:
                # print(new_sample)
                pass
                
            merge_data.append(new_sample)

        return merge_data

            # break




    def _transfer_entity(self, sent, entities):
        sent_entities = []
        idx_sent_entities = []
        for idx, entity in enumerate(entities):
            # print(entity)
            s_e = entity['start']
            e_e = entity['end']
            if s_e >= sent.start and e_e <= sent.end-1:
                sent_entities.append(entity)
                idx_sent_entities.append(idx)
            # print(s_e, e_e)

        return sent_entities, idx_sent_entities
    
    def _trasfer_relation(self, sent, relations, idx_sent_entities):
        # print(idx_sent_entities)
        sent_relations = []
        for rel in relations:
            if rel['head'] and rel['tail'] in idx_sent_entities:
                sent_relations.append(rel)
                # print(rel)

        return sent_relations


    def load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)
        return data

if __name__ == "__main__":
    num_sentence = 1
    padding = 0
    json_path = "experimental_ovp_dataset/half_sem_half_sci/half_set.json"
    saved_path = "experimental_ovp_dataset/half_sem_half_sci/half_set_sentence_{}_padding_{}.json".format(num_sentence, padding)
    visualized_path =  "/Users/dongpham/Documents/NII_internship/brat-1.3p1/data/examples/check_separate_sentence"

    separator = SeparateSentence(num_sentence=1, padding=0, visualized_path=visualized_path)
    separator._run_separte_(json_path, saved_path)