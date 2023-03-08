import os
import json
import sys  

class ConvertJson2Brat():
    def __init__(self) -> None:
        self.sep = "\t"

    def _run_convert(self, path, saved_path):
        data = self.load_json(path)

        for sample in data:
            self.convert_each_sample(sample, saved_path)
    

    def convert_each_sample(self, sample, saved_path):
        orig_id = sample['orig_id']
        tokens = sample['tokens']
        text = " ".join(tokens)
        entities = sample['entities']
        relations = sample['relations']
        with open(os.path.join(saved_path, orig_id+".txt"), "w") as f:
            f.write(text)

        with open(os.path.join(saved_path, orig_id+".ann"), "w") as f:
            for idx_entity, entity in enumerate(entities):
                entity_type = entity['type']
                start = entity['start']
                end = entity['end']

                entity_text = " ".join(tokens[start:end])
                len_char = len(entity_text)
                try:
                    s_char = self.get_location_char(tokens, start)
                except:
                    print("ERROR at get_location_char()")
                    print(entity)
                    print("start", start)
                    print("len tokens", len(tokens))
                    # print("tokens", tokens)
                e_char = s_char + len_char
                s_char = str(s_char)
                e_char = str(e_char)
                
                entiry_content = "T{}".format(str(idx_entity)) + self.sep + entity_type + " " + s_char + " " + e_char + self.sep + entity_text +"\n"
                f.write(entiry_content)

            for idx_rel, relation in enumerate(relations):
                # print(relation)
                rel_type = relation['type']
                head = relation['head']
                tail = relation['tail']

                relation_content = "R{}".format(str(idx_rel)) + self.sep + rel_type + " " + "Arg1:T{}".format(head) + " " + "Arg2:T{}".format(tail) + "\n"
                f.write(relation_content)


    def get_location_char(self, tokens, start):
        # print("len", len(tokens))
        # print(tokens)
        count_char = 0
        for i in range(start):
            # print(i)
            token = tokens[i]
            # print(i, token, len(token))
            count_char = count_char + (len(token)+1)

        return count_char


    def load_json(self, path):
        with open(path, "r") as file:
            data = json.load(file)

        return data


if __name__ == "__main__":
    json_path = "test/semeval-2018_test.json"
    saved_path = "test/annotation"
    # saved_path = "/Users/phamdong/Documents/NII_internship/repo/brat-1.3p1/data/examples/SemEval-2018-task7"
    if os.path.exists(saved_path) is False:
        os.mkdir(saved_path)

    convertor = ConvertJson2Brat()
    convertor._run_convert(json_path, saved_path)