import os
import json
import sys  
from spacy import displacy
from pathlib import Path
import re
colors = {'Task': "#85C1E9", "OtherScientificTerm": "#ff6961", "Material":"#D59B00", "Method":"#50A625", "Generic":"#D7D7D7", "Metric": "#CC99FF"}
options = {"ents": ["Task", "OtherScientificTerm", "Material", "Method", "Generic"], "colors": colors}

class Visualize_HTML():
    def __init__(self) -> None:
        pass
    
    def _run_visualize(self, path, saved_path):
        data = self.load_json(path)

        for sample in data:
            orig_id = sample['orig_id']
            tokens = sample['tokens']
            text = " ".join(tokens)
            entities = sample['entities']
            relations = sample['relations']

            # print(relations)

            self.visualize_sentence(sentence=tokens, 
                                    entities=entities, 
                                    relations=relations, 
                                    saved_name=os.path.join(saved_path, orig_id))


            break
        

    def load_json(self, path):
            with open(path, "r") as file:
                data = json.load(file)

            return data

    def visualize_sentence(self, sentence, entities, relations, style="span", saved_name=None):
        doc = {}
        if style=="span":
            doc['text'] = " ".join(sentence)
            doc['tokens'] = sentence

            spans = []
            for entity in entities:
                span = {}
                span['start_token'] = entity['start']
                span['end_token'] = entity["end"]
                span['label'] = entity['type']
                spans.append(span)
            doc['spans'] = spans

        html_entity = displacy.render(doc, manual=True, style=style, options=options, page=True, jupyter=False)


        relation_tags = []
        for rel in relations:
            # print("rel", rel)
            s_entity_1 = entities[rel['head']]['start']
            e_entity_1 = entities[rel['head']]['end']
            s_entity_2 = entities[rel['tail']]['start']
            e_entity_2 = entities[rel['tail']]['end']
            relation_type = rel['type']
            for entity in entities:
                if s_entity_1==entity['start'] and e_entity_1==entity['end']:
                    label_entity_1 = entity['type']
                    # break
                if s_entity_2==entity['start'] and e_entity_2==entity['end']:
                    label_entity_2 = entity['type']
                
            try:
                text_colors = [colors[label_entity_1], colors[label_entity_2]]
            except:
                text_colors = ['#85C1E9', '#85C1E9']
            entity_list = [dict(text=" ".join(sentence[s_entity_1:e_entity_1]), tag=label_entity_1),
                            dict(text=" ".join(sentence[s_entity_2:e_entity_2]), tag=label_entity_2)]
            relation_list = [dict(start=0, end=1, label=relation_type, dir="right")]

            rel_doc = dict(words=entity_list, arcs=relation_list)
            distance = ((e_entity_1+1-s_entity_1) + (e_entity_2+1-s_entity_2)) * 50
            # print(text_colors)
            rel_options = {"compact": True, "color": "black", "distance":distance, "arrow_stroke": 5, "offset_x":distance/2,
                        "arrow_width":16, "font":"sans-serif", "font-weight":"bold"}
            html_relation = displacy.render(rel_doc, manual=True, style="dep", options=rel_options, page=True, jupyter=False)
            # print(html_relation)
            html_relation = self.fill_color_relations(html=html_relation, text_color=text_colors)
            svg_tag = self.get_svg_tag(html=html_relation)

            relation_tags.append(svg_tag)

            # output_path = Path("visualize_html/"+"relation.html")
            # output_path.open("w", encoding="utf-8").write(html_relation)

            # break
        relation_html = "\n".join(relation_tags) + "\n" + "</figure>"

        html_entity = html_entity.replace("</figure>", relation_html)

        if saved_name:
            output_path = Path(saved_name+".html")
            output_path.open("w", encoding="utf-8").write(html_entity)

        return html_entity

    def get_text_in_html(self, html):
        partern = r'<text class=\"displacy-token\".*>[\s\S]*?</text>'
        matches = re.findall(partern, html)

        return matches

    def get_svg_tag(self, html):
        partern = r'<svg .*>[\s\S]*?</svg>'
        matches = re.findall(partern, html)

        return matches[0]

    def fill_color_relations(self, html, text_color):
        text_tags = self.get_text_in_html(html)
        for i, tag in enumerate(text_tags):
            original_text = tag
            fill_color = tag.replace("currentColor", text_color[i])
            fill_color = fill_color.replace("text class=\"displacy-token\"", "text class=\"displacy-token\" font-weight=\"bold\"")
            html = html.replace(original_text, fill_color)

        return html


if __name__ == "__main__":
    

    json_path = "data/eval/json/measeval_eval.json"
    saved_path = "visualize_html"
    if os.path.exists(saved_path) is False:
        os.mkdir(saved_path)

    convertor = Visualize_HTML(saved_path)
    convertor._run_visualize(json_path)
    pass