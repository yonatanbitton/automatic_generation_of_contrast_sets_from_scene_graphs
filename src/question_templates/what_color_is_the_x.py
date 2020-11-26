import os
import os.path as osp
import random

from config import output_dir
from question_templates.question_templates import question_template
from utils.utils import isplural, colors_names, words_similarity, colorless_objects


class what_color_is_the_x(question_template):
    """what_color_is_the_x"""
    regular_expressions = [r"What color is the (.*)", r"What is the color of the (.*)",
                           r"Which color are the (.*)",  r"What color do you think the (.*)",
                           r"Which color is the (.*)", r"The (.*) has what color", r"The (.*) is of which color",
                           r"What color are the (.*)"
                           ]

    IMAGE_ID_EXCEPTION_LIST = []
    QA_KEY_EXCEPTION_LIST = []

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = osp.join(output_dir, 'what_color_is_the_x')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_atoms(self):
        return (self.x, )

    def init_after_match(self):
        assert len(self.groups) >= 1
        self.x_section = self.groups[0].replace("?", "")
        x_section_words = self.x_section.split(" ")
        self.suffix_has = x_section_words[-1] == 'has'
        if self.suffix_has:
            x_section_words.remove('has')

        if ',' in self.question:
            self.comma_options = True
            words_after_comma = self.question.replace('?', '').split(",")[1].split(" ")
            self.comma_colors_options_list = [x for x in words_after_comma if x in colors_names]
        else:
            self.comma_options = False

        if len(x_section_words) == 1:
            self.x = " ".join(x_section_words).replace(",", "")
        else:
            relevant_x_section = self.x_section
            if 'on top' in self.x_section:
                relevant_x_section = relevant_x_section[:self.x_section.index('on top') - 1]
            if ',' in self.question:
                relevant_x_section = relevant_x_section[:self.x_section.index(',')]
            self.x = relevant_x_section
        self.scene_graph_objects = self.scene_graph_for_image_id['objects']
        self.all_scene_graphs_for_x = {k: v for k, v in self.scene_graph_objects.items()
                                       if self.objects_are_the_same(v['name'], self.x)}
        if len(list(self.all_scene_graphs_for_x.values())) > 0:
            self.all_scene_graphs_for_x = list(self.all_scene_graphs_for_x.values())[0]
        else:
            return False


        return True

    def produce_output(self):
        outputs = {'reg_exp': self.reg_exp, 'reg_exp_match_groups': self.groups, 'x': self.x,
                   'suffix': self.suffix if hasattr(self, 'suffix') else ''
                   }
        return outputs

    def augment_question_template(self):
        success = self.init_after_match()
        if not success:
            return None

        all_items_with_color = {k:v for k,v in self.scene_graph_objects.items()
                                if len(v['attributes']) > 0
                                and any(vi in colors_names for vi in v['attributes'])
                                and v['name'] not in colorless_objects
                                and not self.objects_are_the_same(v['name'], self.x)}
        if len(all_items_with_color) == 0:
            return None

        text_sim_to_x = {}
        for k, v in all_items_with_color.items():
            color = [vi for vi in v['attributes'] if vi in colors_names][0]
            if color == self.answer:
                continue
            if v['name'] in self.produced_objects:
                continue
            text_sim_to_x[v['name'], color] = words_similarity(v['name'], self.x)

        if len(text_sim_to_x) == 0:
            return None

        most_similar_obj, most_similar_obj_color = max(text_sim_to_x, key=text_sim_to_x.get)
        self.produced_objects.append(most_similar_obj)

        self.question = self.question.replace("?", "")
        augmented_question = self.question.replace(" " + self.x_section, " " + most_similar_obj)
        augmented_answer = most_similar_obj_color

        if self.suffix_has:
            augmented_question += ' has'

        if " color is the " in augmented_question and isplural(most_similar_obj):
            augmented_question = augmented_question.replace(" is ", " are ")
        elif " color are the " in self.question and not isplural(most_similar_obj):
            augmented_question = augmented_question.replace(" are ", " is ")

        if self.comma_options:
            new_colors_list = [most_similar_obj_color]
            colors_from_original_q_lst_not_same_as_new_color = [x for x in self.comma_colors_options_list if x != most_similar_obj_color]
            if len(colors_from_original_q_lst_not_same_as_new_color) > 0:
                new_colors_list.append(colors_from_original_q_lst_not_same_as_new_color[0])
            else:
                sampled_color = random.sample([x for x in colors_names if x != most_similar_obj_color], 1)[0]
                new_colors_list.append(sampled_color)
            random.shuffle(new_colors_list)
            augmented_question += f", {new_colors_list[0]} or {new_colors_list[1]}"

        augmented_question += "?"

        return (augmented_question, augmented_answer)

    def questions_classes_are_equivalent(self, other):
        question_are_matched = self.objects_are_the_same(self.x, other.x)
        return question_are_matched

    def test_augmentation(self, other):
        assert self.answer != other.answer
        other_question = other.question
        other_question = other_question.replace("?","")
        if other.comma_options:
            other_question = other_question.split(',')[0]
            other_x_section = other.x_section.split(",")[0]
        else:
            other_x_section = other.x_section

        if self.comma_options:
            self_question = self.question.split(',')[0]
            self_x_section = self.x_section.split(",")[0]
        else:
            self_question = self.question
            self_x_section = self.x_section

        x_first_word = self.x.split(" ")[0]

        if " color is the " in other_question and isplural(x_first_word):
            other_question = other_question.replace(" is ", " are ")
        elif " color are the " in other_question and not isplural(x_first_word):
            other_question = other_question.replace(" are ", " is ")

        other_question += "?"
        other_question_by_replacing_x_section = other_question.replace(" " + other_x_section, " " + self_x_section)

        if self_question[-1] != "?":
            self_question += "?"
        if other_question[-1] != "?":
            other_question += "?"

        assert self_question == other_question_by_replacing_x_section
        return True