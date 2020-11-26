import os
import os.path as osp

from config import output_dir
from question_templates.question_templates import question_template
from utils.utils import words_similarity, not_replaceable_objects, isplural


class on_which_side_is_the_x(question_template):
    """on_which_side_is_the_x"""
    regular_expressions = [r"On which side of the photo is the (.*)", r"On which side of the photo is the (.*)",
                           r"On which side of the image is the (.*)",  r"On which side of the picture is the (.*)",
                           r"On which side is the (.*)",
                           r"On which side of the photo are the (.*)", r"On which side of the photo are the (.*)",
                           r"On which side of the image are the (.*)", r"On which side of the picture are the (.*)",
                           r"On which side are the (.*)",
                           r"On which side of the image are his (.*)", r"On which side are his (.*)",
                           ]

    IMAGE_ID_EXCEPTION_LIST = ['2400938', '2354103', '2371405', '2400661', '2364217', '2411252']
    QA_KEY_EXCEPTION_LIST = ['03144574', '041026809', '03614228', '1755390']

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = osp.join(output_dir, 'on_which_side_is_the_x')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_atoms(self):
        return (self.x, )

    def init_after_match(self):
        assert len(self.groups) >= 1
        split_by_comma = self.groups[0].split(",")
        self.x = split_by_comma[0].replace("?", "")
        if len(split_by_comma) >= 2:
            self.suffix = split_by_comma[1].strip()
            self.left_or_right = all(x in self.suffix for x in ['right', 'or', 'left'])
        else:
            self.left_or_right = False
        return True

    def produce_output(self):
        outputs = {'reg_exp': self.reg_exp, 'reg_exp_match_groups': self.groups, 'x': self.x,
                   'suffix': self.suffix if hasattr(self, 'suffix') else '',
                   'left_or_right': self.left_or_right if hasattr(self, 'left_or_right') else False,
                   }
        return outputs

    def augment_question_template(self):
        self.init_after_match()

        if len([v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, self.x)]) > 1:
            return None

        x_object_data = self.get_x_object_data()
        if x_object_data == None:
            print(f"not finding similar objects to: {self.x}, objects: {[x['name'] for x in self.scene_graph_objects]}")
            return None
        half_width, x_side = self.define_width_middle()

        all_objects_on_another_side = self.get_all_objects_that_are_exclusive_to_the_other_side(half_width, x_side)
        all_objects_on_another_side = [o for o in all_objects_on_another_side if o not in self.produced_objects]

        if len(all_objects_on_another_side) == 0:
            return None
        all_objects_textual_similarity_to_x = {obj_name: words_similarity(self.x, obj_name) for obj_name in all_objects_on_another_side}
        object_most_similar_to_x = max(all_objects_textual_similarity_to_x, key=all_objects_textual_similarity_to_x.get)
        self.produced_objects.append(object_most_similar_to_x)
        if self.left_or_right:
            question_replacement = self.question.replace(self.x, object_most_similar_to_x)
        else:
            question_replacement = self.question.replace(self.x + "?", object_most_similar_to_x + "?")
        question_replacement = self.fix_plural_form(object_most_similar_to_x, question_replacement)
        if self.answer == 'left':
            answer_replacement = 'right'
        elif self.answer == 'right':
            answer_replacement = 'left'
        else:
            raise Exception(f"Unknown answer {self.answer}")
        return (question_replacement, answer_replacement)

    def define_width_middle(self):
        image_width = self.scene_graph_for_image_id['width']
        half_width = image_width / 2
        x_side = self.answer
        # Add margin - not to be on the middle
        margin_pct = 0.2
        if x_side == 'left':
            half_width = half_width * (1 + margin_pct)
        elif x_side == 'right':
            half_width = half_width * (1 - margin_pct)
        else:
            raise Exception(f"Didn't find side for obj {self.x}")
        return half_width, x_side

    def fix_plural_form(self, object_most_similar_to_x, question_replacement):
        quest_replacement_words = question_replacement.split(" ")
        index_words_options = [object_most_similar_to_x, object_most_similar_to_x + ",",object_most_similar_to_x + "?"]
        index_of_replacement = -1
        for op in index_words_options:
            if len(op.split(" ")) > 1:
                op = op.split(" ")[0]
            if op in quest_replacement_words:
                index_of_replacement = quest_replacement_words.index(op)
        if index_of_replacement == -1:
            raise Exception(f"Couldn't find replacement index")

        if isplural(object_most_similar_to_x) and quest_replacement_words[index_of_replacement - 2] == 'is':
            quest_replacement_words[index_of_replacement - 2] = 'are'
            question_replacement = " ".join(quest_replacement_words)
        elif not isplural(object_most_similar_to_x) and quest_replacement_words[index_of_replacement - 2] == 'are':
            quest_replacement_words[index_of_replacement - 2] = 'is'
            question_replacement = " ".join(quest_replacement_words)
        return question_replacement

    def get_all_objects_that_are_exclusive_to_the_other_side(self, half_width, x_side):
        all_objects_on_left = []
        all_objects_on_right = []
        for obj in [x for x in self.scene_graph_objects if x['name'] != self.x]:
            obj_side = self.get_object_side(half_width, obj)
            if obj_side == 'left':
                all_objects_on_left.append(obj['name'])
            elif obj_side == 'right':
                all_objects_on_right.append(obj['name'])
        objects_on_both_sides = list(set(all_objects_on_left).intersection(set(all_objects_on_right)))
        same_objects = []
        for a in all_objects_on_left:
            for b in all_objects_on_right:
                if self.objects_are_the_same(a, b):
                    same_objects.append(a)
                    same_objects.append(b)
        for o in objects_on_both_sides + not_replaceable_objects + same_objects:
            all_objects_on_left = [x for x in all_objects_on_left if x != o]
            all_objects_on_right = [x for x in all_objects_on_right if x != o]

        if x_side == 'left':
            all_objects_on_another_side = all_objects_on_right
        elif x_side == 'right':
            all_objects_on_another_side = all_objects_on_left
        else:
            raise Exception(f"No such side {x_side}")
        return all_objects_on_another_side

    def get_x_object_data(self):
        x_object_data_list = [x for x in self.scene_graph_objects if self.objects_are_the_same(x['name'], self.x)]
        if len(x_object_data_list) == 0:
            print(f"not finding similar objects to: {self.x}, objects: {[x['name'] for x in self.scene_graph_objects]}")
            x_object_data = None
        elif len(x_object_data_list) == 1:
            x_object_data = x_object_data_list[0]
        else:
            x_object_data_list_with_same_name = [x for x in x_object_data_list if x['name'] == self.x]
            if len(x_object_data_list_with_same_name) >= 1:
                x_object_data = x_object_data_list_with_same_name[0]
            else:
                x_object_data = x_object_data_list[0]
        return x_object_data

    def get_object_side(self, half_width, obj_data):
        if obj_data['x'] <= half_width and obj_data['x'] + obj_data['w'] <= half_width:
            obj_side = 'left'
        elif obj_data['x'] >= half_width and obj_data['x'] + obj_data['w'] >= half_width:
            obj_side = 'right'
        else:
            return None
        return obj_side

    def questions_classes_are_equivalent(self, other):
        question_are_matched = self.objects_are_the_same(self.x, other.x)
        return question_are_matched

    def test_augmentation(self, other):
        assert self.answer != other.answer
        self.question = self.question.replace("?", "").replace(",", "")
        other.question = other.question.replace("?", "").replace(",", "")

        curr_words = self.question.split(" ")
        other_words = other.question.split(" ")
        index_of_is_are_self = self.remove_is_are_x(curr_words)

        if self.x in ['picture', 'photo']:
            index_of_x = index_of_is_are_self + 1
            curr_words[index_of_x] = other.x
            curr_quest_no_is_are_replace_x = " ".join(curr_words)
        else:
            curr_quest_no_is_are_replace_x = " ".join(curr_words).replace(self.x, other.x)

        other_q_no_is_are = " ".join(other_words)

        assert curr_quest_no_is_are_replace_x == other_q_no_is_are
        return True

    def remove_is_are_x(self, words):
        index_of_is_are = 3
        if self.question.startswith("On which side of the"):
            index_of_is_are = 6
        if isplural(self.x):
            if 'are' in words and words.index("are") == index_of_is_are:
                words.remove('are')
            else:
                print("Bug")
        elif not isplural(self.x):
            if 'is' in words and words.index("is") == index_of_is_are:
                words.remove('is')
            else:
                print("Bug")
        return index_of_is_are