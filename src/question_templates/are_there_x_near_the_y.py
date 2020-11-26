import os
import os.path as osp
import random

from config import output_dir
from question_templates.question_templates import question_template
from utils.geometry import Rect
from utils.utils import distribution_for_each_scene_graph_object, not_replaceable_objects, words_similarity, \
    REPLACEMENT_SIMILARITY_THRESHOLD, vowels, isplural


class are_there_x_near_the_y(question_template):
    """are_there_x_near_the_y"""
    regular_expressions = [r"Are there (.*) near the (.*)", r"is there (.*) near the (.*)",
                           r"Is there (.*) near the (.*)"]

    IMAGE_ID_EXCEPTION_LIST = ['2363794', '2409481']
    QA_KEY_EXCEPTION_LIST = []

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = osp.join(output_dir, 'are_there_x_near_the_y')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_atoms(self):
        return (self.x, self.y)

    def init_after_match(self):
        assert len(self.groups) >= 2
        self.x_section = self.groups[0]
        self.x_prefix = None
        for x_pref in ['any', 'a']:
            if self.x_section.split(" ")[0] == x_pref:
                self.x_prefix = x_pref
                self.x = " ".join(self.x_section.split(" ")[1:])
        if self.x_prefix is None:
            self.x = self.x_section
        self.y = self.groups[1]
        if "?" in self.y:
            self.y = self.y.replace("?", "")
        if len(self.groups) > 2:
            self.suffix = self.groups[2]
        if self.scene_graph_for_image_id:
            self.scene_graph_objects = self.scene_graph_for_image_id['objects']
            self.all_scene_graphs_for_y = {k: v for k, v in self.scene_graph_objects.items()
                                           if self.objects_are_the_same(v['name'], self.y)}
            if len(list(self.all_scene_graphs_for_y.values())) > 0:
                self.scene_graph_for_y = list(self.all_scene_graphs_for_y.values())[0]
            else:
                return False
        return True

    def produce_output(self):
        outputs = {'reg_exp': self.reg_exp, 'reg_exp_match_groups': self.groups, 'x': self.x, 'y': self.y,
                   'suffix': self.suffix if hasattr(self, 'suffix') else ''
                   }
        return outputs

    def augment_question_template(self):
        success = self.init_after_match()
        if not success:
            return None

        items_that_are_near_y = self.get_items_that_are_near_y()
        if len(items_that_are_near_y) == 0:
            return None

        if self.answer == 'no':
            # which x' does exists near y?
            chosen_replacement = self.get_replacement_that_is_near_y(items_that_are_near_y)
            answer_replacement = 'yes'
        elif self.answer == 'yes':
            # which object is not near y?
            chosen_replacement = self.get_replacement_that_is_not_near_y()
            answer_replacement = 'no'
        else:
            raise Exception("No such answer")

        if chosen_replacement:
            question_replacement = self.create_replacements(chosen_replacement)
            return (question_replacement, answer_replacement)
        else:
            return None

    def create_replacements(self, chosen_replacement):
        plural = isplural(chosen_replacement)
        prefix = 'Are there' if plural else "Is there"
        first_x_word = self.x_section.split(" ")[0]
        if first_x_word == 'any':
            chosen_replacement = 'any ' + chosen_replacement
        else:
            if not plural:
                if any(chosen_replacement.lower().startswith(v) for v in vowels):
                    chosen_replacement = 'an ' + chosen_replacement
                else:
                    chosen_replacement = 'a ' + chosen_replacement
        question_replacement = f"{prefix} {chosen_replacement} near the {self.y}?"
        return question_replacement

    def get_items_that_are_near_y(self):
        relations = self.scene_graph_for_y['relations']
        items_that_are_near_y = [d for d in relations if d['name'] in ['near', 'next to']]
        return items_that_are_near_y


    def get_replacement_that_is_near_y(self, items_that_are_near_y):
        object_near_y_names = [x['object'] for x in items_that_are_near_y]

        objects_that_near_y_scene_graphs = {k: v for k, v in self.scene_graph_objects.items() if k in object_near_y_names}

        objects_that_near_y_scene_graphs = {k:v for k,v in objects_that_near_y_scene_graphs.items() if v['name'] not in self.produced_objects}

        distances_from_y = self.get_distances_for_y(objects_that_near_y_scene_graphs, without_intersection=False)
        if len(distances_from_y) == 0:
            return None

        item_closest_to_y = min(distances_from_y, key=distances_from_y.get)
        chosen_replacement_obj = item_closest_to_y
        chosen_replacement = self.scene_graph_objects[chosen_replacement_obj]['name']
        self.produced_objects.append(chosen_replacement)
        return chosen_replacement

    def get_distances_for_y(self, objects_that_are_not_near_y, without_intersection):
        distances_from_y = {}
        y_rect = Rect(self.scene_graph_for_y['x'], self.scene_graph_for_y['y'],
                      self.scene_graph_for_y['w'], self.scene_graph_for_y['h'])
        for obj_key, obj_dict in objects_that_are_not_near_y.items():
            obj_rect = Rect(obj_dict['x'], obj_dict['y'], obj_dict['w'], obj_dict['h'])
            try:
                distances_from_y[obj_key] = y_rect.distance_to_rect(obj_rect)
            except Exception as ex:
                print(f"Exception in y_rect.distance_to_rect, continue")
                continue
        return distances_from_y

    def get_replacement_that_is_not_near_y(self):
        objects_in_curr_image = [self.scene_graph_objects[k]['name'] for k in self.scene_graph_objects]

        objects_in_image_related_to_y = [v['name'] for v in self.scene_graph_objects.values() if self.objects_are_the_same(v['name'], self.y)]
        y_related_obj = objects_in_image_related_to_y[0]
        if not y_related_obj in distribution_for_each_scene_graph_object:
            return None
        dist_for_y = distribution_for_each_scene_graph_object[y_related_obj]
        for obj in objects_in_curr_image + not_replaceable_objects + self.produced_objects:
            if obj in dist_for_y:
                del dist_for_y[obj]
            similar_objs_in_dist = [k for k in dist_for_y.keys() if words_similarity(k, obj) >= REPLACEMENT_SIMILARITY_THRESHOLD or self.objects_are_the_same(k, obj)]
            for k in similar_objs_in_dist:
                del dist_for_y[k]
        weights = list(dist_for_y.values())

        chosen_replacement = random.choices(population=list(dist_for_y.keys()), weights=weights, k=1)[0]

        self.produced_objects.append(chosen_replacement)

        print(f"Possible replacement connected to y-{self.y}, v-objects-{objects_in_image_related_to_y}: {chosen_replacement}")
        return chosen_replacement

    def questions_classes_are_equivalent(self, other):
        question_are_matched = self.objects_are_the_same(self.x, other.x) and \
                               self.objects_are_the_same(self.y, other.y)
        return question_are_matched

    def test_augmentation(self, other):
        assert self.answer != other.answer
        curr_words = self.question.split(" ")
        other_words = other.question.split(" ")

        self.remove_prefix_is_are(self.x, curr_words)
        self.remove_prefix_is_are(other.x, other_words)

        self.remove_pref_x(self, other, curr_words)
        self.remove_pref_x(other, self, other_words)

        self_x_section_words = self.get_x_words_after_removing_prefix(self)
        in_curr_but_not_in_other = [x for x in curr_words if x not in other_words if x != 'the']
        assert " ".join(in_curr_but_not_in_other) == " ".join(self_x_section_words)

        other_x_section_words = self.get_x_words_after_removing_prefix(other)
        in_other_but_not_in_curr = [x for x in other_words if x not in curr_words if x != 'the']
        assert " ".join(in_other_but_not_in_curr) == " ".join(other_x_section_words)

        symmetric_diff = set(curr_words).symmetric_difference(set(other_words))
        assert len(symmetric_diff) == len(self_x_section_words) + len(other_x_section_words)
        return True

    def remove_pref_x(self, first, second, words):
        if 'any' in words:
            pref = 'any'
            words.remove(pref)
        else:
            if not isplural(first.x) and isplural(second.x):
                if any(first.x.lower().startswith(v) for v in vowels):
                    pref = 'an'
                else:
                    pref = 'a'
                words.remove(pref)

