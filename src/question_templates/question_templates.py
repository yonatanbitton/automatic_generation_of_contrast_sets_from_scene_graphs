import re
from abc import ABC, abstractmethod
from copy import deepcopy

from utils.utils import SIMILARITY_THRESHOLD, all_animals, all_vehicles, all_utensil, synonyms, \
    words_similarity, isplural, wnl, colors_names


class question_template(ABC):
    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        self.question = question
        self.answer = answer
        self.image_id = image_id
        self.qa_key = qa_key
        self.scene_graph_for_image_id = scene_graph_for_image_id
        if scene_graph_for_image_id:
            self.scene_graph_objects = [self.scene_graph_for_image_id['objects'][k] for k in self.scene_graph_for_image_id['objects'].keys()]
            self.scene_graph_objects_names = [v['name'] for v in self.scene_graph_objects]
        self.produced_objects = deepcopy(produced_objects_for_image_id)
        self.can_produce_only_one_sample_for_match = False

    @abstractmethod
    def get_atoms(self):
        pass

    @abstractmethod
    def augment_question_template(self):
        pass

    @abstractmethod
    def questions_classes_are_equivalent(self, other):
        pass

    @abstractmethod
    def test_augmentation(self, other):
        pass

    @abstractmethod
    def produce_output(self):
        pass

    def objects_are_the_same(self, v, y):
        return self.inner_objects_are_the_same(v, y) or self.inner_objects_are_the_same(y, v)

    def inner_objects_are_the_same(self, v, y):
        v = v.lower()
        y = y.lower()
        if v == y or words_similarity(v, y) > SIMILARITY_THRESHOLD:
            return True
        y_parts = y.split()
        y_first_word = y_parts[0]
        if y_first_word == v or (len(y_parts) > 1 and y_parts[1] == v) or (
                len(y_parts) > 1 and words_similarity(v, y) > 0.6 and v in y):
            return True
        # plural
        if (len(y_parts) > 1 and y_parts[1] == v + "s") or (len(y_parts) > 1 and y_parts[1] + "s" == v):
            return True
        if len(y_parts) == 1 and len(v.split(" ")) == 2 and y == v.split(" ")[0]:  # remote, remote control
            return True
        if y_first_word == 'animal' or (len(y_parts) > 1 and y_parts[1] == 'animal'):
            return v in all_animals
        if y_first_word == 'vehicle' or (len(y_parts) > 1 and y_parts[1] == 'vehicle'):
            return v in all_vehicles
        if y_first_word == 'utensil' and v in all_utensil or (
                len(y_parts) > 1 and y_parts[1] == 'utensil' and v in all_utensil):
            return True
        if y_first_word in colors_names and len(y_parts) >= 2 and self.objects_are_the_same(v, " ".join(y_parts[1:])):
            return True
        y_in_length_of_v = " ".join(y.split(" ")[:len(v.split(" "))])
        for syn in synonyms:
            if (y_in_length_of_v, v) == syn or (v, y_in_length_of_v) == syn:
                return True
        if y.lower() == v:
            return True
        if " ".join(y_parts[:2]) == v:
            return True
        if wnl.lemmatize(v) == wnl.lemmatize(y):
            return True
        return False

    def try_to_match_question_to_template(self):
        if self.image_id in self.IMAGE_ID_EXCEPTION_LIST:
            return False
        if self.qa_key in self.QA_KEY_EXCEPTION_LIST:
            return False
        match_success = False
        for reg_exp in self.regular_expressions:
            match_obj = re.match(reg_exp, self.question)
            if match_obj:
                self.reg_exp = reg_exp
                self.match_obj = match_obj
                self.groups = match_obj.groups()
                match_success = True
                break
        return match_success


    def get_x_words_after_removing_prefix(self, cls):
        x_part = cls.x if not hasattr(cls, 'x_section') else cls.x_section
        x_section_words = x_part.split(" ")
        if hasattr(cls, 'x_prefix') and cls.x_prefix is not None:
            x_section_words.remove(cls.x_prefix)
        if 'the' in x_section_words:
            x_section_words = [x for x in x_section_words if x != 'the']
        return x_section_words

    def remove_prefix_is_are(self, x, words):
        if isplural(x):
            assert words[0] == 'Are'
        else:
            assert words[0] == 'Is'
        words.remove(words[0])