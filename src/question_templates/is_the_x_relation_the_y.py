import os
import re

from config import output_dir
from question_templates.question_templates import question_template
from utils.utils import words_similarity, not_replaceable_objects, vowels, isplural, \
    colors_names, verbs

reached_no_num = 0

class is_the_x_relation_the_y_abstract(question_template):
    regular_expressions = [r"is the (.*) behind (.*)", r"Is the (.*) behind (.*)",
                           r"are the (.*) behind (.*)", r"Are the (.*) behind (.*)",
                           r"is the (.*) in front (.*)", r"Is the (.*) in front (.*)",
                           r"are the (.*) in front (.*)", r"Are the (.*) in front (.*)",
                           r"is the (.*) to the left (.*)", r"Is the (.*) to the left (.*)",
                           r"are the (.*) to the left (.*)", r"Are the (.*) to the left (.*)",
                           r"is the (.*) to the right (.*)", r"Is the (.*) to the right (.*)",
                           r"are the (.*) to the right (.*)", r"Are the (.*) to the right (.*)",
                           r"Is the (.*) on top (.*)", r"Are the (.*) on top (.*)",
                           r"is the (.*) on top (.*)", r"are the (.*) on top (.*)",
                           r"Is the (.*) below (.*)", r"Are the (.*) below (.*)",
                           r"is the (.*) below (.*)", r"are the (.*) below (.*)",
                           r"Is the (.*) above (.*)", r"Are the (.*) above (.*)",
                           r"is the (.*) above (.*)", r"are the (.*) above (.*)"
                           ]

    relations = ['behind', 'in front', 'to the left', 'to the right', 'on top', 'below', 'above']

    negative_sentences = ['to the right or to the left', 'to the left or to the right']

    y_suffixes = ['on the left side of the photo', 'on the right side of the photo',
                  'in the picture', 'on the left', 'on the right', 'in the middle of the image',
                  'in the center of the image']
    parts_of_photo_spans = ['in the top', 'on the right', 'on the left', 'in the bottom']

    opposite_relations = {'behind': 'in front', 'to the left': 'to the right', 'on top': 'below', 'above': 'below',
                          'below': 'above'}
    opposite_relations = {**opposite_relations, **{v:k for k,v in opposite_relations.items()}}

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)

    def get_atoms(self):
        return (self.x, self.y)

    def try_to_match_question_to_template(self):
        if self.image_id in self.IMAGE_ID_EXCEPTION_LIST:
            return False
        if self.qa_key in self.QA_KEY_EXCEPTION_LIST:
            return False
        if self.answer not in ['yes', 'no']:
            return False
        if any(neg_sent in self.question for neg_sent in is_the_x_relation_the_y_abstract.negative_sentences):
            return False
        match_success = False
        for reg_exp in self.regular_expressions:
            match_obj = re.match(reg_exp, self.question)
            if match_obj:
                groups = match_obj.groups()
                x = groups[0]
                if any(rel in x for rel in self.relations):
                    continue
                if "that is to the left" in self.question or "that is to the right" in self.question:
                    continue
                if self.question.endswith("is behind of?") or self.question.endswith("in front of?"):
                    continue
                question_words = self.question.split(" ")
                question_words[-1] = question_words[-1].replace("?", "")
                adjs = ['small', 'large', 'tall', 'short', 'modern']
                if (question_words[-3] in adjs and question_words[-2] == 'and') or (question_words[-1] in adjs and question_words[-2] == 'and'):
                    continue
                if (question_words[-3] in colors_names and question_words[-2] == 'and') or (question_words[-1] in colors_names and question_words[-2] == 'and'):
                    continue
                y_section = groups[1].replace("?", "")
                y_suffix = self.get_suffix(y_section)
                y, y_a_or_an_or_the, question_connector = self.init_y_a_or_an_or_the_and_question_connector(y_section, y_suffix)
                y_indirect, y = self.get_y_indirect(y_section, y_suffix, y_a_or_an_or_the, question_connector, y)
                if "made of" in y:
                    continue

                contains_bad_phrase = False
                for phrase in verbs:
                    if "that" not in y_section and phrase in y_section:
                        contains_bad_phrase = True
                        break
                if contains_bad_phrase:
                    continue
                relation = self.get_relation(reg_exp)
                last_rel_w = relation.split(" ")[-1]
                if question_words[question_words.index(last_rel_w) + 1:] == ['of']:
                    continue

                if len([v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, x)]) > 1 \
                        or len([v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, y)]) > 1:
                    continue

                self.init_products(match_obj, reg_exp, relation, y, y_indirect, y_suffix, groups, x, y_section, y_a_or_an_or_the, question_connector)
                match_success = True
                break
        return match_success

    def init_products(self, match_obj, reg_exp, relation, y, y_indirect, y_suffix, groups, x, y_section, question_connector, y_a_or_an_or_the):
        self.x = x
        self.groups = groups
        self.y_indirect = y_indirect
        self.y_suffix = y_suffix
        self.reg_exp = reg_exp
        self.relation = relation
        self.match_obj = match_obj
        self.groups = match_obj.groups()
        self.y_section = y_section
        self.question_connector = question_connector
        self.y_a_or_an_or_the_or_of = y_a_or_an_or_the
        self.y = y

    def init_y_a_or_an_or_the_and_question_connector(self, y_section, y_suffix):
        y_a_or_an_or_the = ""
        question_connector = ""
        if y_section.startswith("of the "):
            y_a_or_an_or_the = 'the'
            question_connector = "of"
        elif y_section.startswith("of an "):
            y_a_or_an_or_the = 'an'
            question_connector = "of"
        elif y_section.startswith("of a "):
            y_a_or_an_or_the = 'a'
            question_connector = "of"
        elif y_section.startswith("a "):
            y_a_or_an_or_the = 'a'
        elif y_section.startswith("an "):
            y_a_or_an_or_the = 'an'
        elif y_section.startswith("of "):
            question_connector = 'of'
        elif y_section.startswith("the "):
            y_a_or_an_or_the = 'the'
        # else:
        #     raise Exception(f"Couldn't find question connector from y_section {y_section}, {self.question}, image_id: {self.image_id}, qa_key: {self.qa_key}")
        y_start_loc = len(question_connector) + 1 if question_connector else 0
        y_start_loc += len(y_a_or_an_or_the) + 1 if y_a_or_an_or_the else 0
        y = y_section[y_start_loc:]
        if y_suffix and y_suffix in y:
            y = y.replace(y_suffix, "").strip()
        if y_a_or_an_or_the == "":
            y_a_or_an_or_the = None
        if question_connector == "":
            question_connector = None
        return y, y_a_or_an_or_the, question_connector

    def produce_output(self):
        outputs = {'reg_exp': self.reg_exp, 'reg_exp_match_groups': self.groups, 'x': self.x, 'y': self.y,
                   'y_indirect': self.y_indirect, 'y_suffix': self.y_suffix,
                   'relation': self.relation, 'y_section': self.y_section
                   }
        return outputs

    def get_relation(self, reg_exp):
        relation = None
        for rel in self.relations:
            if rel in reg_exp:
                relation = rel
                break
        if not relation:
            raise Exception(f"Relation wasn't found {reg_exp}")
        return relation

    def get_y_indirect(self, y_section, y_suffix, y_a_or_an_or_the, question_connector, y):
        y_prefix_parts_len = 0
        if y_a_or_an_or_the:
            y_prefix_parts_len += len(y_a_or_an_or_the.split(" "))
        if question_connector:
            y_prefix_parts_len += len(question_connector.split(" "))
        if "that" in y_section:
            y_section_parts = y_section.split(" ")
            y_and_y_indirect = " ".join(y_section_parts[y_prefix_parts_len:])
            y_and_y_indirect_parts = y_and_y_indirect.split(" ")
            y_indirect = " ".join(y_and_y_indirect_parts[y_and_y_indirect_parts.index("that"):])
        elif any(x in y_section for x in self.y_suffixes + self.parts_of_photo_spans):
            relevant_part = [x for x in self.y_suffixes + self.parts_of_photo_spans if x in y_section][0]
            relevant_part_first_word = relevant_part.split(" ")[0]
            y_section_parts = y_section.split(" ")
            y_and_y_indirect = " ".join(y_section_parts[y_prefix_parts_len:])
            y_and_y_indirect_parts = y_and_y_indirect.split(" ")
            y_indirect = " ".join(y_and_y_indirect_parts[y_and_y_indirect_parts.index(relevant_part_first_word):])
        else:
            y_indirect = None
        if y_indirect and y_indirect in y:
            y = y.replace(y_indirect, "").strip()
        y_stripes = ['that is shown', 'that are shown', 'that is', 'that are']
        for y_st in y_stripes:
            if y_st in y:
                y = y.replace(y_st, "").strip()
        return y_indirect, y

    def get_suffix(self, y_section):
        y_suffix = ""
        for suff in self.y_suffixes:
            if y_section.endswith(suff):
                y_suffix = suff
                break
        return y_suffix

    def get_objects_of_relation_to_x(self, relation):
        x_relevant_objects = [v for v in self.scene_graph_objects if self.objects_are_the_same(v['name'], self.x)]
        if len(x_relevant_objects) == 0:
            return None
        x_obj = x_relevant_objects[0]
        x_relations = {self.scene_graph_for_image_id['objects'][x['object']]['name']: x['name'] for x in x_obj['relations']}
        x_relations_that_holds_relation = {k: v for k, v in x_relations.items() if relation in v}
        x_relations_that_holds_relation_names = list(x_relations_that_holds_relation.keys())
        return x_relations_that_holds_relation_names


    def questions_classes_are_equivalent(self, other):
        question_are_matched = self.objects_are_the_same(self.x, other.x) and \
                               self.objects_are_the_same(self.y, other.y) and \
                               self.relation == other.relation
        return question_are_matched

class is_the_x_relation_the_y_change_x(is_the_x_relation_the_y_abstract):
    """is_the_x_relation_the_y_change_x"""
    IMAGE_ID_EXCEPTION_LIST = []
    QA_KEY_EXCEPTION_LIST = ['05678811', '352847', '06347351', '09587171']

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = os.path.join(output_dir, 'is_the_x_relation_the_y_change_x')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)


    def augment_question_template(self):
        """ Example:
        ('Is the couch to the right of the pillow on the left side of the photo?', 'yes') ->
        ('Is the couch to the right of the shelf?', 'no')
        """
        # is the x relation to the y, we want to change y with y'
        if self.answer == "yes":
            # change y with object of the opposite relation (now the relation won't hold, => answer = no
            objects_to_replace = self.get_objects_of_relation_to_x(self.opposite_relations[self.relation])
            answer_replacement = "no"
        elif self.answer == "no":
            # change y with object of the same relation that do hold (now the relation will hold, => answer = yes
            objects_to_replace = self.get_objects_of_relation_to_x(self.relation)
            answer_replacement = "yes"
        else:
            raise Exception(f"no such answer: {self.answer}")

        objects_to_replace = self.get_objects_to_replace_without_multiple_objects(objects_to_replace)
        objects_to_replace = self.get_objects_to_replace_without_not_replaceable_objects(objects_to_replace)
        if objects_to_replace:
            objects_to_replace = [x for x in objects_to_replace if x not in self.produced_objects]

        if objects_to_replace is None:
            print(f"Didn't find objects same as x: {self.x, self.scene_graph_objects_names}")
            return None
        elif len(objects_to_replace) == 0:
            print(f"No objects to replace: {self.question, self.answer}")
            return None

        all_objects_textual_similarity_to_y = {obj_name: words_similarity(self.y, obj_name) for obj_name in objects_to_replace}
        object_most_similar_to_y = max(all_objects_textual_similarity_to_y, key=all_objects_textual_similarity_to_y.get)
        self.produced_objects.append(object_most_similar_to_y)
        is_or_are = self.reg_exp.split(' ')[0]

        if self.y_a_or_an_or_the_or_of == 'the' or 'the' in self.y_section:
            most_similar_prefix = 'the'
        else:
            most_similar_prefix = ""
            if not isplural(object_most_similar_to_y):
                if any(object_most_similar_to_y.lower().startswith(v) for v in vowels):
                    most_similar_prefix = 'an'
                else:
                    most_similar_prefix = 'a'
        if self.relation == "behind":
            question_replacement = f"{is_or_are} the {self.x} {self.relation} {most_similar_prefix} {object_most_similar_to_y}?"
        else:
            question_replacement = f"{is_or_are} the {self.x} {self.relation} of {most_similar_prefix} {object_most_similar_to_y}?"
        question_replacement = question_replacement.replace("  ", " ")
        return (question_replacement, answer_replacement)

    def get_objects_to_replace_without_multiple_objects(self, objects_to_replace):
        if objects_to_replace == None:
            return None
        objects_to_replace_without_mult_objects = []
        for o in objects_to_replace:
            if len([v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, o)]) > 1:
                continue
            objects_to_replace_without_mult_objects.append(o)
        return objects_to_replace_without_mult_objects

    def get_objects_to_replace_without_not_replaceable_objects(self, objects_to_replace):
        if objects_to_replace == None:
            return None
        objects_to_replace = [x for x in objects_to_replace if x not in not_replaceable_objects]
        return objects_to_replace


    def test_augmentation(self, other):
        assert self.answer != other.answer
        self.question = self.question.replace("?", "")
        other.question = other.question.replace("?", "")

        if self.y_indirect and self.y_indirect != "":
            self.question = self.question.replace(self.y_indirect, "").strip()

        curr_words = self.question.split(" ")
        other_words = other.question.split(" ")

        self.remove_pref_y(curr_words)
        other.remove_pref_y(other_words)

        curr_question_remove_pref_and_replacing_y = " ".join(curr_words).replace(self.y, other.y)
        other_question_remove_pref = " ".join(other_words)
        assert curr_question_remove_pref_and_replacing_y == other_question_remove_pref
        return True

    def remove_pref_y(self, words):
        if not isplural(self.y):
            if any(self.y.lower().startswith(v) for v in vowels):
                pref = 'an'
            else:
                pref = 'a'
            if pref in words:
                words.remove(pref)

class is_the_x_relation_the_y_change_relation(is_the_x_relation_the_y_abstract):
    """is_the_x_relation_the_y_change_relation"""

    IMAGE_ID_EXCEPTION_LIST = ['2336635', '2347445', '2354914']
    QA_KEY_EXCEPTION_LIST = ['02889125', '00352847', '05515888', '0987053', '06895727', '08703687', '14298187',
                             '02551133', '09618258', '13324502', '19840207', '07997570', '01445527', '13663448',
                             '08724727', '12785167', '05888517', '10226098', '16909175']


    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = os.path.join(output_dir, 'is_the_x_relation_the_y_change_relation')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.can_produce_only_one_sample_for_match = True


    def augment_question_template(self):
        """ Example:
        ('Is the couch to the right of the pillow on the left side of the photo?', 'yes') ->
        ('Is the couch to the right of the shelf?', 'no')
        """

        # is the x relation to the y, we want to change the relation
        if self.answer == "yes":
            # if the relation holds, the opposite relation won't
            relation_replacement = self.opposite_relations[self.relation]
            answer_replacement = "no"
        elif self.answer == "no":
            global reached_no_num
            reached_no_num += 1
            if reached_no_num % 100 == 0:
                print(f"reached_no_num: {reached_no_num}")
            # if the relation does not hold, which relation does hold?
            relation_replacement = self.handle_no_answer_find_relations_that_do_hold()
            answer_replacement = "yes"
            if relation_replacement is None:
                return None
        else:
            raise Exception(f"no such answer: {self.answer}")

        question_replacement = self.build_question_replacement(relation_replacement)
        question_replacement = question_replacement.replace("  ", " ")
        return (question_replacement, answer_replacement)

    def handle_no_answer_find_relations_that_do_hold(self):
        x_relevant_objects = [v for v in self.scene_graph_objects if self.objects_are_the_same(self.x, v['name'])]
        y_relevant_objects = [v for v in self.scene_graph_objects if self.objects_are_the_same(self.y, v['name'])]
        if len(x_relevant_objects) == 0 or len(y_relevant_objects) == 0:
            return None
        else:
            x_obj = x_relevant_objects[0]
            y_obj = y_relevant_objects[0]

            x_relations = {self.scene_graph_for_image_id['objects'][x['object']]['name']: x['name'] for x in
                           x_obj['relations']}
            y_relations = {self.scene_graph_for_image_id['objects'][y['object']]['name']: y['name'] for y in
                           y_obj['relations']}

            relation_replacement = None
            if self.y in x_relations:
                relation_cand = x_relations[self.y]
                if self.object_in_known_relations(relation_cand):
                    relation_replacement = relation_cand
            if relation_replacement is None and y_obj['name'] in x_relations:
                relation_cand = x_relations[y_obj['name']]
                if self.object_in_known_relations(relation_cand):
                    relation_replacement = relation_cand
            if relation_replacement is None and self.x in y_relations:
                if y_relations[self.x] in self.opposite_relations:
                    relation_cand = self.opposite_relations[y_relations[self.x].replace(" of", "")] + " of"
                    if self.object_in_known_relations(relation_cand):
                        relation_replacement = relation_cand
            if relation_replacement is None and x_obj['name'] in y_relations:
                if y_relations[x_obj['name']] in self.opposite_relations:
                    relation_cand = self.opposite_relations[y_relations[x_obj['name']].replace(" of", "")] + " of"
                    if self.object_in_known_relations(relation_cand):
                        relation_replacement = relation_cand
            if not relation_replacement:
                print(
                    f"Couldn't find relation between x, y: {self.x, self.y}, x_relations: {x_relations}, y_relations: {y_relations}")
                return None

            if self.relation in relation_replacement or relation_replacement in self.relation:
                raise Exception(
                    f"Bad relation exchange: {self.relation, relation_replacement}, question: {self.question}, "
                    f"qid: {self.qa_key}, image_id: {self.image_id}")
        return relation_replacement

    def object_in_known_relations(self, obj):
        return any(obj in rel for rel in self.relations) or any(rel in obj for rel in self.relations)

    def build_question_replacement(self, relation_replacement):
        is_or_are = self.reg_exp.split(' ')[0]
        question_replacement = f"{is_or_are} the {self.x} {relation_replacement}"
        if self.question_connector != 'of' and relation_replacement in ['in front', 'on top']:
            question_replacement += " of"
        if relation_replacement not in ['behind', 'below'] and self.y_a_or_an_or_the_or_of:
            question_replacement += f" {self.y_a_or_an_or_the_or_of}"
        if self.question_connector:
            question_replacement += f" {self.question_connector}"
        question_replacement += f" {self.y}"
        if self.y_indirect and self.y_indirect != "":
            question_replacement += f" {self.y_indirect}"
        if not self.y_indirect and self.y_suffix and self.y_suffix != "":
            question_replacement += f" {self.y_suffix}"
        question_replacement += "?"
        if ' of of ' in question_replacement:
            question_replacement = question_replacement.replace(' of of ', ' of ')
        return question_replacement

    def test_augmentation(self, other):
        assert self.answer != other.answer
        relation_replacement = self.opposite_relations[self.relation]
        if relation_replacement == 'in front':
            relation_replacement += " of"
        other_question_by_replacing_rel = self.question.replace(self.relation, relation_replacement)
        if self.relation == 'in front':
            quest_words = self.question.split(" ")
            front_index = quest_words.index('front')
            repl_words = other_question_by_replacing_rel.split(" ")
            other_question_by_replacing_rel = (" ".join(repl_words[:front_index + 1]) + " ").replace(" of ", " ") + \
                                              " ".join(repl_words[front_index + 1:])
        assert other_question_by_replacing_rel == other.question
        return True

