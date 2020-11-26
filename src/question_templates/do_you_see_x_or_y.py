import os
import os.path as osp
import random

from config import output_dir
from question_templates.question_templates import question_template
from utils.utils import distribution_for_each_scene_graph_object, not_replaceable_objects, words_similarity, \
    REPLACEMENT_SIMILARITY_THRESHOLD, isplural, vowels, existing_items_in_val


class do_you_see_x_or_y(question_template):
    """do_you_see_x_or_y"""
    regular_expressions = [r"Do you see (.*) or (.*)"]

    IMAGE_ID_EXCEPTION_LIST = []
    QA_KEY_EXCEPTION_LIST = ['171046198', '00627571', '01951355']

    def __init__(self, question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id=None):
        super().__init__(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
        self.output_dir = osp.join(output_dir, 'do_you_see_x_or_y')
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def get_atoms(self):
        return (self.x, self.y)

    def clean_txt(self, g):
        g = g.replace("?","")
        g_words = g.split(" ")
        if len(g_words) == 1:
            return g
        bad_words = ['either', 'any', 'a']
        g_clean_txt = " ".join([w for w in g_words if w not in bad_words])
        return g_clean_txt

    def init_after_match(self):
        assert len(self.groups) >= 2
        question_no_qmark = self.question.replace("?","")
        clean_groups = [self.clean_txt(g) for g in self.groups]

        # get y suffix
        possible_suffixes = ['in the image', 'in the picture', 'in the photo', 'in the scene', 'in the photograph',
                             'in this image', 'in this picture', 'in this photo', 'in this scene', 'in this photograph',
                             'there']
        self.y_suffix = None
        for suff in possible_suffixes:
            if question_no_qmark.endswith(suff):
                self.y_suffix = suff

        self.x_section = clean_groups[0].strip()
        self.y_section = clean_groups[1].strip()

        bad_phrases_in_y = ['that are', 'that is']
        for p in bad_phrases_in_y:
            if p in self.y_section:
                return False

        if self.y_suffix:
            self.y_section = self.y_section.replace(self.y_suffix, "").strip()

        x_section_words = self.x_section.split(" ")
        y_section_words = self.y_section.split(" ")
        if x_section_words[0] in ['a', 'an']:
            x_section_words = x_section_words[1:]
        if y_section_words[0] in ['a', 'an']:
            y_section_words = y_section_words[1:]
        self.x = " ".join(x_section_words)
        self.y = " ".join(y_section_words)

        self.scene_graph_objects = self.scene_graph_for_image_id['objects']

        self.x_objects = [v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, self.x)]
        self.y_objects = [v for v in self.scene_graph_objects_names if self.objects_are_the_same(v, self.y)]

        self.x_exists = len(self.x_objects) > 0
        self.y_exists = len(self.y_objects) > 0

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

        if self.answer == 'no':
            # no x and no y. I can sample z that exists instead x or y.
            object_to_replace = random.sample([self.x, self.y], 1)[0]
            replaced_object_plural = isplural(object_to_replace)
            if replaced_object_plural:
                objects_to_sample = [v for v in self.scene_graph_objects_names if v not in self.produced_objects and v not in [self.x, self.y] and isplural(v)]
            else:
                objects_to_sample = [v for v in self.scene_graph_objects_names if v not in self.produced_objects and v not in [self.x, self.y] and not isplural(v)]
            if len(objects_to_sample) == 0:
                objects_to_sample = [v + "s" for v in self.scene_graph_objects_names if v not in self.produced_objects and v not in [self.x, self.y] and v + "s" in existing_items_in_val]
                if len(objects_to_sample) == 0:
                    return None
            z_that_exists = random.sample(objects_to_sample, 1)[0]
            self.produced_objects.append(object_to_replace)
            augmented_question = self.question.replace(" " + object_to_replace, " " + z_that_exists)
            augmented_question = self.fix_an_or_a_if_needed(augmented_question, z_that_exists)
            augmented_answer = 'yes'
        elif self.answer == 'yes':
            # there is x or y. I need to 'turn off' the thing that exists.
            augmented_question = self.question
            if not self.x_exists and not self.y_exists:
                raise Exception(f'No X and no Y {self.image_id}, {self.qa_key}, {self.question}, {self.scene_graph_objects_names}')
            if self.x_exists:
                x_replacement = self.get_new_correlated_object(self.x, self.y)
                if not x_replacement:
                    return None
                augmented_question = augmented_question.replace(" " + self.x, " " + x_replacement)
                augmented_question = self.fix_an_or_a_if_needed(augmented_question, x_replacement)
            if self.y_exists:
                y_replacement = self.get_new_correlated_object(self.y, self.x)
                if not y_replacement:
                    return None
                augmented_question = augmented_question.replace(" " + self.y, " " + y_replacement)
                augmented_question = self.fix_an_or_a_if_needed(augmented_question, y_replacement)
            augmented_answer = 'no'
        else:
            raise Exception("No such answer")

        return (augmented_question, augmented_answer)

    def fix_a_or_an_if_existed_answer_yes(self, augmented_question, x_replacement):
        question_words = augmented_question.split(" ")
        if question_words[3] == 'a' or question_words[3] == 'an':  # only if existed - fix a or an
            if isplural(x_replacement):  # plural case - remove if existed
                augmented_question = " ".join(question_words[:3] + question_words[4:])
            elif not isplural(x_replacement):
                if any(x_replacement.lower().startswith(v) for v in vowels):
                    to_add = "an"
                else:
                    to_add = "a"
                augmented_question = " ".join(question_words[:3]) + " " + to_add + " " + " ".join(
                    question_words[4:])
        return augmented_question


    def fix_an_or_a_if_needed(self, augmented_question, chosen_object_replacement):
        question_words = augmented_question.replace("?", "").split(" ")
        if 'any' in question_words:
            return augmented_question

        split_by_or = augmented_question.split(" or ")
        if chosen_object_replacement in split_by_or[0]:
            part_to_replace = split_by_or[0]
        else:
            part_to_replace = split_by_or[1]

        words_in_part_to_replace = part_to_replace.split(" ")
        if 'a' not in words_in_part_to_replace and 'an' not in words_in_part_to_replace:
            index_of_a_an = -1
            return augmented_question

        if 'a' in words_in_part_to_replace:
            index_of_a_an = words_in_part_to_replace.index('a')
        elif 'an' in question_words:
            index_of_a_an = words_in_part_to_replace.index('an')

        if any(chosen_object_replacement.lower().startswith(v) for v in vowels):
            words_in_part_to_replace[index_of_a_an] = 'an'
        else:
            words_in_part_to_replace[index_of_a_an] = 'a'
        part_to_replace_fixed = " ".join(words_in_part_to_replace)

        if chosen_object_replacement in split_by_or[0]:
            augmented_question = part_to_replace_fixed + " or " + split_by_or[1]
        else:
            augmented_question = split_by_or[0] + " or " + part_to_replace_fixed

        return augmented_question

    def questions_classes_are_equivalent(self, other):
        question_are_matched = self.objects_are_the_same(self.x, other.x) and \
                               self.objects_are_the_same(self.y, other.y)
        return question_are_matched


    def test_augmentation(self, other):
        assert self.answer != other.answer
        index_of_or = other.question.index(' or ')
        part_till_or = other.question[:index_of_or]
        part_after_or = other.question[index_of_or:]
        if not isplural(self.x):
            if any(self.x.lower().startswith(v) for v in vowels):
                part_till_or = part_till_or.replace(" an ", " ")
            else:
                part_till_or = part_till_or.replace(" a ", " ")
        if not isplural(self.y):
            if any(self.y.lower().startswith(v) for v in vowels):
                part_after_or = part_after_or.replace(" an ", " ")
            else:
                part_after_or = part_after_or.replace(" a ", " ")
        part_till_or_replace_x = part_till_or.replace(" " + other.x, " " + self.x)
        part_after_or_replace_y = part_after_or.replace(" " + other.y, " " + self.y)
        other_question_by_replace = part_till_or_replace_x + part_after_or_replace_y

        assert other_question_by_replace == self.question.replace(" a " , " ").replace(" an ", " ")
        return True

    def get_new_correlated_object(self, item, other_item):
        objects_in_image_related_to_item = [v['name'] for v in self.scene_graph_objects.values() if
                                         self.objects_are_the_same(v['name'], item)]
        item_related_obj = objects_in_image_related_to_item[0]
        dist_for_item = distribution_for_each_scene_graph_object[item_related_obj]
        if isplural(item):
            dist_for_item_same_plurality = {k:v for k,v in dist_for_item.items() if isplural(k)}
        else:
            dist_for_item_same_plurality = {k: v for k, v in dist_for_item.items() if not isplural(k)}
        objects_to_delete_from_dist = objects_in_image_related_to_item + not_replaceable_objects + self.produced_objects + [item] + [other_item] + self.scene_graph_objects_names
        for obj in objects_to_delete_from_dist:
            if obj in dist_for_item_same_plurality:
                del dist_for_item_same_plurality[obj]
            similar_objs_in_dist = [k for k in dist_for_item_same_plurality.keys()
                                    if words_similarity(k, obj) >= REPLACEMENT_SIMILARITY_THRESHOLD
                                    or self.objects_are_the_same(k, obj)]
            for k in similar_objs_in_dist:
                del dist_for_item_same_plurality[k]

        if len(dist_for_item_same_plurality) == 0:
            return None

        weights = list(dist_for_item_same_plurality.values())

        chosen_replacement = random.choices(population=list(dist_for_item_same_plurality.keys()), weights=weights, k=1)[0]
        self.produced_objects.append(chosen_replacement)


        return chosen_replacement

