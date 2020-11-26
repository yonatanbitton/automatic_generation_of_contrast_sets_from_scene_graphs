import json
import os

import pandas as pd

from config import duplicates_dir, AUGMENT_TRAIN
from question_templates.question_template_factory import get_question_template_constructor
from work_on_train.augment_gqa_train import get_train_data
from work_on_valid.augment_gqa import get_val_data


def produce_intersection_csv(df_path, intersection_train_valid=False, intersection_valid_train=False):
    csv_name = df_path.split('/')[-1]
    print(f"produce_intersection_csv. Running {csv_name}")
    df = get_augmented_df(df_path)
    balanced_data, scene_graphs = get_balanced_data(intersection_train_valid, intersection_valid_train)

    all_image_ids = set(df['image_id'].values)

    intersection_for_image_id = {}
    for idx, image_id in enumerate(all_image_ids):
        if idx % 100 == 0:
            print(f'{idx}/{len(all_image_ids)}')
        existing_questions_for_image_id = {question_key: question_dict for question_key, question_dict in
                                           balanced_data.items()
                                           if question_dict['imageId'] == str(image_id)}

        augmented_questions_for_image_id = df[df['image_id'] == str(image_id)]

        scene_graph_for_image_id = scene_graphs[image_id]

        questions_intersection = check_questions_intersection(existing_questions_for_image_id,
                                                              augmented_questions_for_image_id, idx,
                                                              image_id, scene_graph_for_image_id)
        if len(questions_intersection) > 0:
            intersection_for_image_id[image_id] = questions_intersection

    data_dups_lst = []
    for image_id, image_id_dups in intersection_for_image_id.items():
        for dup in image_id_dups:
            image_id_d = {**{'image_id': image_id}, **dup}
            data_dups_lst.append(image_id_d)
    data_dups_df = pd.DataFrame(data_dups_lst, columns=['image_id', 'original_question', 'augmented_qa',
                                                        'existing_similar_qa', 'existing_qa_key'])
    if 'augmented_qa' in data_dups_df.columns:
        data_dups_df['augmented_qa'] = data_dups_df['augmented_qa'].apply(json.dumps)
    dup_path = os.path.join(duplicates_dir, f"duplicates_{csv_name}")
    data_dups_df.to_csv(dup_path, index=False)

    print(f"Number of intersections: {len(intersection_for_image_id)}, for: {csv_name}")
    print(f"Done - check intersection, wrote dup csv: {dup_path}")


def get_balanced_data(intersection_train_valid, intersection_valid_train):
    if not intersection_train_valid and not intersection_valid_train:
        if AUGMENT_TRAIN:
            balanced_data, scene_graphs = get_train_data()
        else:
            balanced_data, scene_graphs = get_val_data()
    elif intersection_train_valid:
        balanced_data, scene_graphs = get_val_data()
    elif intersection_valid_train:
        balanced_data, scene_graphs = get_train_data()
    else:
        raise Exception(f"No such split")
    return balanced_data, scene_graphs


def check_questions_intersection(existing_questions_for_image_id, augmented_questions_for_image_id, idx,
                                 image_id, scene_graph_for_image_id):
    if len(augmented_questions_for_image_id) == 0:
        return []
    else:
        intersection_list = []
        for row_idx, row in augmented_questions_for_image_id.iterrows():
            if row['image_id'] == '2385456' and row['qa_key'] == '352847':
                continue
            for augmented_qa_idx, augmented_qa in enumerate(row['augmented_qas']):
                augmented_answer, augmented_question, augmented_template_class, original_question, question_template_constructor = init_augmented_question(
                    augmented_qa, augmented_qa_idx, idx, image_id, row, row_idx, scene_graph_for_image_id)

                for existing_qa_key, existing_qa_dict in existing_questions_for_image_id.items():
                    existing_match_success, existing_similar_question, existing_similar_answer, existing_template_class = try_to_init_existing_question(
                        existing_qa_dict, existing_qa_key, image_id, question_template_constructor,
                        scene_graph_for_image_id)

                    if existing_match_success:

                        existing_init_after_match_success = existing_template_class.init_after_match() if hasattr(existing_template_class, 'init_after_match') else True

                        if existing_init_after_match_success:
                            try_to_match(augmented_answer, augmented_question, augmented_template_class, existing_similar_question, existing_similar_answer,
                 existing_template_class, intersection_list, original_question, image_id, existing_qa_key)
        return intersection_list


def try_to_match(augmented_answer, augmented_question, augmented_template_class, existing_similar_question, existing_similar_answer,
                 existing_template_class, intersection_list, original_question, image_id, existing_qa_key):
    question_are_matched = existing_template_class.questions_classes_are_equivalent(augmented_template_class)
    if question_are_matched:
        print(f'Found intersection - image_id: {image_id}, existing_qa_key: {existing_qa_key}:\n{augmented_question}\n{existing_similar_question}\n\n')
        intersection_dict = {'original_question': original_question,
                             'augmented_qa': (augmented_question, augmented_answer),
                             'existing_similar_qa': (existing_similar_question, existing_similar_answer),
                             'image_id': image_id,
                             'existing_qa_key': existing_qa_key
                             }
        intersection_list.append(intersection_dict)


def try_to_init_existing_question(existing_qa_dict, existing_qa_key, image_id, question_template_constructor,
                                  scene_graph_for_image_id):
    existing_similar_question = existing_qa_dict['question'].replace("?", "")
    existing_similar_answer = existing_qa_dict['answer']
    existing_template_class = question_template_constructor(existing_similar_question, existing_similar_answer,
                                                            scene_graph_for_image_id, image_id, existing_qa_key)
    existing_match_success = existing_template_class.try_to_match_question_to_template()
    return existing_match_success, existing_similar_question, existing_similar_answer, existing_template_class


def init_augmented_question(augmented_qa, augmented_qa_idx, idx, image_id, row, row_idx, scene_graph_for_image_id):
    original_question = row['question']
    augmented_question, augmented_answer = augmented_qa['qa_augmentations']

    question_template_constructor = get_question_template_constructor(augmented_qa['question_template'])
    augmented_template_class = question_template_constructor(augmented_question, augmented_answer,
                                                             scene_graph_for_image_id, image_id, row['qa_key'])
    augmented_match_success = augmented_template_class.try_to_match_question_to_template()
    if not augmented_match_success:
        raise Exception(
            f'Template class did not match, idx: {idx}, row_idx: {row_idx}, augmented_qa_idx: {augmented_qa_idx},\noriginal_question: {original_question} augmented_question: {augmented_question}, image_id:{image_id}, qa_key: {row["qa_key"]}')
    augmented_init_after_match_success = augmented_template_class.init_after_match() if hasattr(
        augmented_template_class, 'init_after_match') else True
    if not augmented_init_after_match_success:
        raise Exception(
            f'augmented_init_after_match_success Template class did not match, idx: {idx}, row_idx: {row_idx}, augmented_qa_idx: {augmented_qa_idx}')
    return augmented_answer, augmented_question, augmented_template_class, original_question, question_template_constructor


def get_augmented_df(df_path):
    df = pd.read_csv(df_path)
    for k in ['qa_key', 'image_id']:
        df[k] = df[k].astype(str)
    df['augmented_qas'] = df['augmented_qas'].apply(lambda x: json.loads(x))
    return df
