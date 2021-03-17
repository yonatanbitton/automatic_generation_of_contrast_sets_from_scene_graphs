import argparse
import json
import os
import os.path as osp

import pandas as pd

from config import scene_graphs_dir, questions_dir, number_of_augmentations_to_produce, experiment_output_dir, \
    experiment_output_name
from mutual_code_train_valid_augmentation.produce_augmentation_stats import output_augmentations_and_stats_report
from question_templates.question_template_factory import get_question_template_constructor
from utils.utils import draw_image_details

DEBUG = False
DRAW_IMAGES_INDEX = 0

exception_num = 0

def main(stats_dict):
    balanced_train_data, train_scene_graphs = get_train_data()

    all_qa_augmentations_df = init_csv_from_checkpoint(df_checkpoint_path)

    train_scene_graphs_items = list(train_scene_graphs.items())

    for idx, (image_id, scene_graph) in enumerate(train_scene_graphs_items):
        augmented_questions_for_image, all_qa_augmentations_df = augment_qas_for_image_id(balanced_train_data, image_id, scene_graph, all_qa_augmentations_df, idx)

        if augmented_questions_for_image > 0:
            stats_dict['total_number_of_augmented_images'] += 1

        stats_dict['total_number_of_augmented_questions'] += augmented_questions_for_image

    output_augmentations_and_stats_report(stats_dict, template_class_constructor, all_qa_augmentations_df)
    print(f"Done - {experiment_output_name}, {template_class_constructor.__name__}")


def init_csv_from_checkpoint(df_checkpoint_path):
    if df_checkpoint_path:
        all_qa_augmentations_df = pd.read_csv(df_checkpoint_path)
        for k in ['image_id', 'qa_key']:
            all_qa_augmentations_df[k] = all_qa_augmentations_df[k].astype(str)
    else:
        all_qa_augmentations_df = pd.DataFrame()
    return all_qa_augmentations_df


def debug_single_vid(balanced_train_data, scene_graphs, all_qa_augmentations_df, chosen_image_id, qa_key):
    scene_graph_for_image_id = scene_graphs[chosen_image_id]
    augmented_questions_for_image = augment_qas_for_image_id(balanced_train_data, chosen_image_id,
                                                             scene_graph_for_image_id, all_qa_augmentations_df, idx=0, skip_exists=False, debug_qa_key=qa_key)
    print(f"Finished debug {augmented_questions_for_image}")


def augment_qas_for_image_id(balanced_train_data, image_id, scene_graph_for_image_id, all_qa_augmentations_df, idx, skip_exists=True, debug_qa_key=None):
    questions_for_image_id = {question_key: question_dict for question_key, question_dict in
                              balanced_train_data.items()
                              if question_dict['imageId'] == image_id}
    augmented_questions_for_image = 0
    produced_objects_for_image_id = []

    for qa_key, qa_dict in questions_for_image_id.items():
        if debug_qa_key and qa_key != debug_qa_key and qa_key[1:] != debug_qa_key:
            continue

        if not needs_to_skip(all_qa_augmentations_df, image_id, qa_key, skip_exists):

            augmented_qas, produced_objects_for_image_id = augment_qa(image_id, qa_key, qa_dict['question'], qa_dict['answer'], scene_graph_for_image_id,
                                       idx, produced_objects_for_image_id)

            if len(augmented_qas) > 0:
                all_qa_augmentations_df, augmented_questions_for_image = \
                    add_augmented_qas(all_qa_augmentations_df, augmented_qas, augmented_questions_for_image,
                                      image_id, qa_dict, qa_key)
    return augmented_questions_for_image, all_qa_augmentations_df


def needs_to_skip(all_qa_augmentations_df, image_id, qa_key, skip_exists):
    if not skip_exists:
        return False
    existing_rows = all_qa_augmentations_df.query(
        f'image_id=="{image_id}" and qa_key=="{qa_key}"') if 'image_id' in all_qa_augmentations_df.columns else []
    skipping = len(existing_rows) > 0
    return skipping


def add_augmented_qas(all_qa_augmentations_df, augmented_qas, augmented_questions_for_image, image_id, qa_dict, qa_key):
    qa_augmentations_data = {'image_id': str(image_id), 'qa_key': str(qa_key), 'question': qa_dict['question'],
                             'answer': qa_dict['answer'], 'augmented_qas': json.dumps(augmented_qas)}
    all_qa_augmentations_df = all_qa_augmentations_df.append(qa_augmentations_data, ignore_index=True)
    if (len(all_qa_augmentations_df) > dump_csv_steps and len(all_qa_augmentations_df) % dump_csv_steps == 0) \
            or len(all_qa_augmentations_df) == 10:
        idx_path = os.path.join(output_csv_path, f"output_{len(all_qa_augmentations_df)}_items_{template_class_constructor.__name__}.csv")
        if not os.path.exists(idx_path):
            print(f'Dumping csv to path: {idx_path}')
            all_qa_augmentations_df['qa_key'] = all_qa_augmentations_df['qa_key'].astype(str)
            all_qa_augmentations_df.to_csv(idx_path)
    augmented_questions_for_image += len(augmented_qas)
    return all_qa_augmentations_df, augmented_questions_for_image


def augment_qa(image_id, qa_key, question, answer, scene_graph_for_image_id, idx, produced_objects_for_image_id):
    augmentation_success_lst = []

    template_class = template_class_constructor(question, answer, scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)

    match_success = template_class.try_to_match_question_to_template()

    if match_success:
        for iter in range(number_of_augmentations_to_produce):

            try:
                qa_augmentations = template_class.augment_question_template()
            except Exception as ex:
                global exception_num
                exception_num += 1
                print(f"exception_num: {exception_num}, Exception {str(ex)}, continue")
                continue

            if idx > DRAW_IMAGES_INDEX and qa_augmentations and idx % 100 == 0:
                draw_image(answer, idx, image_id, qa_augmentations, qa_key, question, scene_graph_for_image_id,
                           template_class, iter)

            if qa_augmentations:
                produced_objects_for_image_id = list(set(produced_objects_for_image_id + template_class.produced_objects))
                print(f'idx: {idx}, image_id: {image_id}, qa_key: {qa_key}')
                print(f'{question}, {answer}')
                print(f'{qa_augmentations[0]}, {qa_augmentations[1]}')
                print()

            if qa_augmentations:

                augmented_template_class = template_class_constructor(qa_augmentations[0], qa_augmentations[1],
                                                                         scene_graph_for_image_id, image_id, qa_key, produced_objects_for_image_id)
                augmented_match_success = augmented_template_class.try_to_match_question_to_template()
                if not augmented_match_success:
                    print(f'Sanity - Template class did not match, idx: {idx}, qa_key: {qa_key}, image_id:{image_id}')
                    continue

                aug_dict = {'question_template': template_class_constructor.__name__, 'qa_augmentations': qa_augmentations,
                            'question_template_outputs': template_class.produce_output()}
                augmentation_success_lst.append(aug_dict)

            if template_class.can_produce_only_one_sample_for_match:
                break

    return augmentation_success_lst, produced_objects_for_image_id


def draw_image(answer, idx, image_id, qa_augmentations, qa_key, question, scene_graph_for_image_id, template_class, iter):
    if qa_augmentations:
        output_path = osp.join(template_class.output_dir,
                               f'idx_{idx}_image_id_{image_id}_qa_{qa_key}_iter_{iter}.png')
        draw_image_details(image_id, scene_graph_for_image_id, (question, answer), qa_augmentations,
                           output_path)
    else:
        output_path = osp.join(template_class.output_dir,
                               f'idx_{idx}_image_id_{image_id}_qa_{qa_key}_FAIL.png')
        draw_image_details(image_id, scene_graph_for_image_id, (question, answer),
                           output_path=output_path)



def get_train_data():
    scene_graphs_path = osp.join(scene_graphs_dir, 'train_sceneGraphs.json')
    with open(scene_graphs_path, 'rb') as f:
        train_scene_graphs = json.load(f)
    balanced_data_path = osp.join(questions_dir, "train_balanced_questions.json")
    with open(balanced_data_path) as json_file:
        balanced_train_data = json.load(json_file)
        print("Loaded {} train examples".format(len(balanced_train_data)))
    print(f"Got train data")
    return balanced_train_data, train_scene_graphs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_template", help="question_template")
    args = parser.parse_args()
    template_class_constructor = get_question_template_constructor(args.question_template)
    print(f"template_class_constructor: {template_class_constructor.__name__}")
    output_dir = os.path.join(experiment_output_dir, 'question_templates_outputs_train')
    output_csv_path = os.path.join(output_dir, 'output_files_train')
    if not os.path.exists(output_csv_path):
        os.mkdir(output_csv_path)
    df_checkpoint_path = None
    dump_csv_steps = 100000
    stats_dict = {'total_images': 72140, 'total_question_answer_pairs': 943000,
                  'total_number_of_augmented_questions': 0, 'total_number_of_augmented_images': 0}  # len(val_scene_graphs), len(balanced_val_data)

    print("RUNNING AUGMENTATION ON TRAIN")
    main(stats_dict)
