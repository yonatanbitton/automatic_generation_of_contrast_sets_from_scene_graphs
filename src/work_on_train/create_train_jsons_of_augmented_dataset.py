import json
import os
import os.path as osp
from collections import defaultdict

import pandas as pd

from config import experiment_output_dir, all_experiments_output_dir, experiment_output_name, \
    number_of_augmentations_to_produce, output_csv_path
from question_templates.are_there_x_near_the_y import are_there_x_near_the_y
from question_templates.do_you_see_x_or_y import do_you_see_x_or_y
from question_templates.is_the_x_relation_the_y import is_the_x_relation_the_y_change_relation, \
    is_the_x_relation_the_y_change_x
from question_templates.on_which_side_is_the_x import on_which_side_is_the_x
from question_templates.what_color_is_the_x import what_color_is_the_x

if number_of_augmentations_to_produce > 1:  # change rel can produce only 1 output per match
    question_templates = [are_there_x_near_the_y, on_which_side_is_the_x, is_the_x_relation_the_y_change_x,
                          what_color_is_the_x, do_you_see_x_or_y]
else:
    question_templates = [do_you_see_x_or_y, are_there_x_near_the_y, on_which_side_is_the_x,
                          is_the_x_relation_the_y_change_relation, is_the_x_relation_the_y_change_x,
                          what_color_is_the_x]

final_augmentation_csv_dir = output_csv_path
all_exp_augmentation_train_dir = osp.join(all_experiments_output_dir, 'all_exp_augmentations_train')
curr_exp_train_dir = osp.join(all_exp_augmentation_train_dir, experiment_output_name)
for d in [all_exp_augmentation_train_dir, curr_exp_train_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
duplicates_dir = osp.join(experiment_output_dir, 'duplicates_train')

def main():
    print(f"Start - {experiment_output_name}")

    stats_data = []

    augmented_data_for_all_q_t = []
    original_data_for_all_q_t = []

    for q_t in question_templates:
        q_t_name = get_question_template_name(q_t)
        augmented_data_for_q_t, original_data_for_q_t, number_of_dups = get_data_for_q_t(q_t_name)
        print(f'Got {len(augmented_data_for_q_t)} augmented data, and {len(original_data_for_q_t)} original data, for {q_t_name}. number_of_dups: {number_of_dups}')
        stats_data.append({'Question template': q_t_name, 'Size contrast set': len(augmented_data_for_q_t), 'Size original set': len(original_data_for_q_t)})

        augmented_q_t_output_json_path = osp.join(curr_exp_train_dir, f'train_{q_t_name}_augmented.json')
        json.dump(augmented_data_for_q_t, open(augmented_q_t_output_json_path, 'w'), indent=2)

        original_q_t_output_json_path = osp.join(curr_exp_train_dir, f'train_{q_t_name}_original.json')
        json.dump(original_data_for_q_t, open(original_q_t_output_json_path, 'w'), indent=2)

        augmented_data_for_all_q_t += augmented_data_for_q_t
        original_data_for_all_q_t += original_data_for_q_t

    stats_data.append({'Question template': 'merged', 'Size contrast set': len(augmented_data_for_all_q_t),
                       'Size original set': len(original_data_for_all_q_t)})
    stats_df = pd.DataFrame(stats_data)
    stats_path = osp.join(curr_exp_train_dir, f'stats_{experiment_output_name}.csv')
    print(f"wrote stats to path: {stats_path}")
    stats_df.to_csv(stats_path, index=False)

    print(f'Got all data df at len: {len(augmented_data_for_all_q_t)}')
    augmented_q_t_output_json_path = osp.join(curr_exp_train_dir, f'train_augmentation_all_augmented.json')
    json.dump(augmented_data_for_all_q_t, open(augmented_q_t_output_json_path, 'w'), indent=2)

    original_q_t_output_json_path = osp.join(curr_exp_train_dir, f'train_augmentation_all_original.json')
    json.dump(original_data_for_all_q_t, open(original_q_t_output_json_path, 'w'), indent=2)

    print("Done")


def get_data_for_q_t(q_t_name):
    duplicates_path = osp.join(duplicates_dir, f"duplicates_output_final_{q_t_name}.csv")
    dup_df = pd.read_csv(duplicates_path)
    dup_df['augmented_qa'] = dup_df['augmented_qa'].apply(json.loads)

    aug_path = osp.join(final_augmentation_csv_dir, f'output_final_{q_t_name}.csv')
    df = pd.read_csv(aug_path)
    augmentation_data_for_q_t = []
    original_data_for_q_t = []

    number_of_dups = 0
    dups_in_questions_for_image_id = 0

    all_questions_for_image_id = defaultdict(list)

    for row_idx, row in df.iterrows():

        augmented_qas = json.loads(row['augmented_qas'])
        got_item_for_row = False
        for qa_aug_idx, qa_aug in enumerate(augmented_qas):

            row_data_in_dup_df = dup_df.apply(lambda x: same_image_id_and_augmented_qa(x, row["image_id"], qa_aug["qa_augmentations"]), axis=1)
            if sum(row_data_in_dup_df.values) > 0:
                number_of_dups += 1
                continue

            got_item_for_row = True
            question_id = f"{q_t_name}_row_idx_{row_idx}_qa_aug_idx_{qa_aug_idx}"
            qa_augmentations = qa_aug['qa_augmentations']
            sent = qa_augmentations[0]
            label = {qa_augmentations[1]: 1.0}
            qa_aug_dict = {'img_id': str(row['image_id']), 'label': label, 'question_id': question_id, 'sent': sent}
            if sent not in all_questions_for_image_id[row['image_id']]:
                augmentation_data_for_q_t.append(qa_aug_dict)
                qa_aug_dict['original_question_id'] = row['qa_key']
                all_questions_for_image_id[row['image_id']].append(sent)
            else:
                dups_in_questions_for_image_id += 1
                print(f"{dups_in_questions_for_image_id} - dups_in_questions_for_image_id - Found potential dup, {sent}, {all_questions_for_image_id[row['image_id']]}")

        if got_item_for_row:
            orig_label = {row['answer']: 1.0}
            original_qa_dict = {'img_id': str(row['image_id']), 'label': orig_label, 'question_id': str(row['qa_key']), 'sent': row['question'], 'original_question_id': row['qa_key']}
            original_data_for_q_t.append(original_qa_dict)

    return augmentation_data_for_q_t, original_data_for_q_t, number_of_dups

def same_image_id_and_augmented_qa(x, image_id, qa_aug):
    if x['image_id'] == image_id and x['augmented_qa'] == qa_aug:
        return True
    return False

def get_question_template_name(q_t):
    return q_t.__doc__

if __name__ == '__main__':
    main()