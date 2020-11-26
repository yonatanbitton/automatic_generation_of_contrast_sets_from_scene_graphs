import json
import os
import os.path as osp
import random
import shutil
from collections import defaultdict
from copy import deepcopy

import pandas as pd

from config import data_dir, experiment_output_dir, all_experiments_output_dir, experiment_output_name, \
    number_of_augmentations_to_produce
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
    question_templates = [what_color_is_the_x, are_there_x_near_the_y, do_you_see_x_or_y, on_which_side_is_the_x,
                          is_the_x_relation_the_y_change_relation, is_the_x_relation_the_y_change_x,
                          ]

final_augmentation_csv_dir = osp.join(experiment_output_dir, 'question_templates_outputs', 'output_files')
all_exp_augmentation_testdev_dir = osp.join(all_experiments_output_dir, 'all_exp_augmentations_testdev')
curr_exp_testdev_dir = osp.join(all_exp_augmentation_testdev_dir, experiment_output_name)
for d in [all_exp_augmentation_testdev_dir, curr_exp_testdev_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
duplicates_dir = osp.join(experiment_output_dir, 'duplicates')

def main(mTurk_process=False):
    print(f"Start - {experiment_output_name}, mTurk_process: {mTurk_process}")

    stats_data = []

    augmented_data_for_all_q_t = []
    original_data_for_all_q_t = []

    all_imgs_ids_to_sample = []
    all_sampled_size = 0

    for q_t in question_templates:
        q_t_name = get_question_template_name(q_t)
        augmented_data_for_q_t, original_data_for_q_t, number_of_dups = get_data_for_q_t(q_t_name)
        print(f'Got {len(augmented_data_for_q_t)} augmented data, and {len(original_data_for_q_t)} original data, for {q_t_name}. number_of_dups: {number_of_dups}')
        stats_data.append({'Question template': q_t_name, 'Size contrast set': len(augmented_data_for_q_t), 'Size original set': len(original_data_for_q_t)})

        augmented_q_t_output_json_path = osp.join(curr_exp_testdev_dir, f'testdev_{q_t_name}_augmented.json')
        json.dump(augmented_data_for_q_t, open(augmented_q_t_output_json_path, 'w'), indent=2)

        original_q_t_output_json_path = osp.join(curr_exp_testdev_dir, f'testdev_{q_t_name}_original.json')
        json.dump(original_data_for_q_t, open(original_q_t_output_json_path, 'w'), indent=2)

        augmented_data_for_all_q_t += augmented_data_for_q_t
        print(f"After adding {len(augmented_data_for_q_t)}, now has: {len(augmented_data_for_all_q_t)}")
        original_data_for_all_q_t += original_data_for_q_t

        if mTurk_process:
            q_t_imgs_ids, q_t_sampled_size = export_sample(deepcopy(augmented_data_for_q_t), deepcopy(original_data_for_q_t), q_t_name)
            all_imgs_ids_to_sample += q_t_imgs_ids
            all_sampled_size += q_t_sampled_size
        print(f"Finished with: {q_t_name}")

    print(f"Finished with question templates")

    if mTurk_process:
        images_dir = os.path.join(data_dir, 'images')
        out_images_dir = os.path.join(data_dir, 'mTurk', 'images')
        if not os.path.exists(out_images_dir):
            os.mkdir(out_images_dir)
        print(f"Should be: {len(set(all_imgs_ids_to_sample))} images")
        for img_id in all_imgs_ids_to_sample:
            src_img_path = os.path.join(images_dir, img_id + '.jpg')
            tgt_img_path = os.path.join(out_images_dir, img_id + '.png')
            shutil.copy(src_img_path, tgt_img_path)
        print(f"After copy: {len(set(os.listdir(out_images_dir)))}")
        print(f"all_sampled_size: {all_sampled_size}")

    stats_data.append({'Question template': 'merged', 'Size contrast set': len(augmented_data_for_all_q_t),
                       'Size original set': len(original_data_for_all_q_t)})
    stats_df = pd.DataFrame(stats_data)
    stats_path = osp.join(curr_exp_testdev_dir, f'stats_{experiment_output_name}.csv')
    print(f"wrote stats to path: {stats_path}")
    stats_df.to_csv(stats_path, index=False)

    print(f'Got all data df at len: {len(augmented_data_for_all_q_t)}')
    augmented_q_t_output_json_path = osp.join(curr_exp_testdev_dir, f'testdev_augmentation_all_augmented.json')
    json.dump(augmented_data_for_all_q_t, open(augmented_q_t_output_json_path, 'w'), indent=2)

    original_q_t_output_json_path = osp.join(curr_exp_testdev_dir, f'testdev_augmentation_all_original.json')
    json.dump(original_data_for_all_q_t, open(original_q_t_output_json_path, 'w'), indent=2)

    print("Done")


def export_sample(augmented_data_for_q_t, original_data_for_all_q_t, q_t_name):
    aug_df = pd.DataFrame(augmented_data_for_q_t)
    aug_df['label'] = aug_df['label'].apply(lambda x: list(x.keys())[0])
    orig_df = pd.DataFrame(original_data_for_all_q_t)
    orig_df['label'] = orig_df['label'].apply(lambda x: list(x.keys())[0])
    sample_pct = 0.07 if q_t_name != "what_color_is_the_x" else 0.02
    sample_num = int(len(aug_df) * sample_pct)
    random.seed(42)
    joint_ids_to_sample = random.sample(list(aug_df['joint_id']), sample_num)
    all_possible_answers = list(set(orig_df['label'].values).union(set(aug_df['label'].values)))
    all_possible_answers = sorted(all_possible_answers)
    aug_df_sample = aug_df[aug_df['joint_id'].isin(joint_ids_to_sample)]
    orig_df_sample = orig_df[orig_df['joint_id'].isin(joint_ids_to_sample)]
    assert list(orig_df_sample['joint_id']) == list(aug_df_sample['joint_id'])
    df_sample_concat = pd.concat([aug_df_sample, orig_df_sample])
    df_sample_concat['all_possible_answers'] = [all_possible_answers for _ in range(len(df_sample_concat))]
    df_sample_concat['question'] = df_sample_concat['sent']
    df_sample_concat.drop(columns=['sent'], inplace=True)
    out_p = os.path.join(data_dir, 'mTurk', f'{q_t_name}_sample.csv')
    sampled_size = len(aug_df_sample)
    print(
        f'Writing sample of size: {sampled_size} each, total of {len(df_sample_concat)}, sampled from df of {len(aug_df)}')
    df_sample_concat.to_csv(out_p)
    q_t_imgs_ids = list(set(df_sample_concat['img_id']))
    outputs_dir = os.path.join(data_dir, 'all_experiments', 'first_exp_1_outputs_for_match',
                               'question_templates_outputs', q_t_name)
    mTurk_vis_dir = os.path.join(data_dir, 'mTurk', 'vis_dir')
    q_t_mTurk_vis_dir = os.path.join(data_dir, 'mTurk', 'vis_dir', q_t_name)
    if not os.path.exists(mTurk_vis_dir):
        os.mkdir(mTurk_vis_dir)
    if not os.path.exists(q_t_mTurk_vis_dir):
        os.mkdir(q_t_mTurk_vis_dir)
    missing_files = []
    for qid in list(orig_df_sample['question_id']):
        relevant_files = [x for x in os.listdir(outputs_dir) if int(qid) == int(x.split("qa_")[1].split("_iter")[0])]
        if len(relevant_files) < 1:
            raise Exception(f"How to proceed? no image. img_id: {qid}")
        relevant_file = relevant_files[0]
        src_qid_image = os.path.join(outputs_dir, relevant_file)
        assert os.path.exists(src_qid_image)
        tgt_qid_image = os.path.join(q_t_mTurk_vis_dir, relevant_file)
        shutil.copy(src_qid_image, tgt_qid_image)

    print(f"{q_t_name} missing files: {len(missing_files)}")
    print(missing_files)

    return q_t_imgs_ids, sampled_size


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

    for joint_id, (row_idx, row) in enumerate(df.iterrows()):

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
                qa_aug_dict['joint_id'] = joint_id
                qa_aug_dict['original_question_id'] = row['qa_key']
                augmentation_data_for_q_t.append(qa_aug_dict)
                all_questions_for_image_id[row['image_id']].append(sent)
            else:
                dups_in_questions_for_image_id += 1
                print(f"{dups_in_questions_for_image_id} - dups_in_questions_for_image_id - Found potential dup, {sent}, {all_questions_for_image_id[row['image_id']]}")

        if got_item_for_row:
            orig_label = {row['answer']: 1.0}
            original_qa_dict = {'img_id': str(row['image_id']), 'label': orig_label, 'question_id': str(row['qa_key']), 'sent': row['question'], 'joint_id': joint_id, 'original_question_id': row['qa_key']}
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