import json
import os
import pickle
from collections import Counter

import pandas as pd

from config import duplicates_dir, experiment_output_dir, output_csv_path, AUGMENT_TRAIN
from question_templates.question_template_factory import get_question_template_constructor

if AUGMENT_TRAIN:
    stats_dict = {'total_images': 72140, 'total_question_answer_pairs': 943000,
                  'total_number_of_augmented_questions': 0, 'total_number_of_augmented_images': 0}  # len(val_scene_graphs), len(balanced_val_data)
else:
    stats_dict = {'total_images': 10696, 'total_question_answer_pairs': 132062,
                  'total_number_of_augmented_questions': 0,
                  'total_number_of_augmented_images': 0}  # len(val_scene_graphs), len(balanced_val_data)

def output_augmentations_and_stats_report(stats_dict, template_class_constructor=None, all_qa_augmentations_df=None, add_reg_exp_match=True):
    if type(all_qa_augmentations_df) != type(None):
        final_csv_path = os.path.join(output_csv_path, f'output_final_{template_class_constructor.__name__}.csv')
        print(f"Wrote final csv: {final_csv_path}")
        all_qa_augmentations_df.to_csv(final_csv_path)
        from mutual_code_train_valid_augmentation.check_intersection_of_augmented_dataset_with_existing import produce_intersection_csv
        produce_intersection_csv(final_csv_path)
        print(f"Wrote intersection csv: {final_csv_path}")

    output_question_templates = [x for x in os.listdir(output_csv_path) if "final" in x]
    all_image_ids_for_subdirs = {}

    if add_reg_exp_match:

        if AUGMENT_TRAIN:
            from work_on_train.augment_gqa_train import get_train_data
            balanced_data, scene_graphs = get_train_data()
        else:
            from work_on_valid.augment_gqa import get_val_data
            balanced_data, scene_graphs = get_val_data()
        answers_dist_for_q_t = {}
        regexp_matches_for_q_t = {}

    subdirs_stats = []
    for p in output_question_templates:
        q_t_name = p.split("final_")[1].split(".csv")[0]
        q_t_df = pd.read_csv(os.path.join(output_csv_path, p))
        q_t_df['augmented_qas'] = q_t_df['augmented_qas'].apply(json.loads)
        success_qa_pairs = q_t_df['augmented_qas'].apply(lambda x: len(x)).sum()
        success_image_ids = set(q_t_df['image_id'])
        all_image_ids_for_subdirs[q_t_name] = success_image_ids

        duplicates_path = os.path.join(duplicates_dir, f"duplicates_output_final_{q_t_name}.csv")
        if os.path.exists(duplicates_path):
            dup_df = pd.read_csv(duplicates_path)
            number_of_duplicates = len(dup_df)
            success_no_dups = success_qa_pairs - number_of_duplicates
        else:
            number_of_duplicates = '?'
            success_no_dups = '?'

        if add_reg_exp_match:
            print(f"q_t_name: {q_t_name}")
            q_t_const = get_question_template_constructor(q_t_name)
            qas_matched_to_regexp = []
            for qid, q_data in balanced_data.items():
                scene_graph_for_image_id = scene_graphs[q_data['imageId']]
                original_template_class = q_t_const(q_data['question'], q_data['answer'], scene_graph_for_image_id, q_data['imageId'], qid)
                success_original = original_template_class.try_to_match_question_to_template()
                if success_original:
                    q_d = {'question': q_data['question'], 'answer': q_data['answer'], 'image_id': q_data['imageId'], 'question_id': qid}
                    qas_matched_to_regexp.append(q_d)

            all_augmented_qids = set([int(x) for x in q_t_df['qa_key'].values])
            all_matched_qids = set([int(x['question_id']) for x in qas_matched_to_regexp])
            assert len([x for x in all_augmented_qids if x not in all_matched_qids]) < 5

            print(f"For {q_t_name}, matches: {len(qas_matched_to_regexp)}")
            regexp_matches_for_q_t[q_t_name] = qas_matched_to_regexp
            print(f"Iterated all q_t data")
            augmented_questions = list(q_t_df['question'].values)
            matched_questions = [x['question'] for x in qas_matched_to_regexp]
            sanity_all_augmented_and_matched = [q for q in augmented_questions if q in matched_questions]
            all_matched_that_are_not_augmented = [x for x in qas_matched_to_regexp if x['question'] not in augmented_questions]
            answers_distribution_of_augmented = Counter(list(q_t_df['answer'].values))
            answers_distribution_of_matched = Counter([x['answer'] for x in qas_matched_to_regexp])
            answers_distribution_of_matched_not_augmented = Counter([x['answer'] for x in all_matched_that_are_not_augmented])
            q_t_dict = {'# Augmented': len(augmented_questions),
                        '# Reg-exp matched': len(matched_questions),
                        '# Augmented that were matched': len(sanity_all_augmented_and_matched),
                        '# Matched that were not augmented': len(all_matched_that_are_not_augmented),
                        'Augmented answers distribution': dict(answers_distribution_of_augmented),
                        'Matched answers distribution': dict(answers_distribution_of_matched),
                        'Matched and not augmented answers distribution': dict(answers_distribution_of_matched_not_augmented)}
            answers_dist_for_q_t[q_t_name] = q_t_dict

        p_stats = {'Question template': q_t_name,
                   '# Success QA pairs': success_qa_pairs,
                   '# Success images': len(success_image_ids),
                   '% Augmented questions': round((success_qa_pairs / stats_dict['total_question_answer_pairs']) * 100, 3),
                   '# QA Duplicates': number_of_duplicates,
                   '# Success QA pairs - No duplicates': success_no_dups,
                   }
        subdirs_stats.append(p_stats)

    if add_reg_exp_match:
        regexp_match_path = os.path.join(experiment_output_dir, 'regexp_matched.pickle')
        print(f"Dumping regexp_match_path: {regexp_match_path}")
        pickle.dump(regexp_matches_for_q_t, open(regexp_match_path, 'wb'))

    answers_dist_df = pd.DataFrame(answers_dist_for_q_t).T
    answers_dist_df.to_csv(os.path.join(experiment_output_dir, 'answers_distribution_report.csv'))

    subdirs_stats_df = pd.DataFrame(subdirs_stats)
    subdirs_stats_df.to_csv(os.path.join(experiment_output_dir, 'subdirs_augmentation_report.csv'))

    total_number_of_success_augmented_qa_pairs = subdirs_stats_df['# Success QA pairs'].sum()

    image_id_sets = list(all_image_ids_for_subdirs.values())
    all_augmented_images = set().union(*image_id_sets)
    total_number_of_success_augmented_images = len(all_augmented_images)

    stats_dict['total_number_of_success_augmented_qa_pairs'] = total_number_of_success_augmented_qa_pairs
    stats_dict['total_number_of_success_augmented_images'] = total_number_of_success_augmented_images
    stats_dict['augmented_success_images_percentage'] = round(
        (total_number_of_success_augmented_images / stats_dict['total_images']) * 100, 3)
    stats_dict['augmented_success_questions_percentage'] = round(
        (total_number_of_success_augmented_qa_pairs / stats_dict['total_question_answer_pairs']) * 100, 3)
    stats_series = pd.Series(stats_dict)
    stats_series.rename({'total_images': '# Images', 'total_question_answer_pairs': '# QA pairs',
                         'total_number_of_success_augmented_qa_pairs': '# Augmented QA pairs',
                         'total_number_of_success_augmented_images': '# Augmented images',
                         'augmented_success_images_percentage': '% Augmented images',
                         'augmented_success_questions_percentage': '% Augmented QA pairs'}, inplace=True)
    for c in ['total_number_of_augmented_questions', 'total_number_of_augmented_images']:
        if c in stats_series.keys():
            stats_series.drop(c, inplace=True)
    stats_series.to_csv(os.path.join(experiment_output_dir, 'total_augmentation_report.csv'))
    print(f'Wrote reports to {experiment_output_dir}\noutput.csv, subdirs_augmentation_report.csv, total_augmentation_report.csv, answers_distribution_report.csv')


if __name__ == '__main__':
    output_augmentations_and_stats_report(stats_dict)
