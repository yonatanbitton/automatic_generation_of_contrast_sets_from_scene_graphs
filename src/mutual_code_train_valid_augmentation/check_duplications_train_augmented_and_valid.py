import os

from config import data_dir
from mutual_code_train_valid_augmentation.check_intersection_of_augmented_dataset_with_existing import \
    produce_intersection_csv

output_files_train = os.path.join(data_dir, 'all_experiments_train/first_exp_1_outputs_for_match_train/question_templates_outputs_train/output_files_train')
output_files_valid = os.path.join(data_dir, 'all_experiments/first_exp_1_outputs_for_match/question_templates_outputs/output_files')


def main():
    question_template_name = 'are_there_x_near_the_y'
    print(f"Running, question_template_name: {question_template_name}")

    output_file_from_train = os.path.join(output_files_train, f'output_final_{question_template_name}.csv')
    produce_intersection_csv(output_file_from_train, intersection_train_valid=True, intersection_valid_train=False)

    output_file_from_valid = os.path.join(output_files_valid, f'output_final_{question_template_name}.csv')
    produce_intersection_csv(output_file_from_valid, intersection_train_valid=False, intersection_valid_train=True)
    print("Finished")

if __name__ == '__main__':
    main()