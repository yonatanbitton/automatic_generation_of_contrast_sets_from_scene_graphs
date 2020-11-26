from question_templates.are_there_x_near_the_y import are_there_x_near_the_y
from question_templates.do_you_see_x_or_y import do_you_see_x_or_y
from question_templates.is_the_x_relation_the_y import is_the_x_relation_the_y_change_x, \
    is_the_x_relation_the_y_change_relation
from question_templates.on_which_side_is_the_x import on_which_side_is_the_x
from question_templates.what_color_is_the_x import what_color_is_the_x


def get_question_template_constructor(question_template_name):
    if question_template_name == 'are_there_x_near_the_y':
        return are_there_x_near_the_y
    elif question_template_name == 'on_which_side_is_the_x':
        return on_which_side_is_the_x
    elif question_template_name == 'is_the_x_relation_the_y_change_x':
        return is_the_x_relation_the_y_change_x
    elif question_template_name == 'is_the_x_relation_the_y_change_relation':
        return is_the_x_relation_the_y_change_relation
    elif question_template_name == 'what_color_is_the_x':
        return what_color_is_the_x
    elif question_template_name == 'do_you_see_x_or_y':
        return do_you_see_x_or_y
    else:
        raise Exception(f'Unknown constructor: {question_template_name}')