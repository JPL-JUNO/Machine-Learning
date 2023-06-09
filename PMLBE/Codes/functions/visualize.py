"""
@Description: functions used to visualize
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-06-01 09:24:46
"""

CONDITION = {'numerical': {'yes': '>=', 'no': '<'},
             'categorical': {'yes': 'is', 'no': 'is not'}}


def visualize_tree(node: dict, depth: int):
    if isinstance(node, dict):
        if type(node['value']) in [int, float]:
            condition = CONDITION['numerical']
        else:
            condition = CONDITION['categorical']

        print('{}|- X{} {} {}'.format(depth * '  ',
              node['index'] + 1, condition['no'], node['value']))
        if 'left' in node:
            visualize_tree(node['left'], depth + 1)

        print('{}|- X{} {} {}'.format(depth * '  ',
              node['index'] + 1, condition['no'], node['value']))
        if 'right' in node:
            visualize_tree(node['right'], depth + 1)
    else:
        print('{}[{}]'.format(depth * '  ', node))


if __name__ == 'main':
    pass
