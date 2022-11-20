import os
import pandas as pd
import shutil

import beta_utils


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'demo_beta_text.xlsx')
df = pd.read_excel(
    data_file_str, names=['label', 'text'], engine='openpyxl')
print(df)
print(f'Unique Labels: {len(df.label.unique())}')

train_ds_dir = os.path.join(data_root_dir, 'train')
test_ds_dir = os.path.join(data_root_dir, 'test')
for ds_dir in (train_ds_dir, test_ds_dir):
    if os.path.exists(ds_dir):
        shutil.rmtree(ds_dir)
    os.makedirs(ds_dir)
test_ds_size = 1
train_copy_size = 4

for label in df.label.unique():

    def write_data_into_dir(data, data_index, dataset_dir, copy_size):
        for copy_index in range(copy_size):
            with open(os.path.join(
                    dataset_dir, f'{data_index}_{copy_index}.txt'),
                    mode='w', encoding='utf-8') as f:
                f.write(beta_utils.replace_token_for_bert(data))

    working_df = df.loc[df.label == label].sample(frac=1)
    for ds_dir in (train_ds_dir, test_ds_dir):
        os.makedirs(os.path.join(ds_dir, label))
    working_dir = os.path.join(test_ds_dir, label)
    for idx, text in working_df.text[:test_ds_size].items():
        write_data_into_dir(text, idx, working_dir, copy_size=1)
    working_dir = os.path.join(train_ds_dir, label)
    for idx, text in working_df.text[test_ds_size:].items():
        write_data_into_dir(text, idx, working_dir, train_copy_size)

exit(0)
