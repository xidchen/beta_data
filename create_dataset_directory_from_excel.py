import os
import pandas as pd
import shutil

root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData')
data_file_str = os.path.join(data_root_dir, 'demo_beta_text.xlsx')
df = pd.read_excel(
    data_file_str, header=1, names=['label', 'text'], engine='openpyxl')
print(df)

train_ds_dir = os.path.join(data_root_dir, 'train')
test_ds_dir = os.path.join(data_root_dir, 'test')
for ds_dir in (train_ds_dir, test_ds_dir):
    if os.path.exists(ds_dir):
        shutil.rmtree(ds_dir)
    os.makedirs(ds_dir)
test_ds_size = 1
train_copy_size = 10

for row_idx, row in df.iterrows():

    for ds_dir in (train_ds_dir, test_ds_dir):
        if not os.path.exists(os.path.join(ds_dir, row['label'])):
            os.makedirs(os.path.join(ds_dir, row['label']))

    def write_data_into_dir(dataset_dir, copy_size):
        for copy_idx in range(copy_size):
            with open(os.path.join(
                    dataset_dir, '{}_{}.txt'.format(row_idx, copy_idx)),
                    mode='w', encoding='utf-8') as f:
                f.write(row['text'])

    working_dir = os.path.join(test_ds_dir, row['label'])
    if len(os.listdir(working_dir)) < test_ds_size:
        write_data_into_dir(working_dir, 1)
    else:
        working_dir = os.path.join(train_ds_dir, row['label'])
        write_data_into_dir(working_dir, train_copy_size)
