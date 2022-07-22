import numpy as np
import os
import pandas as pd
import re
import shutil


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData', 'coach')
excel_root_dir = os.path.join(data_root_dir, 'xlsx')

ml_dir = os.path.join(data_root_dir, 'ml', 'fluency')
if os.path.exists(ml_dir):
    shutil.rmtree(ml_dir)
os.makedirs(ml_dir)
data_count = 0

for excel_file in os.listdir(excel_root_dir):
    date = re.match(r'coach_(\d+).xlsx', excel_file).group(1)
    excel_path = os.path.join(excel_root_dir, excel_file)
    df = pd.read_excel(excel_path, usecols=['fluency', 'audio'],
                       dtype={'fluency': str}, engine='openpyxl')
    df = df[df['fluency'].notna()]
    for fluency, audio in zip(df['fluency'], df['audio']):
        fluency_dir = os.path.join(ml_dir, fluency)
        if not os.path.exists(fluency_dir):
            os.makedirs(fluency_dir)
        audio_path = os.path.join(data_root_dir, 'wav', date, audio)
        shutil.copy(audio_path, fluency_dir)
    print(f'{date}: {len(df.index)} files created;')
    data_count += len(df.index)

print()
print(f'Totally:  {data_count} files created.')
print()

print('Distribution:')
grades, file_counts = [], []
for fluency_dir in os.listdir(ml_dir):
    grade = int(fluency_dir)
    file_count = len(os.listdir(os.path.join(ml_dir, fluency_dir)))
    print(f'grade {grade}: {file_count} files;')
    grades = np.append(grades, grade)
    file_counts = np.append(file_counts, file_count)

print()
print(f'Average grade: {np.inner(grades, file_counts) / data_count:.2f}')
