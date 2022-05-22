import os
import pandas as pd
import re
import shutil


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_dir = os.path.join(root_dir, 'BetaData', 'coach')
excel_root_dir = os.path.join(data_root_dir, 'xlsx')

ml_dir = os.path.join(data_root_dir, 'ml', 'articulation')
if os.path.exists(ml_dir):
    shutil.rmtree(ml_dir)
os.makedirs(ml_dir)
data_count = 0

for excel_file in os.listdir(excel_root_dir):
    date = re.match(r'coach_(\d+).xlsx', excel_file).group(1)
    excel_path = os.path.join(excel_root_dir, excel_file)
    df = pd.read_excel(excel_path, usecols=['articulation', 'audio'],
                       dtype={'articulation': str}, engine='openpyxl')
    df = df[df['articulation'].notna()]
    for articulation, audio in zip(df['articulation'], df['audio']):
        articulation_dir = os.path.join(ml_dir, articulation)
        if not os.path.exists(articulation_dir):
            os.makedirs(articulation_dir)
        audio_path = os.path.join(data_root_dir, 'wav', date, audio)
        shutil.copy(audio_path, articulation_dir)
    print(f'{date}: {len(df.index)} files created;')
    data_count += len(df.index)

print()
print(f'Totally:  {data_count} files created.')
print()
