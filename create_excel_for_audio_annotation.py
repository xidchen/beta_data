import os
import pandas as pd


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'coach')


for date_dir in os.listdir(os.path.join(data_root_path, 'wav')):

    audio_files = os.listdir(os.path.join(data_root_path, 'wav', date_dir))

    transcripts = []
    for audio_file in audio_files:
        transcript_file = os.path.splitext(audio_file)[0] + '.txt'
        transcript_path = os.path.join(
            data_root_path, 'txt', date_dir, transcript_file)
        if os.path.exists(transcript_path):
            with open(transcript_path, encoding='utf-8') as f:
                transcripts.append(f.read())
        else:
            transcripts.append('')

    labels = [''] * len(audio_files)

    df = pd.DataFrame(data={'fluency': labels,
                            'articulation': labels,
                            'audio': audio_files,
                            'transcript': transcripts})

    excel_file = 'coach_' + date_dir + '.xlsx'
    excel_path = os.path.join(data_root_path, excel_file)
    df.to_excel(excel_writer=excel_path, sheet_name=date_dir, index=False,
                engine='openpyxl', freeze_panes=(1, 0))
    print(f'{excel_path} written.')
