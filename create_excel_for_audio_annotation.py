import os
import pandas as pd

import beta_utils


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'coach')


def designate_date(date_dir=None) -> None:
    """designate a date to create Excel file
    :param date_dir: directory name in a date form
    :return: None
    """

    audio_dir_path = os.path.join(data_root_path, 'wav', date_dir)

    if os.path.exists(audio_dir_path):
        audio_files = os.listdir(audio_dir_path)
        audio_files = beta_utils.sort_numerically(audio_files)

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

    else:
        print(f'{audio_dir_path} does not exist.')
