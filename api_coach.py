import flask
import os
import random
import werkzeug.utils

import beta_audio
import beta_coach


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'coach', 'dev')
upload_files_dir = os.path.join(data_root_path, 'upload')
result_files_dir = os.path.join(data_root_path, 'result')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    a = flask.request.files.get('audio')
    r = flask.request.form.get('rhetoric')
    k = flask.request.form.get('keywords')
    if a and r and k and beta_audio.allowed_file(a.filename):
        audio_file = werkzeug.utils.secure_filename(a.filename)
        res_file = beta_audio.replace_ext_to_txt(audio_file)
        audio_file_path = os.path.join(upload_files_dir, audio_file)
        res_file_path = os.path.join(result_files_dir, res_file)
        a.save(audio_file_path)
        asr_res = beta_audio.run_bidu_asr(audio_file_path)
        transcript = asr_res['result'][0] if 'result' in asr_res else None
        if transcript:
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(str(transcript))
            rhetoric_score = beta_coach.rhetoric_score(r, transcript)
            keywords_score = beta_coach.keywords_score(k, transcript)
            speed_score = beta_coach.speed_score(audio_file_path, transcript)
            res = {'transcript': transcript,
                   'scores': {'rhetoric': rhetoric_score,
                              'keywords': keywords_score,
                              'fluency': random.randint(10, 99),
                              'articulation': random.randint(10, 99),
                              'speed': speed_score}}
        else:
            res = {'transcript': transcript,
                   'scores': {'rhetoric': 0,
                              'keywords': 0,
                              'fluency': 0,
                              'articulation': 0,
                              'speed': 0}}
        return flask.jsonify(res)
    else:
        res = {'error_msg': 'input error'}
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5600)
