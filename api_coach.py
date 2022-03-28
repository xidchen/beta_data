import flask
import os
import werkzeug.utils

import beta_audio
import beta_coach


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'coach', 'dev')
upload_dir = os.path.join(data_root_path, 'upload')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    u = flask.request.form.get('url')
    r = flask.request.form.get('rhetoric')
    k = flask.request.form.get('keywords')
    t = flask.request.form.get('transcript')
    print(f'R: {r}')
    print(f'K: {k}')
    print(f'T: {t}')
    if u and r and k and t:
        audio_file = u.rsplit('/', 1)[-1]
        audio_file = werkzeug.utils.secure_filename(audio_file)
        amr_file_path = beta_audio.get_amr_audio(u, audio_file, upload_dir)
        wav_file_path = beta_audio.convert_amr_to_wav(amr_file_path)
        transcript_path = beta_audio.replace_ext_to_txt(wav_file_path)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(str(t))
        duration = beta_audio.get_duration(wav_file_path)
        rhetoric_score = beta_coach.rhetoric_score(r, t)
        keywords_score = beta_coach.keywords_score(k, t)
        speed_score = beta_coach.speed_score(duration, t)
        fluency_score = beta_coach.fluency_score(wav_file_path)
        articulation_score = beta_coach.articulation_score(wav_file_path)
        res = {'scores': {'rhetoric': rhetoric_score,
                          'keywords': keywords_score,
                          'speed': speed_score,
                          'fluency': fluency_score,
                          'articulation': articulation_score},
               'transcript': t}
        return flask.jsonify(res)
    else:
        res = {'error_msg': 'input error'}
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5600)
