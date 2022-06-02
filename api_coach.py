import flask
import hashlib
import os
import tensorflow as tf
import werkzeug.utils as wu

import beta_audio
import beta_coach

for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
tf.get_logger().setLevel('ERROR')


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'coach', 'dev')
file_dir = os.path.join(data_root_path, 'upload')
tmp_dir = os.path.join(data_root_path, 'tmp')


r1 = r2 = tf.random.normal([1, 8])
tf.einsum('ij,kj->ik', r1, r2)


md_dict = {}


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    v = flask.request.form.get('url')
    u = flask.request.form.get('urls')
    r = flask.request.form.get('rhetoric')
    k = flask.request.form.get('keywords')
    s = flask.request.form.get('transcript')
    t = flask.request.form.get('transcripts')
    if u and r and k and t:
        m = f'U: {u}\nR: {r}\nK: {k}\nT: {t}'
        print(m)
        md = hashlib.md5(m.encode('utf-8')).hexdigest()
        if md in md_dict:
            if 'error_msg' in md_dict[md]:
                res = md_dict[md]
                print(res)
                return flask.jsonify(res)
            if 'rhetoric' in md_dict[md]:
                res = {'scores': md_dict[md],
                       'transcript': t}
                print(res['scores'])
                return flask.jsonify(res)
        u = beta_coach.split_by_semicolon(u)
        t = beta_coach.split_by_semicolon(t)
        if not u or not t:
            res = {'error_msg': 'input error'}
            print(res)
            md_dict[md] = res
            return flask.jsonify(res)
        try:
            wav_path = beta_audio.get_wav_from_urls(u, file_dir, tmp_dir)
        except FileNotFoundError:
            res = {'error_msg': 'input error'}
            print(res)
            md_dict[md] = res
            return flask.jsonify(res)
        transcript_path = beta_audio.replace_ext_to_txt(wav_path)
        t = ''.join(t)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(t)
        duration = beta_audio.get_duration(wav_path)
        rhetoric_score = beta_coach.rhetoric_score(r, t)
        keywords_score = beta_coach.keywords_score(k, t)
        speed_score = beta_coach.speed_score(duration, t)
        fluency_score = beta_coach.fluency_score(duration, wav_path)
        articulation_score = beta_coach.articulation_score(duration, wav_path)
        res = {'scores': {'rhetoric': rhetoric_score,
                          'keywords': keywords_score,
                          'speed': speed_score,
                          'fluency': fluency_score,
                          'articulation': articulation_score},
               'transcript': t}
        print(res['scores'])
        md_dict[md] = res['scores']
        return flask.jsonify(res)
    elif v and r and k and s:
        print(f'V: {v}')
        print(f'R: {r}')
        print(f'K: {k}')
        print(f'S: {s}')
        audio_file = v.rsplit('/', 1)[-1]
        audio_file = wu.secure_filename(audio_file)
        amr_file_path = beta_audio.get_amr_audio(v, audio_file, file_dir)
        wav_path = beta_audio.convert_amr_to_wav(amr_file_path)
        transcript_path = beta_audio.replace_ext_to_txt(wav_path)
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(str(s))
        duration = beta_audio.get_duration(wav_path)
        rhetoric_score = beta_coach.rhetoric_score(r, s)
        keywords_score = beta_coach.keywords_score(k, s)
        speed_score = beta_coach.speed_score(duration, s)
        fluency_score = beta_coach.fluency_score(duration, wav_path)
        articulation_score = beta_coach.articulation_score(duration, wav_path)
        res = {'scores': {'rhetoric': rhetoric_score,
                          'keywords': keywords_score,
                          'speed': speed_score,
                          'fluency': fluency_score,
                          'articulation': articulation_score},
               'transcript': s}
        print(res['scores'])
        return flask.jsonify(res)
    else:
        res = {'error_msg': 'input error'}
        print(res)
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5600)
