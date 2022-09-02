import flask
import os

import gamma_chatbot


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'GammaData', 'chatbot', 'data')


file_name = 'faq_export2022-08-05-10_57_07.xlsx'
file_path = os.path.join(data_root_path, file_name)
sheet_name = '完整版本'

default_p = '默认'
default_t = .8

df = gamma_chatbot.load_excel(file_path=file_path, sheet_name=sheet_name)
df2 = gamma_chatbot.extract_kt_and_sq(df)
se = gamma_chatbot.run_sentence_encoder_on_df(df2)


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    q = flask.request.form.get('q', '')
    p = flask.request.form.get('p', '')
    t = flask.request.form.get('t', '')
    p = p if p else default_p
    t = t if t else default_t
    print({'q': q, 'p': p, 't': t})
    res = {}
    try:
        t = float(t)
    except ValueError:
        res['status'] = '400 Bad Request'
        print(res)
        return flask.jsonify(res)
    if q and 0 <= t <= 1:
        qe = gamma_chatbot.run_sentence_encoder_on_str(q)
        ss = gamma_chatbot.calculate_similarity_scores(se, qe)
        top_kt = gamma_chatbot.collect_top_kt(df2, ss, t)
        res = gamma_chatbot.join_answer(top_kt, df, p)
        res['status'] = '200 OK'
        print(res)
        return flask.jsonify(res)
    else:
        res['status'] = '400 Bad Request'
        print(res)
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5110)