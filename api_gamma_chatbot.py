import flask
import hashlib
import os

import gamma_chatbot


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'GammaData', 'chatbot', 'data')


file_name = 'faq_export2022-08-05-10_57_07.xlsx'
file_path = os.path.join(data_root_path, file_name)
sheet_name = '完整版本'

default_p = '默认'
default_t = .8

df = gamma_chatbot.load_excel(file_path, sheet_name)
df2 = gamma_chatbot.extract_kt_and_sq(df)
se = gamma_chatbot.run_sentence_encoder_on_df(df2, False)
md_dict = {}


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    global df, df2, se, md_dict
    q = flask.request.form.get('q', '')
    p = flask.request.form.get('p', '')
    t = flask.request.form.get('t', '')
    x = flask.request.files.get('x')
    res = {}
    if x:
        if gamma_chatbot.allowed_file(x.filename):
            x_path = gamma_chatbot.saved_excel_path(x, data_root_path)
            m = {'x': x.filename, 'x_path': x_path}
            print(m)
            df = gamma_chatbot.load_excel(x_path, sheet_name)
            df2 = gamma_chatbot.extract_kt_and_sq(df)
            se = gamma_chatbot.run_sentence_encoder_on_df(df2, True)
            md_dict = {}
            res['status'] = '200 OK'
            print(res)
            return flask.jsonify(res)
        else:
            m = {'x': x.filename}
            print(m)
            res['status'] = '400 Bad Request'
            print(res)
            return flask.jsonify(res)
    p = p if p else default_p
    t = t if t else default_t
    m = {'q': q, 'p': p, 't': t}
    print(m)
    md = hashlib.md5(str(m).encode('utf-8')).hexdigest()
    if md in md_dict:
        res = md_dict[md]
        print(res)
        return flask.jsonify(res)
    try:
        t = float(t)
    except ValueError:
        res['status'] = '400 Bad Request'
        print(res)
        return flask.jsonify(res)
    if q and 0 < t <= 1:
        qe = gamma_chatbot.run_sentence_encoder_on_str(q)
        ss = gamma_chatbot.calculate_similarity_scores(se, qe)
        top_kt = gamma_chatbot.collect_top_kt(df2, ss, t)
        res = gamma_chatbot.join_answer(top_kt, df, p)
        res['status'] = '200 OK'
        print(res)
        md_dict[md] = res
        return flask.jsonify(res)
    else:
        res['status'] = '400 Bad Request'
        print(res)
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5110)
