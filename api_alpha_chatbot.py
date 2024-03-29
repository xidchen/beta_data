import flask
import hashlib
import os

import alpha_chatbot


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'alpha_chatbot', 'data')


file_name = 'demo_alpha_text.xlsx'
file_path = os.path.join(data_root_path, file_name)
sheet_name = 'Sheet1'

default_t = .5

df = alpha_chatbot.load_excel(file_path, sheet_name)
df2 = alpha_chatbot.extract_kt_and_sq(df)
se = alpha_chatbot.run_sentence_encoder_on_df(df2, False)
kt_name_to_id = alpha_chatbot.get_kt_code()
md_dict = {}


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    global default_t, df, df2, se, kt_name_to_id, md_dict
    q = flask.request.form.get('query', '')
    t = flask.request.form.get('threshold', '')
    x = flask.request.files.get('excel')
    q = q.replace('\r', '').replace('\n', '').strip()
    t = t.replace('\r', '').replace('\n', '').strip()
    res = {}
    if x:
        if alpha_chatbot.allowed_file(x.filename):
            x_path = alpha_chatbot.saved_excel_path(x, data_root_path)
            m = {'x': x.filename, 'x_path': x_path}
            print(m)
            try:
                df = alpha_chatbot.load_excel(x_path, sheet_name)
            except KeyError as e:
                res['status'] = f'400 Bad Request: {e}'
                print(res)
                return flask.jsonify(res)
            try:
                df2 = alpha_chatbot.extract_kt_and_sq(df)
            except KeyError as e:
                res['status'] = f'400 Bad Request: {e}'
                print(res)
                return flask.jsonify(res)
            try:
                se = alpha_chatbot.run_sentence_encoder_on_df(df2, True)
            except KeyError as e:
                res['status'] = f'400 Bad Request: {e}'
                print(res)
                return flask.jsonify(res)
            kt_name_to_id = alpha_chatbot.get_kt_code()
            md_dict = {}
            res['status'] = '200 OK: excel updated'
            print(res)
            return flask.jsonify(res)
        else:
            m = {'x': x.filename}
            print(m)
            res['status'] = f'400 Bad Request: {x.filename}'
            print(res)
            return flask.jsonify(res)
    if t and not q:
        m = {'t': t}
        print(m)
        try:
            t = float(t)
        except ValueError as e:
            res['status'] = f'400 Bad Request: {e}'
            print(res)
            return flask.jsonify(res)
        if 0 < t <= 1:
            default_t = t
            md_dict = {}
            res['status'] = f'200 OK: threshold updated to {t}'
            print(res)
            return flask.jsonify(res)
        else:
            res['status'] = '400 Bad Request'
            print(res)
            return flask.jsonify(res)
    t = t if t else default_t
    m = f'Q: {q}' if t == default_t else f'Q: {q}\nT: {t}'
    print(m)
    md = hashlib.md5(str(m).encode('utf-8')).hexdigest()
    if md in md_dict:
        res = md_dict[md]
        print(res)
        return flask.jsonify(res)
    try:
        t = float(t)
    except ValueError as e:
        res['status'] = f'400 Bad Request: {e}'
        print(res)
        return flask.jsonify(res)
    if q and 0 < t <= 1:
        res = alpha_chatbot.request_beta_chatbot(q)
        if res['intent']['name'] in alpha_chatbot.allowed_beta_intents():
            print(res)
            md_dict[md] = res
            return flask.jsonify(res)
        qe = alpha_chatbot.run_sentence_encoder_on_str(q)
        ss = alpha_chatbot.calculate_similarity_scores(se, qe)
        top_kt = alpha_chatbot.collect_top_kt(df2, ss, t)
        res = alpha_chatbot.organize_result(top_kt, kt_name_to_id)
        res['status'] = '200 OK'
        print(res)
        res = alpha_chatbot.organize_final_result(res, kt_name_to_id)
        md_dict[md] = res
        return flask.jsonify(res)
    else:
        res['status'] = '400 Bad Request'
        print(res)
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5110)
