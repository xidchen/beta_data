import flask
import os
import werkzeug.utils

import beta_ocr


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'ocr', 'dev')
upload_files_dir = os.path.join(data_root_path, 'upload')
result_files_dir = os.path.join(data_root_path, 'result')


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    file = flask.request.files.get('file')
    if file and beta_ocr.allowed_file(file.filename):
        img_file = werkzeug.utils.secure_filename(file.filename)
        res_file = beta_ocr.replace_ext_to_txt(img_file)
        img_file_path = os.path.join(upload_files_dir, img_file)
        res_file_path = os.path.join(result_files_dir, res_file)
        file.save(img_file_path)
        res = beta_ocr.run_ocr(img_file_path, 'bidu')
        if 'words_result' in res:
            res = {'words_result': res['words_result']}
            print({'bidu': img_file_path})
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(str(res))
            return flask.jsonify(res)
        if 'error_msg' in res:
            print({'error_msg': res['error_msg']})
            res = beta_ocr.run_ocr(img_file_path, 'tess')
            print({'tess': img_file_path})
            return flask.jsonify(res)
    else:
        res = {'error_msg': 'image error'}
        return flask.jsonify(res)


@app.route('/basic', methods=['POST'])
def basic():
    file = flask.request.files.get('file')
    if file and beta_ocr.allowed_file(file.filename):
        img_file = werkzeug.utils.secure_filename(file.filename)
        res_file = beta_ocr.replace_ext_to_txt(img_file)
        img_file_path = os.path.join(upload_files_dir, img_file)
        res_file_path = os.path.join(result_files_dir, res_file)
        file.save(img_file_path)
        res = beta_ocr.run_ocr(img_file_path, 'bidu_basic')
        if 'words_result' in res:
            res = {'words_result': res['words_result']}
            print({'bidu_basic': img_file_path})
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(str(res))
            return flask.jsonify(res)
        if 'error_msg' in res:
            print({'error_msg': res['error_msg']})
            res = beta_ocr.run_ocr(img_file_path, 'bidu_basic_general')
        if 'words_result' in res:
            res = {'words_result': res['words_result']}
            print({'bidu_basic_general': img_file_path})
            return flask.jsonify(res)
        if 'error_msg' in res:
            print({'error_msg': res['error_msg']})
            res = beta_ocr.run_ocr(img_file_path, 'tess_basic')
            print({'tess_basic': img_file_path})
            return flask.jsonify(res)
    else:
        res = {'error_msg': 'image error'}
        return flask.jsonify(res)


@app.route('/table', methods=['POST'])
def table():
    file = flask.request.files.get('file')
    if file and beta_ocr.allowed_file(file.filename):
        img_file = werkzeug.utils.secure_filename(file.filename)
        res_file = beta_ocr.replace_ext_to_txt(img_file)
        img_file_path = os.path.join(upload_files_dir, img_file)
        res_file_path = os.path.join(result_files_dir, res_file)
        file.save(img_file_path)
        res = beta_ocr.run_ocr(img_file_path, 'bidu_table')
        if 'tables_result' in res:
            res = {'tables_result': res['tables_result']}
            print({'bidu': img_file_path})
            with open(res_file_path, 'w', encoding='utf-8') as f:
                f.write(str(res))
            return flask.jsonify(res)
        if 'error_msg' in res:
            print({'error_msg': res['error_msg']})
            res = {'error_msg': res['error_msg']}
            return flask.jsonify(res)
    else:
        res = {'error_msg': 'image error'}
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5500)
