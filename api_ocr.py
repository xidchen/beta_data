import flask
import os
import shutil
import werkzeug.utils

import beta_ocr


root_dir = os.path.dirname(os.path.realpath(__file__))
data_root_path = os.path.join(root_dir, 'BetaData', 'ocr', 'dev')
upload_files_dir = os.path.join(data_root_path, 'upload')
result_files_dir = os.path.join(data_root_path, 'result')
num_ocr_requests = 0
MONTHLY_LIMIT = 2000


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    global num_ocr_requests
    file = flask.request.files.get('file')
    if file and beta_ocr.allowed_file(file.filename):
        img_file = werkzeug.utils.secure_filename(file.filename)
        res_file = beta_ocr.replace_ext_to_txt(img_file)
        img_file_path = os.path.join(upload_files_dir, img_file)
        res_file_path = os.path.join(result_files_dir, res_file)
        file.save(img_file_path)
        if num_ocr_requests < MONTHLY_LIMIT:
            res = beta_ocr.run_ocr(img_file_path, 'bidu')
            if 'error_msg' in res:
                res = {'error_msg': res['error_msg']}
            if 'words_result' in res:
                num_ocr_requests += 1
                res = {'words_result': res['words_result']}
                with open(res_file_path, 'w', encoding='utf-8') as f:
                    f.write(str(res))
        else:
            res = beta_ocr.run_ocr(img_file_path, 'tess')
        return flask.jsonify(res)
    else:
        res = {'error_msg': 'image error'}
        return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5500)
