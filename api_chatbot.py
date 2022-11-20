import flask
import hashlib

import beta_chatbot
import beta_code


i_name_to_id, _ = beta_code.get_intent_code()
print('Intent code loaded')
entity_class_path = './entity_classes.txt'
with open(entity_class_path, encoding='utf-8') as f:
    entity_classes = f.read().strip().split('\n')
e_name_to_code, e_code_to_name = {}, {}
for e in entity_classes:
    try:
        e_name_to_code[e], e_code_to_name[e] = beta_code.get_entity_code(e)
        print(f'Entity {e} code loaded')
    except KeyError:
        print(f'Entity {e} code not loaded')


md_dict = {}


app = flask.Flask(__name__)


@app.route('/', methods=['POST'])
def main():
    q = flask.request.form.get('q', '').strip()
    v = flask.request.form.get('v', '').strip()
    q = q.replace('\r', '').replace('\n', '')
    v = True if v else False
    m = f'Q: {q}\nV: {v}' if v else f'Q: {q}'
    print(m)
    res = {}
    md = hashlib.md5(str(m).encode('utf-8')).hexdigest()
    if md in md_dict:
        res = md_dict[md]
        print(res)
        return flask.jsonify(res)
    if q:
        res['status'] = '200 OK'
        intent = beta_chatbot.run_bert_intent(q)
        res['intent'] = {'id': i_name_to_id.get(intent, ''), 'name': intent}
        entities = beta_chatbot.run_bert_entity(q)
        res['entities'] = beta_chatbot.get_query_entity(
            q, v, entities, e_name_to_code, e_code_to_name)
    else:
        res['status'] = '400 Bad Request'
    print(res)
    md_dict[md] = res
    return flask.jsonify(res)


if __name__ == '__main__':
    app.run('0.0.0.0', 5100)
