#!/bin/sh
nohup gunicorn -w 8 -b 0:5000 api:app --access-logfile - > api.out 2>&1 &
nohup python -u api_chatbot.py > api_chatbot.out 2>&1 &
nohup python -u api_ols.py > api_ols.out 2>&1 &
nohup gunicorn -w 4 -b 0:5300 api_useml:app --access-logfile - > api_useml.out 2>&1 &
nohup gunicorn -w 4 -b 0:5310 api_usem:app --access-logfile - > api_usem.out 2>&1 &
nohup python -u api_bert_tokenizer.py > api_bert_tokenizer.out 2>&1 &
nohup python -u api_ocr.py > api_ocr.out 2>&1 &
nohup gunicorn -w 4 -b 0:5600 api_coach:app --access-logfile - > api_coach.out 2>&1 &
nohup gunicorn -w 4 -b 0:5610 api_fluency:app --access-logfile - > api_fluency.out 2>&1 &
nohup gunicorn -w 4 -b 0:5620 api_articulation:app --access-logfile - > api_articulation.out 2>&1 &
