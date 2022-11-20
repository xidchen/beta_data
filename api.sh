#!/bin/sh
nohup gunicorn -w 4 -b 0:5000 api:app --access-logfile - -t 0 > api.out 2>&1 &
nohup gunicorn -w 1 -b 0:5100 api_chatbot:app --access-logfile - -t 0 > api_chatbot.out 2>&1 &
nohup gunicorn -w 1 -b 0:5110 api_alpha_chatbot:app --access-logfile - -t 0 > api_alpha_chatbot.out 2>&1 &
nohup gunicorn -w 2 -b 0:5200 api_ols:app --access-logfile - -t 0 > api_ols.out 2>&1 &
nohup gunicorn -w 4 -b 0:5300 api_useml:app --access-logfile - -t 0 > api_useml.out 2>&1 &
nohup gunicorn -w 2 -b 0:5310 api_usem:app --access-logfile - -t 0 > api_usem.out 2>&1 &
nohup gunicorn -w 1 -b 0:5410 api_bert_intent:app --access-logfile - -t 0 > api_bert_intent.out 2>&1 &
nohup gunicorn -w 1 -b 0:5420 api_bert_entity:app --access-logfile - -t 0 > api_bert_entity.out 2>&1 &
nohup gunicorn -w 4 -b 0:5600 api_coach:app --access-logfile - -t 0 > api_coach.out 2>&1 &
nohup gunicorn -w 4 -b 0:5610 api_fluency:app --access-logfile - -t 0 > api_fluency.out 2>&1 &
nohup gunicorn -w 4 -b 0:5620 api_articulation:app --access-logfile - -t 0 > api_articulation.out 2>&1 &