#!/bin/sh
nohup python -u api.py > api.out 2>&1 &
nohup python -u api_chatbot.py > api_chatbot.out 2>&1 &
nohup python -u api_ols.py > api_ols.out 2>&1 &
nohup python -u api_useml.py > api_useml.out 2>&1 &
nohup python -u api_bert_tokenizer.py > api_bert_tokenizer.out 2>&1 &
nohup python -u api_ocr.py > api_ocr.out 2>&1 &
nohup python -u api_coach.py > api_coach.out 2>&1 &
