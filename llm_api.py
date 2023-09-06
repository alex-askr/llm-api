import os
import time

import torch
import transformers
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
from trl import SFTTrainer

from constants import (
    HISTORY_MAX_TOKEN,
    PRETRAINED_MODEL_NAME,
    BEARER,
    START_SENTENCE_TOKEN,
    END_SENTENCE_TOKEN,
    START_INSTRUCTION_TOKEN,
    END_INSTRUCTION_TOKEN,
    START_SYSTEM_TOKEN,
    END_SYSTEM_TOKEN,
    SYSTEM_INSTRUCTION,
    FINE_TUNED_MODEL_NAME
)

global_history = dict()

if os.path.exists(FINE_TUNED_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_NAME)
    pipeline = transformers.pipeline(
        "text-generation",
        model=FINE_TUNED_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("USING FINE_TUNED_MODEL")
else:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    pipeline = transformers.pipeline(
        "text-generation",
        model=PRETRAINED_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("USING PRETRAINED_MODEL")

history = []


def format_response(response):
    return response[response.rindex(END_INSTRUCTION_TOKEN) + 7:]


def get_bot_answer(question, user_email, system_prompt):
    if system_prompt != '':
        sp = system_prompt
    else:
        sp = SYSTEM_INSTRUCTION
    if user_email != '':
        instruction = get_instruction_from_history(question, user_email, sp)
    else:
        instruction = get_instruction(question, sp)
    print(instruction)
    start_time = time.time()
    sequences = pipeline(instruction, do_sample=True, top_k=10, num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, max_length=2000, )
    bot_response = ''
    for seq in sequences:
        bot_response = format_response(seq['generated_text'])
    execution_time = time.time() - start_time
    return bot_response, execution_time


def append_user_history(user, question, answer):
    if global_history.get(user) is not None:
        q_and_a_history = global_history.get(user)
    else:
        q_and_a_history = []
    q_and_a = dict()
    q_and_a.update({'question': question})
    q_and_a.update({'answer': answer})
    q_and_a_history.append(q_and_a)
    global_history.update({user: get_filtered_history(q_and_a_history)})
    return


def get_filtered_history(qa_list):
    while str(qa_list).__len__() > HISTORY_MAX_TOKEN:
        qa_list.pop(0)
    return qa_list


def get_instruction_from_history(question, user_email, system_prompt):
    if global_history.get(user_email) is not None:
        question_history = global_history.get(user_email)
        instruction = START_SENTENCE_TOKEN + START_INSTRUCTION_TOKEN + START_SYSTEM_TOKEN + system_prompt \
                      + END_SYSTEM_TOKEN
        for qa in question_history:
            q = qa.get('question')
            a = qa.get('answer')
            instruction = instruction + q + END_INSTRUCTION_TOKEN + a + END_SENTENCE_TOKEN
        return instruction + START_SENTENCE_TOKEN + START_INSTRUCTION_TOKEN + question + END_INSTRUCTION_TOKEN

    else:
        return get_instruction(question)


def get_instruction(question, system_prompt):
    return START_SENTENCE_TOKEN + START_INSTRUCTION_TOKEN + START_SYSTEM_TOKEN + system_prompt + END_SYSTEM_TOKEN \
           + question + END_INSTRUCTION_TOKEN


def fine_tune_model(data):
    trainer = SFTTrainer(
        PRETRAINED_MODEL_NAME,
        train_dataset=data,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer.train()
    return


app = Flask(__name__)


# request { 'question': 'this is the user question', 'user_email': 'to keep previous context - not mandatory', 'system_prompt': 'system prompt to use - not mandatory' }
# response { 'answer': 'this is the bot answer', 'execution_time': 'as the name implies',
# 'user_email': 'to keep previous context - empty if unknown' }
# 'system_prompt': 'system instruction for the model - empty to use the one by default' }
@app.route('/ask', methods=['POST'])
def answer_question():
    print('Http POST Request received')
    authorization = request.headers.get('Authorization')
    content_type = request.headers.get('Content-Type')
    try:
        if authorization == 'Bearer ' + BEARER and content_type == 'application/json':
            print('Request authorized')
            req = request.get_json()
            q = req.get('question')
            u = req.get('user_email')
            s = req.get('system_prompt')
            if q != '':
                print('Question:' + q)
                bot_answer, execution_time = get_bot_answer(q, u, s)
                if u is not None:
                    append_user_history(u, q, bot_answer.strip())
                return jsonify({'answer': bot_answer.strip(), 'execution_time': execution_time, 'user_email': u})
            else:
                return 'Missing question argument', 400
        else:
            return '', 401
    except NameError:
        return '', 401

    # request { 'fine_tuning_data': [{'instruction', 'answer'}, {'instruction', 'answer'}] }


# response { 'answer': 'this is the bot answer', 'execution_time': 'as the name implies',
# 'user_email': 'to keep previous context - empty if unknown' }
@app.route('/train', methods=['POST'])
def fine_tune():
    print('Http POST Request received')
    authorization = request.headers.get('Authorization')
    content_type = request.headers.get('Content-Type')
    try:
        if authorization == 'Bearer ' + BEARER and content_type == 'application/json':
            print('Request authorized')
            req = request.get_json()
            data = req.get('fine_tuning_data')
            if data.len() > 1:
                fine_tune_model(data)
                return '', 200
            else:
                return 'Missing fine_tuning_data argument', 400
        else:
            return '', 401
    except NameError:
        return '', 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
