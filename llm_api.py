import os
import time
import uuid

import torch
import transformers
from flask import Flask, jsonify, request
from transformers import AutoTokenizer


from constants import (
    HISTORY_MAX_TOKEN,
    PRETRAINED_MODEL_NAME,
    START_SENTENCE_TOKEN,
    END_SENTENCE_TOKEN,
    START_INSTRUCTION_TOKEN,
    END_INSTRUCTION_TOKEN,
    START_SYSTEM_TOKEN,
    END_SYSTEM_TOKEN,
    SYSTEM_INSTRUCTION,
    FINE_TUNED_MODEL_NAME,
    TEMPERATURE_DEFAULT_VALUE,
    PIPELINE_TYPE,
    REPEAT_PENALTY_DEFAULT_VALUE,
    LENGTH_PENALTY_DEFAULT_VALUE
)

global_history = dict()

BEARER = str(uuid.uuid4())
print("**** Authentication BEARER = " + BEARER)

if os.path.exists(FINE_TUNED_MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_NAME)
    pipeline = transformers.pipeline(
        PIPELINE_TYPE,
        model=FINE_TUNED_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("USING FINE_TUNED_MODEL")
else:
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    pipeline = transformers.pipeline(
        PIPELINE_TYPE,
        model=PRETRAINED_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("USING PRETRAINED_MODEL")

history = []


def format_response(response):
    r = response[response.rindex(END_INSTRUCTION_TOKEN) + 7:]
    return r[:r.rfind(".") + 1]


def get_bot_answer(question, user_email, system_prompt, max_answer_length, temperature, repetition_penalty,
                   length_penalty):
    if system_prompt != '':
        sp = system_prompt
    else:
        sp = SYSTEM_INSTRUCTION
    if user_email != '':
        instruction = get_instruction_from_history(question, user_email, sp)
    else:
        instruction = get_instruction(question, sp)
    print(instruction)
    temperature = check_temperature_value(temperature)
    print("temperature: " + str(temperature))
    start_time = time.time()
    sequences = pipeline(instruction, do_sample=True, top_k=10, num_return_sequences=1,
                         eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
                         max_new_tokens=max_answer_length, temperature=temperature,
                         repetition_penalty=repetition_penalty, length_penalty=length_penalty)
    bot_response = ''
    for seq in sequences:
        bot_response = format_response(seq['generated_text'])
    execution_time = time.time() - start_time
    return bot_response, execution_time


def check_temperature_value(t):
    if type(t) == float:
        if t > 0:
            return t
        else:
            return TEMPERATURE_DEFAULT_VALUE
    if type(t) == str:
        return check_temperature_value(float(t))
    return TEMPERATURE_DEFAULT_VALUE


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
        return get_instruction(question, system_prompt)


def get_instruction(question, system_prompt):
    return START_SENTENCE_TOKEN + START_INSTRUCTION_TOKEN + START_SYSTEM_TOKEN + system_prompt + END_SYSTEM_TOKEN \
           + question + END_INSTRUCTION_TOKEN


app = Flask(__name__)


# request { 'question': 'this is the user question', 'user_email': 'to keep previous context - not mandatory',
#    'system_prompt': 'system prompt to use - not mandatory' }
#    'max_length': 'max answer length in number of token - not mandatory' }

# response { 'answer': 'this is the bot answer', 'execution_time': 'as the name implies',

@app.route('/ask', methods=['POST'])
def answer_question():
    print('Http POST Request received')
    authorization = request.headers.get('Authorization')
    content_type = request.headers.get('Content-Type')
    try:
        if authorization == 'Bearer ' + BEARER and content_type == 'application/json':
            print('Request authorized')
            req = request.get_json()
            q = req.get('question', '')
            u = req.get('user_email', '')
            s = req.get('system_prompt', '')
            ml = req.get('max_length', 200)
            t = req.get('temperature', TEMPERATURE_DEFAULT_VALUE)
            rp = req.get('repeat_penalty', REPEAT_PENALTY_DEFAULT_VALUE)
            lp = req.get('length_penalty', LENGTH_PENALTY_DEFAULT_VALUE)
            if q != '':
                print('Question:' + q)
                bot_answer, execution_time = get_bot_answer(q, u, s, ml, t, rp, lp)
                if u is not None:
                    append_user_history(u, q, bot_answer.strip())
                return jsonify({'answer': bot_answer.strip(), 'execution_time': execution_time, 'user_email': u})
            else:
                return 'Missing question argument', 400
        else:
            return '', 401
    except NameError:
        return '', 401


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
