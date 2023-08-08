
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
import transformers
import torch
import time



global_history = dict()

model = "meta-llama/Llama-2-7b-chat-hf"
bearer = "123456789"

start_sentence_token = "<s>"
end_sentence_token = "</s>"
start_instruction_token = "[INST]"
end_instruction_token = "[/INST]"
start_system_token = "<<SYS>>"
end_system_token = "<</SYS>>"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
          )

history = []

system_instruction = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information.\n"

def format_response(response):
    return response[response.index("[/INST]")+7:]

def get_bot_answer(question, user_email):
    if (user_email != ''):
        instruction = get_instruction_from_history(question, user_email)
    else:
        instruction = get_instruction(question)
    print(instruction)
    start_time = time.time()
    sequences = pipeline(instruction, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, max_length=2000,)
    for seq in sequences:
        bot_response = format_response(seq['generated_text'])
    execution_time = time.time() - start_time
    return bot_response, execution_time

def append_user_history(user, question, answer):
    q_and_a_history = []
    q_and_a = dict()
    q_and_a.update({'question': question})
    q_and_a.update({'answer': answer})
    q_and_a_history.append(q_and_a)
    global_history.update({user: q_and_a_history})
    return

def get_instruction_from_history(question, user_email):
    if (global_history.get(user_email) != None):
        question_history = global_history.get(user_email)
        instruction = start_sentence_token + start_instruction_token + start_system_token + system_instruction + end_system_token
        for qa in question_history:
            q = qa.get('question')
            a = qa.get('answer')
            instruction = instruction + q + end_instruction_token + a + end_sentence_token
        return instruction + start_sentence_token + start_instruction_token + question + end_instruction_token

    else:
        return get_instruction(question)

def get_instruction(question):
    return start_sentence_token + start_instruction_token + start_system_token + system_instruction + end_system_token + question + end_instruction_token

def fine_tune_model(data):
    return

app = Flask(__name__)

#request { 'question': 'this is the user question', 'user_email': 'to keep previous context - not mandatory' }
#response { 'answer': 'this is the bot answer', 'execution_time': 'as the name implies', 'user_email': 'to keep previous context - empty if unknown' }
@app.route('/ask', methods=['POST'])
def answer_question():
    print('Http POST Request received')
    authorization = request.headers.get('Authorization')
    content_type = request.headers.get('Content-Type')
    try:
        if (authorization == 'Bearer ' + bearer and content_type == 'application/json'):
            print('Request authorized')
            req = request.get_json()
            q = req.get('question')
            u = req.get('user_email')
            if (q != ''):
                print('Question:' + q)
                bot_answer, execution_time = get_bot_answer(q, u)
                if (u != None):
                    append_user_history(u, q, bot_answer.strip())
                return jsonify({ 'answer': bot_answer.strip(), 'execution_time': execution_time, 'user_email': u })
            else:
                return 'Missing question argument', 400
        else:
            return '', 401 
    except NameError:
        return '', 401 
    
#request { 'fine_tuning_data': [{'instruction', 'answer'}, {'instruction', 'answer'}] }
#response { 'answer': 'this is the bot answer', 'execution_time': 'as the name implies', 'user_email': 'to keep previous context - empty if unknown' }
@app.route('/train', methods=['POST'])
def fine_tune():
    print('Http POST Request received')
    authorization = request.headers.get('Authorization')
    content_type = request.headers.get('Content-Type')
    try:
        if (authorization == 'Bearer ' + bearer and content_type == 'application/json'):
            print('Request authorized')
            req = request.get_json()
            data = req.get('fine_tuning_data')
            if (data.len() > 1):
                fine_tune_model(data)
                return '', 200 
            else:
                return 'Missing fine_tuning_data argument', 400
        else:
            return '', 401 
    except NameError:
        return '', 401 



    