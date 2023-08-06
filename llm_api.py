
from flask import Flask, jsonify, request
from transformers import AutoTokenizer
import transformers
import torch
import time

model = "meta-llama/Llama-2-7b-chat-hf"
bearer = "123456789"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
          )

system_instruction = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you do not know the answer to a question, please do not share false information.<</SYS>>\n"

def format_response(response):
    return response[response.index("[/INST]")+7:]

def get_bot_answer(question, user_email):
    instruction = system_instruction + question + '[/INST]'
    start_time = time.time()
    sequences = pipeline(instruction, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, max_length=2000,)
    for seq in sequences:
        bot_response = format_response(seq['generated_text'])
    execution_time = time.time() - start_time
    return bot_response, execution_time

app = Flask(__name__)

#{ 'question': 'this is the user question', 'user_email': 'to keep previous context - not mandatory' }
#{ 'answer': 'this is the bot answer', 'execution_time': 'as the name implies', 'user_email': 'to keep previous context - empty if unknown' }
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
                return jsonify({ 'answer': bot_answer, 'execution_time': execution_time, 'user_email': '' })
            else:
                return 'Missing question argument', 400
        else:
            return '', 401 
    except NameError:
        return '', 401 
    



    