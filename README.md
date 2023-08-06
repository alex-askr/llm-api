# llm-api


This project is intended to make LLM (Large Language Model) easy to train and to use by exposing it via REST APIs

First versions of this project is based on Facebook LLAMA2 LLM.


To run the script:
- Make sure you have access to LLAMA model - see your huggingface account
- Enter you huggingface token using `huggingface-cli login``
- Execute `flask --app llm_api run -h 0.0.0.0`
