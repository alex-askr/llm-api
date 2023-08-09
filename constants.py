HISTORY_MAX_TOKEN = 4092
PRETRAINED_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_FILE_NAME = 'prepared_dataset.json'
BEARER = "123456789"

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"
START_INSTRUCTION_TOKEN = "[INST]"
END_INSTRUCTION_TOKEN = "[/INST]"
START_SYSTEM_TOKEN = "<<SYS>>"
END_SYSTEM_TOKEN = "<</SYS>>"
SYSTEM_INSTRUCTION = "\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, " \
                     "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, " \
                     "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased " \
                     "and positive in nature.If a question does not make any sense, or is not factually coherent, " \
                     "explain why instead of answering something not correct. If you do not know the answer to " \
                     "a question, please do not share false information.\n"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "finetuned_models"
