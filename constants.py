HISTORY_MAX_TOKEN = 4092
PRETRAINED_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
DATASET_FILE_NAME = 'prepared_dataset.csv'
BEARER = "123456789"

START_SENTENCE_TOKEN = "<s>"
END_SENTENCE_TOKEN = "</s>"
START_INSTRUCTION_TOKEN = "[INST]"
END_INSTRUCTION_TOKEN = "[/INST]"
START_SYSTEM_TOKEN = "<<SYS>>"
END_SYSTEM_TOKEN = "<</SYS>>"

SYSTEM_INSTRUCTION = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, " \
                     "while being safe.  Your answers should not include any harmful, unethical, racist, sexist, " \
                     "toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased " \
                     "and positive in nature.If a question does not make any sense, or is not factually coherent, " \
                     "explain why instead of answering something not correct. If you do not know the answer to " \
                     "a question, please do not share false information.\n"

LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

PER_DEVICE_BATCH_SIZE = 1
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCH = 100
MAX_STEPS = 20
LOGGING_STEPS = 10
OUTPUT_DIR = "working_dir"
FINE_TUNED_MODEL_NAME = "fine_tuned_model"
TEMPERATURE_DEFAULT_VALUE = 0.2
REPEAT_PENALTY_DEFAULT_VALUE = 1
LENGTH_PENALTY_DEFAULT_VALUE = 0
TEMPERATURE_DEFAULT_VALUE
PIPELINE_TYPE = "text-generation"
