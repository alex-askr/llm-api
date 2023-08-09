import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from constants import (
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    TRAIN_STEPS,
    OUTPUT_DIR,
    PRETRAINED_MODEL_NAME, LORA_ALPHA, LORA_R, HISTORY_MAX_TOKEN, DATASET_FILE_NAME
)


def training():
    # Step 1: Load the model
    print('# Step 1: Load the model')
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False, load_in_4bit=True
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Step 2: Load the dataset
    print('# Step 2: Load the dataset')
    dataset = load_dataset('csv', data_files=DATASET_FILE_NAME, split="train")

    # Step 3: Define the training arguments
    print('# Step 3: Define the training arguments')
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=LEARNING_RATE,
        num_train_epochs=TRAIN_STEPS,
    )

    # Step 4: Define the LoraConfig
    print('# Step 4: Define the LoraConfig')
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Step 5: Define the Trainer
    print('# Step 5: Define the Trainer')
    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        max_seq_length=HISTORY_MAX_TOKEN,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field='data',
    )

    trainer.train()

    # Step 6: Save the model
    print('# Step 6: Save the model')
    trainer.save_model(OUTPUT_DIR)
    return


if __name__ == '__main__':
    training()
