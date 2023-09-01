import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from trl import SFTTrainer

from constants import (
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    TRAIN_STEPS,
    FINE_TUNED_MODELS_DIRECTORY,
    FINE_TUNED_MODEL_NAME,
    PRETRAINED_MODEL_NAME, LORA_ALPHA, LORA_R, HISTORY_MAX_TOKEN, DATASET_FILE_NAME, PER_DEVICE_BATCH_SIZE, MAX_STEPS,
    LOGGING_STEPS
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
        output_dir=FINE_TUNED_MODELS_DIRECTORY,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=TRAIN_STEPS,
        max_step=MAX_STEPS,
        logging_steps=LOGGING_STEPS,
        fp16=True,
        optim="paged_adamw_8bit",
    )

    # Step 4: Define the LoraConfig
    print('# Step 4: Define the LoraConfig')
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)
    base_model.config.use_cache = False
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
    trainer.save_model(FINE_TUNED_MODELS_DIRECTORY)

    # Step 7: Merge
    model = AutoPeftModelForCausalLM.from_pretrained(
        FINE_TUNED_MODELS_DIRECTORY,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Merge LoRA and base model
    merged_model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    # Save the merged model
    merged_model.save_pretrained(FINE_TUNED_MODEL_NAME, safe_serialization=True)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_NAME)

    print("Successful model fine-tuning")
    return


if __name__ == '__main__':
    training()
