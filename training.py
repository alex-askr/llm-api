import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, AutoTokenizer
from trl import SFTTrainer

from constants import (
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    NUM_TRAIN_EPOCH,
    OUTPUT_DIR,
    FINE_TUNED_MODEL_NAME,
    PRETRAINED_MODEL_NAME, LORA_ALPHA, LORA_R, HISTORY_MAX_TOKEN, DATASET_FILE_NAME, PER_DEVICE_BATCH_SIZE, MAX_STEPS,
    LOGGING_STEPS
)


def training():
    # Step 1: Load the model
    print('# Step 1: Load the model')
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=False,
        bnb_4bit_compute_type=torch.float16,
        load_in_4bit=True
    )


    # Activate 4-bit precision base model loading
    use_4bit = True
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"
    # Activate nested quantization for 4-bit base models (double quantization)
    use_double_nested_quant = False
    # LoRA attention dimension
    lora_r = 64
    # Alpha parameter for LoRA scaling
    lora_alpha = 16
    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_use_double_quant=use_double_nested_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        PRETRAINED_MODEL_NAME,
        quantization_config=bnb_config,
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
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_TRAIN_EPOCH,
        logging_steps=LOGGING_STEPS,
        fp16=True,
        optim="paged_adamw_8bit",
    )

    # or num_train_epochs=NUM_TRAIN_EPOCH,

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
    trainer.save_model(OUTPUT_DIR)

    # Step 7: Merge
    model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
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
