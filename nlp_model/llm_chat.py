# nlp_model/llm_chat.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from peft import LoraConfig, get_peft_model

class ConversationalLLM:
    def __init__(self, model_name="mistralai/Mistral-7B-v0.1", use_lora=True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Apply LoRA for efficient fine-tuning
        if use_lora:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
        
        # Create conversation pipeline
        self.chat_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
    
    def generate_response(self, prompt, max_length=200, temperature=0.7, top_p=0.9):
        # Format the prompt with chat template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Generate response with casual tone parameters
        outputs = self.chat_pipeline(
            formatted_prompt,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            num_return_sequences=1
        )
        
        response = outputs[0]['generated_text'][len(formatted_prompt):].strip()
        return self._add_disfluencies(response)
    
    def _add_disfluencies(self, text):
        """Add natural speech disfluencies to make text more conversational"""
        import random
        if random.random() < 0.3:  # 30% chance to add a disfluency
            disfluencies = ["uh", "um", "you know", "I mean", "like", "well", "I guess"]
            words = text.split()
            if len(words) > 4:
                insert_pos = random.randint(1, len(words)-2)
                words.insert(insert_pos, random.choice(disfluencies))
                return ' '.join(words)
        return text
    
    def fine_tune(self, dataset_path, epochs=3, batch_size=4):
        from datasets import load_dataset
        from transformers import TrainingArguments, Trainer
        
        # Load and preprocess dataset
        dataset = load_dataset('json', data_files=dataset_path)
        dataset = dataset.map(self._preprocess_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./llm_finetuned",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            save_steps=10_000,
            save_total_limit=2,
            learning_rate=2e-5,
            fp16=torch.cuda.is_available(),
            logging_dir='./logs',
            report_to="tensorboard"
        )
        
        # Start training
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            data_collator=self._data_collator
        )
        
        trainer.train()
        trainer.save_model("./llm_finetuned")
    
    def _preprocess_function(self, examples):
        return self.tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    def _data_collator(self, features):
        batch = {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in features]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in features]),
            "labels": torch.stack([torch.tensor(f["input_ids"]) for f in features])
        }
        return batch