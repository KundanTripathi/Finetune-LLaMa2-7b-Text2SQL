# Finetune-LLaMa2-7b-Text2SQL

Llama2-7b is not good at generating SQL Queries even with the context, hence it's good to use a finetuned Llama2-7b model 

---

## Steps of Finetuning 

1. Load data set from Hugging Face from "b-mc2/sql-create-context" <br>
2. Prepare Data for Finetuning LLaMa2-7b <br>
3. Load Model (can load the model in float16 for lower memory overhead) <br>
4. Use Bitsandbytes Config for quantization <br>
5. Use PEFT for LoraConfig <br>
6. Define SFT Training Args <br>
7. Once trained merge the trained model with FP16 base model <br>

---

### QLoRA Training Configuration
This section outlines the configuration used for fine-tuning a model using QLoRA with 4-bit quantization via bitsandbytes, optimized for memory-efficient training on A100 GPUs (or other GPU types via Colab Pro/Pro+).

#### 📌 QLoRA Parameters <br>
|Parameter|	Value | Description |
|---------|-------|-------------|
|lora_r	 | 64 |	Rank of the low-rank adaptation matrix |
|lora_alpha | 16 |	LoRA scaling factor |
|lora_dropout |	0.1	 | Dropout applied to LoRA layers |

#### 📦 bitsandbytes (4-bit Quantization) <br>
|Parameter	| Value	 | Description |
|-----------|--------|-------------|
|use_4bit	| True	| Enables 4-bit quantization |
|bnb_4bit_quant_type |	nf4	| Normal Float 4-bit — most accurate quant type |
|bnb_4bit_compute_dtype	| float16	| Compute type for forward/backward pass |
|use_nested_quant |	False	| Enables nested quantization (optional, saves more memory) |

#### 🛠️ Training Arguments (HF Transformers) <br>
|Parameter	| Value	| Description |
|-----------|-------|-------------|
|output_dir	| "./results"	| Where model checkpoints and logs are saved |
|num_train_epochs |	1	| Number of training epochs |
|fp16 |	False |	Use FP16 training (disabled) |
|bf16 |	True | (→ use True on A100)	Use bfloat16 (recommended for A100 GPUs) |
|per_device_train_batch_size |	4	| Batch size per GPU/device during training |
|per_device_eval_batch_size |	4	| Batch size for evaluation |
|gradient_accumulation_steps |	1	| Steps to accumulate before optimizer step |
|gradient_checkpointing	| True	| Save memory by recomputing intermediate activations |
|max_grad_norm	| 0.3	| Gradient clipping norm |
|learning_rate	| 2e-4	| Peak learning rate |
|weight_decay	| 0.001	| Weight decay for regularization |
|optim	| paged_adamw_32bit	| Memory-efficient optimizer |
|lr_scheduler_type	| cosine	| Learning rate schedule |
|max_steps	| -1	| Train for full epochs (no hard step limit) |
|warmup_ratio	| 0.03	| Proportion of warmup steps relative to total steps |
|group_by_length	| True	| Bucket sequences by length for efficient batching |
|save_steps	| 0	| Disable checkpoint saving during training |
|logging_steps	| 25	| Logging frequency in steps |

---

### Code Snippet to Load Model and Use 

<pre> """ model_new_name = "kundan05/Llama-2-7b-sql-chat-finetuned-1k"

model_new_sql = AutoModelForCausalLM.from_pretrained(
    model_new_name,
    #quantization_config=bnb_config,
    load_in_8bit=True,
    torch_dtype=torch.bfloat16,
    device_map=device_map
    )

tokenizer_new = AutoTokenizer.from_pretrained(model_new_name, trust_remote_code=True)
tokenizer_new.pad_token = tokenizer.eos_token
tokenizer_new.padding_side = "right"

# Ignore warnings
logging.set_verbosity(logging.CRITICAL)

# Run text generation pipeline with our next model
input= "List the name, born state and age of the heads of departments ordered by age."
context= "CREATE TABLE head (name VARCHAR, born_state VARCHAR, age VARCHAR)"
prompt = f'''You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables.

You must output the SQL query that answers the question.

### Input:
{input}

### Context:
{context}'''

pipe = pipeline(task="text-generation", model=model_new_sql, tokenizer=tokenizer_new, max_length=700)
result = pipe(prompt)
print(result[0]['generated_text'])""" </pre>

### Further Improvement in the process

---

1. Having eval set <br>
2. Using fsdp for sharding <br>
3. Using Custom Kernel for Layer Normalization and Attention in Triton or CUDA <br>