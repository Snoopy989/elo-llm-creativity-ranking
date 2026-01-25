import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # set GPU import quant import torch import torch.nn as nn
import bitsandbytes as bnb
import evaluate
import numpy as np
import pandas as pd
import sys
import torch
torch.cuda.empty_cache()
from datasets import Dataset, DatasetDict
from dataprocessing import preprocess_llm_data
from functools import partial
from lora_misc import *
from pynvml import *
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, LlamaForSequenceClassification, TrainingArguments, Trainer, LlamaConfig
from transformers.trainer_pt_utils import get_parameter_names
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForSequenceClassification
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig,DataCollatorForLanguageModeling, Trainer, TrainingArguments

#  SETTINGS
np.random.seed(42) # sets a randomization seed for reproducibility
model_name = 'meta-llama/Llama-2-7b-hf'
checkpoints_dirs = ['sctt_results_LORA_10_epochs_Llama-2-7b-hf',
                    'sctt_results_LORA_10_epochs_Llama-2-7b-chat-hf']
config = LlamaConfig(model_name, problem_type = "regression")
# model_names = ['meta-llama/Llama-2-7b-hf', 'meta-llama/Llama-2-7b-chat-hf']
epochs = 10
val_pct = 0.10 # proportion of total dataset allocated to validation
test_pct = 0.20  # proportion of the dataset to devote to held-out test set
val_train_pct = (1.0/(1.0-test_pct))*val_pct # we have to get the val set from training subset, so pct needs to be modified
prefix = "A creative "
connector1 = " for "
connector2 = " is " # we'll use prefix/conn to construct inputs to the model
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")  # setting for whether to use gpu or cpu
scaler = MinMaxScaler()
expname = "LORA_{}_epochs".format(epochs)
test_args = TrainingArguments(
  do_train = False,
  do_predict = True,
  per_device_eval_batch_size=4,
  fp16 = True,
  output_dir='./sctt_results_{}_{}'.format(expname,model_name.split('/')[1]),
)

#  LOAD & PREPARE DATA
d = pd.read_csv('all_sctt_jrt.csv')
gen = pd.read_csv('sctt_item-generalization_jrt.csv')

#  PREPROCESS DATA
d = preprocess_llm_data(d, gen, val_pct, val_train_pct, test_pct, prefix, connector1, connector2)

# STORAGE LIST
datadict = []

# LOOP THRU MODELS (foundation/chat)
for model_type in checkpoints_dirs:
  checkpoints = os.listdir(model_type)
  # LOOP THRU CHECKPOINTS WTIHIN MODEL TYPE
  for ind, checkpoint in enumerate(checkpoints):
    peft_model_id = '{}/{}'.format(model_type, checkpoint)
    config = PeftConfig.from_pretrained(peft_model_id)
    inference_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels = 1)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize_function(examples): #  define wrapper tokenizer function (for batch training)
      return tokenizer(examples['text'], padding = 'max_length', truncation = True)
    tokenized_datasets = d.map(tokenize_function, batched = True) # applies wrapper to our dataset
    eval_dataset = tokenized_datasets['validation']
    model = PeftModel.from_pretrained(inference_model, peft_model_id)

    trainer = Trainer(
      model=model,
      args=test_args,
      compute_metrics=compute_metrics,
      tokenizer=tokenizer,
    )
    epoch = trainer.state.epoch
    steps = trainer.state.global_step
    prediction = trainer.predict(eval_dataset)
    predictions_clean = prediction.predictions.flatten()
    labels_clean = prediction.label_ids.flatten()
    num_preds = len(predictions_clean)
    for i in range(num_preds):
      datadict.append({'peft_model_id': peft_model_id, 'steps': steps, 'epoch': epoch, 'predictions': predictions_clean[i], 'ratings': labels_clean[i]})
    del model
    del tokenizer
    del trainer
    del tokenized_datasets
    del eval_dataset
    torch.cuda.empty_cache()

out_df = pd.DataFrame.from_dict(datadict)
out_df.to_csv('epoch_wise_LORA_results.csv', index = False)
