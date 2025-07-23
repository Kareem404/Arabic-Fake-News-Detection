import numpy as np 
import torch 
import transformers
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType 
import evaluate
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from transformers import AutoTokenizer
import os

def convert_credibility(example):
    example['labels'] = 1 if example['labels'] == 'credible' else 0
    return example

def load_training_data(dir:str = "./Split_test") -> tuple:
    X_train_text_tokenize = torch.load(f'{dir}/X_train_text_tokenize.pt')
    y_train = torch.load(f'{dir}/y_train.pt')

    # split the training data into training and validation
    train_data = Dataset.from_dict(X_train_text_tokenize)
    train_data = train_data.add_column('credibility',y_train)
    train_dataset, val_dataset = train_data.train_test_split(test_size=0.2, seed=42).values()
    
    train_dataset = train_dataset.map(lambda x: {'labels': x['credibility']})
    val_dataset = val_dataset.map(lambda x: {'labels': x['credibility']})

    train_dataset = train_dataset.remove_columns(['credibility'])
    val_dataset = val_dataset.remove_columns(['credibility'])   

    train_dataset = train_dataset.map(convert_credibility)
    val_dataset = val_dataset.map(convert_credibility)

    print('loaded data successfully')
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    
    f1_metric = evaluate.load('f1')
    recall_metric = evaluate.load('recall')
    precision_metric = evaluate.load("precision")
    accuracy_metric = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"] # weighted f1 score 

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_lora_model(model_name = 'aubmindlab/bert-base-arabertv02'):

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) # two classes  

    print('loaded model successfully')

    # apply low rank adapatation
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, # sequence classification 
        r=16,  # LoRA rank
        lora_alpha=32,
        lora_dropout=0.2
    )
    
    model = get_peft_model(model, lora_config)
    return model

def train_model(model):
    training_args = TrainingArguments(output_dir="./Split_test/results", 
                                  eval_strategy="epoch",
                                  num_train_epochs=1, # 8
                                  per_device_train_batch_size=16)
    
    arabert_tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv02')
    data_collator = DataCollatorWithPadding(tokenizer=arabert_tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    trainer.train()

if __name__ == '__main__':

    # load training datasets
    train_dataset, val_dataset = load_training_data(dir='./Split_test')

    model = get_lora_model(model_name='aubmindlab/bert-base-arabertv02')
    print(model.print_trainable_parameters()) # prints the number of trainable parameters

    # create a results directory
    os.mkdir(os.path.join('Split_test', 'results'))

    # train the model 
    train_model(model)