# from threadpoolctl import threadpool_limits

# with threadpool_limits(limits=1, user_api='blas'):

import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




# Load the DataFrame from the CSV file
df = pd.read_csv('videos.csv')
print(df.head())

    # # Preprocess the text and labels
    # df['text'] = df['title'] + " " + df['description'] + " " + df['transcript']

    # # Map domain labels to numeric labels
    # domain_to_id = {domain: idx for idx, domain in enumerate(domains)}
    # df['label'] = df['domain'].apply(lambda x: domain_to_id[x])

    # # Split the dataset
    # train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'])

    # # Load tokenizer and model
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(domains))

    # # Tokenize the data
    # def tokenize_function(examples):
    #     return tokenizer(examples['text'], padding='max_length', truncation=True)

    # train_dataset = Dataset.from_pandas(train_df)
    # val_dataset = Dataset.from_pandas(val_df)

    # train_dataset = train_dataset.map(tokenize_function, batched=True)
    # val_dataset = val_dataset.map(tokenize_function, batched=True)

    # train_dataset = train_dataset.rename_column("label", "labels")
    # val_dataset = val_dataset.rename_column("label", "labels")

    # # Set training arguments
    # training_args = TrainingArguments(
    #     output_dir='./results',
    #     evaluation_strategy="epoch",
    #     learning_rate=2e-5,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=3,
    #     weight_decay=0.01,
    # )

    # # Initialize Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    # )

    # # Train the model
    # trainer.train()

    # # Evaluate the model
    # results = trainer.evaluate()

    # print(results)
