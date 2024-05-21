from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)
import torch
from datasets import Dataset
import gradio as gr
import pandas as pd
from tabulate import tabulate

pegasus_model_name = "./results/pegasus"

pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)

DECODER_MAX_LENGTH = 64

torch.manual_seed(1234)


def select_model(model_name):
    return model_name


def generate_headline(test_samples, model):
    inputs = pegasus_tokenizer(
        test_samples["body"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask)
    output_str = pegasus_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


def clean_data(df):
    # Remove rows with missing headlines/body
    df = df.dropna(axis=0).copy()

    # Convert to lowercase
    df["body"] = df["body"].apply(lambda row: row.lower())
    df["title"] = df["title"].apply(lambda row: row.lower())

    # Remove headline from body
    df["body"] = df.apply(lambda row: row["body"].replace(row["title"], ""), axis=1)

    return df


df = pd.read_csv("dataset/news_articles.csv", dtype=pd.StringDtype(), usecols=range(3))
df = df.drop(columns=["category"])
df = clean_data(df)
dataset = Dataset.from_pandas(df)
train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1).values()

test_samples = validation_data_txt.select(range(16))
generated_headlines = generate_headline(test_samples, pegasus_model)[1]
print(
    tabulate(
        zip(
            range(len(generated_headlines)),
            generated_headlines,
            list(test_samples["title"]),
        ),
        headers=["Id", "Generated headlines", "Ground Truth"],
    )
)
