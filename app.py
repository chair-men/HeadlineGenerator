from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

import gradio as gr

pegasus_model_name = "./results/pegasus"

bart_model_lowercase_name = "./results/lowercase"
bart_model_lowercase = BartForConditionalGeneration.from_pretrained(
    bart_model_lowercase_name, device_map="auto"
)
bart_tokenizer_lowercase = BartTokenizer.from_pretrained(
    bart_model_lowercase_name, device_map="auto"
)

bart_model_uppercase_name = "./results/no_lowercase"
bart_model_uppercase = BartForConditionalGeneration.from_pretrained(
    bart_model_uppercase_name, device_map="auto"
)
bart_tokenizer_uppercase = BartTokenizer.from_pretrained(
    bart_model_uppercase_name, device_map="auto"
)

pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)

DECODER_MAX_LENGTH = 64


def select_model(model_name):
    return model_name


def generate_headline(news, model_name):
    if model_name == "Bart Lowercase":
        model = bart_model_lowercase
        tokenizer = bart_tokenizer_lowercase
    elif model_name == "Bart Uppercase":
        model = bart_model_uppercase
        tokenizer = bart_tokenizer_uppercase
    else:
        model = pegasus_model
        tokenizer = pegasus_tokenizer
    inp = tokenizer(
        news,
        padding="max_length",
        truncation=True,
        max_length=DECODER_MAX_LENGTH,
        return_tensors="pt",
    )
    input_id = inp.input_ids.to(model.device)
    attention_mask = inp.attention_mask.to(model.device)
    output = model.generate(input_id, attention_mask=attention_mask)
    output_str = tokenizer.batch_decode(output, skip_special_tokens=True)
    return output_str


model_selector = gr.Dropdown(choices=["Bart Lowercase", "Bart Uppercase", "Pegasus"])

demo = gr.Interface(
    fn=generate_headline,
    inputs=["text", model_selector],
    outputs=["text"],
)

demo.launch(share=True)
