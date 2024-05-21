from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)

import gradio as gr

# Update the paths to point to the directory of the weights
pegasus_model_name = "./weights/pegasus"
bart_model_name = "./weughts/bart"

# Load pegasus model and tokenizer
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_model_name)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_model_name)

# Load bart model and tokenizer
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name, device_map="auto")
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name, device_map="auto")


DECODER_MAX_LENGTH = 64


def select_model(model_name):
    return model_name


def generate_headline(news, model_name):
    if model_name == "Bart":
        model = bart_model
        tokenizer = bart_tokenizer
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


model_selector = gr.Dropdown(choices=["Bart", "Pegasus"])

demo = gr.Interface(
    fn=generate_headline,
    inputs=["text", model_selector],
    outputs=["text"],
)

demo.launch(share=True)
