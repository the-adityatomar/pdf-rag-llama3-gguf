import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import streamlit as st
import time

MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


@st.cache_resource
def load_model():
    """
    Loads Llama 3 8B in 4-bit quantized mode.
    Intended for GPU-enabled Hugging Face Space.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    model.eval()

    return tokenizer, model


def generate_response(tokenizer, model, prompt, temperature=0.2, max_tokens=512):
    """
    Generates response from Llama 3 model.
    """

    start_time = time.time()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    end_time = time.time()
    latency = round(end_time - start_time, 2)

    return decoded, latency

