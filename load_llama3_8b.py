"""
This file is used to test the impact of quantisation on the output ofthe LLama3 Model.
"""
from huggingface_hub import login
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_n_word_prob_dict, get_model_stats, get_layer_parameters, flush, bytes_to_giga_bytes, predict
import sys
from transformers import set_seed
import torch
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":

    # read in command line arguments
    quant_flag = sys.argv[1]
    device_map = sys.argv[2]

    if quant_flag == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        torch_dtype=None
    elif quant_flag == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype=None
    elif quant_flag == "fp16":
        quant_config = None
        torch_dtype=torch.float16
    elif quant_flag == "bfp16":
        quant_config = None
        torch_dtype=torch.bfloat16


    TOKEN_PATH = ".secrets/hf_token.txt"
    MODEL_ID = "meta-llama/Meta-Llama-3-8B"
    MODEL_NAME = "Meta-Llama-3-8B"

    set_seed(42)

    # read huggingface token from hf_token.txt
    with open(TOKEN_PATH, "r") as f:
        hf_token = f.read().strip()

    login(hf_token)

    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map=device_map, torch_dtype=torch_dtype)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        raise Exception(f"Error loading model and tokenizer: {e}")

    # Get the model weights
    model_stats = get_model_stats(model)
    # Save model stats as a csv
    model_stats_df = pd.DataFrame().from_dict([model_stats])


    # Repeat a prompt and measure inference speed
    prompt = "Once upon a time"
    inference_times = []
    for i in range(10):
        _, time_taken = predict(prompt, model, tokenizer)
        inference_times.append(time_taken)

    # Save inference times
    mean_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)

    # add to dataframe
    model_stats_df['Mean Inference Time (s)'] = mean_inference_time
    model_stats_df['Std Inference Time (s)'] = std_inference_time

    model_stats_df.to_csv(f"model_stats_{MODEL_NAME}_{quant_flag}_{device_map}.csv", index=False)

