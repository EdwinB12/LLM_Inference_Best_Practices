"""
This file is used to test the impact of quantisation on the output ofthe LLama3 Model.
"""
from huggingface_hub import login
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utils import get_n_word_prob_dict
import sys


if __name__ == "__main__":

    # read in command line arguments
    quant_flag = sys.argv[1]

    if quant_flag == "4bit":
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
    elif quant_flag == "8bit":
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant_flag == "16bit":
        quant_config = None


    TOKEN_PATH = ".secrets/hf_token.txt"
    MODEL_ID = "meta-llama/Meta-Llama-3-8B"


    # read huggingface token from hf_token.txt
    with open(TOKEN_PATH, "r") as f:
        hf_token = f.read().strip()

    login(hf_token)

    list_of_prompts = [
        "The Hitchhiker's Guide to the",
        "The capital of France is",
        "Once upon a time, there was a",
        "Putting pineapple on pizza is",
        "The best way to make money is to",
        "I went to the supermarket to buy some",
        "The most important thing in life is"
    ]

    try:
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=quant_config, device_map=0)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    except Exception as e:
        raise Exception(f"Error loading model and tokenizer: {e}")

    df_list = []
    for prompt in list_of_prompts:
        print(f"Prompt: {prompt}")
        _, prob_dict = get_n_word_prob_dict(prompt, model, tokenizer, n=50)
        data_dict = {
            "Prompt": prompt,
            "Top n Words": list(prob_dict.keys()),
            "Top n Probabilities": list(prob_dict.values())
        }

        df = pd.DataFrame(data_dict)
        df_list.append(df)
    main_df = pd.concat(df_list)
    main_df.reset_index(drop=False, inplace=True, names=["n"])
    main_df["n"] += 1
    main_df["Model"] = MODEL_ID
    main_df["Quantisation"] = quant_flag
    main_df.to_csv(f"output_{quant_flag}.csv", index=False)
