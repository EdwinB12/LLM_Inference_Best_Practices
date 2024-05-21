import matplotlib.pyplot as plt
import seaborn as sns
import torch
import gc
import time
def get_n_word_prob_dict(prompt, model, tokenizer, n=5):
    """
    Returns a dictionary of the top n most likely words to be predicted next with the corresponding probability
    """

    # Tokenize the input prompt
    encoded_input = tokenizer.encode(prompt, return_tensors='pt').to(model.device)

    now = time.monotonic()
    # predict next tokens
    outputs = model(encoded_input)
    time_taken = time.monotonic() - now

    # Get logits from  the final output and convert to probabilities
    probs = outputs.logits[0, -1:].softmax(dim=1).detach().cpu().flatten().numpy()

    # Sort probabilities and pick top n examples
    top_n_tokens = probs.argsort()[::-1][:n]

    # Decode all top n words
    top_n_words = [tokenizer.decode(token) for token in top_n_tokens]

    # Output
    output_seq = tokenizer.decode(model.generate(encoded_input, max_length=len(encoded_input[0]) + 1)[0], skip_special_tokens=True)

    # Return dictionary of words and corresponding probability
    return  output_seq, dict(zip(top_n_words, probs[top_n_tokens])), time_taken

def plot_logits(df, prompt, top_n=5):
    unique_quantisation = df["Quantisation"].unique()
    num_of_plots = len(unique_quantisation)

    fig, axes = plt.subplots(1, num_of_plots, figsize=(10, 5), sharey=True)
    fig.suptitle(f"Top n words and their probabilities for prompt: {prompt}")

    df = df[((df['Prompt'] == prompt) & (df['n'] <= top_n))]

    for i, quantisation in enumerate(unique_quantisation):
        tmp_df = df[df["Quantisation"] == quantisation]
        ax = axes[i]
        sns.barplot(tmp_df, x='Top n Words', y='Top n Probabilities', ax=ax)
        ax.set_title(f"Quantisation: {quantisation}")
    return fig

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024

def flush():
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()

def get_layer_parameters(model, layer_idx=6):
  layer_params = [params for params in model.model.layers[layer_idx].parameters()]
  return layer_params

def get_model_stats(model):
  return {
    "Peak Memory Allocated": bytes_to_giga_bytes(torch.cuda.max_memory_allocated()),
    "Peak Memory Cached": bytes_to_giga_bytes(torch.cuda.max_memory_cached()),
    "Current Memory Allocated": bytes_to_giga_bytes(torch.cuda.memory_allocated()),
    "Current Memory Cached": bytes_to_giga_bytes(torch.cuda.memory_reserved()),
    "Num of Parameters": model.model.num_parameters(),
  }