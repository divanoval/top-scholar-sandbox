# ARENA_Content.ipynb

```python
# NBVAL_IGNORE_OUTPUT
import os

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
IN_GITHUB = True
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False

if not IN_GITHUB and not IN_COLAB:
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_GITHUB or IN_COLAB:
    %pip install torch
    %pip install git+https://github.com/TransformerLensOrg/TransformerLens.git@dev

from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch as t

device = t.device("cuda" if t.cuda.is_available() else "cpu")
```

```python
# NBVAL_IGNORE_OUTPUT

reference_gpt2 = HookedTransformer.from_pretrained(
    "gpt2-small",
    fold_ln=False,
    center_unembed=False,
    center_writing_weights=False,
    device=device,
)
```

```python

# [1.1] Transformer From Scratch
# 1️⃣ UNDERSTANDING INPUTS & OUTPUTS OF A TRANSFORMER

sorted_vocab = sorted(list(reference_gpt2.tokenizer.vocab.items()), key=lambda n: n[1])
first_vocab = sorted_vocab[0]
assert isinstance(first_vocab, tuple)
assert isinstance(first_vocab[0], str)
first_vocab[1]
```

```python
reference_gpt2.to_str_tokens("Ralph")
```

```python
reference_gpt2.to_str_tokens(" Ralph")
```

```python

reference_gpt2.to_str_tokens(" ralph")

```

```python
reference_gpt2.to_str_tokens("ralph")
```

```python

reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
tokens.shape

```

```python

logits, cache = reference_gpt2.run_with_cache(tokens, device=device)
logits.shape

```

```python

most_likely_next_tokens = reference_gpt2.tokenizer.batch_decode(logits.argmax(dim=-1)[0])
most_likely_next_tokens[-1]

```

```python
# 2️⃣ CLEAN TRANSFORMER IMPLEMENTATION

layer_0_hooks = [
    (name, tuple(tensor.shape)) for name, tensor in cache.items() if ".0." in name
]
non_layer_hooks = [
    (name, tuple(tensor.shape)) for name, tensor in cache.items() if "blocks" not in name
]

sorted(non_layer_hooks, key=lambda x: x[0])

```

```python

sorted(layer_0_hooks, key=lambda x: x[0])
```

```python
# NBVAL_IGNORE_OUTPUT
# [1.2] Intro to mech interp
# 2️⃣ FINDING INDUCTION HEADS

cfg = HookedTransformerConfig(
    d_model=768,
    d_head=64,
    n_heads=12,
    n_layers=2,
    n_ctx=2048,
    d_vocab=50278,
    attention_dir="causal",
    attn_only=True, # defaults to False
    tokenizer_name="EleutherAI/gpt-neox-20b",
    seed=398,
    use_attn_result=True,
    normalization_type=None, # defaults to "LN", i.e. layernorm with weights & biases
    positional_embedding_type="shortformer"
)
model = HookedTransformer(cfg)
```

```python

text = "We think that powerful, significantly superhuman machine intelligence is more likely than not to be created this century. If current machine learning techniques were scaled up to this level, we think they would by default produce systems that are deceptive or manipulative, and that no solid plans are known for how to avoid this."

logits, cache = model.run_with_cache(text, remove_batch_dim=True)

logits.shape
```

```python
cache["embed"].ndim
```

---

# Activation_Patching_in_TL_Demo.ipynb


---

# Attribution_Patching_Demo.ipynb


---

# BERT.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/BERT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# BERT in TransformerLens
This demo shows how to use BERT in TransformerLens for the Masked Language Modelling and Next Sentence Prediction task.

# Setup
(No need to read)

```python
# NBVAL_IGNORE_OUTPUT
import os

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False

if not IN_GITHUB and not IN_COLAB:
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB:
    %pip install transformer_lens
    %pip install circuitsvis
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")
```

```python
import circuitsvis as cv

# Testing that the library works
cv.examples.hello("Neel")
```

```python
# Import stuff
import torch

from transformers import AutoTokenizer

from transformer_lens import HookedEncoder, BertNextSentencePrediction
```

```python
torch.set_grad_enabled(False)
```

# BERT

In this section, we will load a pretrained BERT model and use it for the Masked Language Modelling and Next Sentence Prediction task

```python
# NBVAL_IGNORE_OUTPUT
bert = HookedEncoder.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

## Masked Language Modelling
Use the "[MASK]" token to mask any tokens which you would like the model to predict.
When specifying return_type="predictions" the prediction of the model is returned, alternatively (and by default) the function returns logits.
You can also specify None as return type for which nothing is returned

```python
prompt = "The [MASK] is bright today."

prediction = bert(prompt, return_type="predictions")

print(f"Prompt: {prompt}")
print(f'Prediction: "{prediction}"')
```

You can also input a list of prompts:

```python
prompts = ["The [MASK] is bright today.", "She [MASK] to the store.", "The dog [MASK] the ball."]

predictions = bert(prompts, return_type="predictions")

print(f"Prompt: {prompts}")
print(f'Prediction: "{predictions}"')
```

## Next Sentence Prediction
To carry out Next Sentence Prediction, you have to use the class BertNextSentencePrediction, and pass a HookedEncoder in its constructor.
Then, create a list with the two sentences you want to perform NSP on as elements and use that as input to the forward function.
The model will then predict the probability of the sentence at position 1 following (i.e. being the next sentence) to the sentence at position 0.

```python
nsp = BertNextSentencePrediction(bert)
sentence_a = "A man walked into a grocery store."
sentence_b = "He bought an apple."

input = [sentence_a, sentence_b]

predictions = nsp(input, return_type="predictions")

print(f"Sentence A: {sentence_a}")
print(f"Sentence B: {sentence_b}")
print(f'Prediction: "{predictions}"')
```

# Inputting tokens directly
You can also input tokens instead of a string or a list of strings into the model, which could look something like this

```python
prompt = "The [MASK] is bright today."

tokens = tokenizer(prompt, return_tensors="pt")["input_ids"]
logits = bert(tokens) # Since we are not specifying return_type, we get the logits
logprobs = logits[tokens == tokenizer.mask_token_id].log_softmax(dim=-1)
prediction = tokenizer.decode(logprobs.argmax(dim=-1).item())

print(f"Prompt: {prompt}")
print(f'Prediction: "{prediction}"')
```

Well done, BERT!

---

# Colab_Compatibility.ipynb

```python
# NBVAL_IGNORE_OUTPUT
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"

try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB or IN_GITHUB:
    # %pip install sentencepiece # Llama tokenizer requires sentencepiece
    %pip install transformers>=4.31.0 # Llama requires transformers>=4.31.0 and transformers in turn requires Python 3.8
    %pip install torch
    %pip install tiktoken
    # %pip install transformer_lens
    %pip install transformers_stream_generator
    # !huggingface-cli login --token NEEL'S TOKEN
```

```python
import torch
from transformer_lens import HookedTransformer, HookedEncoderDecoder, HookedEncoder, BertNextSentencePrediction, loading
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
from typing import List
import gc

untested_models = []
untested_models.extend(loading.OFFICIAL_MODEL_NAMES)

print("TransformerLens currently supports " + str(len(untested_models)) + " models out of the box.")

GENERATE = True
# Fill this in if you have llama weights uploaded, and you with to test those models
LLAMA_MODEL_PATH = ""
```

```python
def mark_models_as_tested(model_set: List[str]) -> None:
    for model in model_set:
        untested_models.remove(model)

def run_set(model_set: List[str], device="cuda") -> None:
    for model in model_set:
        print("Testing " + model)
        tl_model = HookedTransformer.from_pretrained_no_processing(model, device=device)
        if GENERATE:
            print(tl_model.generate("Hello my name is"))
        del tl_model
        gc.collect()
        if IN_COLAB:
            %rm -rf /root/.cache/huggingface/hub/models*

def run_llama_set(model_set: List[str], weight_root: str, device="cuda") -> None:
    for model in model_set:
        print("Testing " + model)
        # to run this, make sure weight root is the root that contains all models with the
        # sub directories sharing the same name as the model in the list of models
        tokenizer = LlamaTokenizer.from_pretrained(weight_root + model)
        hf_model = LlamaForCausalLM.from_pretrained(weight_root + model, low_cpu_mem_usage=True)
        tl_model = HookedTransformer.from_pretrained_no_processing(
            model,
            hf_model=hf_model,
            device=device,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            tokenizer=tokenizer,
        )
        if GENERATE:
            print(tl_model.generate("Hello my name is"))
        del tl_model
        gc.collect()
        if IN_COLAB:
            %rm -rf /root/.cache/huggingface/hub/models*

def run_encoder_decoder_set(model_set: List[str], device="cuda") -> None:
    for model in model_set:
        print("Testing " + model)
        tokenizer = AutoTokenizer.from_pretrained(model)
        tl_model = HookedEncoderDecoder.from_pretrained(model, device=device)
        if GENERATE:
            # Originally from the t5 demo
            prompt = "Hello, how are you? "
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            decoder_input_ids = torch.tensor([[tl_model.cfg.decoder_start_token_id]]).to(input_ids.device)

            while True:
                logits = tl_model.forward(input=input_ids, one_zero_attention_mask=attention_mask, decoder_input=decoder_input_ids)
                # logits.shape == (batch_size (1), predicted_pos, vocab_size)

                token_idx = torch.argmax(logits[0, -1, :]).item()
                print("generated token: \"", tokenizer.decode(token_idx), "\", token id: ", token_idx, sep="")

                # append token to decoder_input_ids
                decoder_input_ids = torch.cat([decoder_input_ids, torch.tensor([[token_idx]]).to(input_ids.device)], dim=-1)

                # break if End-Of-Sequence token generated
                if token_idx == tokenizer.eos_token_id:
                    break
        del tl_model
        gc.collect()
        if IN_COLAB:
            %rm -rf /root/.cache/huggingface/hub/models*

def run_encoder_only_set(model_set: List[str], device="cuda") -> None:
    for model in model_set:
        print("Testing " + model)
        tl_model = HookedEncoder.from_pretrained(model, device=device)
        tl_model_nsp = NextSentencePrediction.from_pretrained(model, device=device)

        if GENERATE:
            print("Testing Masked Language Modelling:")
            # Slightly adapted version of the BERT demo
            prompt = "The capital of France is [MASK]."

            prediction = tl_model(prompt, return_type="predictions")

            print(f"Prompt: {prompt}")
            print(f'Prediction: "{prediction}"')

            print("Testing Next Sentence Prediction:")
            sentence_a = "She went to the grocery store."
            sentence_b = "She bought some milk."

            prediction = tl_model_nsp([sentence_a, sentence_b], return_type="predictions")

            print(f"Sentence A: {sentence_a}")
            print(f"Sentence B: {sentence_b}")
            print(f"Prediction: {prediction}")

        del tl_model
        gc.collect()
        if IN_COLAB:
            %rm -rf /root/.cache/huggingface/hub/models*
```

```python
# The following models can run in the T4 free environment
free_compatible = [
    "ai-forever/mGPT",
    "ArthurConmy/redwood_attn_2l",
    "bigcode/santacoder",
    "bigscience/bloom-1b1",
    "bigscience/bloom-560m",
    "distilgpt2",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-2.7B",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-1.4b-deduped-v0",
    "EleutherAI/pythia-1.4b-v0",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3",
    "EleutherAI/pythia-160m-v0",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1b-deduped-v0",
    "EleutherAI/pythia-1b-v0",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/pythia-70m-v0",
    "facebook/opt-1.3b",
    "facebook/opt-125m",
    "gpt2",
    "gpt2-large",
    "gpt2-medium",
    "gpt2-xl",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/phi-1",
    "microsoft/phi-1_5",
    "NeelNanda/Attn-Only-2L512W-Shortformer-6B-big-lr",
    "NeelNanda/Attn_Only_1L512W_C4_Code",
    "NeelNanda/Attn_Only_2L512W_C4_Code",
    "NeelNanda/Attn_Only_3L512W_C4_Code",
    "NeelNanda/Attn_Only_4L512W_C4_Code",
    "NeelNanda/GELU_1L512W_C4_Code",
    "NeelNanda/GELU_2L512W_C4_Code",
    "NeelNanda/GELU_3L512W_C4_Code",
    "NeelNanda/GELU_4L512W_C4_Code",
    "NeelNanda/SoLU_10L1280W_C4_Code",
    "NeelNanda/SoLU_10L_v22_old",
    "NeelNanda/SoLU_12L1536W_C4_Code",
    "NeelNanda/SoLU_12L_v23_old",
    "NeelNanda/SoLU_1L512W_C4_Code",
    "NeelNanda/SoLU_1L512W_Wiki_Finetune",
    "NeelNanda/SoLU_1L_v9_old",
    "NeelNanda/SoLU_2L512W_C4_Code",
    "NeelNanda/SoLU_2L_v10_old",
    "NeelNanda/SoLU_3L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_C4_Code",
    "NeelNanda/SoLU_4L512W_Wiki_Finetune",
    "NeelNanda/SoLU_4L_v11_old",
    "NeelNanda/SoLU_6L768W_C4_Code",
    "NeelNanda/SoLU_6L_v13_old",
    "NeelNanda/SoLU_8L1024W_C4_Code",
    "NeelNanda/SoLU_8L_v21_old",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "roneneldan/TinyStories-1Layer-21M",
    "roneneldan/TinyStories-1M",
    "roneneldan/TinyStories-28M",
    "roneneldan/TinyStories-2Layers-33M",
    "roneneldan/TinyStories-33M",
    "roneneldan/TinyStories-3M",
    "roneneldan/TinyStories-8M",
    "roneneldan/TinyStories-Instruct-1M",
    "roneneldan/TinyStories-Instruct-28M",
    "roneneldan/TinyStories-Instruct-2Layers-33M",
    "roneneldan/TinyStories-Instruct-33M",
    "roneneldan/TinyStories-Instruct-3M",
    "roneneldan/TinyStories-Instruct-8M",
    "roneneldan/TinyStories-Instuct-1Layer-21M",
    "stanford-crfm/alias-gpt2-small-x21",
    "stanford-crfm/arwen-gpt2-medium-x21",
    "stanford-crfm/battlestar-gpt2-small-x49",
    "stanford-crfm/beren-gpt2-medium-x49",
    "stanford-crfm/caprica-gpt2-small-x81",
    "stanford-crfm/celebrimbor-gpt2-medium-x81",
    "stanford-crfm/darkmatter-gpt2-small-x343",
    "stanford-crfm/durin-gpt2-medium-x343",
    "stanford-crfm/eowyn-gpt2-medium-x777",
    "stanford-crfm/expanse-gpt2-small-x777",
]

if IN_COLAB:
    run_set(free_compatible)

mark_models_as_tested(free_compatible)
```

```python
paid_gpu_models = [
    "01-ai/Yi-6B",
    "01-ai/Yi-6B-Chat",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-7b-Python-hf",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-2.8b-deduped-v0",
    "EleutherAI/pythia-2.8b-v0",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-6.9b-deduped-v0",
    "EleutherAI/pythia-6.9b-v0",
    "facebook/opt-2.7b",
    "facebook/opt-6.7b",
    "google/gemma-2-2b",
    "google/gemma-2-2b-it",
    "google/gemma-2b",
    "google/gemma-2b-it",
    "google/gemma-7b",
    "google/gemma-7b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "microsoft/phi-2",
    "microsoft/Phi-3-mini-4k-instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-Nemo-Base-2407",
    "mistralai/Mistral-Small-24B-Base-2501",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "stabilityai/stablelm-base-alpha-3b",
    "stabilityai/stablelm-base-alpha-7b",
    "stabilityai/stablelm-tuned-alpha-3b",
    "stabilityai/stablelm-tuned-alpha-7b",
]

if IN_COLAB:
    run_set(paid_gpu_models)

mark_models_as_tested(paid_gpu_models)
```

```python
paid_cpu_models = [
    "EleutherAI/gpt-j-6B",
    "EleutherAI/gpt-neox-20b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-12b-deduped-v0",
    "EleutherAI/pythia-12b-v0",
    "facebook/opt-13b",
    "google/gemma-2-9b",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "microsoft/phi-4",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
]

if IN_COLAB:
    run_set(paid_cpu_models, "cpu")

mark_models_as_tested(paid_cpu_models)
```

```python
incompatible_models = [
    "01-ai/Yi-34B",
    "01-ai/Yi-34B-Chat",
    "facebook/opt-30b",
    "facebook/opt-66b",
    "google/gemma-2-27b",
    "google/gemma-2-27b-it",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x7B-v0.1",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen3-14B",
    "Qwen/QwQ-32B-Preview",
]

mark_models_as_tested(incompatible_models)
```

```python
# The following models take a few extra steps to function. Check the official demo for more
# information on how to use. 7b and 13b will work in the paid environment. 30b and 65b will not work
# in Colab
not_hosted_models = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]

if LLAMA_MODEL_PATH:
    run_llama_set(not_hosted_models, LLAMA_MODEL_PATH)

mark_models_as_tested(not_hosted_models)
```

```python
# These all work on the free version of Colab
encoder_decoders = [
    "google-t5/t5-base",
    "google-t5/t5-large",
    "google-t5/t5-small",
]
if IN_COLAB:
    run_encoder_decoder_set(encoder_decoders)

mark_models_as_tested(encoder_decoders)
```

```python
# This model works on the free version of Colab
encoder_only_models = [
    "google-bert/bert-base-cased",
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-cased",
    "google-bert/bert-large-uncased",
]

if IN_COLAB:
    run_encoder_only_set(encoder_only_models)

mark_models_as_tested(encoder_only_models)
```

```python
broken_models = [
    "Baidicoot/Othello-GPT-Transformer-Lens",
]
```

```python
# Any models listed in the cell below have not been tested. This should always remain blank. If your
# PR fails due to this notebook, most likely you need to check any new model changes to ensure that
# this notebook is up to date.
print(*untested_models, sep="\n")
```

---

# Config_Overhaul.ipynb

# Overview

The current way configuration is designed in TransformerLens has a lot of limitations. It does not
allow for outside people to pass through configurations that are not officially supported, and it
is very bug prone with something as simple as typo potentially giving you a massive headache. There
are also a number of hidden rules that are not clearly documented, which can go hidden until
different pieces of TransformerLens are activated. Allowing to pass in an optional object of configuration
with no further changes does solve a couple of these problems, but it does not solve the bigger
issues. It also introduces new problems with users potentially passing in architectures that are not
supported without having a clear way to inform the user what isn't supported.

My proposal for how all of these problems can be resolved is to fundamentally revamp the
configuration to allow for something that I like to call configuration composition. From a technical
perspective, this involves creating a centralized class that describes all supported configurations
by TransformerLens. This class would then be used to construct specific configurations for all models
that are currently supported, and it would then allow anyone to easily see in a single place all
configuration features supported by TransformerLens while also being able to read the code to
understand how they can create their own configurations for the purpose of either submitting new
models into TransformerLens, or configuring an unofficially supported model by TransformerLens,
when TransformerLens already happens to support all of the architectural pieces separately.

This could simple be an overhaul of the existing HookedTransformerConfig. Everything I am
describing here could be made compatible with that class to give it a more usable interface that is
then directly interacted with by the end user. At the moment, that class is not really built to be
interacted with, and is instead used as a wrapper around spreading configured anonymous objects.
Overhauling this class to do what I am about to describe is a viable path, but keeping it as it is,
and making a new class as something meant to be used by the end user would be a way to maintain
compatibility, avoid refactors, and keep model configuration only focused on putting together
configuration for models, as opposed to configuring full settings needed by HookedTransformer, which
includes checking the available environment.

A very unscientific basic example of how this would look in code by the end user can be seen
immediately below. I will delve into details of each piece in this document.

```python
config = ModelConfig(
    d_model=4096,
    d_head=8192 // 64,
    n_heads=64,
    act_fn="silu"
    # Other universally required properties across all models go here in the constructor
)
# Enabling specific features not universal among all models
config.enabled_gated_mlp()
# Customizing optional attributes
config.set_positional_embedding_type("alibi")

# and so on, until the full configuration is set

```

## The constructor

The first piece of this I want to talk about is what will be injected into the constructor. It
should basically take everything absolutely required by all models. This keeps the code easy for
someone to understand, without adding too much clutter. All fields should be required, and if there
is ever an idea that a field should be in the constructor as an option, then that is probably an
indication that there is a good case to add a function to configure that variable in a different
point in the class. An example of what this would look like can be seen below...

```python
# make it easy for someone to see what activation functions are supported, this would be moved from
# HookedTransformerConfig
ActivationFunction = "silu" | "gelu"

class ModelConfig:
    def __init__(
        self,
        d_model: int,
        eps: int,
        act_fn: ActivationFunction,
        remaining_required_attributes,
    ):
        self.d_model = d_model
        self.eps = eps
        self.act_fn = act_fn
        # Set defaults for any remaining supported attributes that are not required here
        self.gated_mlp = False

```

## Boolean Variables

Within TransformerLens config, anything that is a boolean variable is essentially a feature flag.
This means that all features at the time of construction would have default values, most likely set
to false. They then get toggled on with an `enable_feature` function call on the config object.
Having these functions will make very clear for someone less familiar with TransformerLens what
features are available. It also allows us to decorate these calls, which is very important. There
are some instances where if a boolean is true, a different one cannot be true, but this requirement
is not clear anywhere without analyzing code. Decorating these functions allows us to make sure
these sort of bugs are not possible. I will use `gated_mlp` as an example here, but it is not
meant to be a real implementation.

```python
def enabled_gated_mlp(self: ModelConfig) -> ModelConfig:
    self.gated_mlp = True
    # Configure any side effects caused by enabling of a feature
    self.another_feature = False
    # Returning self allows someone to chain together config calls
    return self

ModelConfig.enabled_gated_mlp = enabled_gated_mlp
```

## Additional Options

Any other options would similarly have their own functions to configure. This allows for similar
decoration as with feature flags, and it also in a way documents the architectural capabilities of
TransformerLens in a single place. If there are groups of options that are also always required
together, this then gives us a way to require all of those options as opposed to having them all be
configured at the root level. This also allows us to make changes to other attributes that may be
affected as a side affect of having some values set, which again makes it both harder for people to
introduce bugs, and also creates code that documents itself. Another off the cuff example of
something like this can be seen below.

```python
def set_rotary_dim(self: ModelConfig, rotary_dim: int) -> ModelConfig:
    self.rotary_dim = rotary_dim
    # Additional settings that seem to be present whenever rotary_dim is set
    self.positional_embedding_type = "rotary"
    self.rotary_adjacent_pairs = False
    return self

ModelConfig.set_rotary_dim = set_rotary_dim
```

## Config Final Thoughts

The best way to describe this idea is configuration composition. The reason being is that the user is
essentially composing a model configuration by setting the base, and then combining various options
from predefined functions. Doing it like this has a lot of advantages. One of those advantages being
that there would need to be a lot less memorization on how architectures should be combined. e.g.
maybe it's not that hard to remember that `rotary_adjacent_pairs` should be False when `rotary_dim`
is set, but these sorts of combinations accumulate. Having it interfaced out gives everyone a
place to look to see how parts of configuration work in isolation without the need to memorize a
large amount of rules.

This would also allow us to more easily mock out fake configurations and enable specific features in
order to test that functionality in isolation. This also should make it easier for someone to at a
glance understand all model compatibilities with TransformerLens, since there would be a single file
where they would all be listed out and documented. It will also allow for people to see
compatibility limitations at a glance.

As for compatibility, this change would be 100% compatible with the existing structure. The objects
I am suggesting are abstractions of the existing configuration dictionaries for the purpose of
communication and ease of use. This means that they can be passed around just like the current
anonymous dictionaries.

## Further Changes

With this, there are a number of changes that I would like to make to the actual
`loading_from_pretrained` file in order to revise it to be ready for the possibility of rapidly
supporting new models. The biggest change in this respect would be to break out what is now a
configuration dictionary for every model into having its own module where one of these configuration
objects would be constructed. That object would then be exposed, so that it can be imported into
`loading_from_pretrained`. We would then create a dictionary where the official name of the
model would have the configuration object as its value, thus completely eliminating that big giant
if else statement, and replacing it with a simple return from the dictionary. The configurations
themselves would then live in a directory structure like so...

config/ <- where the ModelConfig file lives
config/meta-llama/ <- directory for all models from the group
config/meta-llama/Llama-2-13b.py <- name matching hugging face to make it really easy to find the
                                    configuration

## Impact on Testing

This change, would allow us to directly interact with these configuration objects to allow us to
more easily assert that configurations are set properly, and to also allow us to more easily access
these configurations in tests for the purposes of writing better unit tests.

## Summary

This change should solve a lot of problems. It may be a big change at first from what currently
exists, but in time I think most people will find it more elegant, and easier to understand.

```python

```

---

# Exploratory_Analysis_Demo.ipynb

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb)

# Exploratory Analysis Demo

This notebook demonstrates how to use the
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/) library to perform exploratory
analysis. The notebook tries to replicate the analysis of the Indirect Object Identification circuit
in the [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) paper.

## Tips for Reading This

* If running in Google Colab, go to Runtime > Change Runtime Type and select GPU as the hardware
accelerator.
* Look up unfamiliar terms in [the mech interp explainer](https://neelnanda.io/glossary)
* You can run all this code for yourself
* The graphs are interactive
* Use the table of contents pane in the sidebar to navigate (in Colab) or VSCode's "Outline" in the
  explorer tab.
* Collapse irrelevant sections with the dropdown arrows
* Search the page using the search in the sidebar (with Colab) not CTRL+F

## Setup

### Environment Setup (ignore)

**You can ignore this part:** It's just for use internally to setup the tutorial in different
environments. You can delete this section if using in your own repo.

```python

# Detect if we're running in Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False

# Install if in Colab
if IN_COLAB:
    %pip install transformer_lens
    %pip install circuitsvis
    # Install a faster Node version
    !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs  # noqa

# Hot reload in development mode & not running on the CD
if not IN_COLAB:
    from IPython import get_ipython
    ip = get_ipython()
    if not ip.extension_manager.loaded:
        ip.extension_manager.load('autoreload')
        %autoreload 2

```

### Imports

```python
from functools import partial
from typing import List, Optional, Union

import einops
import numpy as np
import plotly.express as px
import plotly.io as pio
import torch
from circuitsvis.attention import attention_heads
from fancy_einsum import einsum
from IPython.display import HTML, IFrame
from jaxtyping import Float

import transformer_lens.utils as utils
from transformer_lens import ActivationCache, HookedTransformer
```

### PyTorch Setup

We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

```python
torch.set_grad_enabled(False)
print("Disabled automatic differentiation")
```

### Plotting Helper Functions (ignore)

Some plotting helper functions are included here (for simplicity).

```python
def imshow(tensor, **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        **kwargs,
    ).show()

def line(tensor, **kwargs):
    px.line(
        y=utils.to_numpy(tensor),
        **kwargs,
    ).show()

def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y,
        x=x,
        labels={"x": xaxis, "y": yaxis, "color": caxis},
        **kwargs,
    ).show()
```

## Introduction

This is a demo notebook for [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), a library for mechanistic interpretability of GPT-2 style transformer language models. A core design principle of the library is to enable exploratory analysis - one of the most fun parts of mechanistic interpretability compared to normal ML is the extremely short feedback loops! The point of this library is to keep the gap between having an experiment idea and seeing the results as small as possible, to make it easy for **research to feel like play** and to enter a flow state.

The goal of this notebook is to demonstrate what exploratory analysis looks like in practice with the library. I use my standard toolkit of basic mechanistic interpretability techniques to try interpreting a real circuit in GPT-2 small. Check out [the main demo](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Main_Demo.ipynb) for an introduction to the library and how to use it.

Stylistically, I will go fairly slowly and explain in detail what I'm doing and why, aiming to help convey how to do this kind of research yourself! But the code itself is written to be simple and generic, and easy to copy and paste into your own projects for different tasks and models.

Details tags contain asides, flavour + interpretability intuitions. These are more in the weeds and you don't need to read them or understand them, but they're helpful if you want to learn how to do mechanistic interpretability yourself! I star the ones I think are most important.
<details><summary>(*) Example details tag</summary>Example aside!</details>

### Indirect Object Identification

The first step when trying to reverse engineer a circuit in a model is to identify *what* capability
I want to reverse engineer. Indirect Object Identification is a task studied in Redwood Research's
excellent [Interpretability in the Wild](https://arxiv.org/abs/2211.00593) paper (see [my interview
with the authors](https://www.youtube.com/watch?v=gzwj0jWbvbo) or [Kevin Wang's Twitter
thread](https://threadreaderapp.com/thread/1587601532639494146.html) for an overview). The task is
to complete sentences like "After John and Mary went to the shops, John gave a bottle of milk to"
with " Mary" rather than " John".

In the paper they rigorously reverse engineer a 26 head circuit, with 7 separate categories of heads
used to perform this capability. Their rigorous methods are fairly involved, so in this notebook,
I'm going to skimp on rigour and instead try to speed run the process of finding suggestive evidence
for this circuit!

The circuit they found roughly breaks down into three parts:
1. Identify what names are in the sentence
2. Identify which names are duplicated
3. Predict the name that is *not* duplicated

The first step is to load in our model, GPT-2 Small, a 12 layer and 80M parameter transformer with `HookedTransformer.from_pretrained`. The various flags are simplifications that preserve the model's output but simplify its internals.

```python
# NBVAL_IGNORE_OUTPUT
model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)

# Get the default device used
device: torch.device = utils.get_device()
```

The next step is to verify that the model can *actually* do the task! Here we use `utils.test_prompt`, and see that the model is significantly better at predicting Mary than John!

<details><summary>Asides:</summary>

Note: If we were being careful, we'd want to run the model on a range of prompts and find the average performance

`prepend_bos` is a flag to add a BOS (beginning of sequence) to the start of the prompt. GPT-2 was not trained with this, but I find that it often makes model behaviour more stable, as the first token is treated weirdly.
</details>

```python
example_prompt = "After John and Mary went to the store, John gave a bottle of milk to"
example_answer = " Mary"
utils.test_prompt(example_prompt, example_answer, model, prepend_bos=True)
```

We now want to find a reference prompt to run the model on. Even though our ultimate goal is to reverse engineer how this behaviour is done in general, often the best way to start out in mechanistic interpretability is by zooming in on a concrete example and understanding it in detail, and only *then* zooming out and verifying that our analysis generalises.

We'll run the model on 4 instances of this task, each prompt given twice - one with the first name as the indirect object, one with the second name. To make our lives easier, we'll carefully choose prompts with single token names and the corresponding names in the same token positions.

<details> <summary>(*) <b>Aside on tokenization</b></summary>

We want models that can take in arbitrary text, but models need to have a fixed vocabulary. So the solution is to define a vocabulary of **tokens** and to deterministically break up arbitrary text into tokens. Tokens are, essentially, subwords, and are determined by finding the most frequent substrings - this means that tokens vary a lot in length and frequency!

Tokens are a *massive* headache and are one of the most annoying things about reverse engineering language models... Different names will be different numbers of tokens, different prompts will have the relevant tokens at different positions, different prompts will have different total numbers of tokens, etc. Language models often devote significant amounts of parameters in early layers to convert inputs from tokens to a more sensible internal format (and do the reverse in later layers). You really, really want to avoid needing to think about tokenization wherever possible when doing exploratory analysis (though, of course, it's relevant later when trying to flesh out your analysis and make it rigorous!). HookedTransformer comes with several helper methods to deal with tokens: `to_tokens, to_string, to_str_tokens, to_single_token, get_token_position`

**Exercise:** I recommend using `model.to_str_tokens` to explore how the model tokenizes different strings. In particular, try adding or removing spaces at the start, or changing capitalization - these change tokenization!</details>

```python
prompt_format = [
    "When John and Mary went to the shops,{} gave the bag to",
    "When Tom and James went to the park,{} gave the ball to",
    "When Dan and Sid went to the shops,{} gave an apple to",
    "After Martin and Amy went to the park,{} gave a drink to",
]
names = [
    (" Mary", " John"),
    (" Tom", " James"),
    (" Dan", " Sid"),
    (" Martin", " Amy"),
]
# List of prompts
prompts = []
# List of answers, in the format (correct, incorrect)
answers = []
# List of the token (ie an integer) corresponding to each answer, in the format (correct_token, incorrect_token)
answer_tokens = []
for i in range(len(prompt_format)):
    for j in range(2):
        answers.append((names[i][j], names[i][1 - j]))
        answer_tokens.append(
            (
                model.to_single_token(answers[-1][0]),
                model.to_single_token(answers[-1][1]),
            )
        )
        # Insert the *incorrect* answer to the prompt, making the correct answer the indirect object.
        prompts.append(prompt_format[i].format(answers[-1][1]))
answer_tokens = torch.tensor(answer_tokens).to(device)
print(prompts)
print(answers)
```

**Gotcha**: It's important that all of your prompts have the same number of tokens. If they're different lengths, then the position of the "final" logit where you can check logit difference will differ between prompts, and this will break the below code. The easiest solution is just to choose your prompts carefully to have the same number of tokens (you can eg add filler words like The, or newlines to start).

There's a range of other ways of solving this, eg you can index more intelligently to get the final logit. A better way is to just use left padding by setting `model.tokenizer.padding_side = 'left'` before tokenizing the inputs and running the model; this way, you can use something like `logits[:, -1, :]` to easily access the final token outputs without complicated indexing. TransformerLens checks the value of `padding_side` of the tokenizer internally, and if the flag is set to be `'left'`, it adjusts the calculation of absolute position embedding and causal masking accordingly.

In this demo, though, we stick to using the prompts of the same number of tokens because we want to show some visualisations aggregated along the batch dimension later in the demo.

```python
for prompt in prompts:
    str_tokens = model.to_str_tokens(prompt)
    print("Prompt length:", len(str_tokens))
    print("Prompt as tokens:", str_tokens)
```

We now run the model on these prompts and use `run_with_cache` to get both the logits and a cache of all internal activations for later analysis

```python
tokens = model.to_tokens(prompts, prepend_bos=True)

# Run the model and cache all activations
original_logits, cache = model.run_with_cache(tokens)
```

We'll later be evaluating how model performance differs upon performing various interventions, so it's useful to have a metric to measure model performance. Our metric here will be the **logit difference**, the difference in logit between the indirect object's name and the subject's name (eg, `logit(Mary)-logit(John)`).

```python
def logits_to_ave_logit_diff(logits, answer_tokens, per_prompt=False):
    # Only the final logits are relevant for the answer
    final_logits = logits[:, -1, :]
    answer_logits = final_logits.gather(dim=-1, index=answer_tokens)
    answer_logit_diff = answer_logits[:, 0] - answer_logits[:, 1]
    if per_prompt:
        return answer_logit_diff
    else:
        return answer_logit_diff.mean()

print(
    "Per prompt logit difference:",
    logits_to_ave_logit_diff(original_logits, answer_tokens, per_prompt=True)
    .detach()
    .cpu()
    .round(decimals=3),
)
original_average_logit_diff = logits_to_ave_logit_diff(original_logits, answer_tokens)
print(
    "Average logit difference:",
    round(logits_to_ave_logit_diff(original_logits, answer_tokens).item(), 3),
)
```

We see that the average logit difference is 3.5 - for context, this represents putting an $e^{3.5}\approx 33\times$ higher probability on the correct answer.

## Brainstorm What's Actually Going On (Optional)

Before diving into running experiments, it's often useful to spend some time actually reasoning about how the behaviour in question could be implemented in the transformer. **This is optional, and you'll likely get the most out of engaging with this section if you have a decent understanding already of what a transformer is and how it works!**

You don't have to do this and forming hypotheses after exploration is also reasonable, but I think it's often easier to explore and interpret results with some grounding in what you might find. In this particular case, I'm cheating somewhat, since I know the answer, but I'm trying to simulate the process of reasoning about it!

Note that often your hypothesis will be wrong in some ways and often be completely off. We're doing science here, and the goal is to understand how the model *actually* works, and to form true beliefs! There are two separate traps here at two extremes that it's worth tracking:
* Confusion: Having no hypotheses at all, getting a lot of data and not knowing what to do with it, and just floundering around
* Dogmatism: Being overconfident in an incorrect hypothesis and being unwilling to let go of it when reality contradicts you, or flinching away from running the experiments that might disconfirm it.

**Exercise:** Spend some time thinking through how you might imagine this behaviour being implemented in a transformer. Try to think through this for yourself before reading through my thoughts!

<details> <summary>(*) <b>My reasoning</b></summary>

<h3>Brainstorming:</h3>

So, what's hard about the task? Let's focus on the concrete example of the first prompt, "When John and Mary went to the shops, John gave the bag to" -> " Mary".

A good starting point is thinking though whether a tiny model could do this, eg a <a href="https://transformer-circuits.pub/2021/framework/index.html">1L Attn-Only model</a>. I'm pretty sure the answer is no! Attention is really good at the primitive operations of looking nearby, or copying information. I can believe a tiny model could figure out that at `to` it should look for names and predict that those names came next (eg the skip trigram " John...to -> John"). But it's much harder to tell how <i>many</i> of each previous name there are - attending 0.3 to each copy of John will look exactly the same as attending 0.6 to a single John token. So this will be pretty hard to figure out on the " to" token!

The natural place to break this symmetry is on the second " John" token - telling whether there is an earlier copy of the <i>current</i> token should be a much easier task. So I might expect there to be a head which detects duplicate tokens on the second " John" token, and then another head which moves that information from the second " John" token to the " to" token.

The model then needs to learn to predict " Mary" and <i>not</i> " John". I can see two natural ways to do this:
1. Detect all preceding names and move this information to " to" and then delete the any name corresponding to the duplicate token feature. This feels easier done with a non-linearity, since precisely cancelling out vectors is hard, so I'd imagine an MLP layer deletes the " John" direction of the residual stream
2. Have a head which attends to all previous names, but where the duplicate token features <i>inhibit</i> it from attending to specific names. So this only attends to Mary. And then the output of this head maps to the logits.

(Spoiler: It's the second one).

<h3>Experiment Ideas</h3>

A test that could distinguish these two is to look at which components of the model add directly to the logits - if it's mostly attention heads which attend to " Mary" and to neither " John" it's probably hypothesis 2, if it's mostly MLPs it's probably hypothesis 1.

And we should be able to identify duplicate token heads by finding ones which attend from " John" to " John", and whose outputs are then moved to the " to" token by V-Composition with another head (Spoiler: It's more complicated than that!)

Note that all of the above reasoning is very simplistic and could easily break in a real model! There'll be significant parts of the model that figure out whether to use this circuit at all (we don't want to inhibit duplicated names when, eg, figuring out what goes at the start of the <i>next</i> sentence), and may be parts towards the end of the model that do "post-processing" just before the final output. But it's a good starting point for thinking about what's going on.

## Direct Logit Attribution

*Look up unfamiliar terms in the [mech interp explainer](https://neelnanda.io/glossary)*

Further, the easiest part of the model to understand is the output - this is what the model is trained to optimize, and so it can always be directly interpreted! Often the right approach to reverse engineering a circuit is to start at the end, understand how the model produces the right answer, and to then work backwards. The main technique used to do this is called **direct logit attribution**

**Background:** The central object of a transformer is the **residual stream**. This is the sum of the outputs of each layer and of the original token and positional embedding. Importantly, this means that any linear function of the residual stream can be perfectly decomposed into the contribution of each layer of the transformer. Further, each attention layer's output can be broken down into the sum of the output of each head (See [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) for details), and each MLP layer's output can be broken down into the sum of the output of each neuron (and a bias term for each layer).

The logits of a model are `logits=Unembed(LayerNorm(final_residual_stream))`. The Unembed is a linear map, and LayerNorm is approximately a linear map, so we can decompose the logits into the sum of the contributions of each component, and look at which components contribute the most to the logit of the correct token! This is called **direct logit attribution**. Here we look at the direct attribution to the logit difference!

<details> <summary>(*) <b>Background and motivation of the logit difference</b></summary>

Logit difference is actually a *really* nice and elegant metric and is a particularly nice aspect of the setup of Indirect Object Identification. In general, there are two natural ways to interpret the model's outputs: the output logits, or the output log probabilities (or probabilities).

The logits are much nicer and easier to understand, as noted above. However, the model is trained to optimize the cross-entropy loss (the average of log probability of the correct token). This means it does not directly optimize the logits, and indeed if the model adds an arbitrary constant to every logit, the log probabilities are unchanged.

But `log_probs == logits.log_softmax(dim=-1) == logits - logsumexp(logits)`, and so `log_probs(" Mary") - log_probs(" John") = logits(" Mary") - logits(" John")` - the ability to add an arbitrary constant cancels out!

Further, the metric helps us isolate the precise capability we care about - figuring out *which* name is the Indirect Object. There are many other components of the task - deciding whether to return an article (the) or pronoun (her) or name, realising that the sentence wants a person next at all, etc. By taking the logit difference we control for all of that.

Our metric is further refined, because each prompt is repeated twice, for each possible indirect object. This controls for irrelevant behaviour such as the model learning that John is a more frequent token than Mary (this actually happens! The final layernorm bias increases the John logit by 1 relative to the Mary logit)

</details>

<details> <summary>Ignoring LayerNorm</summary>

LayerNorm is an analogous normalization technique to BatchNorm (that's friendlier to massive parallelization) that transformers use. Every time a transformer layer reads information from the residual stream, it applies a LayerNorm to normalize the vector at each position (translating to set the mean to 0 and scaling to set the variance to 1) and then applying a learned vector of weights and biases to scale and translate the normalized vector. This is *almost* a linear map, apart from the scaling step, because that divides by the norm of the vector and the norm is not a linear function. (The `fold_ln` flag when loading a model factors out all the linear parts).

But if we fixed the scale factor, the LayerNorm would be fully linear. And the scale of the residual stream is a global property that's a function of *all* components of the stream, while in practice there is normally just a few directions relevant to any particular component, so in practice this is an acceptable approximation. So when doing direct logit attribution we use the `apply_ln` flag on the `cache` to apply the global layernorm scaling factor to each constant. See [my clean GPT-2 implementation](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=Clean_Transformer_Implementation) for more on LayerNorm.
</details>

Getting an output logit is equivalent to projecting onto a direction in the residual stream. We use `model.tokens_to_residual_directions` to map the answer tokens to that direction, and then convert this to a logit difference direction for each batch

```python
answer_residual_directions = model.tokens_to_residual_directions(answer_tokens)
print("Answer residual directions shape:", answer_residual_directions.shape)
logit_diff_directions = (
    answer_residual_directions[:, 0] - answer_residual_directions[:, 1]
)
print("Logit difference directions shape:", logit_diff_directions.shape)
```

To verify that this works, we can apply this to the final residual stream for our cached prompts (after applying LayerNorm scaling) and verify that we get the same answer.

<details> <summary>Technical details</summary>

`logits = Unembed(LayerNorm(final_residual_stream))`, so we technically need to account for the centering, and then learned translation and scaling of the layernorm, not just the variance 1 scaling.

The centering is accounted for with the preprocessing flag `center_writing_weights` which ensures that every weight matrix writing to the residual stream has mean zero.

The learned scaling is folded into the unembedding weights `model.unembed.W_U` via `W_U_fold = layer_norm.weights[:, None] * unembed.W_U`

The learned translation is folded to `model.unembed.b_U`, a bias added to the logits (note that GPT-2 is not trained with an existing `b_U`). This roughly represents unigram statistics. But we can ignore this because each prompt occurs twice with names in the opposite order, so this perfectly cancels out.

Note that rather than using layernorm scaling we could just study cache["ln_final.hook_normalised"]

</details>

```python
# cache syntax - resid_post is the residual stream at the end of the layer, -1 gets the final layer. The general syntax is [activation_name, layer_index, sub_layer_type].
final_residual_stream = cache["resid_post", -1]
print("Final residual stream shape:", final_residual_stream.shape)
final_token_residual_stream = final_residual_stream[:, -1, :]
# Apply LayerNorm scaling
# pos_slice is the subset of the positions we take - here the final token of each prompt
scaled_final_token_residual_stream = cache.apply_ln_to_stack(
    final_token_residual_stream, layer=-1, pos_slice=-1
)

average_logit_diff = einsum(
    "batch d_model, batch d_model -> ",
    scaled_final_token_residual_stream,
    logit_diff_directions,
) / len(prompts)
print("Calculated average logit diff:", round(average_logit_diff.item(), 3))
print("Original logit difference:", round(original_average_logit_diff.item(), 3))
```

### Logit Lens

We can now decompose the residual stream! First we apply a technique called the [**logit lens**](https://www.alignmentforum.org/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - this looks at the residual stream after each layer and calculates the logit difference from that. This simulates what happens if we delete all subsequence layers.

```python
def residual_stack_to_logit_diff(
    residual_stack: Float[torch.Tensor, "components batch d_model"],
    cache: ActivationCache,
) -> float:
    scaled_residual_stack = cache.apply_ln_to_stack(
        residual_stack, layer=-1, pos_slice=-1
    )
    return einsum(
        "... batch d_model, batch d_model -> ...",
        scaled_residual_stack,
        logit_diff_directions,
    ) / len(prompts)
```

Fascinatingly, we see that the model is utterly unable to do the task until layer 7, almost all performance comes from attention layer 9, and performance actually *decreases* from there.

**Note:** Hover over each data point to see what residual stream position it's from!

<details> <summary>Details on `accumulated_resid`</summary>
**Key:** `n_pre` means the residual stream at the start of layer n, `n_mid` means the residual stream after the attention part of layer n (`n_post` is the same as `n+1_pre` so is not included)

* `layer` is the layer for which we input the residual stream (this is used to identify *which* layer norm scaling factor we want)
* `incl_mid` is whether to include the residual stream in the middle of a layer, ie after attention & before MLP
* `pos_slice` is the subset of the positions used. See `utils.Slice` for details on the syntax.
* return_labels is whether to return the labels for each component returned (useful for plotting)
</details>

```python
accumulated_residual, labels = cache.accumulated_resid(
    layer=-1, incl_mid=True, pos_slice=-1, return_labels=True
)
logit_lens_logit_diffs = residual_stack_to_logit_diff(accumulated_residual, cache)
line(
    logit_lens_logit_diffs,
    x=np.arange(model.cfg.n_layers * 2 + 1) / 2,
    hover_name=labels,
    title="Logit Difference From Accumulate Residual Stream",
)
```

### Layer Attribution

We can repeat the above analysis but for each layer (this is equivalent to the differences between adjacent residual streams)

Note: Annoying terminology overload - layer k of a transformer means the kth **transformer block**, but each block consists of an **attention layer** (to move information around) *and* an **MLP layer** (to process information).

We see that only attention layers matter, which makes sense! The IOI task is about moving information around (ie moving the correct name and not the incorrect name), and less about processing it. And again we note that attention layer 9 improves things a lot, while attention 10 and attention 11 *decrease* performance

```python
per_layer_residual, labels = cache.decompose_resid(
    layer=-1, pos_slice=-1, return_labels=True
)
per_layer_logit_diffs = residual_stack_to_logit_diff(per_layer_residual, cache)
line(per_layer_logit_diffs, hover_name=labels, title="Logit Difference From Each Layer")
```

## Head Attribution

We can further break down the output of each attention layer into the sum of the outputs of each attention head. Each attention layer consists of 12 heads, which each act independently and additively.

<details> <summary>Decomposing attention output into sums of heads</summary>
The standard way to compute the output of an attention layer is by concatenating the mixed values of each head, and multiplying by a big output weight matrix. But as described in [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) this is equivalent to splitting the output weight matrix into a per-head output (here `model.blocks[k].attn.W_O`) and adding them up (including an overall bias term for the entire layer)
</details>

We see that only a few heads really matter - heads L9H6 and L9H9 contribute a lot positively (explaining why attention layer 9 is so important), while heads L10H7 and L11H10 contribute a lot negatively (explaining why attention layer 10 and layer 11 are actively harmful). These correspond to (some of) the name movers and negative name movers discussed in the paper. There are also several heads that matter positively or negatively but less strongly (other name movers and backup name movers)

There are a few meta observations worth making here - our model has 144 heads, yet we could localise this behaviour to a handful of specific heads, using straightforward, general techniques. This supports the claim in [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) that attention heads are the right level of abstraction to understand attention. It also really surprising that there are *negative* heads - eg L10H7 makes the incorrect logit 7x *more* likely. I'm not sure what's going on there, though the paper discusses some possibilities.

```python
per_head_residual, labels = cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_logit_diffs = residual_stack_to_logit_diff(per_head_residual, cache)
per_head_logit_diffs = einops.rearrange(
    per_head_logit_diffs,
    "(layer head_index) -> layer head_index",
    layer=model.cfg.n_layers,
    head_index=model.cfg.n_heads,
)
imshow(
    per_head_logit_diffs,
    labels={"x": "Head", "y": "Layer"},
    title="Logit Difference From Each Head",
)
```

## Attention Analysis

Attention heads are particularly easy to study because we can look directly at their attention patterns and study from what positions they move information from and two. This is particularly easy here as we're looking at the direct effect on the logits so we need only look at the attention patterns from the final token.

We use Alan Cooney's circuitsvis library to visualize the attention patterns! We visualize the top 3 positive and negative heads by direct logit attribution, and show these for the first prompt (as an illustration).

<details> <summary>Interpreting Attention Patterns</summary>
An easy mistake to make when looking at attention patterns is thinking that they must convey information about the <i>token</i> looked at (maybe accounting for the context of the token). But actually, all we can confidently say is that it moves information from the *residual stream position* corresponding to that input token. Especially later on in the model, there may be components in the residual stream that are nothing to do with the input token! Eg the period at the end of a sentence may contain summary information for that sentence, and the head may solely move that, rather than caring about whether it ends in ".", "!" or "?"
</details>

```python
def visualize_attention_patterns(
    heads: Union[List[int], int, Float[torch.Tensor, "heads"]],
    local_cache: ActivationCache,
    local_tokens: torch.Tensor,
    title: Optional[str] = "",
    max_width: Optional[int] = 700,
) -> str:
    # If a single head is given, convert to a list
    if isinstance(heads, int):
        heads = [heads]

    # Create the plotting data
    labels: List[str] = []
    patterns: List[Float[torch.Tensor, "dest_pos src_pos"]] = []

    # Assume we have a single batch item
    batch_index = 0

    for head in heads:
        # Set the label
        layer = head // model.cfg.n_heads
        head_index = head % model.cfg.n_heads
        labels.append(f"L{layer}H{head_index}")

        # Get the attention patterns for the head
        # Attention patterns have shape [batch, head_index, query_pos, key_pos]
        patterns.append(local_cache["attn", layer][batch_index, head_index])

    # Convert the tokens to strings (for the axis labels)
    str_tokens = model.to_str_tokens(local_tokens)

    # Combine the patterns into a single tensor
    patterns: Float[torch.Tensor, "head_index dest_pos src_pos"] = torch.stack(
        patterns, dim=0
    )

    # Circuitsvis Plot (note we get the code version so we can concatenate with the title)
    plot = attention_heads(
        attention=patterns, tokens=str_tokens, attention_head_names=labels
    ).show_code()

    # Display the title
    title_html = f"<h2>{title}</h2><br/>"

    # Return the visualisation as raw code
    return f"<div style='max-width: {str(max_width)}px;'>{title_html + plot}</div>"
```

Inspecting the patterns, we can see that both types of name movers attend to the indirect object - this suggests they're simply copying the name attended to (with the OV circuit) and that the interesting part is the circuit behind the attention pattern that calculates *where* to move information from (the QK circuit)

```python
top_k = 3

top_positive_logit_attr_heads = torch.topk(
    per_head_logit_diffs.flatten(), k=top_k
).indices

positive_html = visualize_attention_patterns(
    top_positive_logit_attr_heads,
    cache,
    tokens[0],
    f"Top {top_k} Positive Logit Attribution Heads",
)

top_negative_logit_attr_heads = torch.topk(
    -per_head_logit_diffs.flatten(), k=top_k
).indices

negative_html = visualize_attention_patterns(
    top_negative_logit_attr_heads,
    cache,
    tokens[0],
    title=f"Top {top_k} Negative Logit Attribution Heads",
)

HTML(positive_html + negative_html)
```

## Activation Patching

**This section explains how to do activation patching conceptually by implementing it from scratch. To use it in practice with TransformerLens, see [this demonstration instead](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Activation_Patching_in_TL_Demo.ipynb)**.

The obvious limitation to the techniques used above is that they only look at the very end of the circuit - the parts that directly affect the logits. Clearly this is not sufficient to understand the circuit! We want to understand how things compose together to produce this final output, and ideally to produce an end-to-end circuit fully explaining this behaviour.

The technique we'll use to investigate this is called **activation patching**. This was first introduced in [David Bau and Kevin Meng's excellent ROME paper](https://rome.baulab.info/), there called causal tracing.

The setup of activation patching is to take two runs of the model on two different inputs, the clean run and the corrupted run. The clean run outputs the correct answer and the corrupted run does not. The key idea is that we give the model the corrupted input, but then **intervene** on a specific activation and **patch** in the corresponding activation from the clean run (ie replace the corrupted activation with the clean activation), and then continue the run. And we then measure how much the output has updated towards the correct answer.

We can then iterate over many possible activations and look at how much they affect the corrupted run. If patching in an activation significantly increases the probability of the correct answer, this allows us to *localise* which activations matter.

The ability to localise is a key move in mechanistic interpretability - if the computation is diffuse and spread across the entire model, it is likely much harder to form a clean mechanistic story for what's going on. But if we can identify precisely which parts of the model matter, we can then zoom in and determine what they represent and how they connect up with each other, and ultimately reverse engineer the underlying circuit that they represent.

Here's an animation from the ROME paper demonstrating this technique (they studied factual recall, and use stars to represent corruption applied to the subject of the sentence, but the same principles apply):

![CT Animation](https://rome.baulab.info/images/small-ct-animation.gif)

See also [the explanation in a mech interp explainer](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=qeWBvs-R-taFfcCq-S_hgMqx) and [this piece](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching#how-to-think-about-activation-patching) describing how to think about patching on a conceptual level

The above was all fairly abstract, so let's zoom in and lay out a concrete example to understand Indirect Object Identification.

Here our clean input will be eg "After John and Mary went to the store, **John** gave a bottle of milk to" and our corrupted input will be eg "After John and Mary went to the store, **Mary** gave a bottle of milk to". These prompts are identical except for the name of the indirect object, and so patching is a causal intervention which will allow us to understand precisely which parts of the network are identifying the indirect object.

One natural thing to patch in is the residual stream at a specific layer and specific position. For example, the model is likely initially doing some processing on the second subject token to realise that it's a duplicate, but then uses attention to move that information to the " to" token. So patching in the residual stream at the " to" token will likely matter a lot in later layers but not at all in early layers.

We can zoom in much further and patch in specific activations from specific layers. For example, we think that the output of head L9H9 on the final token is significant for directly connecting to the logits

We can patch in specific activations, and can zoom in as far as seems reasonable. For example, if we patch in the output of head L9H9 on the final token, we would predict that it will significantly affect performance.

Note that this technique does *not* tell us how the components of the circuit connect up, just what they are.

<details> <summary>Technical details</summary>
The choice of clean and corrupted prompt has both pros and cons. By carefully setting up the counterfactual, that <i>only</i> differs in the second subject, we avoid detecting the parts of the model doing irrelevant computation like detecting that the indirect object task is relevant at all or that it should be outputting a name rather than an article or pronoun. Or even context like that John and Mary are names at all.

However, it *also* bakes in some details that *are* relevant to the task. Such as finding the location of the second subject, and of the names in the first clause. Or that the name mover heads have learned to copy whatever they look at.

Some of these could be patched by also changing up the order of the names in the original sentence - patching in "After <b>John and Mary</b> went to the store, John gave a bottle of milk to" vs "After <b>Mary and John</b> went to the store, John gave a bottle of milk to".

In the ROME paper they take a different tack. Rather than carefully setting up counterfactuals between two different but related inputs, they **corrupt** the clean input by adding Gaussian noise to the token embedding for the subject. This is in some ways much lower effort (you don't need to set up a similar but different prompt) but can also introduce some issues, such as ways this noise might break things. In practice, you should take care about how you choose your counterfactuals and try out several. Try to reason beforehand about what they will and will not tell you, and compare the results between different counterfactuals.

I discuss some of these limitations and how the author's solved them with much more refined usage of these techniques <a href="https://www.youtube.com/watch?v=gzwj0jWbvbo">in our interview</a>
</details>

## Residual Stream

Lets begin by patching in the residual stream at the start of each layer and for each token position.

We first create a set of corrupted tokens - where we swap each pair of prompts to have the opposite answer.

```python
corrupted_prompts = []
for i in range(0, len(prompts), 2):
    corrupted_prompts.append(prompts[i + 1])
    corrupted_prompts.append(prompts[i])
corrupted_tokens = model.to_tokens(corrupted_prompts, prepend_bos=True)
corrupted_logits, corrupted_cache = model.run_with_cache(
    corrupted_tokens, return_type="logits"
)
corrupted_average_logit_diff = logits_to_ave_logit_diff(corrupted_logits, answer_tokens)
print("Corrupted Average Logit Diff", round(corrupted_average_logit_diff.item(), 2))
print("Clean Average Logit Diff", round(original_average_logit_diff.item(), 2))
```

```python
model.to_string(corrupted_tokens)
```

We now intervene on the corrupted run and patch in the clean residual stream at a specific layer and position.

We do the intervention using TransformerLens's `HookPoint` feature. We can design a hook function that takes in a specific activation and returns an edited copy, and temporarily add it in with `model.run_with_hooks`.

```python
def patch_residual_component(
    corrupted_residual_component: Float[torch.Tensor, "batch pos d_model"],
    hook,
    pos,
    clean_cache,
):
    corrupted_residual_component[:, pos, :] = clean_cache[hook.name][:, pos, :]
    return corrupted_residual_component

def normalize_patched_logit_diff(patched_logit_diff):
    # Subtract corrupted logit diff to measure the improvement, divide by the total improvement from clean to corrupted to normalise
    # 0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance
    return (patched_logit_diff - corrupted_average_logit_diff) / (
        original_average_logit_diff - corrupted_average_logit_diff
    )

patched_residual_stream_diff = torch.zeros(
    model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for position in range(tokens.shape[1]):
        hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("resid_pre", layer), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

        patched_residual_stream_diff[layer, position] = normalize_patched_logit_diff(
            patched_logit_diff
        )
```

We can immediately see that, exactly as predicted, originally all relevant computation happens on the second subject token, and at layers 7 and 8, the information is moved to the final token. Moving the residual stream at the correct position near *exactly* recovers performance!

For reference, tokens and their index from the first prompt are on the x-axis. In an abuse of notation, note that the difference here is averaged over *all* 8 prompts, while the labels only come from the *first* prompt.

To be easier to interpret, we normalise the logit difference, by subtracting the corrupted logit difference, and dividing by the total improvement from clean to corrupted to normalise
0 means zero change, negative means actively made worse, 1 means totally recovered clean performance, >1 means actively *improved* on clean performance

```python
prompt_position_labels = [
    f"{tok}_{i}" for i, tok in enumerate(model.to_str_tokens(tokens[0]))
]
imshow(
    patched_residual_stream_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched Residual Stream",
    labels={"x": "Position", "y": "Layer"},
)
```

## Layers

We can apply exactly the same idea, but this time patching in attention or MLP layers. These are also residual components with identical shapes to the residual stream terms, so we can reuse the same hooks.

```python
patched_attn_diff = torch.zeros(
    model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
)
patched_mlp_diff = torch.zeros(
    model.cfg.n_layers, tokens.shape[1], device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for position in range(tokens.shape[1]):
        hook_fn = partial(patch_residual_component, pos=position, clean_cache=cache)
        patched_attn_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("attn_out", layer), hook_fn)],
            return_type="logits",
        )
        patched_attn_logit_diff = logits_to_ave_logit_diff(
            patched_attn_logits, answer_tokens
        )
        patched_mlp_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("mlp_out", layer), hook_fn)],
            return_type="logits",
        )
        patched_mlp_logit_diff = logits_to_ave_logit_diff(
            patched_mlp_logits, answer_tokens
        )

        patched_attn_diff[layer, position] = normalize_patched_logit_diff(
            patched_attn_logit_diff
        )
        patched_mlp_diff[layer, position] = normalize_patched_logit_diff(
            patched_mlp_logit_diff
        )
```

We see that several attention layers are significant but that, matching the residual stream results, early layers matter on the second subject token, and later layers matter on the final token, and layers essentially don't matter on any other token. Extremely localised! As with direct logit attribution, layer 9 is positive and layers 10 and 11 are not, suggesting that the late layers only matter for direct logit effects, but we also see that layers 7 and 8 matter significantly. Presumably these are the heads that move information about which name is duplicated from the second subject token to the final token.

```python
imshow(
    patched_attn_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched Attention Layer",
    labels={"x": "Position", "y": "Layer"},
)
```

In contrast, the MLP layers do not matter much. This makes sense, since this is more a task about moving information than about processing it, and the MLP layers specialise in processing information.

The one exception is MLP 0, which matters a lot, but I think this is misleading and just a generally true statement about MLP 0 rather than being about the circuit on this task.

<details> <summary>My takes on MLP0</summary>
It's often observed on GPT-2 Small that MLP0 matters a lot, and that ablating it utterly destroys performance. My current best guess is that the first MLP layer is essentially acting as an extension of the embedding (for whatever reason) and that when later layers want to access the input tokens they mostly read in the output of the first MLP layer, rather than the token embeddings. Within this frame, the first attention layer doesn't do much.

In this framing, it makes sense that MLP0 matters on the second subject token, because that's the one position with a different input token!

I'm not entirely sure why this happens, but I would guess that it's because the embedding and unembedding matrices in GPT-2 Small are the same. This is pretty unprincipled, as the tasks of embedding and unembedding tokens are <i>not</i> inverses, but this is common practice, and plausibly models want to dedicate some parameters to overcoming this.

I only have suggestive evidence of this, and would love to see someone look into this properly!
</details>

```python
imshow(
    patched_mlp_diff,
    x=prompt_position_labels,
    title="Logit Difference From Patched MLP Layer",
    labels={"x": "Position", "y": "Layer"},
)
```

## Heads

We can refine the above analysis by patching in individual heads! This is somewhat more annoying, because there are now three dimensions (head_index, position and layer), so for now lets patch in a head's output across all positions.

The easiest way to do this is to patch in the activation `z`, the "mixed value" of the attention head. That is, the average of all previous values weighted by the attention pattern, ie the activation that is then multiplied by `W_O`, the output weights.

```python
def patch_head_vector(
    corrupted_head_vector: Float[torch.Tensor, "batch pos head_index d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_vector[:, :, head_index, :] = clean_cache[hook.name][
        :, :, head_index, :
    ]
    return corrupted_head_vector

patched_head_z_diff = torch.zeros(
    model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for head_index in range(model.cfg.n_heads):
        hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("z", layer, "attn"), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

        patched_head_z_diff[layer, head_index] = normalize_patched_logit_diff(
            patched_logit_diff
        )
```

We can now see that, in addition to the name mover heads identified before, in mid-late layers the heads L8H6, L8H10, L7H9 matter and are presumably responsible for moving information from the second subject to the final token. And heads L5H5, L6H9, L3H0 also matter a lot, and are presumably involved in detecting duplicated tokens.

```python
imshow(
    patched_head_z_diff,
    title="Logit Difference From Patched Head Output",
    labels={"x": "Head", "y": "Layer"},
)
```

## Decomposing Heads

Decomposing attention layers into patching in individual heads has already helped us localise the behaviour a lot. But we can understand it further by decomposing heads. An attention head consists of two semi-independent operations - calculating *where* to move information from and to (represented by the attention pattern and implemented via the QK-circuit) and calculating *what* information to move (represented by the value vectors and implemented by the OV circuit). We can disentangle which of these is important by patching in just the attention pattern *or* the value vectors. (See [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) or [my walkthrough video](https://www.youtube.com/watch?v=KV5gbOmHbjU) for more on this decomposition. If you're not familiar with the details of how attention is implemented, I recommend checking out [my clean transformer implementation](https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/clean-transformer-demo/Clean_Transformer_Demo.ipynb#scrollTo=3Pb0NYbZ900e) to see how the code works))

First let's patch in the value vectors, to measure when figuring out what to move is important. . This has the same shape as z ([batch, pos, head_index, d_head]) so we can reuse the same hook.

```python
patched_head_v_diff = torch.zeros(
    model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for head_index in range(model.cfg.n_heads):
        hook_fn = partial(patch_head_vector, head_index=head_index, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("v", layer, "attn"), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

        patched_head_v_diff[layer, head_index] = normalize_patched_logit_diff(
            patched_logit_diff
        )
```

We can plot this as a heatmap and it's initially hard to interpret.

```python
imshow(
    patched_head_v_diff,
    title="Logit Difference From Patched Head Value",
    labels={"x": "Head", "y": "Layer"},
)
```

But it's very easy to interpret if we plot a scatter plot against patching head outputs. Here we see that the earlier heads (L5H5, L6H9, L3H0) and late name movers (L9H9, L10H7, L11H10) don't matter at all now, while the mid-late heads (L8H6, L8H10, L7H9) do.

Meta lesson: Plot things early, often and in diverse ways as you explore a model's internals!

```python
head_labels = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
scatter(
    x=utils.to_numpy(patched_head_v_diff.flatten()),
    y=utils.to_numpy(patched_head_z_diff.flatten()),
    xaxis="Value Patch",
    yaxis="Output Patch",
    caxis="Layer",
    hover_name=head_labels,
    color=einops.repeat(
        np.arange(model.cfg.n_layers), "layer -> (layer head)", head=model.cfg.n_heads
    ),
    range_x=(-0.5, 0.5),
    range_y=(-0.5, 0.5),
    title="Scatter plot of output patching vs value patching",
)
```

When we patch in attention patterns, we see the opposite effect - early and late heads matter a lot, middle heads don't. (In fact, the sum of value patching and pattern patching is approx the same as output patching)

```python
def patch_head_pattern(
    corrupted_head_pattern: Float[torch.Tensor, "batch head_index query_pos d_head"],
    hook,
    head_index,
    clean_cache,
):
    corrupted_head_pattern[:, head_index, :, :] = clean_cache[hook.name][
        :, head_index, :, :
    ]
    return corrupted_head_pattern

patched_head_attn_diff = torch.zeros(
    model.cfg.n_layers, model.cfg.n_heads, device=device, dtype=torch.float32
)
for layer in range(model.cfg.n_layers):
    for head_index in range(model.cfg.n_heads):
        hook_fn = partial(patch_head_pattern, head_index=head_index, clean_cache=cache)
        patched_logits = model.run_with_hooks(
            corrupted_tokens,
            fwd_hooks=[(utils.get_act_name("attn", layer, "attn"), hook_fn)],
            return_type="logits",
        )
        patched_logit_diff = logits_to_ave_logit_diff(patched_logits, answer_tokens)

        patched_head_attn_diff[layer, head_index] = normalize_patched_logit_diff(
            patched_logit_diff
        )
```

```python
imshow(
    patched_head_attn_diff,
    title="Logit Difference From Patched Head Pattern",
    labels={"x": "Head", "y": "Layer"},
)
head_labels = [
    f"L{l}H{h}" for l in range(model.cfg.n_layers) for h in range(model.cfg.n_heads)
]
scatter(
    x=utils.to_numpy(patched_head_attn_diff.flatten()),
    y=utils.to_numpy(patched_head_z_diff.flatten()),
    hover_name=head_labels,
    xaxis="Attention Patch",
    yaxis="Output Patch",
    title="Scatter plot of output patching vs attention patching",
)
```

## Consolidating Understanding

OK, let's zoom out and reconsolidate. At a high-level, we find that all the action is on the second subject token until layer 7 and then transitions to the final token. And that attention layers matter a lot, MLP layers not so much (apart from MLP0, likely as an extended embedding).

We've further localised important behaviour to several categories of heads. We've found 3 categories of heads that matter a lot - early heads (L5H5, L6H9, L3H0) whose output matters on the second subject and whose behaviour is determined by their attention patterns, mid-late heads (L8H6, L8H10, L7H9, L7H3) whose output matters on the final token and whose behaviour is determined by their value vectors, and late heads (L9H9, L10H7, L11H10) whose output matters on the final token and whose behaviour is determined by their attention patterns.

A natural speculation is that early heads detect both that the second subject is a repeated token and *which* is repeated (ie the " John" token is repeated), middle heads compose with this and move this duplicated token information from the second subject token to the final token, and the late heads compose with this to *inhibit* their attention to the duplicated token, and then attend to the correct indirect object name and copy that directly to the logits.

### Visualizing Attention Patterns

We can validate this by looking at the attention patterns of these heads! Let's take the top 10 heads by output patching (in absolute value) and split it into early, middle and late.

We see that middle heads attend from the final token to the second subject, and late heads attend from the final token to the indirect object, which is completely consistent with the above speculation! But weirdly, while *one* early head attends from the second subject to its first copy, the other two mysteriously attend to the word *after* the first copy.

```python
top_k = 10
top_heads_by_output_patch = torch.topk(
    patched_head_z_diff.abs().flatten(), k=top_k
).indices
first_mid_layer = 7
first_late_layer = 9
early_heads = top_heads_by_output_patch[
    top_heads_by_output_patch < model.cfg.n_heads * first_mid_layer
]
mid_heads = top_heads_by_output_patch[
    torch.logical_and(
        model.cfg.n_heads * first_mid_layer <= top_heads_by_output_patch,
        top_heads_by_output_patch < model.cfg.n_heads * first_late_layer,
    )
]
late_heads = top_heads_by_output_patch[
    model.cfg.n_heads * first_late_layer <= top_heads_by_output_patch
]

early = visualize_attention_patterns(
    early_heads, cache, tokens[0], title=f"Top Early Heads"
)
mid = visualize_attention_patterns(
    mid_heads, cache, tokens[0], title=f"Top Middle Heads"
)
late = visualize_attention_patterns(
    late_heads, cache, tokens[0], title=f"Top Late Heads"
)

HTML(early + mid + late)
```

### Comparing to the Paper

We can now refer to the (far, far more rigorous and detailed) analysis in the paper to compare our results! Here's the diagram they give of their results.

![IOI1](https://pbs.twimg.com/media/FghGkTAWAAAmkhm.jpg)

(Head 1.2 in their notation is L1H2 in my notation etc. And note - in the [latest version of the paper](https://arxiv.org/pdf/2211.00593.pdf) they add 9.0 as a backup name mover, and remove 11.3)

The heads form three categories corresponding to the early, middle and late categories we found and we did fairly well! Definitely not perfect, but with some fairly generic techniques and some a priori reasoning, we found the broad strokes of the circuit and what it looks like. We focused on the most important heads, so we didn't find all relevant heads in each category (especially not the heads in brackets, which are more minor), but this serves as a good base for doing more rigorous and involved analysis, especially for finding the *complete* circuit (ie all of the parts of the model which participate in this behaviour) rather than just a partial and suggestive circuit. Go check out [their paper](https://arxiv.org/abs/2211.00593) or [our interview](https://www.youtube.com/watch?v=gzwj0jWbvbo) to learn more about what they did and what they found!

Breaking down their categories:

* Early: The duplicate token heads, previous token heads and induction heads. These serve the purpose of detecting that the second subject is duplicated and which earlier name is the duplicate.
    * We found a direct duplicate token head which behaves exactly as expected, L3H0. Heads L5H0 and L6H9 are induction heads, which explains why they don't attend directly to the earlier copy of John!
    * Note that the duplicate token heads and induction heads do not compose with each other - both directly add to the S-Inhibition heads. The diagram is somewhat misleading.
* Middle: They call these S-Inhibition heads - they copy the information about the duplicate token from the second subject to the to token, and their output is used to *inhibit* the attention paid from the name movers to the first subject copy. We found all these heads, and had a decent guess for what they did.
    * In either case they attend to the second subject, so the patch that mattered was their value vectors!
* Late: They call these name movers, and we found some of them. They attend from the final token to the indirect object name and copy that to the logits, using the S-Inhibition heads to inhibit attention to the first copy of the subject token.
    * We did find their surprising result of *negative* name movers - name movers that inhibit the correct answer!
    * They have an entire category of heads we missed called backup name movers - we'll get to these later.

So, now, let's dig into the two anomalies we missed - induction heads and backup name mover heads

## Bonus: Exploring Anomalies

### Early Heads are Induction Heads(?!)

A really weird observation is that some of the early heads detecting duplicated tokens are induction heads, not just direct duplicate token heads. This is very weird! What's up with that?

First off, what's an induction head? An induction head is an important type of attention head that can detect and continue repeated sequences. It is the second head in a two head induction circuit, which looks for previous copies of the current token and attends to the token *after* it, and then copies that to the current position and predicts that it will come next. They're enough of a big deal that [we wrote a whole paper on them](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html).

![Move image demo](https://pbs.twimg.com/media/FNWAzXjVEAEOGRe.jpg)

Second, why is it surprising that they come up here? It's surprising because it feels like overkill. The model doesn't care about *what* token comes after the first copy of the subject, just that it's duplicated. And it already has simpler duplicate token heads. My best guess is that it just already had induction heads around and that, in addition to their main function, they *also* only activate on duplicated tokens. So it was useful to repurpose this existing machinery.

This suggests that as we look for circuits in larger models life may get more and more complicated, as components in simpler circuits get repurposed and built upon.

We can verify that these are induction heads by running the model on repeated text and plotting the heads.

```python
example_text = "Research in mechanistic interpretability seeks to explain behaviors of machine learning models in terms of their internal components."
example_repeated_text = example_text + example_text
example_repeated_tokens = model.to_tokens(example_repeated_text, prepend_bos=True)
example_repeated_logits, example_repeated_cache = model.run_with_cache(
    example_repeated_tokens
)
induction_head_labels = [81, 65]
```

```python
code = visualize_attention_patterns(
    induction_head_labels,
    example_repeated_cache,
    example_repeated_tokens,
    title="Induction Heads",
    max_width=800,
)
HTML(code)
```

#### Implications

One implication of this is that it's useful to categories heads according to whether they occur in
simpler circuits, so that as we look for more complex circuits we can easily look for them. This is
easy to do here! An interesting fact about induction heads is that they work on a sequence of
repeated random tokens - notable for being wildly off distribution from the natural language GPT-2
was trained on. Being able to predict a model's behaviour off distribution is a good mark of success
for mechanistic interpretability! This is a good sanity check for whether a head is an induction
head or not.

We can characterise an induction head by just giving a sequence of random tokens repeated once, and
measuring the average attention paid from the second copy of a token to the token after the first
copy. At the same time, we can also measure the average attention paid from the second copy of a
token to the first copy of the token, which is the attention that the induction head would pay if it
were a duplicate token head, and the average attention paid to the previous token to find previous
token heads.

Note that this is a superficial study of whether something is an induction head - we totally ignore
the question of whether it actually does boost the correct token or whether it composes with a
single previous head and how. In particular, we sometimes get anti-induction heads which suppress
the induction-y token (no clue why!), and this technique will find those too . But given the
previous rigorous analysis, we can be pretty confident that this picks up on some true signal about
induction heads.

<details> <summary>Technical Implementation Details</summary>
We can do this again by using hooks, this time just to access the attention patterns rather than to intervene on them.

Our hook function acts on the attention pattern activation. This has the name
"blocks.{layer}.{layer_type}.hook_{activation_name}" in general, here it's
"blocks.{layer}.attn.hook_attn". And it has shape [batch, head_index, query_pos, token_pos]. Our
hook function takes in the attention pattern activation, calculates the score for the relevant type
of head, and write it to an external cache.

We add in hooks using `model.run_with_hooks(tokens, fwd_hooks=[(names_filter, hook_fn)])` to
temporarily add in the hooks and run the model, getting the resulting output. Previously
names_filter was the name of the activation, but here it's a boolean function mapping activation
names to whether we want to hook them or not. Here it's just whether the name ends with hook_attn.
hook_fn must take in the two inputs activation (the activation tensor) and hook (the HookPoint
object, which contains the name of the activation and some metadata such as the current layer).

Internally our hooks use the function `tensor.diagonal`, this takes the diagonal between two
dimensions, and allows an arbitrary offset - offset by 1 to get previous tokens, seq_len to get
duplicate tokens (the distance to earlier copies) and seq_len-1 to get induction heads (the distance
to the token *after* earlier copies). Different offsets give a different length of output tensor,
and we can now just average to get a score in [0, 1] for each head
</details>

```python
seq_len = 100
batch_size = 2

prev_token_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)

def prev_token_hook(pattern, hook):
    layer = hook.layer()
    diagonal = pattern.diagonal(offset=1, dim1=-1, dim2=-2)
    # print(diagonal)
    # print(pattern)
    prev_token_scores[layer] = einops.reduce(
        diagonal, "batch head_index diagonal -> head_index", "mean"
    )

duplicate_token_scores = torch.zeros(
    (model.cfg.n_layers, model.cfg.n_heads), device=device
)

def duplicate_token_hook(pattern, hook):
    layer = hook.layer()
    diagonal = pattern.diagonal(offset=seq_len, dim1=-1, dim2=-2)
    duplicate_token_scores[layer] = einops.reduce(
        diagonal, "batch head_index diagonal -> head_index", "mean"
    )

induction_scores = torch.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)

def induction_hook(pattern, hook):
    layer = hook.layer()
    diagonal = pattern.diagonal(offset=seq_len - 1, dim1=-1, dim2=-2)
    induction_scores[layer] = einops.reduce(
        diagonal, "batch head_index diagonal -> head_index", "mean"
    )

torch.manual_seed(0)
original_tokens = torch.randint(
    100, 20000, size=(batch_size, seq_len), device="cpu"
).to(device)
repeated_tokens = einops.repeat(
    original_tokens, "batch seq_len -> batch (2 seq_len)"
).to(device)

pattern_filter = lambda act_name: act_name.endswith("hook_pattern")

loss = model.run_with_hooks(
    repeated_tokens,
    return_type="loss",
    fwd_hooks=[
        (pattern_filter, prev_token_hook),
        (pattern_filter, duplicate_token_hook),
        (pattern_filter, induction_hook),
    ],
)
print(torch.round(utils.get_corner(prev_token_scores).detach().cpu(), decimals=3))
print(torch.round(utils.get_corner(duplicate_token_scores).detach().cpu(), decimals=3))
print(torch.round(utils.get_corner(induction_scores).detach().cpu(), decimals=3))
```

We can now plot the head scores, and instantly see that the relevant early heads are induction heads or duplicate token heads (though also that there's a lot of induction heads that are *not* use - I have no idea why!).

```python
imshow(
    prev_token_scores, labels={"x": "Head", "y": "Layer"}, title="Previous Token Scores"
)
imshow(
    duplicate_token_scores,
    labels={"x": "Head", "y": "Layer"},
    title="Duplicate Token Scores",
)
imshow(
    induction_scores, labels={"x": "Head", "y": "Layer"}, title="Induction Head Scores"
)
```

The above suggests that it would be a useful bit of infrastructure to have a "wiki" for the heads of a model, giving their scores according to some metrics re head functions, like the ones we've seen here. TransformerLens makes this easy to make, as just changing the name input to `HookedTransformer.from_pretrained` gives a different model but in the same architecture, so the same code should work. If you want to make this, I'd love to see it!

As a proof of concept, [I made a mosaic of all induction heads across the 40 models then in TransformerLens](https://www.neelnanda.io/mosaic).

![induction scores as proof of concept](https://firebasestorage.googleapis.com/v0/b/firescript-577a2.appspot.com/o/imgs%2Fapp%2FNeelNanda%2F5vtuFmdzt_.png?alt=media&token=4d613de4-9d14-48d6-ba9d-e591c562d429)

### Backup Name Mover Heads

Another fascinating anomaly is that of the **backup name mover heads**. A standard technique to apply when interpreting model internals is ablations, or knock-out. If we run the model but intervene to set a specific head to zero, what happens? If the model is robust to this intervention, then naively we can be confident that the head is not doing anything important, and conversely if the model is much worse at the task this suggests that head was important. There are several conceptual flaws with this approach, making the evidence only suggestive, eg that the average output of the head may be far from zero and so the knockout may send it far from expected activations, breaking internals on *any* task. But it's still an easy technique to apply to give some data.

But a wild finding in the paper is that models have **built in redundancy**. If we knock out one of the name movers, then there are some backup name movers in later layers that *change their behaviour* and do (some of) the job of the original name mover head. This means that naive knock-out will significantly underestimate the importance of the name movers.

Let's test this! Let's ablate the most important name mover (head L9H9) on just the final token using a custom ablation hook and then cache all new activations and compared performance. We focus on the final position because we want to specifically ablate the direct logit effect. When we do this, we see that naively, removing the top name mover should reduce the logit diff massively, from 3.55 to 0.57. **But actually, it only goes down to 2.99!**

<details> <summary>Implementation Details</summary>
Ablating heads is really easy in TransformerLens! We can just define a hook on the z activation in the relevant attention layer (recall, z is the mixed values, and comes immediately before multiplying by the output weights $W_O$). z has a head_index axis, so we can set the component for the relevant head and for position -1 to zero, and return it. (Technically we could just edit in place without returning it, but by convention we always return an edited activation).

We now want to compare all internal activations with a hook, which is hard to do with the nice `run_with_hooks` API. So we can directly access the hook on the z activation with `model.blocks[layer].attn.hook_z` and call its `add_hook` method. This adds in the hook to the *global state* of the model. We can now use run_with_cache, and don't need to care about the global state, because run_with_cache internally adds a bunch of caching hooks, and then removes all hooks after the run, *including* the previously added ablation hook. This can be disabled with the reset_hooks_end flag, but here it's useful!
</details>

```python
top_name_mover = per_head_logit_diffs.flatten().argmax().item()
top_name_mover_layer = top_name_mover // model.cfg.n_heads
top_name_mover_head = top_name_mover % model.cfg.n_heads
print(f"Top Name Mover to ablate: L{top_name_mover_layer}H{top_name_mover_head}")

def ablate_top_head_hook(z: Float[torch.Tensor, "batch pos head_index d_head"], hook):
    z[:, -1, top_name_mover_head, :] = 0
    return z

# Adds a hook into global model state
model.blocks[top_name_mover_layer].attn.hook_z.add_hook(ablate_top_head_hook)
# Runs the model, temporarily adds caching hooks and then removes *all* hooks after running, including the ablation hook.
ablated_logits, ablated_cache = model.run_with_cache(tokens)
print(f"Original logit diff: {original_average_logit_diff:.2f}")
print(
    f"Post ablation logit diff: {logits_to_ave_logit_diff(ablated_logits, answer_tokens).item():.2f}"
)
print(
    f"Direct Logit Attribution of top name mover head: {per_head_logit_diffs.flatten()[top_name_mover].item():.2f}"
)
print(
    f"Naive prediction of post ablation logit diff: {original_average_logit_diff - per_head_logit_diffs.flatten()[top_name_mover].item():.2f}"
)
```

So what's up with this? As before, we can look at the direct logit attribution of each head to see what's going on. It's easiest to interpret if plotted as a scatter plot against the initial per head logit difference.

And we can see a *really* big difference in a few heads! (Hover to see labels) In particular the negative name mover L10H7 decreases its negative effect a lot, adding +1 to the logit diff, and the backup name mover L10H10 adjusts its effect to be more positive, adding +0.8 to the logit diff (with several other marginal changes). (And obviously the ablated head has gone down to zero!)

```python
per_head_ablated_residual, labels = ablated_cache.stack_head_results(
    layer=-1, pos_slice=-1, return_labels=True
)
per_head_ablated_logit_diffs = residual_stack_to_logit_diff(
    per_head_ablated_residual, ablated_cache
)
per_head_ablated_logit_diffs = per_head_ablated_logit_diffs.reshape(
    model.cfg.n_layers, model.cfg.n_heads
)
imshow(per_head_ablated_logit_diffs, labels={"x": "Head", "y": "Layer"})
scatter(
    y=per_head_logit_diffs.flatten(),
    x=per_head_ablated_logit_diffs.flatten(),
    hover_name=head_labels,
    range_x=(-3, 3),
    range_y=(-3, 3),
    xaxis="Ablated",
    yaxis="Original",
    title="Original vs Post-Ablation Direct Logit Attribution of Heads",
)
```

One natural hypothesis is that this is because the final LayerNorm scaling has changed, which can scale up or down the final residual stream. This is slightly true, and we can see that the typical head is a bit off from the x=y line. But the average LN scaling ratio is 1.04, and this should uniformly change *all* heads by the same factor, so this can't be sufficient

```python
print(
    "Average LN scaling ratio:",
    round(
        (
            cache["ln_final.hook_scale"][:, -1]
            / ablated_cache["ln_final.hook_scale"][:, -1]
        )
        .mean()
        .item(),
        3,
    ),
)
print(
    "Ablation LN scale",
    ablated_cache["ln_final.hook_scale"][:, -1].detach().cpu().round(decimals=2),
)
print(
    "Original LN scale",
    cache["ln_final.hook_scale"][:, -1].detach().cpu().round(decimals=2),
)
```

**Exercise to the reader:** Can you finish off this analysis? What's going on here? Why are the backup name movers changing their behaviour? Why is one negative name mover becoming significantly less important?

---

# Grokking_Demo.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Grokking Demo Notebook

<b style="color: red">To use this notebook, go to Runtime > Change Runtime Type and select GPU as the hardware accelerator.</b>

# Setup
(No need to read)

```python
TRAIN_MODEL = True
```

```python
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

DEVELOPMENT_MODE = True
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB or IN_GITHUB:
    %pip install transformer_lens
    %pip install circuitsvis
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")
```

```python
pio.templates['plotly'].layout.xaxis.title.font.size = 20
pio.templates['plotly'].layout.yaxis.title.font.size = 20
pio.templates['plotly'].layout.title.font.size = 30
```

```python
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import os
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
```

```python
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Plotting helper functions:

```python
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
```

```python
# Define the location to save the model, using a relative path
PTH_LOCATION = "workspace/_scratch/grokking_demo.pth"

# Create the directory if it does not exist
os.makedirs(Path(PTH_LOCATION).parent, exist_ok=True)
```

# Model Training

## Config

```python
p = 113
frac_train = 0.3

# Optimizer config
lr = 1e-3
wd = 1.
betas = (0.9, 0.98)

num_epochs = 25000
checkpoint_every = 100

DATA_SEED = 598
```

## Define Task
* Define modular addition
* Define the dataset & labels

Input format:
|a|b|=|

```python
a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
equals_vector = einops.repeat(torch.tensor(113), " -> (i j)", i=p, j=p)

```

```python
dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(device)
print(dataset[:5])
print(dataset.shape)
```

```python
labels = (dataset[:, 0] + dataset[:, 1]) % p
print(labels.shape)
print(labels[:5])
```

Convert this to a train + test set - 30% in the training set

```python
torch.manual_seed(DATA_SEED)
indices = torch.randperm(p*p)
cutoff = int(p*p*frac_train)
train_indices = indices[:cutoff]
test_indices = indices[cutoff:]

train_data = dataset[train_indices]
train_labels = labels[train_indices]
test_data = dataset[test_indices]
test_labels = labels[test_indices]
print(train_data[:5])
print(train_labels[:5])
print(train_data.shape)
print(test_data[:5])
print(test_labels[:5])
print(test_data.shape)
```

## Define Model

```python

cfg = HookedTransformerConfig(
    n_layers = 1,
    n_heads = 4,
    d_model = 128,
    d_head = 32,
    d_mlp = 512,
    act_fn = "relu",
    normalization_type=None,
    d_vocab=p+1,
    d_vocab_out=p,
    n_ctx=3,
    init_weights=True,
    device=device,
    seed = 999,
)
```

```python
model = HookedTransformer(cfg)
```

Disable the biases, as we don't need them for this task and it makes things easier to interpret.

```python
for name, param in model.named_parameters():
    if "b_" in name:
        param.requires_grad = False

```

## Define Optimizer + Loss

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
```

```python
def loss_fn(logits, labels):
    if len(logits.shape)==3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
    return -correct_log_probs.mean()
train_logits = model(train_data)
train_loss = loss_fn(train_logits, train_labels)
print(train_loss)
test_logits = model(test_data)
test_loss = loss_fn(test_logits, test_labels)
print(test_loss)
```

```python
print("Uniform loss:")
print(np.log(p))
```

## Actually Train

**Weird Decision:** Training the model with full batch training rather than stochastic gradient descent. We do this so to make training smoother and reduce the number of slingshots.

```python
train_losses = []
test_losses = []
model_checkpoints = []
checkpoint_epochs = []
if TRAIN_MODEL:
    for epoch in tqdm.tqdm(range(num_epochs)):
        train_logits = model(train_data)
        train_loss = loss_fn(train_logits, train_labels)
        train_loss.backward()
        train_losses.append(train_loss.item())

        optimizer.step()
        optimizer.zero_grad()

        with torch.inference_mode():
            test_logits = model(test_data)
            test_loss = loss_fn(test_logits, test_labels)
            test_losses.append(test_loss.item())

        if ((epoch+1)%checkpoint_every)==0:
            checkpoint_epochs.append(epoch)
            model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")
```

```python
torch.save(
    {
        "model":model.state_dict(),
        "config": model.cfg,
        "checkpoints": model_checkpoints,
        "checkpoint_epochs": checkpoint_epochs,
        "test_losses": test_losses,
        "train_losses": train_losses,
        "train_indices": train_indices,
        "test_indices": test_indices,
    },
    PTH_LOCATION)
```

```python
if not TRAIN_MODEL:
    cached_data = torch.load(PTH_LOCATION)
    model.load_state_dict(cached_data['model'])
    model_checkpoints = cached_data["checkpoints"]
    checkpoint_epochs = cached_data["checkpoint_epochs"]
    test_losses = cached_data['test_losses']
    train_losses = cached_data['train_losses']
    train_indices = cached_data["train_indices"]
    test_indices = cached_data["test_indices"]
```

## Show Model Training Statistics, Check that it groks!

```python
%pip install git+https://github.com/neelnanda-io/neel-plotly.git
from neel_plotly.plot import line
line([train_losses[::100], test_losses[::100]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Training Curve for Modular Addition", line_labels=['train', 'test'], toggle_x=True, toggle_y=True)
```

# Analysing the Model

## Standard Things to Try

```python
original_logits, cache = model.run_with_cache(dataset)
print(original_logits.numel())
```

Get key weight matrices:

```python
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
```

```python
original_loss = loss_fn(original_logits, labels).item()
print("Original Loss:", original_loss)
```

### Looking at Activations

Helper variable:

```python
pattern_a = cache["pattern", 0, "attn"][:, :, -1, 0]
pattern_b = cache["pattern", 0, "attn"][:, :, -1, 1]
neuron_acts = cache["post", 0, "mlp"][:, -1, :]
neuron_pre_acts = cache["pre", 0, "mlp"][:, -1, :]
```

Get all shapes:

```python
for param_name, param in cache.items():
    print(param_name, param.shape)
```

```python
imshow(cache["pattern", 0].mean(dim=0)[:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])
```

```python
imshow(cache["pattern", 0][5][:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])
```

```python
dataset[:4]
```

```python
imshow(cache["pattern", 0][:, 0, -1, 0].reshape(p, p), title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a")
```

```python
imshow(
    einops.rearrange(cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p),
    title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a", facet_col=0)
```

Plotting neuron activations

```python
cache["post", 0, "mlp"].shape
```

```python
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p),
    title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)
```

### Singular Value Decomposition

```python
W_E.shape
```

```python
U, S, Vh = torch.svd(W_E)
line(S, title="Singular Values")
imshow(U, title="Principal Components on the Input")
```

```python
# Control - random Gaussian matrix
U, S, Vh = torch.svd(torch.randn_like(W_E))
line(S, title="Singular Values Random")
imshow(U, title="Principal Components Random")
```

## Explaining Algorithm

### Analyse the Embedding - It's a Lookup Table!

```python
U, S, Vh = torch.svd(W_E)
line(U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary")
```

```python
fourier_basis = []
fourier_basis_names = []
fourier_basis.append(torch.ones(p))
fourier_basis_names.append("Constant")
for freq in range(1, p//2+1):
    fourier_basis.append(torch.sin(torch.arange(p)*2 * torch.pi * freq / p))
    fourier_basis_names.append(f"Sin {freq}")
    fourier_basis.append(torch.cos(torch.arange(p)*2 * torch.pi * freq / p))
    fourier_basis_names.append(f"Cos {freq}")
fourier_basis = torch.stack(fourier_basis, dim=0).to(device)
fourier_basis = fourier_basis/fourier_basis.norm(dim=-1, keepdim=True)
imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)
```

```python
line(fourier_basis[:8], xaxis="Input", line_labels=fourier_basis_names[:8], title="First 8 Fourier Components")
line(fourier_basis[25:29], xaxis="Input", line_labels=fourier_basis_names[25:29], title="Middle Fourier Components")
```

```python
imshow(fourier_basis @ fourier_basis.T, title="All Fourier Vectors are Orthogonal")
```

### Analyse the Embedding

```python
imshow(fourier_basis @ W_E, yaxis="Fourier Component", xaxis="Residual Stream", y=fourier_basis_names, title="Embedding in Fourier Basis")
```

```python
line((fourier_basis @ W_E).norm(dim=-1), xaxis="Fourier Component", x=fourier_basis_names, title="Norms of Embedding in Fourier Basis")
```

```python
key_freqs = [17, 25, 32, 47]
key_freq_indices = [33, 34, 49, 50, 63, 64, 93, 94]
fourier_embed = fourier_basis @ W_E
key_fourier_embed = fourier_embed[key_freq_indices]
print("key_fourier_embed", key_fourier_embed.shape)
imshow(key_fourier_embed @ key_fourier_embed.T, title="Dot Product of embedding of key Fourier Terms")
```

### Key Frequencies

```python
line(fourier_basis[[34, 50, 64, 94]], title="Cos of key freqs", line_labels=[34, 50, 64, 94])
```

```python
line(fourier_basis[[34, 50, 64, 94]].mean(0), title="Constructive Interference")
```

## Analyse Neurons

```python
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p),
    title="First 5 neuron acts", xaxis="b", yaxis="a", facet_col=0)
```

```python
imshow(
    einops.rearrange(neuron_acts[:, 0], "(a b) -> a b", a=p, b=p),
    title="First neuron act", xaxis="b", yaxis="a",)
```

```python
imshow(fourier_basis[94][None, :] * fourier_basis[94][:, None], title="Cos 47a * cos 47b")
```

```python
imshow(fourier_basis[94][None, :] * fourier_basis[0][:, None], title="Cos 47a * const")
```

```python
imshow(fourier_basis @ neuron_acts[:, 0].reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of neuron 0", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
```

```python
imshow(fourier_basis @ neuron_acts[:, 5].reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of neuron 5", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
```

```python
imshow(fourier_basis @ torch.randn_like(neuron_acts[:, 0]).reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of RANDOM", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
```

### Neuron Clusters

```python
fourier_neuron_acts = fourier_basis @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p) @ fourier_basis.T
# Center these by removing the mean - doesn't matter!
fourier_neuron_acts[:, 0, 0] = 0.
print("fourier_neuron_acts", fourier_neuron_acts.shape)
```

```python
neuron_freq_norm = torch.zeros(p//2, model.cfg.d_mlp).to(device)
for freq in range(0, p//2):
    for x in [0, 2*(freq+1) - 1, 2*(freq+1)]:
        for y in [0, 2*(freq+1) - 1, 2*(freq+1)]:
            neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y]**2
neuron_freq_norm = neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
imshow(neuron_freq_norm, xaxis="Neuron", yaxis="Freq", y=torch.arange(1, p//2+1), title="Neuron Frac Explained by Freq")
```

```python
line(neuron_freq_norm.max(dim=0).values.sort().values, xaxis="Neuron", title="Max Neuron Frac Explained over Freqs")
```

## Read Off the Neuron-Logit Weights to Interpret

```python
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
```

```python
line((W_logit @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit in the Fourier Basis")
```

```python
neurons_17 = neuron_freq_norm[17-1]>0.85
neurons_17.shape
```

```python
neurons_17.sum()
```

```python
line((W_logit[neurons_17] @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit for freq 17 neurons in the Fourier Basis")
```

Study sin 17

```python
freq = 17
W_logit_fourier = W_logit @ fourier_basis
neurons_sin_17 = W_logit_fourier[:, 2*freq-1]
line(neurons_sin_17)
```

```python
neuron_acts.shape
```

```python
inputs_sin_17c = neuron_acts @ neurons_sin_17
imshow(fourier_basis @ inputs_sin_17c.reshape(p, p) @ fourier_basis.T, title="Fourier Heatmap over inputs for sin17c", x=fourier_basis_names, y=fourier_basis_names)
```

# Black Box Methods + Progress Measures

## Setup Code

Code to plot embedding freqs

```python
def embed_to_cos_sin(fourier_embed):
    if len(fourier_embed.shape) == 1:
        return torch.stack([fourier_embed[1::2], fourier_embed[2::2]])
    else:
        return torch.stack([fourier_embed[:, 1::2], fourier_embed[:, 2::2]], dim=1)

from neel_plotly.plot import melt

def plot_embed_bars(
    fourier_embed,
    title="Norm of embedding of each Fourier Component",
    return_fig=False,
    **kwargs
):
    cos_sin_embed = embed_to_cos_sin(fourier_embed)
    df = melt(cos_sin_embed)
    # display(df)
    group_labels = {0: "sin", 1: "cos"}
    df["Trig"] = df["0"].map(lambda x: group_labels[x])
    fig = px.bar(
        df,
        barmode="group",
        color="Trig",
        x="1",
        y="value",
        labels={"1": "$w_k$", "value": "Norm"},
        title=title,
        **kwargs
    )
    fig.update_layout(dict(legend_title=""))

    if return_fig:
        return fig
    else:
        fig.show()
```

Code to test a tensor of edited logits

```python
def test_logits(logits, bias_correction=False, original_logits=None, mode="all"):
    # Calculates cross entropy loss of logits representing a batch of all p^2
    # possible inputs
    # Batch dimension is assumed to be first
    if logits.shape[1] == p * p:
        logits = logits.T
    if logits.shape == torch.Size([p * p, p + 1]):
        logits = logits[:, :-1]
    logits = logits.reshape(p * p, p)
    if bias_correction:
        # Applies bias correction - we correct for any missing bias terms,
        # independent of the input, by centering the new logits along the batch
        # dimension, and then adding the average original logits across all inputs
        logits = (
            einops.reduce(original_logits - logits, "batch ... -> ...", "mean") + logits
        )
    if mode == "train":
        return loss_fn(logits[train_indices], labels[train_indices])
    elif mode == "test":
        return loss_fn(logits[test_indices], labels[test_indices])
    elif mode == "all":
        return loss_fn(logits, labels)
```

Code to run a metric over every checkpoint

```python
metric_cache = {}
```

```python
def get_metrics(model, metric_cache, metric_fn, name, reset=False):
    if reset or (name not in metric_cache) or (len(metric_cache[name]) == 0):
        metric_cache[name] = []
        for c, sd in enumerate(tqdm.tqdm((model_checkpoints))):
            model.reset_hooks()
            model.load_state_dict(sd)
            out = metric_fn(model)
            if type(out) == torch.Tensor:
                out = utils.to_numpy(out)
            metric_cache[name].append(out)
        model.load_state_dict(model_checkpoints[-1])
        try:
            metric_cache[name] = torch.tensor(metric_cache[name])
        except:
            metric_cache[name] = torch.tensor(np.array(metric_cache[name]))

```

## Defining Progress Measures

### Loss Curves

```python
memorization_end_epoch = 1500
circuit_formation_end_epoch = 13300
cleanup_end_epoch = 16600
```

```python
def add_lines(figure):
    figure.add_vline(memorization_end_epoch, line_dash="dash", opacity=0.7)
    figure.add_vline(circuit_formation_end_epoch, line_dash="dash", opacity=0.7)
    figure.add_vline(cleanup_end_epoch, line_dash="dash", opacity=0.7)
    return figure
```

```python
fig = line([train_losses[::100], test_losses[::100]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Training Curve for Modular Addition", line_labels=['train', 'test'], toggle_x=True, toggle_y=True, return_fig=True)
add_lines(fig)
```

### Logit Periodicity

```python
all_logits = original_logits[:, -1, :]
print(all_logits.shape)
all_logits = einops.rearrange(all_logits, "(a b) c -> a b c", a=p, b=p)
print(all_logits.shape)
```

```python
coses = {}
for freq in key_freqs:
    print("Freq:", freq)
    a = torch.arange(p)[:, None, None]
    b = torch.arange(p)[None, :, None]
    c = torch.arange(p)[None, None, :]
    cube_predicted_logits = torch.cos(freq * 2 * torch.pi / p * (a + b - c)).to(device)
    cube_predicted_logits /= cube_predicted_logits.norm()
    coses[freq] = cube_predicted_logits
```

```python
approximated_logits = torch.zeros_like(all_logits)
for freq in key_freqs:
    print("Freq:", freq)
    coeff = (all_logits * coses[freq]).sum()
    print("Coeff:", coeff)
    cosine_sim = coeff / all_logits.norm()
    print("Cosine Sim:", cosine_sim)
    approximated_logits += coeff * coses[freq]
residual = all_logits - approximated_logits
print("Residual size:", residual.norm())
print("Residual fraction of norm:", residual.norm()/all_logits.norm())
```

```python
random_logit_cube = torch.randn_like(all_logits)
print((all_logits * random_logit_cube).sum()/random_logit_cube.norm()/all_logits.norm())
```

```python
test_logits(all_logits)
```

```python
test_logits(approximated_logits)
```

#### Look During Training

```python
cos_cube = []
for freq in range(1, p//2 + 1):
    a = torch.arange(p)[:, None, None]
    b = torch.arange(p)[None, :, None]
    c = torch.arange(p)[None, None, :]
    cube_predicted_logits = torch.cos(freq * 2 * torch.pi / p * (a + b - c)).to(device)
    cube_predicted_logits /= cube_predicted_logits.norm()
    cos_cube.append(cube_predicted_logits)
cos_cube = torch.stack(cos_cube, dim=0)
print(cos_cube.shape)
```

```python
def get_cos_coeffs(model):
    logits = model(dataset)[:, -1]
    logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
    vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
    return vals

get_metrics(model, metric_cache, get_cos_coeffs, "cos_coeffs")
print(metric_cache["cos_coeffs"].shape)
```

```python
fig = line(metric_cache["cos_coeffs"].T, line_labels=[f"Freq {i}" for i in range(1, p//2+1)], title="Coefficients with Predicted Logits", xaxis="Epoch", x=checkpoint_epochs, yaxis="Coefficient", return_fig=True)
add_lines(fig)
```

```python
def get_cos_sim(model):
    logits = model(dataset)[:, -1]
    logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
    vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
    return vals / logits.norm()

get_metrics(model, metric_cache, get_cos_sim, "cos_sim") # You may need a big GPU. If you don't have one and can't work around this, raise an issue for help!
print(metric_cache["cos_sim"].shape)

fig = line(metric_cache["cos_sim"].T, line_labels=[f"Freq {i}" for i in range(1, p//2+1)], title="Cosine Sim with Predicted Logits", xaxis="Epoch", x=checkpoint_epochs, yaxis="Cosine Sim", return_fig=True)
add_lines(fig)
```

```python
def get_residual_cos_sim(model):
    logits = model(dataset)[:, -1]
    logits = einops.rearrange(logits, "(a b) c -> a b c", a=p, b=p)
    vals = (cos_cube * logits[None, :, :, :]).sum([-3, -2, -1])
    residual = logits - (vals[:, None, None, None] * cos_cube).sum(dim=0)
    return residual.norm() / logits.norm()

get_metrics(model, metric_cache, get_residual_cos_sim, "residual_cos_sim")
print(metric_cache["residual_cos_sim"].shape)

fig = line([metric_cache["cos_sim"][:, i] for i in range(p//2)]+[metric_cache["residual_cos_sim"]], line_labels=[f"Freq {i}" for i in range(1, p//2+1)]+["residual"], title="Cosine Sim with Predicted Logits + Residual", xaxis="Epoch", x=checkpoint_epochs, yaxis="Cosine Sim", return_fig=True)
add_lines(fig)
```

## Restricted Loss

```python
neuron_acts.shape
```

```python
neuron_acts_square = einops.rearrange(neuron_acts, "(a b) neur -> a b neur", a=p, b=p).clone()
# Center it
neuron_acts_square -= einops.reduce(neuron_acts_square, "a b neur -> 1 1 neur", "mean")
neuron_acts_square_fourier = einsum("a b neur, fa a, fb b -> fa fb neur", neuron_acts_square, fourier_basis, fourier_basis)
imshow(neuron_acts_square_fourier.norm(dim=-1), xaxis="Fourier Component b", yaxis="Fourier Component a", title="Norms of neuron activations by Fourier Component", x=fourier_basis_names, y=fourier_basis_names)
```

```python
original_logits, cache = model.run_with_cache(dataset)
print(original_logits.numel())
neuron_acts = cache["post", 0, "mlp"][:, -1, :]
```

```python
approx_neuron_acts = torch.zeros_like(neuron_acts)
approx_neuron_acts += neuron_acts.mean(dim=0)
a = torch.arange(p)[:, None]
b = torch.arange(p)[None, :]
for freq in key_freqs:
    cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
    cos_apb_vec /= cos_apb_vec.norm()
    cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
    approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
    sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
    sin_apb_vec /= sin_apb_vec.norm()
    sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
    approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
restricted_logits = approx_neuron_acts @ W_logit
print(loss_fn(restricted_logits[test_indices], test_labels))
```

```python
print(loss_fn(all_logits, labels)) # This bugged on models not fully trained
```

### Look During Training

```python
def get_restricted_loss(model):
    logits, cache = model.run_with_cache(dataset)
    logits = logits[:, -1, :]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    approx_neuron_acts = torch.zeros_like(neuron_acts)
    approx_neuron_acts += neuron_acts.mean(dim=0)
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    restricted_logits = approx_neuron_acts @ model.blocks[0].mlp.W_out @ model.unembed.W_U
    # Add bias term
    restricted_logits += logits.mean(dim=0, keepdim=True) - restricted_logits.mean(dim=0, keepdim=True)
    return loss_fn(restricted_logits[test_indices], test_labels)
get_restricted_loss(model)
```

```python
get_metrics(model, metric_cache, get_restricted_loss, "restricted_loss", reset=True)
print(metric_cache["restricted_loss"].shape)
```

```python
fig = line([train_losses[::100], test_losses[::100], metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Restricted Loss Curve", line_labels=['train', 'test', "restricted_loss"], toggle_x=True, toggle_y=True, return_fig=True)
add_lines(fig)
```

```python
fig = line([torch.tensor(test_losses[::100])/metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Restricted Loss to Test Loss Ratio", toggle_x=True, toggle_y=True, return_fig=True)
# WARNING: bugged when cancelling training half way thr ough
add_lines(fig)
```

## Excluded Loss

```python
approx_neuron_acts = torch.zeros_like(neuron_acts)
# approx_neuron_acts += neuron_acts.mean(dim=0)
a = torch.arange(p)[:, None]
b = torch.arange(p)[None, :]
for freq in key_freqs:
    cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
    cos_apb_vec /= cos_apb_vec.norm()
    cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
    approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
    sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
    sin_apb_vec /= sin_apb_vec.norm()
    sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
    approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
excluded_neuron_acts = neuron_acts - approx_neuron_acts
excluded_logits = excluded_neuron_acts @ W_logit
print(loss_fn(excluded_logits[train_indices], train_labels))
```

```python
def get_excluded_loss(model):
    logits, cache = model.run_with_cache(dataset)
    logits = logits[:, -1, :]
    neuron_acts = cache["post", 0, "mlp"][:, -1, :]
    approx_neuron_acts = torch.zeros_like(neuron_acts)
    # approx_neuron_acts += neuron_acts.mean(dim=0)
    a = torch.arange(p)[:, None]
    b = torch.arange(p)[None, :]
    for freq in key_freqs:
        cos_apb_vec = torch.cos(freq * 2 * torch.pi / p * (a + b)).to(device)
        cos_apb_vec /= cos_apb_vec.norm()
        cos_apb_vec = einops.rearrange(cos_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * cos_apb_vec).sum(dim=0) * cos_apb_vec
        sin_apb_vec = torch.sin(freq * 2 * torch.pi / p * (a + b)).to(device)
        sin_apb_vec /= sin_apb_vec.norm()
        sin_apb_vec = einops.rearrange(sin_apb_vec, "a b -> (a b) 1")
        approx_neuron_acts += (neuron_acts * sin_apb_vec).sum(dim=0) * sin_apb_vec
    excluded_neuron_acts = neuron_acts - approx_neuron_acts
    residual_stream_final = excluded_neuron_acts @ model.blocks[0].mlp.W_out + cache["resid_mid", 0][:, -1, :]
    excluded_logits = residual_stream_final @ model.unembed.W_U
    return loss_fn(excluded_logits[train_indices], train_labels)
get_excluded_loss(model)
```

```python
get_metrics(model, metric_cache, get_excluded_loss, "excluded_loss", reset=True)
print(metric_cache["excluded_loss"].shape)
```

```python
fig = line([train_losses[::100], test_losses[::100], metric_cache["excluded_loss"], metric_cache["restricted_loss"]], x=np.arange(0, len(train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Excluded and Restricted Loss Curve", line_labels=['train', 'test', "excluded_loss", "restricted_loss"], toggle_x=True, toggle_y=True, return_fig=True)

add_lines(fig)
```

---

# Head_Detector_Demo.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Head_Detector_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# TransformerLens Head Detector Demo

A common technique in mechanistic interpretability of transformer-based neural networks is identification of specialized attention heads, based on the attention patterns elicited by one or more prompts. The most basic examples of such heads are: previous token head, duplicate token head, or induction head ([more info](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_Jzi6YHRHKP1JziwdE02qdYZ)). Usually, such heads are identified manually, by through visualizations of attention patterns layer by layer, head by head, and trying to recognize the patterns by eye.

The purpose of the `TransformerLens.head_detector` feature is to automate a part of that workflow. The pattern characterizing a head of particular type/function is specified as a `Tensor` being a `seq_len x seq_len` [lower triangular matrix](https://en.wikipedia.org/wiki/Triangular_matrix). It can be either passed to the `detect_head` function directly or by giving a string identifying of several pre-defined detection patterns.

## How to use this notebook

Go to Runtime > Change Runtime Type and select GPU as the hardware accelerator.

Tips for reading this Colab:

* You can run all this code for yourself!
* The graphs are interactive!
* Use the table of contents pane in the sidebar to navigate
* Collapse irrelevant sections with the dropdown arrows
* Search the page using the search in the sidebar, not CTRL+F

## Setup (Ignore)

```python
# NBVAL_IGNORE_OUTPUT
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

DEVELOPMENT_MODE = True
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB or IN_GITHUB:
    %pip install git+https://github.com/TransformerLensOrg/TransformerLens.git
    # Install Neel's personal plotting utils
    %pip install git+https://github.com/neelnanda-io/neel-plotly.git
    # Install another version of node that makes PySvelte work way faster
    !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    %pip install git+https://github.com/neelnanda-io/PySvelte.git
    # Needed for PySvelte to work, v3 came out and broke things...
    %pip install typeguard==2.13.3
    %pip install typing-extensions
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

if IN_COLAB or not DEBUG_MODE:
    # Thanks to annoying rendering issues, Plotly graphics will either show up in colab OR Vscode depending on the renderer - this is bad for developing demos! Thus creating a debug mode.
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "png"
```

```python
import torch
import einops
import pysvelte
from tqdm import tqdm

import transformer_lens
from transformer_lens import HookedTransformer, ActivationCache
from neel_plotly import line, imshow, scatter
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")
```

### Some plotting utils

```python
# Util for plotting head detection scores

def plot_head_detection_scores(
    scores: torch.Tensor,
    zmin: float = -1,
    zmax: float = 1,
    xaxis: str = "Head",
    yaxis: str = "Layer",
    title: str = "Head Matches"
) -> None:
    imshow(scores, zmin=zmin, zmax=zmax, xaxis=xaxis, yaxis=yaxis, title=title)

def plot_attn_pattern_from_cache(cache: ActivationCache, layer_i: int):
    attention_pattern = cache["pattern", layer_i, "attn"].squeeze(0)
    attention_pattern = einops.rearrange(attention_pattern, "heads seq1 seq2 -> seq1 seq2 heads")
    print(f"Layer {layer_i} Attention Heads:")
    return pysvelte.AttentionMulti(tokens=model.to_str_tokens(prompt), attention=attention_pattern)
```

## Head detector

Utils: these will be in `transformer_lens.utils` after merging the fork to the main repo

```python
def is_square(x: torch.Tensor) -> bool:
    """Checks if `x` is a square matrix."""
    return x.ndim == 2 and x.shape[0] == x.shape[1]

def is_lower_triangular(x: torch.Tensor) -> bool:
    """Checks if `x` is a lower triangular matrix."""
    if not is_square(x):
        return False
    return x.equal(x.tril())
```

The code below is copy-pasted from the expanded (not yet merged) version of `transformer_lens.head_detector`.

After merging the code below can be replaced with simply

```py
from transformer_lens.head_detector import *
```

(but please don't use star-imports in production ;))

```python
from collections import defaultdict
import logging
from typing import cast, Dict, List, Optional, Tuple, Union
from typing_extensions import get_args, Literal

import numpy as np
import torch

from transformer_lens import HookedTransformer, ActivationCache
# from transformer_lens.utils import is_lower_triangular, is_square

HeadName = Literal["previous_token_head", "duplicate_token_head", "induction_head"]
HEAD_NAMES = cast(List[HeadName], get_args(HeadName))
ErrorMeasure = Literal["abs", "mul"]

LayerHeadTuple = Tuple[int, int]
LayerToHead = Dict[int, List[int]]

INVALID_HEAD_NAME_ERR = (
    f"detection_pattern must be a Tensor or one of head names: {HEAD_NAMES}; got %s"
)

SEQ_LEN_ERR = (
    "The sequence must be non-empty and must fit within the model's context window."
)

DET_PAT_NOT_SQUARE_ERR = "The detection pattern must be a lower triangular matrix of shape (sequence_length, sequence_length); sequence_length=%d; got detection patern of shape %s"

def detect_head(
    model: HookedTransformer,
    seq: Union[str, List[str]],
    detection_pattern: Union[torch.Tensor, HeadName],
    heads: Optional[Union[List[LayerHeadTuple], LayerToHead]] = None,
    cache: Optional[ActivationCache] = None,
    *,
    exclude_bos: bool = False,
    exclude_current_token: bool = False,
    error_measure: ErrorMeasure = "mul",
) -> torch.Tensor:
    """Searches the model (or a set of specific heads, for circuit analysis) for a particular type of attention head.
    This head is specified by a detection pattern, a (sequence_length, sequence_length) tensor representing the attention pattern we expect that type of attention head to show.
    The detection pattern can be also passed not as a tensor, but as a name of one of pre-specified types of attention head (see `HeadName` for available patterns), in which case the tensor is computed within the function itself.

    There are two error measures available for quantifying the match between the detection pattern and the actual attention pattern.

    1. `"mul"` (default) multiplies both tensors element-wise and divides the sum of the result by the sum of the attention pattern.
    Typically, the detection pattern should in this case contain only ones and zeros, which allows a straightforward interpretation of the score:
    how big fraction of this head's attention is allocated to these specific query-key pairs?
    Using values other than 0 or 1 is not prohibited but will raise a warning (which can be disabled, of course).
    2. `"abs"` calculates the mean element-wise absolute difference between the detection pattern and the actual attention pattern.
    The "raw result" ranges from 0 to 2 where lower score corresponds to greater accuracy. Subtracting it from 1 maps that range to (-1, 1) interval,
    with 1 being perfect match and -1 perfect mismatch.

    **Which one should you use?** `"abs"` is likely better for quick or exploratory investigations. For precise examinations where you're trying to
    reproduce as much functionality as possible or really test your understanding of the attention head, you probably want to switch to `"abs"`.

    The advantage of `"abs"` is that you can make more precise predictions, and have that measured in the score.
    You can predict, for instance, 0.2 attention to X, and 0.8 attention to Y, and your score will be better if your prediction is closer.
    The "mul" metric does not allow this, you'll get the same score if attention is 0.2, 0.8 or 0.5, 0.5 or 0.8, 0.2.

    Args:
    ----------
        model: Model being used.
        seq: String or list of strings being fed to the model.
        head_name: Name of an existing head in HEAD_NAMES we want to check. Must pass either a head_name or a detection_pattern, but not both!
        detection_pattern: (sequence_length, sequence_length) Tensor representing what attention pattern corresponds to the head we're looking for **or** the name of a pre-specified head. Currently available heads are: `["previous_token_head", "duplicate_token_head", "induction_head"]`.
        heads: If specific attention heads is given here, all other heads' score is set to -1. Useful for IOI-style circuit analysis. Heads can be spacified as a list tuples (layer, head) or a dictionary mapping a layer to heads within that layer that we want to analyze.
        cache: Include the cache to save time if you want.
        exclude_bos: Exclude attention paid to the beginning of sequence token.
        exclude_current_token: Exclude attention paid to the current token.
        error_measure: `"mul"` for using element-wise multiplication (default). `"abs"` for using absolute values of element-wise differences as the error measure.

    Returns:
    ----------
    A (n_layers, n_heads) Tensor representing the score for each attention head.

    Example:
    --------
    .. code-block:: python

        >>> from transformer_lens import HookedTransformer,  utils
        >>> from transformer_lens.head_detector import detect_head
        >>> import plotly.express as px

        >>> def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
        >>>     px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

        >>> model = HookedTransformer.from_pretrained("gpt2-small")
        >>> sequence = "This is a test sequence. This is a test sequence."

        >>> attention_score = detect_head(model, sequence, "previous_token_head")
        >>> imshow(attention_score, zmin=-1, zmax=1, xaxis="Head", yaxis="Layer", title="Previous Head Matches")
    """

    cfg = model.cfg
    tokens = model.to_tokens(seq).to(cfg.device)
    seq_len = tokens.shape[-1]

    # Validate error_measure

    assert error_measure in get_args(ErrorMeasure), f"Invalid {error_measure=}; valid values are {get_args(ErrorMeasure)}"

    # Validate detection pattern if it's a string
    if isinstance(detection_pattern, str):
        assert detection_pattern in HEAD_NAMES, (
            INVALID_HEAD_NAME_ERR % detection_pattern
        )
        if isinstance(seq, list):
            batch_scores = [detect_head(model, seq, detection_pattern) for seq in seq]
            return torch.stack(batch_scores).mean(0)
        detection_pattern = cast(
            torch.Tensor,
            eval(f"get_{detection_pattern}_detection_pattern(tokens.cpu())"),
        ).to(cfg.device)

    # if we're using "mul", detection_pattern should consist of zeros and ones
    if error_measure == "mul" and not set(detection_pattern.unique().tolist()).issubset(
        {0, 1}
    ):
        logging.warning(
            "Using detection pattern with values other than 0 or 1 with error_measure 'mul'"
        )

    # Validate inputs and detection pattern shape
    assert 1 < tokens.shape[-1] < cfg.n_ctx, SEQ_LEN_ERR
    assert (
        is_lower_triangular(detection_pattern) and seq_len == detection_pattern.shape[0]
    ), DET_PAT_NOT_SQUARE_ERR % (seq_len, detection_pattern.shape)

    if cache is None:
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

    if heads is None:
        layer2heads = {
            layer_i: list(range(cfg.n_heads)) for layer_i in range(cfg.n_layers)
        }
    elif isinstance(heads, list):
        layer2heads = defaultdict(list)
        for layer, head in heads:
            layer2heads[layer].append(head)
    else:
        layer2heads = heads

    matches = -torch.ones(cfg.n_layers, cfg.n_heads)

    for layer, layer_heads in layer2heads.items():
        # [n_heads q_pos k_pos]
        layer_attention_patterns = cache["pattern", layer, "attn"]
        for head in layer_heads:
            head_attention_pattern = layer_attention_patterns[head, :, :]
            head_score = compute_head_attention_similarity_score(
                head_attention_pattern,
                detection_pattern=detection_pattern,
                exclude_bos=exclude_bos,
                exclude_current_token=exclude_current_token,
                error_measure=error_measure,
            )
            matches[layer, head] = head_score
    return matches

# Previous token head
def get_previous_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [previous token heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=0O5VOHe9xeZn8Ertywkh7ioc).

    Args:
      tokens: Tokens being fed to the model.
    """
    detection_pattern = torch.zeros(tokens.shape[-1], tokens.shape[-1])
    # Adds a diagonal of 1's below the main diagonal.
    detection_pattern[1:, :-1] = torch.eye(tokens.shape[-1] - 1)
    return torch.tril(detection_pattern)

# Duplicate token head
def get_duplicate_token_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [duplicate token heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=2UkvedzOnghL5UHUgVhROxeo).

    Args:
      sequence: String being fed to the model.
    """
    # [pos x pos]
    token_pattern = tokens.repeat(tokens.shape[-1], 1).numpy()

    # If token_pattern[i][j] matches its transpose, then token j and token i are duplicates.
    eq_mask = np.equal(token_pattern, token_pattern.T).astype(int)

    np.fill_diagonal(
        eq_mask, 0
    )  # Current token is always a duplicate of itself. Ignore that.
    detection_pattern = eq_mask.astype(int)
    return torch.tril(torch.as_tensor(detection_pattern).float())

# Induction head
def get_induction_head_detection_pattern(
    tokens: torch.Tensor,  # [batch (1) x pos]
) -> torch.Tensor:
    """Outputs a detection score for [induction heads](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=_tFVuP5csv5ORIthmqwj0gSY).

    Args:
      sequence: String being fed to the model.
    """
    duplicate_pattern = get_duplicate_token_head_detection_pattern(tokens)

    # Shift all items one to the right
    shifted_tensor = torch.roll(duplicate_pattern, shifts=1, dims=1)

    # Replace first column with 0's
    # we don't care about bos but shifting to the right moves the last column to the first,
    # and the last column might contain non-zero values.
    zeros_column = torch.zeros(duplicate_pattern.shape[0], 1)
    result_tensor = torch.cat((zeros_column, shifted_tensor[:, 1:]), dim=1)
    return torch.tril(result_tensor)

def get_supported_heads() -> None:
    """Returns a list of supported heads."""
    print(f"Supported heads: {HEAD_NAMES}")

def compute_head_attention_similarity_score(
    attention_pattern: torch.Tensor,  # [q_pos k_pos]
    detection_pattern: torch.Tensor,  # [seq_len seq_len] (seq_len == q_pos == k_pos)
    *,
    exclude_bos: bool,
    exclude_current_token: bool,
    error_measure: ErrorMeasure,
) -> float:
    """Compute the similarity between `attention_pattern` and `detection_pattern`.

    Args:
      attention_pattern: Lower triangular matrix (Tensor) representing the attention pattern of a particular attention head.
      detection_pattern: Lower triangular matrix (Tensor) representing the attention pattern we are looking for.
      exclude_bos: `True` if the beginning-of-sentence (BOS) token should be omitted from comparison. `False` otherwise.
      exclude_bcurrent_token: `True` if the current token at each position should be omitted from comparison. `False` otherwise.
      error_measure: "abs" for using absolute values of element-wise differences as the error measure. "mul" for using element-wise multiplication (legacy code).
    """
    assert is_square(
        attention_pattern
    ), f"Attention pattern is not square; got shape {attention_pattern.shape}"

    # mul

    if error_measure == "mul":
        if exclude_bos:
            attention_pattern[:, 0] = 0
        if exclude_current_token:
            attention_pattern.fill_diagonal_(0)
        score = attention_pattern * detection_pattern
        return (score.sum() / attention_pattern.sum()).item()

    # abs

    abs_diff = (attention_pattern - detection_pattern).abs()
    assert (abs_diff - torch.tril(abs_diff).to(abs_diff.device)).sum() == 0

    size = len(abs_diff)
    if exclude_bos:
        abs_diff[:, 0] = 0
    if exclude_current_token:
        abs_diff.fill_diagonal_(0)

    return 1 - round((abs_diff.mean() * size).item(), 3)

```

## Using Head Detector For Premade Heads

Load the model

```python
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
```

See what heads are supported out of the box

```python
get_supported_heads()
```

Let's test detecting previous token head in the following prompt.

```python
prompt = "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery."
head_scores = detect_head(model, prompt, "previous_token_head")
plot_head_detection_scores(head_scores, title="Previous Head Matches")
```

We can see both L2H2 and L4H11 are doing a fair bit of previous token detection. Let's take a look and see if that pans out.

```python
_, cache = model.run_with_cache(prompt)
```

```python
plot_attn_pattern_from_cache(cache, 2)
```

```python
plot_attn_pattern_from_cache(cache, 4)
```

As we expected, L2H2 is doing a lot of previous token detection, but doesn't appear to be a sharp previous token detection head. L4H11, on the other hand, is pretty much perfect. In fact, the only place it seems to be putting any other attention is the very first token, where it pays attention to the BOS (*beginning-of-sentence*) token.

Mechanistic interpretability is still a very new field, and we don't know the best ways to measure things yet. Ignoring attention paid to BOS allows us to solve problems like the above, but may also give us artifically high results for a head like L4H10, which doesn't appear to be doing much of anything, but does have a bit of previous token attention going on if you squint carefully.

As such, the head detector supports both an `exclude_bos` and `exclude_current_token` argument, which ignores all BOS attention and all current token attention respectively. By default these are `False`, but this is a pretty arbitrary decision, so feel free to try things out! You don't need a good reason to change these arguments - pick whatever best helps you find out useful things!

```python
head_scores = detect_head(model, prompt, "previous_token_head", exclude_bos=True, exclude_current_token=True)
plot_head_detection_scores(head_scores, title="Previous Head Matches")
```

Now we have a lot more detection, including L0H3 and L5H6 which were unremarkable before. Let's check them out!

```python
plot_attn_pattern_from_cache(cache, 5)
```

```python
plot_attn_pattern_from_cache(cache, 0)
```

Here, we see some interesting results. L5H6 does very little, but happens to react quite strongly to the first token of "Trans|former". (Capital letters? Current word detection? We don't know)

L0H3 reacts almost entirely to the current token, but what little it does outside of this pays attention to the previous token. Again, it seems to be caring about the first token of "Trans|former".

In order to more fully automate these heads, we'll need to discover more principled ways of expressing these scores. For now, you can see how while scores may be misleading, different scores lead us to interesting results.

## Using Head Detector for Custom Heads

These heads are great, but sometimes there are more than three things going on in Transformers. [citation needed] As a result, we may want to use our head detector for things that aren't pre-included in TransformerLens. Fortunately, the head detector provides support for this, via **detection patterns**.

A detection pattern is simply a matrix of the same size as our attention pattern, which specifies the attention pattern exhibited by the kind of head we're looking for.

There are two error measures available for quantifying the match between the detection pattern and the actual attention pattern. You can choose it by passing the right value to the `error_measure` argument.

### 1. `"mul"` (default) multiplies both tensors element-wise and divides the sum of the result by the sum of the attention pattern.

Typically, the detection pattern should in this case contain only ones and zeros, which allows a straightforward interpretation of the score: how big fraction of this head's attention is allocated to these specific query-key pairs? Using values other than 0 or 1 is not prohibited but will raise a warning (which can be disabled, of course).

<br>

$$
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0.5 & 0.5 & 0 & 0 \\
0.2 & 0.3 & 0.5 & 0 \\
0.1 & 0.15 & 0.5 & 0.25
\end{pmatrix}
\odot
\begin{pmatrix}
0 & 0 & 0 & 0 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
=
\begin{pmatrix}
0 & 0 & 0 & 0 \\
0.5 & 0 & 0 & 0 \\
0 & 0.3 & 0 & 0 \\
0 & 0 & 0.5 & 0
\end{pmatrix}
$$

<br>

0.5, 0.3, and 0.5 all get multiplied by 1, so they get kept. All the others go to 0 and are removed. (Note: You can use values other than 0 or 1 when creating your own heads)

Our total score would then be 1.3 / 4, or 0.325. If we ignore bos and current token, it would be 0.8 / 0.95 instead, or ~0.842. (This is a large difference, but the difference generally gets smaller as the matrices get bigger)

This is how the head detector works under the hood - each existing head just has its own detection pattern. Thus, we can pass in our own detection pattern using the `detection_pattern` argument.

### 2. `"abs"` calculates the mean element-wise absolute difference between the detection pattern and the actual attention pattern.

The "raw result" ranges from 0 to 2 where lower score corresponds to greater accuracy. Subtracting it from 1 maps that range to (-1, 1) interval, with 1 being perfect match and -1 perfect mismatch.

We take the attention pattern and compute its absolute element-wise difference with our detection pattern. Since every number in any of the two patterns has a value between -1 and 1, the maximum absolute difference of any pair is 2 and the minimum is 0:

$$|-1-1|=|1-(-1)|=2$$

$$|x-x|=0$$

That number tells us how much our expectation and the real attention pattern diverge, i.e., the error.

$$
M_{diff}=
\left|
\begin{pmatrix}
1 & 0 & 0 & 0
\\
0.5 & 0.5 & 0 & 0
\\
0.2 & 0.3 & 0.5 & 0
\\
0.1 & 0.15 & 0.5 & 0.25
\end{pmatrix}
-
\begin{pmatrix}
0 & 0 & 0 & 0
\\
1 & 0 & 0 & 0
\\
0 & 1 & 0 & 0
\\
0 & 0 & 1 & 0
\end{pmatrix}
\right|
=
\begin{pmatrix}
1 & 0 & 0 & 0
\\
0.5 & 0.5 & 0 & 0
\\
0.2 & 0.7 & 0.5 & 0
\\
0.1 & 0.15 & 0.5 & 0.25
\end{pmatrix}
$$

We take the mean and multiply it by the number of rows.

We subtract the result from 1 in order to map the (0, 2) interval where lower is better to the (-1, 1) interval where higher is better.

$$1 - \text{n_rows} \times \text{mean}(M_{diff}) = 1 - 4 \times 0.275 = 1 - 1.1 = -.1$$

Our final score would then be -1. If we ignore `BOS` and current token, it would be 0.6625. (This is a large difference, but the difference generally gets smaller as the matrices get bigger.)

This is how the head detector works under the hood - each existing head just has its own detection pattern. Thus, we can pass in our own detection pattern using the `detection_pattern` argument.

I'm curious what's going on with this L0H3 result, where we mostly focus on the current token but occasionally focus on the "Trans" token in "Trans|former". Let's make a **current word head** detection pattern, which returns 1 for previous tokens that are part of the current word being looked at, and 0 for everything else.

### **Which one should you use?**

`"abs"` is likely better for quick or exploratory investigations. For precise examinations where you're trying to reproduce as much functionality as possible or really test your understanding of the attention head, you probably want to switch to `"abs"`.

The advantage of `"abs"` is that you can make more precise predictions, and have that measured in the score. You can predict, for instance, 0.2 attention to X, and 0.8 attention to Y, and your score will be better if your prediction is closer. The "mul" metric does not allow this, you'll get the same score if attention is 0.2, 0.8 or 0.5, 0.5 or 0.8, 0.2.

Below we show how different scores these two measures can give on the same prompt. After that, we will proceed with using `"abs"` and will get back to `"mul"` at the end of the notebook.

```python
prompt = "The following lexical sequence has been optimised for the maximisation of loquaciously multitoken letter combinations."
tokens = model.to_str_tokens(prompt)
print(len(tokens), tokens)
detection_pattern = []
for i in range(2):
  detection_pattern.append([0 for t in tokens]) # Ignore BOS token and first token.
for i in range(2, len(tokens)):
    current_token = i
    previous_tokens_in_word = 0
    while not tokens[current_token].startswith(' '): # If the current token does not start with a space (and is not the first token) it's part of a word.
      previous_tokens_in_word += 1
      current_token -= 1
    # Hacky code that adds in some 1's where needed, and fills the rest of the row with 0's.
    detection_pattern.append([0 for j in range(i - previous_tokens_in_word)] + [1 for j in range(previous_tokens_in_word)] + [0 for j in range(i+1, len(tokens)+1)])
detection_pattern = torch.as_tensor(detection_pattern).to(device)
detection_pattern.shape
```

```python
_, cache = model.run_with_cache(prompt)
```

`"mul"`

```python
head_scores = detect_head(
    model,
    prompt,
    detection_pattern=detection_pattern,
    exclude_bos=False,
    exclude_current_token=True,
    error_measure="mul"
)
plot_head_detection_scores(head_scores, title="Current Word Head Matches (mul)")
```

`"abs"`

```python
head_scores = detect_head(
    model,
    prompt,
    detection_pattern=detection_pattern,
    exclude_bos=False,
    exclude_current_token=True,
    error_measure="abs"
)
plot_head_detection_scores(head_scores, title="Current Word Head Matches (abs)")
```

75% match for L0H3 - only 16% for L5H6. Let's check them out with our new sequence!

```python
plot_attn_pattern_from_cache(cache, 5)
```

```python
plot_attn_pattern_from_cache(cache, 0)
```

As we can see, L5H6 appears to be doing something totally different than we expected, whereas L0H3 is mostly doing what we expected - by our original hypothesis, we would expect "lo|qu|aciously" to have a lot of attention paid to, and "combinations|." the same, which didn't happen. However, our two-token words were exactly as we expected. Could this be a two-token detector (that doesn't work on punctuation)? A "current word" detector that just doesn't understand an obscure word like "loquaciously"? The field is full of such problems, just waiting to be answered!

So, why do this at all? For just a couple of sentences, it's easier to just look at the attention patterns directly and see what we get. But as we can see, heads react differently to different sentences. What we might want to do is give an entire dataset or distribution of sentences to our attention head and see that it consistently does what we want - that's something that would be much harder without this feature!

So what if we gave it a whole distribution? Rather than actually create one, which is not the point of this demo, we're just going to repeat our last sentence a thousand times.

```python
scores = []
for i in tqdm(range(100)):
    scores.append(detect_head(model, prompt, detection_pattern=detection_pattern, exclude_bos=False, exclude_current_token=True, error_measure="abs"))
scores = torch.stack(scores).mean(dim=0)
plot_head_detection_scores(scores, title="Current Word Head Matches")
```

## Processing Many Prompts

`detect_head` can also take more than one prompt. The resulting attention score is the mean of scores for each prompt.

```python
prompts = [
    "This is the first the test prompt.",
    "This is another test prompt, being just a sequence of tokens.",
    "If you're interested in mechanistic interpretability, this is how the sausage REALLY is made."
]
```

```python
head_scores = detect_head(model, prompts, "previous_token_head", error_measure="abs")
plot_head_detection_scores(head_scores, title="Previous token head; average across 3 prompts")
```

L4H11 emerges again as the dominant head, exactly as expected.

What about duplicate token heads?

```python
head_scores = detect_head(model, prompts, "duplicate_token_head", error_measure="abs")
plot_head_detection_scores(head_scores, title="Duplicate token head; average across 3 prompts")
```

Nothing but this should be expected, in hindsight, since our prompts don't contain too many duplicate tokens. Let's try three other prompts that do.

```python
prompts = [
    "one two three one two three one two three",
    "1 2 3 4 5 1 2 3 4 1 2 3 1 2 3 4 5 6 7",
    "green ideas sleep furiously; green ideas don't sleep furiously"
]
```

```python
head_scores = detect_head(model, prompts, "duplicate_token_head", exclude_bos=False, exclude_current_token=False, error_measure="abs")
plot_head_detection_scores(head_scores, title="Duplicate token head; average across 3 prompts")
```

3 or 4 heads seem to do something that we would expected from a duplicate token head but the signal is not very strong. You can tweak the `exclude_bos` and `exclude_current_token` flags if you want, but it doesn't change much.

Let's hunt for induction heads now!

```python
head_scores = detect_head(model, prompts, "induction_head", exclude_bos=False, exclude_current_token=False, error_measure="abs")
plot_head_detection_scores(head_scores, title="Duplicate token head; average across 3 prompts")
```

Similarly, at least on average.

Try running the script on different prompts and see if you can get high values for duplicate token or induction heads.

## Why not element-wise multiplication - robustness against [Goodharting](https://en.wikipedia.org/wiki/Goodhart%27s_law)

Initially, the error measure was not the mean element-wise absolute value error (normalized to the number of rows) but the mean [element-wise product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)). However, it had its problems, such as susceptibility to Goodharting. You can specify a pattern consisting of all ones and in this way achieve a perfect match for all layers and heads in the model.

More generally, using element-wise product causes the score to go down when we narrow our hypothesis. We can get a maximum score by just predicting 1 for everything.

```python
prompt = "The head detector feature for TransformerLens allows users to check for various common heads automatically, reducing the cost of discovery."
seq_len = len(model.to_str_tokens(prompt))
# torch.tril to make the pattern lower triangular
ones_detection_pattern = torch.tril(torch.ones(seq_len, seq_len).to(device))
```

```python
ones_head_scores = detect_head(
    model,
    prompt,
    ones_detection_pattern,
    exclude_bos=True,
    exclude_current_token=True,
)
plot_head_detection_scores(ones_head_scores, title="Transformers Have Now Been Solved, We Can All Go Home")
```

The new error measure also achieves uniform score but this time its uniformly extremely negative because **not a single head in the model matches this pattern**.

*(It's true that the scores descend below -9 whereas in theory they should remain within the (-1, 1) range. It's not yet clear if that matters for real-world uses.)*

An alternative would be to demand that *predictions add up to 1 for each row* but that seems unnecessarily nitpicky considering that your score will get reduced in general for not doing that anyway.

Mean squared errors have also bean tried before converging on the absolute ones. The problem with MSE is that the scores get lower as attention gets more diffuse. Error value of 1 would become 1, 0.5 would become 0.25 etc.

```python
ones_head_scores = detect_head(
    model,
    prompt,
    ones_detection_pattern,
    exclude_bos=True,
    exclude_current_token=True,
    error_measure="abs" # we specify the error measure here
)
plot_head_detection_scores(ones_head_scores, title="Transformers Have Not Been Solved Yet, Get Back To Work!")
```

## Further improvements

**Performance for large distributions** isn't as good as it could be. The head detector could be rewritten to support taking in a list of sequences and performing these computations in parallel, but 1000 sequences per minute is certainly adequate for most use cases. If having this be faster would help your research, please write up an issue on TransformerLens, mention it on the Open Source Mechanistic Interpretability Slack, or e-mail jaybaileycs@gmail.com.

### Other

- Extending to few-shot learning/translation heads
- More pre-specified heads?
- For inspiration, see [this post from Neel](https://www.lesswrong.com/s/yivyHaCAmMJ3CqSyj/p/btasQF7wiCYPsr5qw)

---

# Interactive_Neuroscope.ipynb

<a target="_blank" href="https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Interactive_Neuroscope.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Interactive Neuroscope

*This is an interactive accompaniment to [neuroscope.io](https://neuroscope.io) and to the [studying learned language features post](https://www.alignmentforum.org/posts/Qup9gorqpd9qKAEav/200-cop-in-mi-studying-learned-features-in-language-models) in [200 Concrete Open Problems in Mechanistic Interpretability](https://neelnanda.io/concrete-open-problems)*

There's a surprisingly rich ecosystem of easy ways to create interactive graphics, especially for ML systems. If you're trying to do mechanistic interpretability, the ability to do web dev and to both visualize data and interact with it seems high value!

This is a demo of how you can combine HookedTransformer and [Gradio](https://gradio.app/) to create an interactive Neuroscope - a visualization of a neuron's activations on text that will dynamically update as you edit the text. I don't particularly claim that this code is any *good*, but the goal is to illustrate what quickly hacking together a custom visualisation (while knowing fuck all about web dev, like me) can look like! (And as such, I try to explain the basic web dev concepts I use)

Note that you'll need to run the code yourself to get the interactive interface, so the cell at the bottom will be blank at first!

To emphasise - the point of this notebook is to be a rough proof of concept that just about works, *not* to be the well executed ideal of interactively studying neurons! You are highly encouraged to write your own (and ideally, to [make a pull request](https://github.com/neelnanda-io/TransformerLens/pulls) with improvements!)

## Setup

```python
# NBVAL_IGNORE_OUTPUT
# Janky code to do different setup when run in a Colab notebook vs VSCode
import os

DEVELOPMENT_MODE = True
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB or IN_GITHUB:
    %pip install transformer_lens
    %pip install gradio
    %pip install datasets==2.19.1
```

```python
import gradio as gr
from transformer_lens import HookedTransformer
from transformer_lens.utils import to_numpy
from IPython.display import HTML
```

## Extracting Model Activations

We first write some code using HookedTransformer's cache to extract the neuron activations on a given layer and neuron, for a given text

```python
# NBVAL_IGNORE_OUTPUT
model_name = "gpt2-small"
model = HookedTransformer.from_pretrained(model_name)
```

```python
def get_neuron_acts(text, layer, neuron_index):
    # Hacky way to get out state from a single hook - we have a single element list and edit that list within the hook.
    cache = {}

    def caching_hook(act, hook):
        cache["activation"] = act[0, :, neuron_index]

    model.run_with_hooks(
        text, fwd_hooks=[(f"blocks.{layer}.mlp.hook_post", caching_hook)]
    )
    return to_numpy(cache["activation"])
```

We can run this function and verify that it gives vaguely sensible outputs

```python
default_layer = 9
default_neuron_index = 652
default_text = "The following is a list of powers of 10: 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000"
print(model.to_str_tokens(default_text))
```

```python
# NBVAL_IGNORE_OUTPUT
print(get_neuron_acts(default_text, default_layer, default_neuron_index))
```

## Visualizing Model Activations

We now write some code to visualize the neuron activations on some text - we're going to hack something together which just does some string processing to make an HTML string, with each token element colored according to the intensity neuron activation. We normalize the neuron activations so they all lie in [0, 1]. You can do much better, but this is a useful proof of concept of what "just hack stuff together" can look like!

I'll be keeping neuron 562 in layer 9 as a running example, as it seems to activate strongly on powers of 10.

Note that this visualization is very sensitive to `max_val` and `min_val`! You can tune those to whatever seems reasonable for the distribution of neuron activations you care about - I generally default to `min_val=0` and `max_val` as the max activation across the dataset.

```python
# This is some CSS (tells us what style )to give each token a thin gray border, to make it easy to see token separation
style_string = """<style>
    span.token {
        border: 1px solid rgb(123, 123, 123)
        }
    </style>"""

def calculate_color(val, max_val, min_val):
    # Hacky code that takes in a value val in range [min_val, max_val], normalizes it to [0, 1] and returns a color which interpolates between slightly off-white and red (0 = white, 1 = red)
    # We return a string of the form "rgb(240, 240, 240)" which is a color CSS knows
    normalized_val = (val - min_val) / max_val
    return f"rgb(240, {240*(1-normalized_val)}, {240*(1-normalized_val)})"

def basic_neuron_vis(text, layer, neuron_index, max_val=None, min_val=None):
    """
    text: The text to visualize
    layer: The layer index
    neuron_index: The neuron index
    max_val: The top end of our activation range, defaults to the maximum activation
    min_val: The top end of our activation range, defaults to the minimum activation

    Returns a string of HTML that displays the text with each token colored according to its activation

    Note: It's useful to be able to input a fixed max_val and min_val, because otherwise the colors will change as you edit the text, which is annoying.
    """
    if layer is None:
        return "Please select a Layer"
    if neuron_index is None:
        return "Please select a Neuron"
    acts = get_neuron_acts(text, layer, neuron_index)
    act_max = acts.max()
    act_min = acts.min()
    # Defaults to the max and min of the activations
    if max_val is None:
        max_val = act_max
    if min_val is None:
        min_val = act_min
    # We want to make a list of HTML strings to concatenate into our final HTML string
    # We first add the style to make each token element have a nice border
    htmls = [style_string]
    # We then add some text to tell us what layer and neuron we're looking at - we're just dealing with strings and can use f-strings as normal
    # h4 means "small heading"
    htmls.append(f"<h4>Layer: <b>{layer}</b>. Neuron Index: <b>{neuron_index}</b></h4>")
    # We then add a line telling us the limits of our range
    htmls.append(
        f"<h4>Max Range: <b>{max_val:.4f}</b>. Min Range: <b>{min_val:.4f}</b></h4>"
    )
    # If we added a custom range, print a line telling us the range of our activations too.
    if act_max != max_val or act_min != min_val:
        htmls.append(
            f"<h4>Custom Range Set. Max Act: <b>{act_max:.4f}</b>. Min Act: <b>{act_min:.4f}</b></h4>"
        )
    # Convert the text to a list of tokens
    str_tokens = model.to_str_tokens(text)
    for tok, act in zip(str_tokens, acts):
        # A span is an HTML element that lets us style a part of a string (and remains on the same line by default)
        # We set the background color of the span to be the color we calculated from the activation
        # We set the contents of the span to be the token
        htmls.append(
            f"<span class='token' style='background-color:{calculate_color(act, max_val, min_val)}' >{tok}</span>"
        )

    return "".join(htmls)
```

```python
# NBVAL_IGNORE_OUTPUT
# The function outputs a string of HTML
default_max_val = 4.0
default_min_val = 0.0
default_html_string = basic_neuron_vis(
    default_text,
    default_layer,
    default_neuron_index,
    max_val=default_max_val,
    min_val=default_min_val,
)

# IPython lets us display HTML
print("Displayed HTML")
display(HTML(default_html_string))

# We can also print the string directly
print("HTML String - it's just raw HTML code!")
print(default_html_string)
```

## Create Interactive UI

We now put all these together to create an interactive visualization in Gradio!

The internal format is that there's a bunch of elements - Textboxes, Numbers, etc which the user can interact with and which return strings and numbers. And we can also define output elements that just display things - in this case, one which takes in an arbitrary HTML string. We call `input.change(update_function, inputs, output)` - this says "if that input element changes, run the update function on the value of each of the elements in `inputs` and set the value of `output` to the output of the function". As a bonus, this gives us live interactivity!

This is also more complex than a typical Gradio intro example - I wanted to use custom HTML to display the nice colours, which made things much messier! Normally you could just make `out` into another Textbox and pass it a string.

```python
# The `with gr.Blocks() as demo:` syntax just creates a variable called demo containing all these components
with gr.Blocks() as demo:
    gr.HTML(value=f"Hacky Interactive Neuroscope for {model_name}")
    # The input elements
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="Text", value=default_text)
            # Precision=0 makes it an int, otherwise it's a float
            # Value sets the initial default value
            layer = gr.Number(label="Layer", value=default_layer, precision=0)
            neuron_index = gr.Number(
                label="Neuron Index", value=default_neuron_index, precision=0
            )
            # If empty, these two map to None
            max_val = gr.Number(label="Max Value", value=default_max_val)
            min_val = gr.Number(label="Min Value", value=default_min_val)
            inputs = [text, layer, neuron_index, max_val, min_val]
        with gr.Column():
            # The output element
            out = gr.HTML(label="Neuron Acts", value=default_html_string)
    for inp in inputs:
        inp.change(basic_neuron_vis, inputs, out)
```

We can now launch our demo element, and we're done! The setting share=True even gives you a public link to the demo (though it just redirects to the backend run by this notebook, and will go away once you turn the notebook off!) Sharing makes it much slower, and can be turned off if you aren't in a colab.

**Exercise:** Explore where this neuron does and does not activate. Is it just powers of ten? Just comma separated numbers? Numbers in any particular sequence?

```python
# NBVAL_IGNORE_OUTPUT
demo.launch(share=True, height=1000)
```

---

# LLaMA.ipynb


---

# LLaMA2_GPU_Quantized.ipynb


---

# LLaVA.ipynb

### LLaVA use case demonstration

At that notebook you can see simple example of how to use TransformerLens for LLaVA interpretability. More specifically you can pass united image patch embeddings and textual embedding to LLaVA language model (Vicuna) with TransformerLens and get logits and cache that contains activations for next analysis. Here we consider the simplest example of LLaVA and TransformerLens sharing.

```python
# import staff
import sys

# Uncomment if use clonned version of TransformerLens
# currently forked version https://github.com/zazamrykh/TransformerLens supports
TL_path = r"../"
if TL_path not in sys.path:
	sys.path.insert(0, TL_path)
	sys.path.insert(0, TL_path + r"/transformer_lens")

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration  # Should update transformer to latest version

# For image loading
from PIL import Image
import requests
from io import BytesIO

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import matplotlib.pyplot as plt
%matplotlib inline

from transformer_lens import HookedTransformer
import circuitsvis as cv

_ = torch.set_grad_enabled(False)
```

Load llava model from hugging face. Load some revision because at this moment newest one is not working.

```python
model_id = "llava-hf/llava-1.5-7b-hf"

llava = LlavaForConditionalGeneration.from_pretrained(
	model_id,
	torch_dtype=torch.float16,
	load_in_4bit=False,
	low_cpu_mem_usage=True,
	revision="a272c74",
	device_map="cpu"
)

for param in llava.parameters():  # At this demo we don't need grads
	param.requires_grad = False

processor = AutoProcessor.from_pretrained(model_id, revision="a272c74")
tokenizer = processor.tokenizer

# Taking model apart
language_model = llava.language_model.eval()
config = language_model.config
print("Base language model:", config._name_or_path)

vision_tower = llava.vision_tower.to(device).eval()
projector = llava.multi_modal_projector.to(device).eval()
```

```python
# You can write your own version of getting language model's input embeddings similar way
# This function will not be working with old transformers library version. Should update transformers library.
def get_llm_input_embeddings(llava, processor, image: Image, text: str, device='cuda'):
    """ Extract features from image, project them to LLM's space and insert them to text embedding sequence.
    Returns:
    	inputs_embeds, attention_mask, labels, position_ids - input for language model of LLaVA
    """
    conversation = [
      {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image"},
          ],
      },
    	]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
    llava.vision_tower.to(device)
    llava.multi_modal_projector.to(device)

    clip_output = llava.vision_tower(inputs['pixel_values'])
    projector_output = llava.multi_modal_projector(clip_output.last_hidden_state)

    before_device = llava.language_model.model.embed_tokens.weight.device
    llava.language_model.model.embed_tokens.to(device)
    text_embeddings = llava.language_model.model.embed_tokens(inputs['input_ids'])
    llava.language_model.model.embed_tokens.to(before_device)

    full_sequence = torch.hstack([projector_output, text_embeddings])

    attention_mask = torch.ones(full_sequence.shape[:-1], device=full_sequence.device, dtype=int)
    inputs_embeds, attention_mask, labels, position_ids = llava._merge_input_ids_with_image_features(
		projector_output, text_embeddings, inputs['input_ids'], attention_mask, labels=None
	)  # Access to private member... Well, but what can i do :-)

    return inputs_embeds, attention_mask, labels, position_ids
```

Okay, now create HookedTransformer model

```python
hooked_llm = HookedTransformer.from_pretrained(
	"llama-7b-hf",  # Use config of llama
	center_unembed=False,
	fold_ln=False,
	fold_value_biases=False,
	device='cuda',
	hf_model=language_model,  # Use Vicuna's weights
	tokenizer=tokenizer,
	center_writing_weights=False,
	dtype=torch.float16,
	vocab_size=language_model.config.vocab_size  # New argument. llama and vicuna have different vocab size, so we pass it here
)

for param in hooked_llm.parameters():
	param.requires_grad = False
```

Now try if hooked model is working

```python
image_url = "https://github.com/zazamrykh/PicFinder/blob/main/images/doge.jpg?raw=true"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))
plt.axis('off')
_ = plt.imshow(image)
```

```python
question = "What do you see on photo?"
inputs_embeds, attention_mask, labels, position_ids = get_llm_input_embeddings(llava, processor, image, question, device=device)

# Return tokens
outputs = hooked_llm.generate(
	inputs_embeds,
	max_new_tokens=30,
	do_sample=True,
    return_type='tokens'
)
generated_text = processor.decode(outputs[0], skip_special_tokens=True)
print('Generated text:', generated_text)
```

```python
# Now return embeddings and then project them on vocab space
outputs = hooked_llm.generate(
	inputs_embeds,
	max_new_tokens=30,
	do_sample=True,
)

logits = outputs[:,-30:,:].to(device) @ language_model.model.embed_tokens.weight.T.to(device)
generated_text = processor.decode(logits.argmax(-1)[0], skip_special_tokens=True)
print('Generated text:', generated_text)
```

As we can see everything is working. Now try visualize attention patterns in generated output.

```python
# Here we visualize attention for the last 30 tokens.
logits, cache = hooked_llm.run_with_cache(inputs_embeds, start_at_layer=0, remove_batch_dim=True)

layer_to_visualize = 16
tokens_to_show = 30
attention_pattern = cache["pattern", layer_to_visualize, "attn"]

product = inputs_embeds @ language_model.model.embed_tokens.weight.T.to(device)  # Project embeddings to vocab
llama_str_tokens = hooked_llm.to_str_tokens(product.argmax(dim=-1)[0])

print(f"Layer {layer_to_visualize} Head Attention Patterns:")
display(cv.attention.attention_patterns(tokens=llama_str_tokens[-tokens_to_show:],
										attention=attention_pattern[:, -tokens_to_show:, -tokens_to_show:]))
```

As we can see image tokens also appears and can be used for multimodal attention exploration.

---

# Main_Demo.ipynb


---

# No_Position_Experiment.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/No_Position_Experiment.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Introduction

The accompanying notebook to my [real-time research](https://www.youtube.com/watch?v=yo4QvDn-vsU) video. Trains a model with no positional embeddings to predict the previous token, and makes a start at analysing what's going on there!

EDIT: The loss spikes were due to the learning rate being max(step/100, 1.0) not min! Thanks to MadHatter for catching that.

# Setup

```python
# NBVAL_IGNORE_OUTPUT
import os

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")

if IN_COLAB or IN_GITHUB:
    %pip install einops
    %pip install transformer_lens@v1.15.0

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git

from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
import numpy as np
import plotly.express as px
import plotly.io as pio

pio.renderers.default = "colab"
import tqdm.auto as tqdm
import einops
from transformer_lens.utils import to_numpy

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Some plotting code. Wrappers around Plotly, not important to understand.

```python
def line(tensor, line_labels=None, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    labels = {"y": yaxis, "x": xaxis}
    fig = px.line(tensor, labels=labels, **kwargs)
    if line_labels:
        for c, label in enumerate(line_labels):
            fig.data[c].name = label
    fig.show()

def imshow(tensor, yaxis="", xaxis="", **kwargs):
    tensor = to_numpy(tensor)
    plot_kwargs = {
        "color_continuous_scale": "RdBu",
        "color_continuous_midpoint": 0.0,
        "labels": {"x": xaxis, "y": yaxis},
    }
    plot_kwargs.update(kwargs)
    px.imshow(tensor, **plot_kwargs).show()
```

# Model Training

## Setup

### Defining the Model

```python
cfg = HookedTransformerConfig(
    n_layers=2,
    d_model=64,
    d_head=64,
    n_heads=1,
    d_mlp=256,
    d_vocab=300,
    n_ctx=50,
    act_fn="relu",
    normalization_type="LN",
    device=device,
)
model = HookedTransformer(cfg)
```

```python
def deactivate_position(model):
    model.pos_embed.W_pos.data[:] = 0.0
    model.pos_embed.W_pos.requires_grad = False

deactivate_position(model)
```

```python
print(model)
```

### Define data + Loss function

```python
def make_data_generator(cfg, batch_size, seed=123, incl_bos_token=True):
    torch.manual_seed(seed)
    while True:
        x = torch.randint(1, cfg.d_vocab, (batch_size, cfg.n_ctx))
        if incl_bos_token:
            x[:, 0] = 0
        yield x

data_generator = make_data_generator(cfg, 2)
print(next(data_generator))
```

```python
def loss_fn(logits, tokens, per_token=False):
    # logit shape: [batch, pos, vocab]
    # token shape: [batch, pos]
    logits = logits[:, 1:]
    tokens = tokens[:, :-1]
    log_probs = logits.log_softmax(-1)
    correct_log_probs = log_probs.gather(-1, tokens[..., None])[..., 0]
    if per_token:
        return -correct_log_probs
    else:
        return -correct_log_probs.mean()
```

```python
# Test the loss function works
test_tokens = torch.arange(5)[None, :]
test_logits = torch.randn(1, 5, 10)
test_logits[:, 1, 0] = 10.0
test_logits[:, 2, 1] = 10.0
test_logits[:, 3, 2] = 10.0
test_logits[:, 4, 3] = 10.0
print(loss_fn(test_logits, test_tokens, per_token=True))
print(loss_fn(test_logits, test_tokens, per_token=False))
```

### Setup Optimizer

```python
batch_size = 256
num_epochs = 4000
lr = 1e-4
betas = (0.9, 0.95)
max_grad_norm = 1.0
wd = 0.1
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 100, 1.0))

data_loader = make_data_generator(cfg, batch_size)
```

## Model Training

```python
losses = []
for epoch in tqdm.tqdm(range(num_epochs)):
    tokens = next(data_loader)
    tokens = tokens.to(device)
    logits = model(tokens)
    loss = loss_fn(logits, tokens)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    losses.append(loss.item())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: {loss.item()}")
px.line(losses, labels={"x": "Epoch", "y": "Loss"})
```

```python
# torch.save(model.state_dict(), "no_pos_experiment_state_dict_v0.pth")
```

# Model Interpretability

```python
model.pos_embed.W_pos.norm()
```

## Look at attention patterns

```python
big_data_loader = make_data_generator(cfg, 4000)
big_tokens = next(big_data_loader)
big_tokens = big_tokens.to(device)
logits, cache = model.run_with_cache(big_tokens)
print("Loss:", loss_fn(logits, big_tokens).item())
```

```python
print(cache)
```

```python
cache["blocks.0.attn.hook_pattern"].shape
```

```python
batch_index = 0
tokens = big_tokens[batch_index]
imshow(
    to_numpy(cache["attn", 0].mean([0, 1])),
    title="Layer 0 Attention Pattern",
    height=500,
    width=500,
)
imshow(
    to_numpy(cache["attn", 1].mean([0, 1])),
    title="Layer 1 Attention Pattern",
    height=500,
    width=500,
)
```

## Look at how different bits of the model directly contribute to the logits

```python
resid_components = [
    cache["embed"],
    cache["attn_out", 0],
    cache["mlp_out", 0],
    cache["attn_out", 1],
    cache["mlp_out", 1],
]
labels = ["embed", "A0", "M0", "A1", "M2"]
resid_stack = torch.stack(resid_components, 0)
resid_stack = resid_stack - resid_stack.mean(-1, keepdim=True)
print(resid_stack.shape)
```

```python
fold_W_U = model.ln_final.w[:, None] * model.unembed.W_U
logit_components = resid_stack[:, batch_index] @ fold_W_U / cache["scale"][batch_index]
print(logit_components.shape)
```

```python
logit_components = logit_components - logit_components.mean(-1, keepdim=True)
line(
    logit_components[:, torch.arange(1, model.cfg.n_ctx).to(device), tokens[:-1]].T,
    line_labels=labels,
)
```

## Folding In LayerNorm

```python
analysis_cfg = HookedTransformerConfig(
    n_layers=2,
    d_model=64,
    d_head=64,
    n_heads=1,
    d_mlp=256,
    d_vocab=300,
    n_ctx=50,
    act_fn="relu",
    normalization_type="LNPre",
    init_weights=False,
)
analysis_model = HookedTransformer(analysis_cfg)
state_dict = model.state_dict()
analysis_model.load_and_process_state_dict(
    state_dict, fold_ln=True, center_writing_weights=True, center_unembed=True
)
deactivate_position(analysis_model)
```

```python
# analysis_model()
```

## Understand Attn 0

```python
QK = model.W_E @ model.W_Q[0, 0] @ model.W_K[0, 0].T @ model.W_E.T
imshow(QK, yaxis="Query", xaxis="Key")
```

```python
OV = model.W_E @ model.W_V[0, 0] @ model.W_O[0, 0] @ model.W_in[0]
imshow(OV, yaxis="Input Vocab", xaxis="Neuron")
```

```python
line(OV[:, torch.randint(0, 256, (5,))])
```

## Understand MLP 0

```python
imshow(cache["post", 0][batch_index], yaxis="Pos", xaxis="Neuron")
imshow(cache["post", 0].mean(0), yaxis="Pos", xaxis="Neuron")
imshow((cache["post", 0] > 0).float()[batch_index], yaxis="Pos", xaxis="Neuron")
imshow((cache["post", 0] > 0).float().mean(0), yaxis="Pos", xaxis="Neuron")
```

## Understand Attn 1

## Understand MLP 1

```python

```

# Experiment

```python
new_token_batch = next(big_data_loader).to(device)
baseline_loss = loss_fn(model(new_token_batch), new_token_batch).item()
print("Baseline loss:", baseline_loss)
```

```python
hook_list = list(model.hook_dict.keys())
losses = []
loss_labels = []
for hook_name in hook_list:
    if (
        hook_name in cache
        and hook_name != "hook_pos_embed"
        and "result" not in hook_name
    ):
        average_act = cache[hook_name].mean(0)

        def replacing_with_average_act(activation, hook):
            activation[:] = einops.repeat(
                average_act, "... -> batch ...", batch=new_token_batch.size(0)
            )
            return activation

        logits = model.run_with_hooks(
            new_token_batch, fwd_hooks=[(hook_name, replacing_with_average_act)]
        )
        loss = loss_fn(logits, new_token_batch)
        print(hook_name, loss.item())
        losses.append(loss.item())
        loss_labels.append(hook_name)
```

```python
line(losses, hover_name=loss_labels)
```

```python
cache.cache_dict.keys()
```

---

# Othello_GPT.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Othello_GPT.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

This is a demo notebook porting the weights of the Othello-GPT Model from the excellent [Emergent World Representations](https://arxiv.org/pdf/2210.13382.pdf) paper to my TransformerLens library. Check out the paper's [blog post](https://thegradient.pub/othello/), [paper](https://arxiv.org/pdf/2210.13382.pdf), and [github](https://github.com/likenneth/othello_world/)

I think this is a super interesting paper, and I want to better enable work trying to reverse-engineer this model! I'm particularly curious about:
* Why non-linear probes work much better than linear probes?
    * Is the model internally representing the board in a usable yet non-linear way?
    * Is there a representation of simpler concepts (eg diagonal lines in the board, number of black pieces, whether a cell is blank)) that the non-linear probe uses to compute board positions, but where the model internally reasons in this simpler representation?
* What's going up with the model editing?
    * The paper edits across many layers at once. What's the minimal edit that works?
        * Can we edit just before the final layer?
        * Can we do a single edit rather than across many layers?
    * If we contrast model activations pre and post edit, what changes?
        * Which components shift their output and how does this affect the logits?
        * Is there significant depth of composition, or does it just affect the output logits?
* Can we find any non-trivial circuits in the model?
    * Start with [exploratory techniques](https://neelnanda.io/exploratory-analysis-demo), like direct logit attribution, or just looking at head attention patterns, and try to get traction
    * Pick a simple sub-task, eg figuring out whether a cell is blank, and try to interpret that.

I uploaded pre-converted checkpoints to HuggingFace, which can be automatically downloaded, and there's a code snippet to do this after the setup.

If you want to use the author's code, I wrote a script to load and convert checkpoints from the author's code, given below this.

To get started, check out the transformer lens [main tutorial](https://neelnanda.io/transformer-lens-demo) and [tutorial on exploratory techniques](https://neelnanda.io/exploratory-analysis-demo), and the author's [excellent Github](https://github.com/likenneth/othello_world/) (Ot**hello world**) for various notebooks demonstrating their code, showing how to load inputs, etc. And check out my [concrete open problems in mechanistic interpretability](https://www.lesswrong.com/s/yivyHaCAmMJ3CqSyj) sequence, especially the algorithmic problems post, for tips on this style of research.

# Setup (Skip)

```python
# NBVAL_IGNORE_OUTPUT
import os

# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
IN_GITHUB = os.getenv("GITHUB_ACTIONS") == "true"
try:
    import google.colab

    IN_COLAB = True
    print("Running as a Colab notebook")

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")

if IN_COLAB or IN_GITHUB:
    %pip install transformer_lens
    %pip install circuitsvis
    %pip install torchtyping
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")
```

```python
import circuitsvis as cv

# Testing that the library works
cv.examples.hello("Neel")
```

```python
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
```

```python
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import (
    HookedTransformer,
    HookedTransformerConfig,
    FactoredMatrix,
    ActivationCache,
)
```

We turn automatic differentiation off, to save GPU memory, as this notebook focuses on model inference not model training.

```python
torch.set_grad_enabled(False)
```

Plotting helper functions:

```python
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)
```

# Othello GPT

```python
LOAD_AND_CONVERT_CHECKPOINT = False
```

```python
import transformer_lens.utils as utils

cfg = HookedTransformerConfig(
    n_layers=8,
    d_model=512,
    d_head=64,
    n_heads=8,
    d_mlp=2048,
    d_vocab=61,
    n_ctx=59,
    act_fn="gelu",
    normalization_type="LNPre",
)
model = HookedTransformer(cfg)
```

```python
# NBVAL_IGNORE_OUTPUT
sd = utils.download_file_from_hf(
    "NeelNanda/Othello-GPT-Transformer-Lens", "synthetic_model.pth"
)
# champion_ship_sd = utils.download_file_from_hf("NeelNanda/Othello-GPT-Transformer-Lens", "championship_model.pth")
model.load_state_dict(sd)
```

Code to load and convert one of the author's checkpoints to TransformerLens:

```python
def convert_to_transformer_lens_format(in_sd, n_layers=8, n_heads=8):
    out_sd = {}
    out_sd["pos_embed.W_pos"] = in_sd["pos_emb"].squeeze(0)
    out_sd["embed.W_E"] = in_sd["tok_emb.weight"]

    out_sd["ln_final.w"] = in_sd["ln_f.weight"]
    out_sd["ln_final.b"] = in_sd["ln_f.bias"]
    out_sd["unembed.W_U"] = in_sd["head.weight"].T

    for layer in range(n_layers):
        out_sd[f"blocks.{layer}.ln1.w"] = in_sd[f"blocks.{layer}.ln1.weight"]
        out_sd[f"blocks.{layer}.ln1.b"] = in_sd[f"blocks.{layer}.ln1.bias"]
        out_sd[f"blocks.{layer}.ln2.w"] = in_sd[f"blocks.{layer}.ln2.weight"]
        out_sd[f"blocks.{layer}.ln2.b"] = in_sd[f"blocks.{layer}.ln2.bias"]

        out_sd[f"blocks.{layer}.attn.W_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_Q"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.query.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_K"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.key.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.weight"],
            "(head d_head) d_model -> head d_model d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_V"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.value.bias"],
            "(head d_head) -> head d_head",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.W_O"] = einops.rearrange(
            in_sd[f"blocks.{layer}.attn.proj.weight"],
            "d_model (head d_head) -> head d_head d_model",
            head=n_heads,
        )
        out_sd[f"blocks.{layer}.attn.b_O"] = in_sd[f"blocks.{layer}.attn.proj.bias"]

        out_sd[f"blocks.{layer}.mlp.b_in"] = in_sd[f"blocks.{layer}.mlp.0.bias"]
        out_sd[f"blocks.{layer}.mlp.W_in"] = in_sd[f"blocks.{layer}.mlp.0.weight"].T
        out_sd[f"blocks.{layer}.mlp.b_out"] = in_sd[f"blocks.{layer}.mlp.2.bias"]
        out_sd[f"blocks.{layer}.mlp.W_out"] = in_sd[f"blocks.{layer}.mlp.2.weight"].T

    return out_sd

if LOAD_AND_CONVERT_CHECKPOINT:
    synthetic_checkpoint = torch.load("/workspace/othello_world/gpt_synthetic.ckpt")
    for name, param in synthetic_checkpoint.items():
        if name.startswith("blocks.0") or not name.startswith("blocks"):
            print(name, param.shape)

    cfg = HookedTransformerConfig(
        n_layers=8,
        d_model=512,
        d_head=64,
        n_heads=8,
        d_mlp=2048,
        d_vocab=61,
        n_ctx=59,
        act_fn="gelu",
        normalization_type="LNPre",
    )
    model = HookedTransformer(cfg)

    model.load_and_process_state_dict(
        convert_to_transformer_lens_format(synthetic_checkpoint)
    )
```

Testing code for the synthetic checkpoint giving the correct outputs

```python
# An example input
sample_input = torch.tensor(
    [
        [
            20,
            19,
            18,
            10,
            2,
            1,
            27,
            3,
            41,
            42,
            34,
            12,
            4,
            40,
            11,
            29,
            43,
            13,
            48,
            56,
            33,
            39,
            22,
            44,
            24,
            5,
            46,
            6,
            32,
            36,
            51,
            58,
            52,
            60,
            21,
            53,
            26,
            31,
            37,
            9,
            25,
            38,
            23,
            50,
            45,
            17,
            47,
            28,
            35,
            30,
            54,
            16,
            59,
            49,
            57,
            14,
            15,
            55,
            7,
        ]
    ]
)
# The argmax of the output (ie the most likely next move from each position)
sample_output = torch.tensor(
    [
        [
            21,
            41,
            40,
            34,
            40,
            41,
            3,
            11,
            21,
            43,
            40,
            21,
            28,
            50,
            33,
            50,
            33,
            5,
            33,
            5,
            52,
            46,
            14,
            46,
            14,
            47,
            38,
            57,
            36,
            50,
            38,
            15,
            28,
            26,
            28,
            59,
            50,
            28,
            14,
            28,
            28,
            28,
            28,
            45,
            28,
            35,
            15,
            14,
            30,
            59,
            49,
            59,
            15,
            15,
            14,
            15,
            8,
            7,
            8,
        ]
    ]
)
model(sample_input).argmax(dim=-1)
```

---

# Patchscopes_Generation_Demo.ipynb


---

# Qwen.ipynb

```python
%pip install transformers_stream_generator plotly circuitsvis huggingface_hub einops tiktoken datasets
```

```python
# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    %pip install git+https://github.com/TransformerLensOrg/TransformerLens.git
    %pip install circuitsvis

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio
if IN_COLAB or not DEVELOPMENT_MODE:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")
```

```python
%cd ~/TransformerLens
import torch
torch.set_grad_enabled(False)

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from functools import partial
```

```python
def assert_hf_and_tl_model_are_close(
    hf_model,
    tl_model,
    tokenizer,
    prompt="This is a prompt to test out",
    atol=1e-3,
):
    prompt_toks = tokenizer(prompt, return_tensors="pt").input_ids

    hf_logits = hf_model(prompt_toks.to(hf_model.device)).logits
    tl_logits = tl_model(prompt_toks).to(hf_logits)

    assert torch.allclose(torch.softmax(hf_logits, dim=-1), torch.softmax(tl_logits, dim=-1), atol=atol)
```

## Qwen, first generation

```python
model_path = "Qwen/Qwen-1_8B-Chat"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    fp32=True,
    use_logn_attn=False,
    use_dynamic_ntk = False,
    scale_attn_weights = False,
    trust_remote_code = True
).eval()

tl_model = HookedTransformer.from_pretrained_no_processing(
    model_path,
    device=device,
    fp32=True,
    dtype=torch.float32,
).to(device)

assert_hf_and_tl_model_are_close(hf_model, tl_model, tokenizer)
```

## Qwen, new generation

```python
model_path = "Qwen/Qwen1.5-1.8B-Chat"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
)

hf_model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
).eval()

tl_model = HookedTransformer.from_pretrained_no_processing(
    model_path,
    device=device,
    dtype=torch.float32,
).to(device)

assert_hf_and_tl_model_are_close(hf_model, tl_model, tokenizer)
```

```python

```

---

# SVD_Interpreter_Demo.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/SVD_Interpreter_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## TransformerLens SVD Interpreter Demo

A few months ago, a Conjecture post came out about how the singular value decompositions of transformer matrices were [surprisingly interpretable](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight#Directly_editing_SVD_representations), leading to recognisable semantic clusters. This seemed like good functionality to add to TransformerLens, which is what the SVD Interpreter feature does. You simply need to pass it a model, the type of matrix you want, and the size of the results you want, then you can plot it using PySvelte. This demo will show you how it's done.

How to use this notebook:

**Go to Runtime > Change Runtime Type and select GPU as the hardware accelerator.**

Tips for reading this Colab:

* You can run all this code for yourself!
* The graphs are interactive!
* Use the table of contents pane in the sidebar to navigate
* Collapse irrelevant sections with the dropdown arrows
* Search the page using the search in the sidebar, not CTRL+F

## Setup (Can be ignored)

```python
# Janky code to do different setup when run in a Colab notebook vs VSCode
DEBUG_MODE = False
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    %pip install git+https://github.com/JayBaileyCS/TransformerLens.git # TODO: Change!
    # Install Neel's personal plotting utils
    %pip install git+https://github.com/neelnanda-io/neel-plotly.git
    # Install another version of node that makes PySvelte work way faster
    !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    %pip install git+https://github.com/neelnanda-io/PySvelte.git
    # Needed for PySvelte to work, v3 came out and broke things...
    %pip install typeguard==2.13.3
    %pip install typing-extensions
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
```

```python
# Plotly needs a different renderer for VSCode/Notebooks vs Colab argh
import plotly.io as pio

if IN_COLAB or not DEBUG_MODE:
    # Thanks to annoying rendering issues, Plotly graphics will either show up in colab OR Vscode depending on the renderer - this is bad for developing demos! Thus creating a debug mode.
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "png"
```

```python
import torch
import pysvelte
import numpy as np
import transformer_lens
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer, SVDInterpreter
```

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"{device = }")
```

## SVD Interpretation

The SVD Interpreter supports interpretation for three types of Transformer matrix:

* OV - The [output-value circuit](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=CLmGoD1pvjmsg0dPyL3wkuGS) of the matrix. (d_model x d_model) in size.
* w_in - Weights passed into the MLP block of the matrix. (d_model x (4 x d_model)) in size.
* w_out - Weights that come out of the MLP block of the matrix. ((4 x d_model) x d_model) in size.

The SVD interpreter handles everything behind the scenes, so you only need to pass in the model and the type of matrix you want. Let's give it a go!

We'll be passing in **fold_ln = False, center_writing_weights+false, and center_unembed=False** here to mimic the existing post as closely as possible in order to demonstrate that this works (and the numerical instability that makes it not *completely* work). You can do interpretability on the default model without these parameters, but you won't be able to replicate the same results. I haven't checked much to see how it affects their quality, though w_out seemed to decay greatly when center_unembed was True - this would be worth testing properly!

Replication with this type of analysis is inherently difficult, because linear dependence is numerically unstable. Very minor numerical changes (Like floating-point discrepancies) can alter the results slightly. (See [this comment](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight?commentId=4e8534hbyWCpZFgFD)) So don't worry if you don't get exactly the same results on different devices - this is, unfortunately, expected. Try to stick to the same device for all your experiments and be sure to point out which one you used when writing them up. (And if anyone has a more stable way to get these results, [let us know](https://github.com/TransformerLensOrg/TransformerLens/issues)!)

```python
model = HookedTransformer.from_pretrained("gpt2-medium", fold_ln=False, center_writing_weights=False, center_unembed=False)
```

```python
all_tokens = [model.to_str_tokens(np.array([i])) for i in range(model.cfg.d_vocab)]
all_tokens = [all_tokens[i][0] for i in range(model.cfg.d_vocab)]

# Utility function to plot values in the same style as the Conjecture post.
def plot_matrix(matrix, tokens, k=10, filter="topk"):
  pysvelte.TopKTable(tokens=all_tokens, activations=matrix, obj_type="SVD direction", k=k, filter=filter).show()
```

```python
svd_interpreter = SVDInterpreter(model)

ov = svd_interpreter.get_singular_vectors('OV', layer_index=22, head_index=10)
w_in = svd_interpreter.get_singular_vectors('w_in', layer_index=20)
w_out = svd_interpreter.get_singular_vectors('w_out', layer_index=16)

plot_matrix(ov, all_tokens)
plot_matrix(w_in, all_tokens)
plot_matrix(w_out, all_tokens)
```

Currently, this is the extent of our support for SVD interpretability. However, this is a very new idea, and we're excited to see how people use it! If you find an interesting use for this type of research that we don't cover, feel free to [open a ticket](https://github.com/TransformerLensOrg/TransformerLens/issues) or contact the code's author at jaybaileycs@gmail.com.

One thing I'd love to see that basically anyone who followed this demo could get started with (I'd consider it an **A-level problem** from Neel's [Concrete Open Problems sequence](https://www.lesswrong.com/s/yivyHaCAmMJ3CqSyj)) is to try different combinations of model parameters (fold_ln, center_writing_weights, center_unembed) and see which ones lead to big changes in the interpretability of the SVD matrices.

Are these changes positive, or negative? Can you pick any set of parameters you want? Are different parameters more or less interpretable in general, or does it vary by head and layer? Can you get two different interpretations of the same head with different parameters? What else can you find? This is very low-hanging fruit that would be immediately tractable and immediately useful!

---

# Santa_Coder.ipynb

```python
# Janky code to do different setup when run in a Colab notebook vs VSCode
DEVELOPMENT_MODE = False
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    %pip install git+https://github.com/TransformerLensOrg/TransformerLens.git``
    %pip install circuitsvis
    %pip install torchtyping

    # PySvelte is an unmaintained visualization library, use it as a backup if circuitsvis isn't working
    # # Install another version of node that makes PySvelte work way faster
    # !curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -; sudo apt-get install -y nodejs
    # %pip install git+https://github.com/neelnanda-io/PySvelte.git
except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    from IPython import get_ipython

    ipython = get_ipython()
    # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    ipython.magic("load_ext autoreload")
    ipython.magic("autoreload 2")
```

```python
# Import stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
from tqdm import tqdm
import random
from pathlib import Path
import plotly.express as px
from torch.utils.data import DataLoader

from torchtyping import TensorType as TT
from typing import List, Union, Optional
from jaxtyping import Float, Int
from functools import partial
import copy

import itertools
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import dataclasses
import datasets
from IPython.display import HTML
# import circuitsvis as cv

import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookedRootModule,
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, HookedTransformerConfig, FactoredMatrix, ActivationCache

torch.set_grad_enabled(False)

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)
```

```python
# load hf model
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-560m")
```

```python
# Disable folding norms and folding norms and biases so that intermediate value
# in between transformer blocks can be compared
bloom = HookedTransformer.from_pretrained("bloom-560m",fold_ln=False, fold_value_biases=False, center_writing_weights=False)
```

```python
text = '''
TransformerLens lets you load in 50+ different open source language models,
and exposes the internal activations of the model to you. You can cache
any internal activation in the model, and add in functions to edit, remove
or replace these activations as the model runs.
'''
input_ids = tokenizer(text, return_tensors='pt')['input_ids']
gt_logits = model(input_ids)['logits'] # ground truth logits from hf
my_logits = bloom(input_ids)
centered_gt_logits = gt_logits - gt_logits.mean(-1, keepdim=True)
mean_diff = (my_logits.cpu() - centered_gt_logits).mean()
print("avg logits difference:", mean_diff.item())
max_diff = (my_logits.cpu() - centered_gt_logits).abs().max()
print("max logits difference:", max_diff.item())
```

```python
gt_cache = model(input_ids, output_hidden_states=True)['hidden_states']
_, my_cache = bloom.run_with_cache(input_ids)
use_loose_bound = False
pass_loose_bound = True
print("*"*5, "Matching hf and T-Lens residual stream in between transformer blocks", "*"*5)
for i in range(24):
    try:
        torch.testing.assert_close(my_cache['resid_pre',i], gt_cache[i].cuda())
    except:
        max_diff = (my_cache['resid_pre',i] - gt_cache[i].cuda()).abs().max()
        print(f"layer {i} \t not close, max difference: {max_diff}")
        use_loose_bound = True

if use_loose_bound:
    atol = rtol = 1e-3
    print("*"*5, f"\ttesting with atol={atol} and rtol={rtol}\t","*"*5)
    for i in range(24):
        try:
            torch.testing.assert_close(my_cache['resid_pre',i], gt_cache[i].cuda(), atol=atol, rtol=rtol)
        except:
            max_diff = (my_cache['resid_pre',i] - gt_cache[i].cuda()).abs().max()
            print(f"layer {i} \t not close, max difference: {max_diff}")
            pass_loose_bound = False

    if pass_loose_bound:
        print(f"All layers match with atol={atol} rtol={rtol}")
else:
    print("All layers match")
```

```python
my_loss = bloom(input_ids, return_type='loss')
print("T-Lens next token loss:", my_loss.item())
gt_outputs = model(input_ids, labels=input_ids)
gt_loss = gt_outputs.loss
print("HF next token loss:", gt_loss.item())
print("diff in loss (abs):", (gt_loss-my_loss).abs().item())
```

---

# T5.ipynb


---

# Tracr_to_Transformer_Lens_Demo.ipynb

<a target="_blank" href="https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Tracr_to_Transformer_Lens_Demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# Tracr to TransformerLens Converter
[Tracr](https://github.com/deepmind/tracr) is a cool new DeepMind tool that compiles a written program in RASP to transformer weights. TransformerLens is a library I've written to easily do mechanistic interpretability on a transformer and to poke around at its internals. This is a (hacky!) script to convert Tracr weights from the JAX form to a TransformerLens HookedTransformer in PyTorch.

See [the TransformerLens tutorial](https://neelnanda.io/transformer-lens-demo) to get started

Python version must be >=3.8 (my fork of Tracr is a bit more backwards compatible, original library is at least 3.9)

```python
!python --version
```

```python
try:
    import google.colab
    IN_COLAB = True
    print("Running as a Colab notebook")
    %pip install transformer_lens
    # Fork of Tracr that's backward compatible with Python 3.8
    %pip install git+https://github.com/neelnanda-io/Tracr

except:
    IN_COLAB = False
    print("Running as a Jupyter notebook - intended for development only!")
    # from IPython import get_ipython

    # ipython = get_ipython()
    # # Code to automatically update the HookedTransformer code as its edited without restarting the kernel
    # ipython.magic("load_ext autoreload")
    # ipython.magic("autoreload 2")
```

```python
from transformer_lens import HookedTransformer, HookedTransformerConfig
import einops
import torch
import numpy as np

from tracr.rasp import rasp
from tracr.compiler import compiling
```

Loads an example RASP program model. This program reverses lists. The model takes as input a list of pre-tokenization elements (here `["BOS", 1, 2, 3]`), these are tokenized (`[3, 0, 1, 2]`), the transformer is applied, and then an argmax is taken over the output and it is detokenized - this can be seen on the `out.decoded` attribute of the output

```python

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

length = make_length()  # `length` is not a primitive in our implementation.
opp_index = length - rasp.indices - 1
flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
reverse = rasp.Aggregate(flip, rasp.tokens)

bos = "BOS"
model = compiling.compile_rasp_to_model(
    reverse,
    vocab={1, 2, 3},
    max_seq_len=5,
    compiler_bos=bos,
)

out = model.apply([bos, 1, 2, 3])
```

Extract the model config from the Tracr model, and create a blank HookedTransformer object

```python

# %%

n_heads = model.model_config.num_heads
n_layers = model.model_config.num_layers
d_head = model.model_config.key_size
d_mlp = model.model_config.mlp_hidden_size
act_fn = "relu"
normalization_type = "LN"  if model.model_config.layer_norm else None
attention_type = "causal"  if model.model_config.causal else "bidirectional"

n_ctx = model.params["pos_embed"]['embeddings'].shape[0]
# Equivalent to length of vocab, with BOS and PAD at the end
d_vocab = model.params["token_embed"]['embeddings'].shape[0]
# Residual stream width, I don't know of an easy way to infer it from the above config.
d_model = model.params["token_embed"]['embeddings'].shape[1]

# Equivalent to length of vocab, WITHOUT BOS and PAD at the end because we never care about these outputs
# In practice, we always feed the logits into an argmax
d_vocab_out = model.params["token_embed"]['embeddings'].shape[0] - 2

cfg = HookedTransformerConfig(
    n_layers=n_layers,
    d_model=d_model,
    d_head=d_head,
    n_ctx=n_ctx,
    d_vocab=d_vocab,
    d_vocab_out=d_vocab_out,
    d_mlp=d_mlp,
    n_heads=n_heads,
    act_fn=act_fn,
    attention_dir=attention_type,
    normalization_type=normalization_type,
)
tl_model = HookedTransformer(cfg)
```

Extract the state dict, and do some reshaping so that everything has a n_heads dimension

```python

# %%
sd = {}
sd["pos_embed.W_pos"] = model.params["pos_embed"]['embeddings']
sd["embed.W_E"] = model.params["token_embed"]['embeddings']
# Equivalent to max_seq_len plus one, for the BOS

# The unembed is just a projection onto the first few elements of the residual stream, these store output tokens
# This is a NumPy array, the rest are Jax Arrays, but w/e it's fine.
sd["unembed.W_U"] = np.eye(d_model, d_vocab_out)

for l in range(n_layers):
    sd[f"blocks.{l}.attn.W_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_K"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/key"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_Q"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/query"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["w"],
        "d_model (n_heads d_head) -> n_heads d_model d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_V"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/value"]["b"],
        "(n_heads d_head) -> n_heads d_head",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.W_O"] = einops.rearrange(
        model.params[f"transformer/layer_{l}/attn/linear"]["w"],
        "(n_heads d_head) d_model -> n_heads d_head d_model",
        d_head = d_head,
        n_heads = n_heads
    )
    sd[f"blocks.{l}.attn.b_O"] = model.params[f"transformer/layer_{l}/attn/linear"]["b"]

    sd[f"blocks.{l}.mlp.W_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["w"]
    sd[f"blocks.{l}.mlp.b_in"] = model.params[f"transformer/layer_{l}/mlp/linear_1"]["b"]
    sd[f"blocks.{l}.mlp.W_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["w"]
    sd[f"blocks.{l}.mlp.b_out"] = model.params[f"transformer/layer_{l}/mlp/linear_2"]["b"]
print(sd.keys())

```

Convert weights to tensors and load into the tl_model

```python

for k, v in sd.items():
    # I cannot figure out a neater way to go from a Jax array to a numpy array lol
    sd[k] = torch.tensor(np.array(v))

tl_model.load_state_dict(sd, strict=False)

```

Create helper functions to do the tokenization and de-tokenization

```python

# %%
INPUT_ENCODER = model.input_encoder
OUTPUT_ENCODER = model.output_encoder

def create_model_input(input, input_encoder=INPUT_ENCODER):
    encoding = input_encoder.encode(input)
    return torch.tensor(encoding).unsqueeze(dim=0)

def decode_model_output(logits, output_encoder=OUTPUT_ENCODER, bos_token=INPUT_ENCODER.bos_token):
    max_output_indices = logits.squeeze(dim=0).argmax(dim=-1)
    decoded_output = output_encoder.decode(max_output_indices.tolist())
    decoded_output_with_bos = [bos_token] + decoded_output[1:]
    return decoded_output_with_bos

```

We can now run the model!

```python

input = [bos, 1, 2, 3]
out = model.apply(input)
print("Original Decoding:", out.decoded)

input_tokens_tensor = create_model_input(input)
logits = tl_model(input_tokens_tensor)
decoded_output = decode_model_output(logits)
print("TransformerLens Replicated Decoding:", decoded_output)
# %%

```

Lets cache all intermediate activations in the model, and check that they're the same:

```python
logits, cache = tl_model.run_with_cache(input_tokens_tensor)

for layer in range(tl_model.cfg.n_layers):
    print(f"Layer {layer} Attn Out Equality Check:", np.isclose(cache["attn_out", layer].detach().cpu().numpy(), np.array(out.layer_outputs[2*layer])).all())
    print(f"Layer {layer} MLP Out Equality Check:", np.isclose(cache["mlp_out", layer].detach().cpu().numpy(), np.array(out.layer_outputs[2*layer+1])).all())
```

Look how pretty and ordered the final residual stream is!

(The logits are the first 3 dimensions of the residual stream, and we can see that they're flipped!)

```python
import plotly.express as px
px.imshow(cache["resid_post", -1].detach().cpu().numpy()[0],
color_continuous_scale="Blues", labels={"x":"Residual Stream", "y":"Position"}, y=[str(i) for i in input]).show("colab" if IN_COLAB else "")
```

---

# comparing-to-huggingface.ipynb

Compare the TransformerLens implementation of a model to the Huggingface implementation. This script was originally use in https://github.com/TransformerLensOrg/TransformerLens/issues/570 to debug Mixtral.

## setup

```python
%pip install transformers matplotlib
```

```python
# Everything can be configured here
model_id = ""
text = "Hello my name is"
device="cpu"
# Set this to true to trigger hugging face login if needed
gated_model = False
```

```python
# If you need a specific head, uncomment this and specify the head
# %pip install git+https://github.com/TransformerLensOrg/TransformerLens.git@head
# Otherwise, for running this on the latest release
%pip install transformer_lens
```

```python
if gated_model:
    %pip install huggingface_hub
    from huggingface_hub import login
    login()
```

```python
import einops
from torch.testing import assert_close
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
```

## TransformerLens model

```python
tl_model = HookedTransformer.from_pretrained_no_processing(
    model_id,
    device=device,
)
```

```python
tl_model.generate(
    text,
    verbose=False,
    max_new_tokens=50,
)
```

## Huggingface Model

```python
tokenizer = AutoTokenizer.from_pretrained(model_id)
hf_model = AutoModelForCausalLM.from_pretrained(model_id)
```

```python
inputs = tokenizer(text, return_tensors="pt")
outputs = hf_model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Compare Model Weights

```python
torch.all(
    einops.rearrange(tl_model.blocks[0].attn.W_Q, "n m h -> (n h) m") ==
    hf_model.model.layers[0].self_attn.q_proj.weight
)
```

```python
tl_model.blocks[0].attn.W_K.shape, hf_model.model.layers[0].self_attn.k_proj.weight.shape
```

```python
torch.all(
    einops.reduce(
        tl_model.blocks[0].attn.W_K, "(n repeat) m h -> (n h) m",
        'max',
        n=tl_model.cfg.n_key_value_heads,
        repeat=4) ==
    hf_model.model.layers[0].self_attn.k_proj.weight
)
```

```python
torch.all(
    einops.reduce(
        tl_model.blocks[0].attn.W_V, "(n repeat) m h -> (n h) m",
        'max',
        n=tl_model.cfg.n_key_value_heads,
        repeat=4) ==
    hf_model.model.layers[0].self_attn.v_proj.weight
)
```

```python
torch.all(
    einops.rearrange(tl_model.blocks[0].attn.W_O, "n h m -> m (n h)") ==
    hf_model.model.layers[0].self_attn.o_proj.weight
)
```

```python
tl_model.blocks[0].attn.b_Q
```

```python
torch.all(hf_model.model.layers[0].block_sparse_moe.gate.weight.T == tl_model.blocks[0].mlp.W_gate)
```

```python
hf_model.model.layers[0].block_sparse_moe.gate.weight.dtype, tl_model.blocks[0].mlp.W_gate.dtype
```

## Compare Layer Outputs

```python
test_tensor = torch.randn((1, 1, 4096,))
```

```python
hf_model.model.layers[0](test_tensor)
```

```python
tl_model.blocks[0](test_tensor)
```

```python
hf_model.model.layers[0](test_tensor)[0] == tl_model.blocks[0](test_tensor)
```

```python
hf_model.model.layers[0](test_tensor)[0][0, 0, -2].item(), tl_model.blocks[0](test_tensor)[0, 0, -2].item()
```

```python
torch.sum(hf_model.model.layers[0](test_tensor)[0] == tl_model.blocks[0](test_tensor))
```

```python
differences = hf_model.model.layers[0](test_tensor)[0] - tl_model.blocks[0](test_tensor)

# Flatten the differences to create a one-dimensional tensor
flattened_differences = differences.flatten().cpu().detach().numpy()

# Plot the histogram of the differences
plt.hist(flattened_differences, bins=50, alpha=0.75, color='blue')
plt.title('Differences Between Layer Outputs')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

## Compare MLP Outputs

```python
torch.all(
    tl_model.blocks[0].mlp.experts[0].W_in ==
    hf_model.model.layers[0].block_sparse_moe.experts[0].w3.weight.T
)
```

```python
test_tensor = torch.randn((1, 1, 4096,))
```

```python
torch.all(
    hf_model.model.layers[0].block_sparse_moe(test_tensor)[0] ==
    tl_model.blocks[0].mlp(test_tensor)
)
```

```python
hf_model.model.layers[0].block_sparse_moe(test_tensor)[0]
```

```python
tl_model.blocks[0].mlp(test_tensor)
```

```python
tl_model.blocks[0].mlp(test_tensor).shape
```

```python
hf_model.model.layers[0].block_sparse_moe(test_tensor)[0] == tl_model.blocks[0].mlp(test_tensor)
```

```python
torch.sum(hf_model.model.layers[0].block_sparse_moe(test_tensor)[0] == tl_model.blocks[0].mlp(test_tensor))
```

```python
differences = hf_model.model.layers[0].block_sparse_moe(test_tensor)[0] - tl_model.blocks[0].mlp(test_tensor)

# Flatten the differences to create a one-dimensional tensor
flattened_differences = differences.flatten().cpu().detach().numpy()

# Plot the histogram of the differences
plt.hist(flattened_differences, bins=50, alpha=0.75, color='blue')
plt.title('Differences Between MLP Outputs')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

```python
hf_model.model.layers[0].block_sparse_moe(test_tensor)[0][0, 0, 0].item()
```

```python
tl_model.blocks[0].mlp(test_tensor)[0, 0, 0].item()
```

## Compare Attention Outputs

```python
tl_model.blocks[0].attn.forward(test_tensor, test_tensor, test_tensor)
```

```python
hf_model.model.layers[0].self_attn.forward(test_tensor)[0]
```

```python
(tl_model.blocks[0].attn.forward(test_tensor, test_tensor, test_tensor) ==
 hf_model.model.layers[0].self_attn.forward(test_tensor)[0])
```

```python
torch.sum(tl_model.blocks[0].attn.forward(test_tensor, test_tensor, test_tensor) ==
 hf_model.model.layers[0].self_attn.forward(test_tensor)[0])
```

```python
differences = tl_model.blocks[0].attn.forward(test_tensor, test_tensor, test_tensor) - hf_model.model.layers[0].self_attn.forward(test_tensor)[0]

# Flatten the differences to create a one-dimensional tensor
flattened_differences = differences.flatten().cpu().detach().numpy()

# Plot the histogram of the differences
plt.hist(flattened_differences, bins=50, alpha=0.75, color='blue')
plt.title('Differences Between Attention Outputs')
plt.xlabel('Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
```

---

# hf-tl-logit-comparator.ipynb

# Logit Comparator for HuggingFace and TransformerLens Outputs
This notebook is a quick and dirty tool to compare the logit outputs of a HuggingFace model and a TransformerLens model via several different metrics. It is intended to help debug issues with the TransformerLens model, such as bugs in the model's implementation. If you identify any issues, please open an issue on the [GitHub repository](https://github.com/TransformerLensOrg/TransformerLens).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import torch
import torch.nn.functional as F

if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_grad_enabled(False)
```

## Comparator Setup

```python
model_name = "EleutherAI/pythia-2.8b"  # You can change this to any model name
sentence = "The quick brown fox"
```

```python
from huggingface_hub import login
login(token="")
```

## Get Transformers Logits

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def get_logits(model, tokenizer, sentence, device):
    # Tokenize the input sentence
    inputs = tokenizer(sentence, return_tensors="pt")

    # Move inputs to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate the logits
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the logits for all tokens
    logits = outputs.logits

    return logits

model, tokenizer = load_model(model_name)
model = model.to(device)

hf_logits = get_logits(model, tokenizer, sentence, device)[:, -1, :]
```

## Get TransformerLens Logits

```python
model = HookedTransformer.from_pretrained_no_processing(model_name, device=device)
tokens = model.to_tokens(sentence, prepend_bos=False)
tl_logits = model(tokens)[:, -1, :]
```

## Compare Logit Distributions
Various metrics are used to compare the logit distributions of the two models. We don't yet have standard values for what constitutes a "good" logit comparison, so we are working on establishing benchmarks.

### Shape

```python
print(f"HF Logits Shape: {hf_logits.shape}")
print(f"TL Logits Shape: {tl_logits.shape}")
```

### Tensor Comparison

```python
are_close = torch.allclose(tl_logits, hf_logits, rtol=1e-5, atol=1e-3)
print(f"Are the logits close? {are_close}")
```

### Mean Squared Error

```python
# Compare the logits with MSE
mse = torch.nn.functional.mse_loss(hf_logits, tl_logits)
print(f"MSE: {mse}")
```

### Maximum Absolute Difference

```python
max_diff = torch.max(torch.abs(tl_logits - hf_logits))
print(f"Max Diff: {max_diff}")
```

### Cosine Similarity

```python
cosine_sim = F.cosine_similarity(tl_logits, hf_logits, dim=-1).mean()
print(f"Cosine Sim: {cosine_sim}")
```

### KL Divergence

```python
def kl_div(logits1: torch.Tensor, logits2: torch.Tensor) -> torch.Tensor:
    probs1 = F.softmax(logits1, dim=-1)
    probs2 = F.softmax(logits2, dim=-1)
    return F.kl_div(probs1.log(), probs2, reduction='batchmean')

kl_tl_hf = kl_div(tl_logits, hf_logits)
kl_hf_tl = kl_div(hf_logits, tl_logits)
print(f"KL(TL||HF): {kl_tl_hf}")
print(f"KL(HF||TL): {kl_hf_tl}")
```

```python

```

---

# stable_lm.ipynb


---

