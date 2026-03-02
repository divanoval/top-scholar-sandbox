# LoRA_tutorial.ipynb

# LoRA for Sentiment Analysis
📗 You can find an interactive Colab version of this tutorial [here](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/LoRA_tutorial.ipynb).

[Low Rank Adaptation (LoRA)](https://github.com/microsoft/LoRA) is a technique used to modify and fine tune large language models in a more efficient way. Rather than modifying all of the model weights, LoRAs find two low dimensional matrices that have the lowest rank. It then multiplies the two matrices to find the fine tuned weight matrix. This fine tuned weight matrix will be the same size as the original pre trained weight matrix. Once the fine tuned matrix has been found it can then be applied to the model's layers.

![TRAIN FIGURE](https://github.com/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/images/LoRA_tutorial_figure_1.png?raw=1)

<br>
<br>

Fine tuning with a LoRA is a part of the [Parameter Efficient Fine Tuning (PEFT)](https://github.com/huggingface/) family because it keeps the original model unchanged and introduces a small number of layers or parameters instead. Once the fine tuned matrix has been calculated, it is applied to the last Multilayer Perceptron (MLP) layer of the model. Once the LoRA has been applied, the model is fine tuned based on a knowledge base or domain specific dataset.

![TEST FIGURE](https://github.com/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/images/LoRA_tutorial_figure_2.png?raw=1)

# Setup

Make sure you have obtained your [NDIF API key](https://login.ndif.us/) and configured your workspace for [remote execution](https://nnsight.net/notebooks/features/remote_execution/).

The following packages need to be installed for this tutorial:
```
!pip install nnsight
!pip install pyarrow==15.0.2
!pip install datasets
!pip install datasets torch
```

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
    !pip install pyarrow==15.0.2
    !pip install datasets
    !pip install datasets torch
```

```python
from nnsight import CONFIG

if is_colab:
    # include your HuggingFace Token and NNsight API key on Colab secrets
    from google.colab import userdata
    NDIF_API = userdata.get('NDIF_API')
    HF_TOKEN = userdata.get('HF_TOKEN')

    CONFIG.set_default_api_key(NDIF_API)
```

Here are the imports needed for this tutorial.

```python
import torch
import torch.nn as nn
import pandas as pd
from nnsight import LanguageModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
```

# Prepare Data

For this tutorial we will be using the The Stanford Sentiment Treebank (SST2). It consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence as being either positive or negative. In the dataset, the positive/negative labels of each phrase are represented by a 0 for each negative statement and a 1 for each positive statement.

```python
# GLUE is a standard Natural Language Processing (NLP) benchmark which is commonly used for sentiment analysis tasks.
# It is responisble for assessing the effectiveness of language models across various NLP tasks.
# It serves as a standard for evaluating a model's ability to understand and process language.
dataset = load_dataset("glue", "sst2")

# 0 = neg, 1 = pos
def label_to_str(example):
    example['label'] = 'positive' if example['label'] == 1 else 'negative'
    return example

train_data = [(dataset['sentence'], 'positive' if dataset['label'] == 1 else 'negative') for dataset in dataset['train']]
validation_data = [(dataset['sentence'], 'positive' if dataset['label'] == 1 else 'negative') for dataset in dataset['validation']]
```

Next, we need to tokenize our data. Tokenizing involves converting text into a numerical representation. It is a popular technique in NLP because it helps the models better understand the text and output a more accurate result.

```python
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2', add_prefix_space=True)
tokenizer.pad_token = tokenizer.eos_token

# Uses the tokenizer from the model to tokenize a given sentence with padding and truncation
def tokenize_function(text):
  return tokenizer(text['sentence'], padding='max_length', truncation=True, max_length=10, return_tensors='pt')

# We use .map() in order to apply the tokenization function to all the training data.
tokenized_train_dataset = dataset['train'].map(tokenize_function, batched=True, batch_size=10)
tokenized_train_dataset = tokenized_train_dataset.map(lambda x: {'input_ids': x['input_ids'], 'attention_mask': x['attention_mask'], 'labels': x['label']})
```

# Prepare our Model

For this tutorial we will be using the [Llama-70B](https://huggingface.co/meta-llama/Llama-2-70b) language model.

```python
# Use the LanguageModel wrapper class to load in the Llama model
model_name = "meta-llama/Meta-Llama-3.1-70B"
model = LanguageModel(model_name, device_map='auto')
```

This is the model architechure before the LoRA has been applied. After the model has been fine tuned with the LoRA, the last MLP layer of the model will be replaced with the LoRA.

<br>
<br>

We’re going to train a very simple LORA that, when applied, will make our model determine whether a sentence is displaying a positive sentiment or a negative sentiment.

```python
from nnsight import Envoy

# We will define a LORA class.
# The LORA class call method operations are simply traced like you would normally do in a .trace.
class LORA(nn.Module):
    def __init__(self, module: Envoy, dim: int, r: int) -> None:
        """Init.

        Args:
            module (Envoy): Which model Module we are adding the LORA to.
            dim (int): Dimension of the layer we are adding to (This could potentially be auto populated if the user scanned first so we know the shape)
            r (int): Inner dimension of the LORA
        """
        super(LORA, self).__init__()
        self.r = r
        self.module = module
        self.WA = torch.nn.Parameter(torch.randn(dim, self.r), requires_grad=True).save()
        self.WB = torch.nn.Parameter(torch.zeros(self.r, dim), requires_grad=True).save()

    # The Call method defines how to actually apply the LORA.
    # happens after the forward pass
    def __call__(self, alpha: float = 1.0):
        """Call.

        Args:
            alpha (float, optional): How much to apply the LORA. Can be altered after training for inference. Defaults to 1.0.
        """

        # We apply WA to the first positional arg (the hidden states)
        A_x = torch.matmul(self.module.input, self.WA)
        BA_x = torch.matmul(A_x, self.WB)

        # LORA is additive
        h = BA_x + self.module.output

        # Replace the output with our new one * alpha
        # Could also have been self.module.output[:] = h * alpha, for in-place
        self.module.output = h * alpha

    def parameters(self):
        # Some way to get all the parameters.
        return [self.WA, self.WB]
```

# LLM Fine Tuning

```python
# Inner LORA dimension
lora_dim = 4

# Module to train LORA on
# Accesses the last mlp layer of the model
module = model.model.layers[-1].mlp
```

We can use the `.scan()` method to get the shape of the module without having to fully run the model.

```python
with model.scan(" "):
    dim = module.output.shape[-1]

print(dim)
```

```python
import nnsight
# The LORA object itself isn't transmitted to the server. Only the forward / call method.
# The parameters are created remotely and never sent only retrieved
with model.session(remote=True) as session:

    dataset = tokenized_train_dataset

    # Smaller chunks to run faster, feel free to increase
    indices = list(range(0, 5000))
    subset = Subset(dataset, indices)

    # Create a dataloader from it.
    dataloader = DataLoader(subset, batch_size=10)

    # Create our LORA on the last mlp and apply it to the model
    lora = LORA(module, dim, lora_dim)

    # Create an optimizer. Use the parameters from LORA
    optimizer = torch.optim.AdamW(lora.parameters(), lr=3)

    # Iterate over dataloader using .iter.
    with session.iter(dataloader) as batch:

        # Accesses the phrase that contains either a positive/negative sentiment
        prompt = batch['sentence']

        # Determines whether the phrase is positive/negative
        correct_token = batch['label']

        # Run .trace with prompt
        with model.trace(prompt) as tracer:

            # Apply LORA to intervention graph just by calling it with .trace
            # This is invoke the __call__() method of the LORA class defined above
            lora()

            # Get logits
            # Logits are the output of the neural network before the
            # activation function has been applied.
            logits = model.lm_head.output

            # Do cross entropy on last predicted token and correct_token
            loss = torch.nn.functional.cross_entropy(logits[:, -1], batch['label'])

            # Call backward
            loss.backward()

        # Call methods on optimizer. Graphs that arent from .trace (so in this case session and iterator both have their own graph) are executed sequentially.
        # The Graph of Iterator here will be:
        # 1.) Index batch at 0 for prompt
        # 2.) Index batch at 1 for correct_token
        # 3.) Execute the .trace using the prompt
        # 4.) Call .step() on optimizer
        optimizer.step()
        # 5.) Call .zero_grad() in optimizer
        optimizer.zero_grad()
        # 6.) Print out the lora WA weights to show they are indeed changing
        nnsight.log(lora.WA)

```

```python
print(model)
```

In addition to the weights changing, we know the LoRA has been applied because there is a difference in the model's architecture. The 11th block of the model no longer has the standard MLP layer and instead contains the LoRA.

Now it is time to test out whether our fine tuned model is able to predict the sentiment of a given sentence.

```python
# With lora. Will output "negative".
with model.generate("I'm upset", remote=True) as generator:
  lora()
  out = model.lm_head.output.save()

# The model outputs the sentiment as tokens first.
token_ids = out.argmax(dim=-1)

# Convert the tokens to either positive or negative
count_positive = (token_ids == 1).sum().item()
count_negative = (token_ids == 0).sum().item()

# Determine the overall sentiment of the entire sentence
if count_positive > count_negative:
  print("\nPrediction with LoRA: Positive\n")
else:
  print("\nPrediction with LoRA: Negative\n")

# Then without. It will try to complete the sentence rather than output the
# sentiment analysis.

with model.generate("I'm upset", remote=True) as generator:
    out = model.lm_head.output.save()

print("\nPrediction without LoRA:", model.tokenizer.decode(out.argmax(dim=-1)[0]))
```

---

# NNsight_Walkthrough.ipynb

<a href="https://colab.research.google.com/github/ndif-team/nnsight/blob/main/NNsight_Walkthrough.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<img src="https://nnsight.net/_static/images/nnsight_logo.svg" alt="drawing" width="200"/>

# **NNsight**

## The API for a transparent science on black-box AI

In this era of large-scale deep learning, the most interesting AI models are
massive black boxes that are hard to run. Ordinary commercial inference service
APIs let us interact with huge models, but they do not let us access model
internals.

The `nnsight` library is different: it provides full access to all the neural
network internals. When used together with a remote service like the
[National Deep Inference Fabric](https://thevisible.net/docs/NDIF-proposal.pdf)
(NDIF), it makes possible to run complex experiments on huge open  models easily,
with fully transparent access.

Our team wants to enable entire labs and independent researchers alike, as we
believe a large, passionate, and collaborative community will produce the next
big insights on a profoundly important field.

# 1 First, let's start small

## Setup

```python
# Install nnsight
!pip install nnsight
!pip install --upgrade transformers torch

from IPython.display import clear_output

clear_output()
```

## Tracing Context

To demonstrate the core functionality and syntax of nnsight, we'll define and
use a tiny two layer neural network.

Our little model here is composed of two submodules – linear layers 'layer1' and 'layer2'. We specify the sizes of each of these modules and create
some complementary example input.

```python
from collections import OrderedDict
import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)
```

The core object of the nnsight package is `NNsight`. This wraps around a given
PyTorch model to enable investigation of its internal parameters.

```python
import nnsight
from nnsight import NNsight

tiny_model = NNsight(net)
```

Printing a PyTorch model shows a named hierarchy of modules which is very useful
when accessing sub-components directly. NNsight reflect the same hierarchy and can be similarly printed.

```python
print(tiny_model)
```

Before we actually get to using the model we just created, let's talk about
Python contexts.

Python contexts define a scope using the `with` statement and are often used to
create some object, or initiate some logic, that you later want to destroy or
conclude.

The most common application is opening files as in the following example:

```python
with open('myfile.txt', 'r') as file:
  text = file.read()
```

Python uses the `with` keyword to enter a context-like object. This object
defines logic to be run at the start of the `with` block, as well as logic to be
run when exiting. When using `with` for a file, entering the context opens the
file and exiting the context closes it. Being within the context means we can
read from the file.

Simple enough! Now we can discuss how `nnsight` uses
contexts to enable intuitive access into the internals of a neural network.

The main tool with `nnsight` is a context for tracing.

We enter the tracing context by calling `model.trace(<input>)` on an `NNsight`
model, which defines how we want to run the model. Inside the context, we will
be able to customize how the neural network runs. The model is actually run upon
exiting the tracing context.

```python
# random input
input = torch.rand((1, input_size))

with tiny_model.trace(input) as tracer:
    pass
```

But where's the output? To get that, we'll have to learn how to request it from
within the tracing context.

## Getting

Earlier, when we wrapped our little neural net with the `NNsight` class. This
added a couple properties to each module in the model (including the root model
itself). The two most important ones are `.input` and `.output`.

```python
model.input
model.output
```

The names are self explanatory. They correspond to the inputs and outputs of
their respective modules during a forward pass of the model. We can use these
attributes inside the `with` block.

However, it is important to understand that the model is not executed until the
end of the tracing context. How can we access inputs and outputs before the
model is run? The trick is deferred execution.

`.input` and `.output` are Proxies for the eventual inputs and outputs of a
module. In other words, when we access `model.output` what we are
communicating to `nnsight` is, "When you compute the output of `model`, please
grab it for me and put the value into its corresponding Proxy object. Let's try it:

```python
with tiny_model.trace(input) as tracer:

    output = tiny_model.output

print(output)
```

Oh no an error! "Accessing value before it's been set."

Why doesn't our `output` have a `value`?

Proxy objects will only have their value at the end of a context if we call
`.save()` on them. This helps to reduce memory costs. Adding `.save()` fixes the
error:

```python
with tiny_model.trace(input) as tracer:

    output = tiny_model.output.save()

print(output)
```

Success! We now have the model output. We just completed out first
intervention using `nnsight`.

Each time we access a module's input or output, we create an _intervention_ in
the neural network's forward pass. Collectively these requests form the
_intervention graph_. We call the process of executing it alongside the model's
normal computation graph, _interleaving_.

<details>
<summary>On Model output</summary>

---

If we don't need to access anything other than the final model output, we can
call the tracing context with `trace=False` and not use it as a context. This could be especially useful for easy remote inference.

```python
  output = model.trace(<inputs>, trace=False)
```

---

</details>

Just like we saved the output of the model as a whole, we can save the output of
any of its submodules. We use normal Python attribute syntax. We can discover
how to access them by name by printing out the model:

```python
print(tiny_model)
```

Let's access the output of the first layer (which we've named 'layer1'):

```python
with tiny_model.trace(input) as tracer:

    l1_output = tiny_model.layer1.output.save()

print(l1_output)
```

Let's do the same for the input of layer2. While we're at it, let's also drop
the `as tracer`, as we won't be needing the tracer object itself for a few
sections:

```python
with tiny_model.trace(input):

    l2_input = tiny_model.layer2.input.save()

print(l2_input)
```

<details>
  <summary>On module inputs</summary>

---

Notice how the value for `l2_input`, is just a single tensor. By default, the `.input` attribute of a module will return the **first** tensor input to the module.

We can also access the full input to a module by using the `.inputs` attribute which will return the values in the form of:

      tuple(tuple(args), dictionary(kwargs))

Where the first index of the tuple is itself a tuple of all positional
arguments, and the second index is a dictionary of the keyword arguments.

---

</details>

Until now we were saving the output of the model and its submodules within the `Trace` context to then print it after exiting the context. We will continuing doing this in the rest of the tutorial since it's a good practice to save the computation results for later analysis.

However, we can also log the outputs of the model and its submodules within the `Trace` context. This is useful for debugging and understanding the model's behavior while saving memory. Let's see how to do this:

```python
with tiny_model.trace(input) as tracer:
  tracer.log("Layer 1 - out: ", tiny_model.layer1.output)
```

## Functions, Methods, and Operations

Now that we can access activations, we also want to do some post-processing on
it. Let's find out which dimension of layer1's output has the highest value.

We could do this by calling `torch.argmax(...)` after the tracing context or we
can just leverage the fact that `nnsight` handles Pytorch functions and methods within
the tracing context, by creating a Proxy request for it:

```python
with tiny_model.trace(input):

    # Note we don't need to call .save() on the output,
    # as we're only using its value within the tracing context.
    l1_output = tiny_model.layer1.output

    # We do need to save the argmax tensor however,
    # as we're using it outside the tracing context.
    l1_amax = torch.argmax(l1_output, dim=1).save()

print(l1_amax[0])
```

Nice! That worked seamlessly, but hold on, how come we didn't need to call
`.value[0]` on the result? In previous sections, we were just being explicit to
get an understanding of Proxies and their value. In practice, however, `nnsight`
knows that when outside of the tracing context we only care about the actual
value, and so printing, indexing, and applying functions all immediately return
and reflect the data in `.value`. So for the rest of the tutorial we won't use
it.

The same principles work for Pytorch methods and all operators as well:

```python
with tiny_model.trace(input):

    value = (tiny_model.layer1.output.sum() + tiny_model.layer2.output.sum()).save()

print(value)
```

The code block above is saying to `nnsight`, "Run the model with
the given `input`. When the output of `tiny_model.layer1` is computed, take its sum. Then do
the same for `tiny_model.layer2`. Now that both of those are computed, add them and make sure
not to delete this value as I wish to use it outside of the tracing context."

## Custom Functions

Everything within the tracing context operates on the intervention graph. Therefore, for `nnsight` to trace a  function it must also be a part of the intervention graph.

Out-of-the-box `nnsight` supports PyTorch functions and methods, all operators, as well the `einops` library. We don't need to do anything special to use them. But what do we do if we want to use custom functions? How do we add them to the intervention graph?

Enter `nnsight.apply()`. It allows us to add new functions to the intervention graph. Let's see how it works:

```python
# Take a tensor and return the sum of its elements
def tensor_sum(tensor):
    flat = tensor.flatten()
    total = 0
    for element in flat:
        total += element.item()

    return torch.tensor(total)

with tiny_model.trace(input) as tracer:

    # Specify the function name and its arguments (in a comma-separated form) to add to the intervention graph
    custom_sum = nnsight.apply(tensor_sum, tiny_model.layer1.output).save()
    sum = tiny_model.layer1.output.sum()
    sum.save()

print(custom_sum, sum)
```

`nnsight.apply()` executes the function it wraps and returns its output as a Proxy object. We can then use this Proxy object as we would any other.

The applications of `nnsight.apply` are wide: it can be used to wrap any custom function or functions from libraries that `nnsight` does not support out-of-the-box.

## Setting

Getting and analyzing the activations from various points in a model can be
really insightful, and a number of ML techniques do exactly that. However, often we not only want to view the computation of a model, but also to influence it.

To demonstrate the effect of editing the flow of information through the model,
let's set the first dimension of the first layer's output to 0. `NNsight` makes
this really easy using the '=' operator:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the 0th index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, 0] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

Seems our change was reflected. Now let's do the same for the last dimension:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the last index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, hidden_dims] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

Oh no, we are getting an error! Ah of course, we needed to index at `hidden_dims - 1` not `hidden_dims`.

If you've been using `nnsight`, you are probably familiar with error messages that can be quite difficult to troubleshoot. In `nnsight 0.4` we've now improved error messaging to be descriptive and line-specific, as you should see in the above example!

<details>

<summary>
Old NNsight error messaging
</summary>

If you've been using NNsight prior to the NNsight 0.4 release, you will be familiar with the following non-descriptive error messaging. If you choose to turn off NNsight 0.4's new error messaging feature, this is how errors within the tracing context will appear.

```
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in execute(self)
    379                 # Call the target to get value.
--> 380                 output = self.target(*args, **kwargs)
    381

IndexError: index 10 is out of bounds for dimension 1 with size 10

The above exception was the direct cause of the following exception:

IndexError                                Traceback (most recent call last)
20 frames
<ipython-input-16-5c81de91fb1f> in <cell line: 0>()
----> 1 with tiny_model.trace(input):
      2
      3     # Save the output before the edit to compare.
      4     # Notice we apply .clone() before saving as the setting operation is in-place.
      5     l1_output_before = tiny_model.layer1.output.clone().save()

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/Tracer.py in __exit__(self, exc_type, exc_val, exc_tb)
    100
    101
--> 102         super().__exit__(exc_type, exc_val, exc_tb)
    103
    104     def invoke(self, *inputs: Any, **kwargs) -> Invoker:

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/GraphBasedContext.py in __exit__(self, exc_type, exc_val, exc_tb)
    215             raise exc_val
    216
--> 217         self.backend(self)
    218
    219     ### BACKENDS ########

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/backends/LocalBackend.py in __call__(self, obj)
     25     def __call__(self, obj: LocalMixin):
     26
---> 27         obj.local_backend_execute()

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/Tracer.py in local_backend_execute(self)
    144         self.graph.execute()
    145
--> 146         self.model.interleave(
    147             self.model._execute,
    148             self.graph,

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in interleave(self, fn, intervention_graph, *inputs, **kwargs)
    467         module_paths = InterventionProtocol.get_interventions(intervention_graph).keys()
    468
--> 469         with HookHandler(
    470             self._model,
    471             list(module_paths),

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in __exit__(self, exc_type, exc_val, exc_tb)
    579
    580         if isinstance(exc_val, Exception):
--> 581             raise exc_val
    582
    583

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in interleave(self, fn, intervention_graph, *inputs, **kwargs)
    478         ):
    479             try:
--> 480                 fn(*inputs, **kwargs)
    481             except protocols.EarlyStopProtocol.EarlyStopException:
    482                 # TODO: Log.

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in _execute(self, *prepared_inputs, **kwargs)
    585             pass
    586
--> 587         return self._model(
    588             *prepared_inputs,
    589             **kwargs,

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1734             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735         else:
-> 1736             return self._call_impl(*args, **kwargs)
   1737
   1738     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1842
   1843         try:
-> 1844             return inner()
   1845         except Exception:
   1846             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1788                 args = bw_hook.setup_input_hook(args)
   1789
-> 1790             result = forward_call(*args, **kwargs)
   1791             if _global_forward_hooks or self._forward_hooks:
   1792                 for hook_id, hook in (

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1734             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735         else:
-> 1736             return self._call_impl(*args, **kwargs)
   1737
   1738     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1842
   1843         try:
-> 1844             return inner()
   1845         except Exception:
   1846             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1801                         hook_result = hook(self, args, kwargs, result)
   1802                     else:
-> 1803                         hook_result = hook(self, args, result)
   1804
   1805                     if hook_result is not None:

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in output_hook(module, input, output, module_path)
    564
    565                 def output_hook(module, input, output, module_path=module_path):
--> 566                     return self.output_hook(output, module_path)
    567
    568                 self.handles.append(

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in <lambda>(activations, module_path)
    473                 activations, module_path, "input", intervention_handler
    474             ),
--> 475             output_hook=lambda activations, module_path: InterventionProtocol.intervene(
    476                 activations, module_path, "output", intervention_handler
    477             ),

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in intervene(cls, activations, module_path, key, intervention_handler)
    454
    455                 # Value injection.
--> 456                 node.set_value(value)
    457
    458                 # Check if through the previous value injection, there was a 'swap' intervention.

/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in set_value(self, value)
    408
    409             if listener.fulfilled() and not self.graph.sequential:
--> 410                 listener.execute()
    411
    412         for dependency in self.arg_dependencies:

/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in execute(self)
    385         except Exception as e:
    386
--> 387             raise type(e)(
    388                 f"Above exception when execution Node: '{self.name}' in Graph: '{self.graph.id}'"
    389             ) from e

IndexError: Above exception when execution Node: 'setitem_0' in Graph: '132147685816016'

```

</details>

The error messaging feature can be toggled using `nnsight.CONFIG.APP.DEBUG` which defaults to true.

<details>

<summary>
Toggle Error Messaging
</summary>

Turn off debugging:
```
import nnsight

nnsight.CONFIG.APP.DEBUG = False
nnsight.CONFIG.save()
```

Turn on debugging:
```
import nnsight

nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()
```
</details>

Now that we know more about NNsight's error messaging, let's try our setting operation again with the correct indexing and view the shape of the output
before leaving the tracing context:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    print(f"Layer 1 output shape: {tiny_model.layer1.output.shape}")

    # Access the last index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, hidden_dims - 1] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

## Scan and Validate
Error codes are helpful, but sometimes you may want to quickly troubleshoot your code without actually running it.

Enter "Scanning" and "Validating"! We can enable this features by setting the `scan=True` and `validate=True` flag in the `trace` method.

"Scanning" runs "fake" inputs throught the model to collect information like shapes and types (i.e., scanning will populate all called `.inputs` and `.outputs`).

"Validating" attempts to execute the intervention proxies with "fake" inputs to check if they work (i.e., executes all interventions in your code with fake tensors).

"Validating" is dependent on "Scanning" to work correctly, so we need to run the scan of the model at least once to debug with validate. Let's try it out on our example above.

```python
# turn on scan and validate
with tiny_model.trace(input, scan=True, validate=True):

    l1_output_before = tiny_model.layer1.output.clone().save()

    # the error is happening here
    tiny_model.layer1.output[:, hidden_dims] = 0

    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

The operations are never executed using tensors with real values so it doesn't incur any memory costs. Then, when creating proxy requests like the setting one above, `nnsight` also attempts to execute the request on the "fake" values we recorded. Hence, it lets us know if our request is feasible before even running the model. [Here](https://nnsight.net/notebooks/features/scan_validate/) is a more detailed example of scan and validate in action!

<details>
<summary>A word of caution</summary>

---

Some pytorch operations and related libraries don't work well with fake tensors

If you are doing anything in a loop where efficiency is important, you should keep scanning and validating off. It's best to use them only when debugging or when you are unsure if your intervention will work.

---

</details>

We can also use the `.scan()` method to get the shape of a module without having to fully run the model. If scan  is enabled, our input is run though the model under its own "fake" context. This means the input makes its way through all of the model operations, allowing `nnsight` to record the shapes and data types of module inputs and outputs!

```python
with tiny_model.scan(input):

    dim = tiny_model.layer1.output.shape[-1]

print(dim)
```

We can also just replace proxy inputs and outputs with tensors of the same shape
and type. Let's use the shape information we have at our disposal to add noise
to the output, and replace it with this new noised tensor:

## Gradients

`NNsight` also lets us apply backpropagation and access gradients with respect to a
loss. Like `.input` and `.output` on modules, `nnsight` exposes `.grad` on
Proxies themselves (assuming they are proxies of tensors):

```python
with tiny_model.trace(input):

    # We need to explicitly have the tensor require grad
    # as the model we defined earlier turned off requiring grad.
    tiny_model.layer1.output.requires_grad = True

    # We call .grad on a tensor Proxy to communicate we want to store its gradient.
    # We need to call .save() since .grad is its own Proxy.
    layer1_output_grad = tiny_model.layer1.output.grad.save()
    layer2_output_grad = tiny_model.layer2.output.grad.save()

    # Need a loss to propagate through the later modules in order to have a grad.
    loss = tiny_model.output.sum()
    loss.backward()

print("Layer 1 output gradient:", layer1_output_grad)
print("Layer 2 output gradient:", layer2_output_grad)
```

All of the features we learned previously, also apply to `.grad`. In other
words, we can apply operations to and edit the gradients. Let's zero the grad of
`layer1` and double the grad of `layer2`.

```python
with tiny_model.trace(input):

    # We need to explicitly have the tensor require grad
    # as the model we defined earlier turned off requiring grad.
    tiny_model.layer1.output.requires_grad = True

    tiny_model.layer1.output.grad[:] = 0
    tiny_model.layer2.output.grad = tiny_model.layer2.output.grad * 2

    layer1_output_grad = tiny_model.layer1.output.grad.save()
    layer2_output_grad = tiny_model.layer2.output.grad.save()

    # Need a loss to propagate through the later modules in order to have a grad.
    loss = tiny_model.output.sum()
    loss.backward()

print("Layer 1 output gradient:", layer1_output_grad)
print("Layer 2 output gradient:", layer2_output_grad)
```

## Early Stopping

If we are only interested in a model's intermediate computations, we can halt a forward pass run at any module level, reducing runtime and conserving compute resources. One examples where this could be particularly useful would if we are working with SAEs - we can train an SAE on one layer and then stop the execution.

```python
with tiny_model.trace(input):
   l1_out = tiny_model.layer1.output.save()
   tiny_model.layer1.output.stop()

# get the output of the first layer and stop tracing
print("L1 - Output: ", l1_out)
```

Interventions within the tracing context do not necessarily execute in the order they are defined. Instead, their execution is tied to the module they are associated with.

As a result, if the forward pass is terminated early any interventions linked to modules beyond that point will be skipped, even if they were defined earlier in the context.

In the example below, the output of layer 2 _**cannot**_ be accessed since the model's execution was stopped at layer 1.

```python
with tiny_model.trace(input):
   l2_out = tiny_model.layer2.output.save()
   tiny_model.layer1.output.stop()

print("L2 - Output: ", l2_out)
```

## Conditional Interventions

Interventions can also be made conditional.

Inside the tracing context we can specify a new - conditional - context. This context will only execute the interventions within it if the condition is met.

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  with tracer.cond(rand_int % 2 == 0):
    tracer.log("Random Integer ", rand_int, " is Even")

  with tracer.cond(rand_int % 2 == 1):
    tracer.log("Random Integer ", rand_int, " is Odd")
```

Conditional contexts can also be nested, if we want our interventions to depend on more than one condition at a time.

```python
with tiny_model.trace(input) as tracer:

  non_rand_int = 8

  with tracer.cond(non_rand_int > 0):
    with tracer.cond(non_rand_int % 2 == 0):
      tracer.log("Rand Int ", non_rand_int, " is Positive and Even")
```

With `nnsight 0.4` we can now also use Python `if` statements within the tracing context to create a conditional context!

*Note: Colab behaves a little strangely with this feature the first time you run it - expect some lagging and warnings*

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")

  if rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

`elif` statements should also work as `if` statements within the tracing context:

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")
  elif rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

## Iterative Interventions

With the iterator context, you can now run an intervention loop at scale. It iteratively executes and updates a single intervention graph. Use a `.session()` to define the Iterator context and pass in a sequence of items that you want to loop over at each iteration

```python
with tiny_model.session() as session:

  li = nnsight.list() # an NNsight built-in list object
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

With `nnsight 0.4` we can now also use Python `for` loops within a tracer context at scale.

*NOTE: inline for loops (i.e., `[x for x in <Proxy object>`]) are not currently supported.*

```python
# New: Using Python for loops for iterative interventions
with tiny_model.session() as session:

    li = nnsight.list()
    [li.append([num]) for num in range(0, 3)]
    li2 = nnsight.list().save()

    # Using regular for loops
    for item in li:
        for item_2 in item: # for loops can be nested!
            li2.append(item_2)

print("\nList: ", li2)
```

# 2️ Bigger

Now that we have the basics of `nnsight` under our belt, we can scale our model
up and combine the techniques we've learned into more interesting experiments.

The `NNsight` class is very bare bones. It wraps a pre-defined model and does no
pre-processing on the inputs we enter. It's designed to be extended with more
complex and powerful types of models, and we're excited to see what can be done
to leverage its features!

However, if you'd like to load a Language Model from HuggingFace with its tokenizer, the`LanguageModel` subclass greatly simplifies this process.

## LanguageModel

`LanguageModel` is a subclass of `NNsight`. While we could define and create a
model to pass in directly, `LanguageModel` includes special support for
Huggingface language models, including automatically loading models from a
Huggingface ID, and loading the model together with the appropriate tokenizer.

Here is how we can use `LanguageModel` to load `GPT-2`:

```python
from nnsight import LanguageModel

llm = LanguageModel("openai-community/gpt2", device_map="auto")

print(llm)
```

<details>
<summary>On Model Initialization</summary>

---

A few important things to note:

Keyword arguments passed to the initialization of `LanguageModel` is forwarded
to HuggingFace specific loading logic. In this case, `device_map` specifies
which devices to use and its value `auto` indicates to evenly distribute it to
all available GPUs (and CPU if no GPUs available). Other arguments can be found
here:
https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM

When we initialize `LanguageModel`, we aren't yet loading the parameters of the
model into memory. We are actually loading a 'meta' version of the model which
doesn't take up any memory, but still allows us to view and trace actions on it.
After exiting the first tracing context, the model is then fully loaded into
memory. To load into memory on initialization, you can pass `dispatch=True` into
`LanguageModel` like
`LanguageModel('openai-community/gpt2', device_map="auto", dispatch=True)`.

---

</details>

Let's now apply some of the features that we used on the small model to `GPT-2`. Unlike `NNsight`, `LanguageModel` does define logic to pre-process
inputs upon entering the tracing context. This makes interacting with the model
simpler (i.e., you can send prompts to the model without having to directly access the tokenizer).

In the following example, we ablate the value coming from the last layer's MLP
module and decode the logits to see what token the model predicts without
influence from that particular module:

```python
with llm.trace("The Eiffel Tower is in the city of"):

    # Access the last layer using h[-1] as it's a ModuleList
    # Access the first index of .output as that's where the hidden states are.
    llm.transformer.h[-1].mlp.output[0][:] = 0

    # Logits come out of model.lm_head and we apply argmax to get the predicted token ids.
    token_ids = llm.lm_head.output.argmax(dim=-1).save()

print("\nToken IDs:", token_ids)

# Apply the tokenizer to decode the ids into words after the tracing context.
print("Prediction:", llm.tokenizer.decode(token_ids[0][-1]))
```

We just ran a little intervention on a much more complex model with many more
parameters! However, we're missing an important piece of information: what the
prediction would have looked like without our ablation.

We could just run two tracing contexts and compare the outputs. However, this would require two forward passes through the model. `NNsight` can do
better than that with batching.

<a name="batching-id"></a>

## Batching

Batching is a way to process multiple inputs in one forward pass. To better understand how batching works, we're going to bring back the `Tracer` object that we dropped before.

When we call `.trace(...)`, it's actually creating two different contexts behind the scenes. The first one is the tracing context that we've discussed previously, and the second one is the invoker context. The invoker context defines the values of the `.input` and `.output` Proxies.

If we call `.trace(...)` with some input, the input is passed on to the invoker. As there is only one input, only one invoker context is created.

If we call `.trace()` without an input, then we can call `tracer.invoke(input1)` to manually create the invoker context with an input, `input1`. We can also repeatedly call `tracer.invoke(...)` to create the invoker context for additional inputs. Every subsequent time we call
`.invoke(...)`, interventions within its context will only refer to the input in that particular invoke statement.

When exiting the tracing context, the inputs from all of the invokers will be batched together, and they will be executed in one forward pass! To test this out, let's do the same ablation experiment, but also add a 'control' output for comparison:

<details>
<summary>More on the invoker context</summary>

---

Note that when injecting data to only the relevant invoker interventions, `nnsight` tries, but can't guarantee, to narrow the data into the right
batch indices. Thus, there are cases
where all invokes will get all of the data. Specifically, if the input or output data is stored
as an object that is not an arbitrary collection of tensors, it will be broadcasted to all invokes.

Just like `.trace(...)` created a `Tracer` object, `.invoke(...)` creates an `Invoker` object. For `LanguageModel` models, the `Invoker` prepares the input by running a tokenizer on it.
`Invoker` stores pre-processed inputs at `invoker.inputs`, which can be accessed to see information about our inputs.
In a case where we pass a single input to `.trace(...)` directly, we can still access the invoker
object at `tracer.invoker` without having to call `tracer.invoke(...)`.

Keyword arguments given to `.invoke(..)` make their way to the input pre-processing.
`LanguageModel` has keyword arguments `max_length` and `truncation` used for tokenization which can be
passed to the invoker. If we want to pass keyword arguments to the invoker for a single-input `.trace(...)`, we can pass `invoker_args` as a dictionary of invoker keyword arguments.

Here is an example to demonstrate everything we've described:

**This snippet**

```
with llm.trace("hello", invoker_args={"max_length":10}) as tracer:
  invoker = tracer.invoker

```
  **does the same as**

```
with llm.trace() as tracer:
  with tracer.invoke("hello", max_length=10) as invoker:
    invoker = invoker
```

---

</details>

```python
with llm.trace() as tracer:

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Ablate the last MLP for only this batch.
        llm.transformer.h[-1].mlp.output[0][:] = 0

        # Get the output for only the intervened on batch.
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Get the output for only the original batch.
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("Original token IDs:", token_ids_original)
print("Modified token IDs:", token_ids_intervention)

print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))
```

Based on our control results, our ablation did end up affecting what the model predicted. That's pretty neat!

Another cool thing with multiple invokes is that Proxies can interact between them.

Here, we transfer the token embeddings from a real prompt into another placeholder prompt. Therefore the latter prompt produces the output of the former prompt:

```python
with llm.trace() as tracer:

    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = llm.transformer.wte.output

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        llm.transformer.wte.output = embeddings
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
      token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("original prediction shape", token_ids_original[0][-1].shape)
print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))

print("modified prediction shape", token_ids_intervention[0][-1].shape)
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))
```

## Multiple Token Generation

### .next()

Some HuggingFace models define methods to generate multiple outputs at a time.
`LanguageModel` wraps that functionality to provide the same tracing features by
using `.generate(...)` instead of `.trace(...)`. This calls the underlying
model's `.generate` method. It passes the output through a `.generator`
module that we've added onto the model, allowing us to get the generate output
at `.generator.output`.

In a case like this, the underlying model is called more than once; the modules
of said model produce more than one output. Which iteration should a given
`module.output` refer to? That's where `Module.next()` comes in!

Each module has a call index associated with it and `.next()` simply increments
that attribute. At the time of execution, data is injected into the intervention
graph only at the iteration that matches the call index.

```python
with llm.generate('The Eiffel Tower is in the city of', max_new_tokens=3) as tracer:

    hidden_states1 = llm.transformer.h[-1].output[0].save()

    # use module.next() to access the next intervention
    hidden_states2 = llm.transformer.h[-1].next().output[0].save()

    # saving the output allows you to save the hidden state across the initial prompt
    out = llm.generator.output.save()

print(hidden_states1.shape)
print(hidden_states2.shape)
print(out)
```

### New! using .all()

With `nnsight 0.4` you can now use `.all()` to recursively apply interventions to a model. Calling `.all()` on a module within a model will recursively apply its `.input` and `.output` across all iterations. Previously, we'd need to loop across each new generated token, saving the intervention for every generated token and calling `.next()` to move forward.

```python
# Old approach:
prompt = 'The Eiffel Tower is in the city of'
layers = llm.transformer.h
n_new_tokens = 3
hidden_states = []
with llm.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    for i in range(n_new_tokens):
        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output.save())

        # Move to next generated token
        layers[0].next()

print("Hidden state length: ",len(hidden_states))
```

We can use also `.all()` to streamline the multiple token generation process. We simply call `.all` on the module where we are applying the intervention (in this case GPT-2's layers), apply our intervention, and append our hidden states (stored in an `nnsight.list()` object).
<br> <br>

Let's test this out for the multiple token generation case:

```python
# using .all():
prompt = 'The Eiffel Tower is in the city of'
layers = llm.transformer.h
n_new_tokens = 3
with llm.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() to apply intervention to each new token
    layers.all()

    # Apply intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(layers[-1].output) # no need to call .save
    # Don't need to loop or call .next()!

print("Hidden state length: ",len(hidden_states))
```

Easy! Note that because `.all()` is recursive, it will only work to append outputs called on children of the module that `.all()` was called on. See example below for more information. TL;DR: apply `.all()` on the highest-level accessed module if interventions and outputs have different hierarchies within model structure.

<details>
<summary>Recursive properties of .all()</summary>

`.all()` recursively acts on model components. In the below code example, only the first token generation is saved, because `.all()` applied to `layers`, while the saved variable `hidden_states` is produced from `model.lm_head`, which is not a child of `layers`.

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on layers
    layers.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save, it's already initialized

print("Hidden state length: ",len(hidden_states)) # length is 1, meaning it only saved the first token generation
```

If you want to apply an intervention during multiple token generation while saving the state of a model component that isn't a child of that module, you can instead apply `.all()` to the full model:

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on model
    model.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save

print("Hidden state length: ",len(hidden_states)) # length is 3, as expected!
```

</details>

## Model Editing

NNsight's model editing feature allows you to create persistently modified versions of a model with a use of `.edit()`. Unlike interventions in a tracing context, which are temporary, the **Editor** context enables you to make lasting changes to a model instance.

This feature is useful for:
* Creating modified model variants without altering the original
* Applying changes that persist across multiple forward passes
* Comparing interventions between original and edited models

Let's explore how to use the **Editor** context to make a simple persistent change to a model:

```python
# we take the hidden states with the expected output "Paris"
with llm.trace("The Eiffel Tower is located in the city of") as tracer:
    hs11 = llm.transformer.h[11].output[0][:, -1, :].save()

# the edited model will now always predict "Paris" as the next token
with llm.edit() as llm_edited:
    llm.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with llm.trace("Vatican is located in the city of") as tracer:
    original_tokens = llm.lm_head.output.argmax(dim=-1).save()

# ...with the output of the edited model
with llm_edited.trace("Vatican is located in the city of") as tracer:
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("\nOriginal Prediction: ", llm.tokenizer.decode(original_tokens[0][-1]))
print("Modified Prediction: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

Edits defined within an **Editor** context create a new, modified version of the model by default, preserving the original. This allows for safe experimentation with model changes. If you wish to modify the original model directly, you can set `inplace=True` when calling `.edit()`.

Use this option cautiously, as in-place edits alter the base model for all the consequent model calls.

```python
# we use the hidden state we saved above (hs11)
with llm.edit(inplace=True) as llm_edited:
    llm.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with llm.trace("Vatican is located in the city of") as tracer:
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("Modified In-place: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

If you've made in-place edits to your model and need to revert these changes, you can apply `.clear_edits()`. This method removes all edits applied to the model, effectively restoring it to its original state.

```python
llm.clear_edits()

with llm.trace("Vatican is located in the city of"):
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("Edits cleared: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

# 3 I thought you said huge models?

`NNsight` is only one part of our project to democratize access to AI internals. The other half is the National Deep Inference Fabric, or `NDIF`. `NDIF` hosts large models for shared access using `NNsight`, so you don't have to worry about any of the headaches of hosting large models yourself!

The interaction between `NDIF` and `NNsight` is fairly straightforward. The
**intervention graph** we create via the tracing context can be encoded into a
custom json format and sent via an http request to the `NDIF` servers. `NDIF`
then decodes the **intervention graph** and **interleaves** it alongside the
specified model.

To see which models are currently being hosted, check out the following status
page: https://nnsight.net/status/

## Remote execution

In its current state, `NDIF` requires you to receive an API key. Therefore, to
run the rest of this walkthrough, you need one of your own. To get one, simply
register at https://login.ndif.us.

With a valid API key, you then can configure `nnsight` as follows:

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")
```

If you're running in a local IDE, this only needs to be run once as it will save the API key as the default in a
.config file along with your `nnsight` installation. You can also add your API key to Google Colab secrets.

To amp things up a few levels, let's demonstrate using `nnsight`'s tracing
context with `Llama-3.1-8b`!

```python
import os

# Llama 3.1 8b is a gated model, so you need to apply for access on HuggingFace and include your token.
os.environ['HF_TOKEN'] = "YOUR_HUGGING_FACE_TOKEN"
```

```python
from nnsight import LanguageModel

# We'll never actually load the parameters locally, so no need to specify a device_map.

llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:

    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])
```

It really is as simple as `remote=True`. All of the techniques we went through
in earlier sections work just the same when running locally or remotely.

## Sessions

NDIF uses a queue to handle concurrent requests from multiple users. To optimize the execution of our experiments we can use the `session` context to efficiently package multiple interventions together as one single request to the server.

This offers the following benefits:
1.   All interventions within a session will be executed one after another without additional wait in the NDIF queue
2.   All intermediate outputs for each intervention are stored on the server and can be accessed by other interventions in the same session without moving the data back and forth between NDIF and the local machine

Let's take a look:

```python
with llama.session(remote=True) as session:

  with llama.trace("The Eiffel Tower is in the city of") as t1:
    # capture the hidden state from layer 32 at the last token
    hs_31 = llama.model.layers[31].output[0][:, -1, :] # no .save()
    t1_tokens_out = llama.lm_head.output.argmax(dim=-1).save()

  with llama.trace("Buckingham Palace is in the city of") as t2:
    llama.model.layers[1].output[0][:, -1, :] = hs_31[:]
    t2_tokens_out = llama.lm_head.output.argmax(dim=-1).save()

print("\nT1 - Original Prediction: ", llama.tokenizer.decode(t1_tokens_out[0][-1]))
print("T2 - Modified Prediction: ", llama.tokenizer.decode(t2_tokens_out[0][-1]))
```

In the example above, we are interested in replacing the hidden state of a later layer with an earlier one. Since we are using a `session`, we don't have to save the hidden state from Tracer 1 to reference it in Tracer 2.

It is important to note that all the traces defined within the `session` context are executed sequentially, strictly following the order of definition (i.e. `t2` being executed after `t1` and `t3` after `t2` etc.).

The `session` context object has its own methods to log values and be terminated early.

```python
with llama.session(remote=True) as session:
  session.log("-- Early Stop --")
  nnsight.stop
```

In addition to the benefits mentioned above, the `session` context also enables interesting experiments not possible with other `nnsight` tools — since every trace is run on its own model, it means that within one session we can run interventions between different models — for example, we could swap activations between base and instruct versions of the Llama model and compare their outputs. And `session` can also be used to run similar experiments entirely locally!

## Streaming

Streaming enables users apply functions and datasets locally during remote model execution. This allows users to stream results for immediate consumption (i.e., seeing tokens as they are generated) or applying non-whitelisted functions such as model tokenizers, large local datasets, and more!

*   `nnsight.local()` context sends values immediately to user's local machine from server
*   Intervention graph is executed locally on downstream nodes
*   Exiting local context uploads data back to server
*   `@nnsight.trace` function decorator enables custom functions to be added to intervention graph when using `nnsight.local()`

### `nnsight.local()`

You may sometimes want to locally access and manipulate values during remote execution. Using `.local()` on a proxy, you can send remote content to your local machine and apply local functions. The intervention graph is then executed locally on downstream nodes (until you send execution back to the remote server by exiting the `.local()` context).

There are a few use cases for streaming with `.local()`, including live chat generation and applying large datasets or non-whitelisted local functions to the intervention graph.

Now let's explore how streaming works. We'll start by grabbing some hidden states of the model and printing their value using `tracer.log()`. Without calling `nnsight.local()`, these operations will all occur remotely.

```python
# This will give you a remote LOG response because it's coming from the remote server
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

Now, let's try the same operation using the `nnsight.local()` context. This will send the operations to get and print the hidden states to your local machine, changing how the logging message is formatted (local formatting instead of remote).

```python
# This will print locally because it's already local
with llama.trace("hello", remote=True) as tracer:

    with nnsight.local():
        hs = llama.model.layers[-1].output[0]
        tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

### `@nnsight.trace` function decorator

We can also use function decorators to create custom functions to be used during `.local` calls. This is a handy way to enable live streaming of a chat or to train probing classifiers on model hidden states.

Let's try out `@nnsight.trace` and `nnsight.local()` to access a custom function during remote execution.

```python
# first, let's define our function
@nnsight.trace # decorator that enables this function to be added to the intervention graph
def my_local_fn(value):
    return value * 0

# We use a local function to ablate some hidden states
# This downloads the data for the .local context, and then uploads it back to set the value.
with llama.generate("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    with nnsight.local():

        hs = my_local_fn(hs)

    llama.model.layers[-1].output[0][:] = hs

    out =  llama.lm_head.output.save()
```

Note that without calling `.local`, the remote API does not know about `my_local_fn` and will throw a whitelist error. A whitelist error occurs because you are being allowed access to the function.

```python
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    hs = my_local_fn(hs) # no .local - will cause an error

    llama.model.layers[-1].output[0][:] = hs * 2

    out =  llama.lm_head.output.save()

print(out)
```

### Example: Live-streaming remote chat

Now that we can access data within the tracing context on our local computer, we can apply non-whitelisted functions, such as the model's tokenizer, within our tracing context.

Let's build a decoding function that will decode tokens into words and print the result.

```python
@nnsight.trace
def my_decoding_function(tokens, model, max_length=80, state=None):
    # Initialize state if not provided
    if state is None:
        state = {'current_line': '', 'current_line_length': 0}

    token = tokens[-1] # only use last token

    # Decode the token
    decoded_token = llama.tokenizer.decode(token).encode("unicode_escape").decode()

    if decoded_token == '\\n':  # Handle explicit newline tokens
        # Print the current line and reset state
        print('',flush=True)
        state['current_line'] = ''
        state['current_line_length'] = 0
    else:
        # Check if adding the token would exceed the max length
        if state['current_line_length'] + len(decoded_token) > max_length:
            print('',flush=True)
            state['current_line'] = decoded_token  # Start a new line with the current token
            state['current_line_length'] = len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line
        else:
            # Add a space if the line isn't empty and append the token
            if state['current_line']:
                state['current_line'] += decoded_token
            else:
                state['current_line'] = decoded_token
            state['current_line_length'] += len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line

    return state
```

Now we can decode and print our model outputs throughout token generation by accessing our decoding function through `nnsight.local()`.

```python
import torch

nnsight.CONFIG.APP.REMOTE_LOGGING = False

prompt = "A press release is an official statement delivered to members of the news media for the purpose of"
# prompt = "Your favorite board game is"

print("Prompt: ",prompt,'\n', end ="")

# Initialize the state for decoding
state = {'current_line': '', 'current_line_length': 0}

with llama.generate(prompt, remote=True, max_new_tokens = 50) as generator:
    # Call .all() to apply to each new token
    llama.all()

    all_tokens = nnsight.list().save()

    # Access model output
    out = llama.lm_head.output.save()

    # Apply softmax to obtain probabilities and save the result
    probs = torch.nn.functional.softmax(out, dim=-1)
    max_probs = torch.max(probs, dim=-1)
    tokens = max_probs.indices.cpu().tolist()
    all_tokens.append(tokens[0]).save()

    with nnsight.local():
        state = my_decoding_function(tokens[0], llama, max_length=20, state=state)
```

## Looping across sessions

We mention earlier that the `session` context enables multi-tracing execution. But how can we optimize a process that would require running an intervention graph in a loop? If we create a simple `for` loop with a **Tracer context** inside, this will result in creating a new intervention graph at each iteration, which is not scalable.

We solve this problem the `nnsight` way via the **Iterator context**: an intervention loop that iteratively executes and updates a single intervention graph.

Use a `session` to define the **Iterator context** and pass in a sequence of items that you want to loop over at each iteration:

```python
with llama.session(remote=True) as session:

  with session.iter([0, 1, 2]) as item:
    # define intervention body here ...

    with llama.trace("_"):
      # define interventions here ...
      pass

    with llama.trace("_"):
      # define interventions here ...
      pass
```

The `Iterator` context extends all the `nnsight` graph-based functionalities, but also closely mimics the conventional `for` loop statement in Python, which allows it to support all kind of iterative operations with a use of `as item` syntax:

```python
with llama.session(remote=True) as session:

  li = nnsight.list()
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

Notice how we used the `nnsight.list()` method to create a list of lists to loop over. This type of method is what we call an **NNsight Built-in**. It is a special type of methods that serve as a wrapper around `nnsight.apply()` to provide a more user-friendly interface for adding common datatypes to the Intervention Graph.

<details>
<summary>A full list of NNsight Built-ins</summary>

`nnsight.bool()` creates a traceable Boolean

`nnsight.bytes()` creates a traceable Bytes

`nnsight.int()` creates a traceable Integer

`nnsight.float()` creates a traceable Float

`nnsight.str()` creates a traceable String

`nnsight.comples()` creates a traceable Complex number

`nnsight.bytearray()` creates a traceable Bytearray

`nnsight.tuple()` creates a traceable Tuple

`nnsight.list()` creates a traceable List

`nnsight.set()` creates a traceable Set

`nnsight.dict()` creates a traceable Dictionary

</details>

We can also expose the `iterator` context object via a `return_context` flag. You can then use it to `exit` out of the Iteration loop early and log the intermediate outputs within the loop:

```python
with llama.session(remote=True) as session:

  # with session.iter([0, 1, 2, 3], return_context=True) as (item, iterator):
  with session.iter([0, 1, 2, 3]) as item:

      nnsight.log(item)

      with nnsight.cond(item == 2):
        nnsight.stop()
```

The **Iterator** context is a niece piece of functionality that allows you to define a bunch of basic code operations that can now be "traceable" by `nnsight`.

But in what kind of experimental scenario would someone need iterators?

In the next section, we delve into a powerful use case of the `Iterator` context and see how it enables it!

## Training a LoRA

Here is an example of a task that uses everything we have covered in the last section - remote execution, **Session** context and iterative interventions. Using session and iterator contexts, we're going apply a very simple fine-tuning approach called low-rank adaptation (LoRA).

Let's try training a LoRA that, when applied, makes our model always predict "Paris" no matter what.

```python
import torch
import torch.nn as nn
import nnsight
# from nnsight.envoy import Envoy # this moved in 0.4
from nnsight import Envoy

# We will define a LORA class.
# The LORA class call method operations are simply traced like you would normally do in a .trace.
class LORA(nn.Module):
    def __init__(self, module: Envoy, dim: int, r: int) -> None:
        """Init.

        Args:
            module (Envoy): Which model Module we are adding the LORA to.
            dim (int): Dimension of the layer we are adding to (This could potentially be auto populated if the user scanned first so we know the shape)
            r (int): Inner dimension of the LORA
        """
        super(LORA, self).__init__()
        self.r = r
        self.module = module
        self.WA = torch.nn.Parameter(torch.randn(dim, self.r), requires_grad=True).save()
        self.WB = torch.nn.Parameter(torch.zeros(self.r, dim), requires_grad=True).save()

    # The Call method defines how to actually apply the LORA.
    def __call__(self, alpha: float = 1.0):
        """Call.

        Args:
            alpha (float, optional): How much to apply the LORA. Can be altered after training for inference. Defaults to 1.0.
        """

        # We apply WA to the first positional arg (the hidden states)
        A_x = torch.matmul(self.module.input[0][0], self.WA)
        BA_x = torch.matmul(A_x, self.WB)

        # LORA is additive
        h = BA_x + self.module.output

        # Replace the output with our new one * alpha
        # Could also have been self.module.output[:] = h * alpha, for in-place
        self.module.output = h * alpha

    def parameters(self):
        # Some way to get all the parameters.
        return [self.WA, self.WB]
```

Let's define all the variables to use in LoRA training.

```python
# We need the token id of the correct answer.
answer = " Paris"
answer_token = llama.tokenizer.encode(answer)[1]
# Inner LORA dimension
lora_dim = 4
# Module to train LORA on
module = llama.model.layers[-1].mlp
```

We can use the `.scan()` method to get the shape of the module without having to fully run the model.

```python
with llama.scan(" "):
    dim = module.output.shape[-1]

print(dim)
```

It's time to run the LORA training loop! We using the **Session** and the **Iterator** contexts to achieve this.

```python
from torch.utils.data import DataLoader

# The LORA object itself isn't transmitted to the server. Only the forward / call method.
# The parameters are created remotely and never sent only retrieved
with llama.session(remote=True) as session:

    # Create dataset of 100 pairs of a blank prompt and the " Paris " id
    dataset = [["_", answer_token]] * 100

    # Create a dataloader from it.
    dataloader = DataLoader(dataset, batch_size=10)

    # Create our LORA on the last mlp
    lora = LORA(module, dim, lora_dim)

    # Create an optimizer. Use the parameters from LORA
    optimizer = torch.optim.AdamW(lora.parameters(), lr=3)

    # Iterate over dataloader using .iter.
    with session.iter(dataloader) as batch:

        prompt = batch[0]
        correct_token = batch[1]

        # Run .trace with prompt
        with llama.trace(prompt) as tracer:

            # Apply LORA to intervention graph just by calling it with .trace
            lora()

            # Get logits
            logits = llama.lm_head.output

            # Do cross entropy on last predicted token and correct_token
            loss = torch.nn.functional.cross_entropy(logits[:, -1], batch[1])
            # Call backward
            loss.backward()

        # Call methods on optimizer. Graphs that arent from .trace (so in this case session and iterator both have their own graph) are executed sequentially.
        # The Graph of Iterator here will be:
        # 1.) Index batch at 0 for prompt
        # 2.) Index batch at 1 for correct_token
        # 3.) Execute the .trace using the prompt
        # 4.) Call .step() on optimizer
        optimizer.step()
        # 5.) Call .zero_grad() in optimizer
        optimizer.zero_grad()
        # 6.) Print out the lora WA weights to show they are indeed changing
        nnsight.log(lora.WA)

```

Now `WA` and `WB` are optimized! So we generate with the LoRA just by calling `lora()` in the `.generate` and save the output to then de-tokenize it.

```python
# With lora. Should produce "Hello Paris"
with llama.generate("Hello", remote=True) as generator:

    lora()

    out = llama.generator.output.save()

print(llama.tokenizer.batch_decode(out.value))

# Then without. Should produce "Hello,"
with llama.generate("Hello", remote=True) as generator:

    out = llama.generator.output.save()

print(llama.tokenizer.batch_decode(out.value))

```

# Next Steps
Check out [nnsight.net/tutorials](https://nnsight.net/tutorials) for more walkthroughs implementating classic interpretability techniques using `nnsight`.

# Getting Involved!

Note that both `nnsight` and `NDIF` are in active development, so changes may be made and errors may arise during use. If you’re interested in following updates to `nnsight`, contributing, giving feedback, or finding collaborators, please join the [NDIF discord](https://discord.gg/6uFJmCSwW7). We’d love to hear about your work using nnsight!

You can also follow us on [LinkedIn](https://www.linkedin.com/company/national-deep-inference-fabric/), Bluesky: [@ndif-team.bsky.social](https://bsky.app/profile/ndif-team.bsky.social), and X: [@ndif_team](https://x.com/ndif_team).

💟


---

# NNsight_v0_3_guide.ipynb

# NNsight 0.3 - User Guide

## Set up

```python
from IPython.display import clear_output

!pip install nnsight
!pip install --upgrade transformers torch

clear_output()
```

```python
from google.colab import userdata
from nnsight import CONFIG

from nnsight.logger import remote_logger
remote_logger.propagate = False

CONFIG.set_default_api_key('422220a9817141e49c5add1868af07a5')
```

```python
from collections import OrderedDict
from nnsight import NNsight
import torch

input_size = 5
hidden_dims = 10
output_size = 2

torch.manual_seed(423)

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, hidden_dims))
        ]
    )
).requires_grad_(False)

input = torch.rand((1, input_size))
input_2 = torch.rand((1, input_size))

tiny_model = NNsight(net)
```

```python
from nnsight import LanguageModel

lm = LanguageModel("openai-community/gpt2", dispatch=True)
llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
```

# Breaking Changes

## input/inputs

Module input access has a syntactic change:

- Old: `nnsight.Envoy.input`

- New: `nnsight.Envoy.inputs`

- Note: `nnsight.Envoy.input` now provides access to the first positional argument of the module's input.

```python
with lm.trace("Hello World"):
  l2_ins = lm.transformer.h[2].inputs.save()
  l2_in = lm.transformer.h[2].input.save()

print("Inputs: ", l2_ins)
print("First Positional Argument Input: ", l2_in)
```

## `scan` and `validate`

`scan` and `validate` are now set to `False` by default in the `Tracer` context.

# New Features

### Scanning

You can scan a model without executing it to gather important insights. This is useful for looking at internal modules' shapes for example. You can pass a dummy input to the model, and it will not be executed. This is also means that you don't have to call `save()` on any variable.

```python
with tiny_model.scan(torch.tensor([0, 0, 0, 0, 0])):
  dim = tiny_model.layer2.input.shape

print(dim)
```

## nnsight builtins

You can now define multiple `Python` builtins to be traceable by the Intervention graph.

Simply use the `nnsight` import to call constructors for these data structures.

```python
import nnsight

with tiny_model.trace(input):
  num = nnsight.int(5).save()
  l = nnsight.list().save()
  l.append(num)
  d = nnsight.dict({"five": num}).save()

print("Interger: ", num)
print("List: ", l)
print("Dictionary: ", d)
```

Here is the complete list of supported `Python` builtins:
- bool
- bytes
- int
- float
- str
- complex
- bytearray
- tuple
- list
- set
- dict

## Proxy Update

For literals created and traced by `nnsight`, there is no direct way of setting their values.

Use our `.update()` method on Intervention Proxies to assign it a new value.

```python
import nnsight

with tiny_model.trace(input):
  input_str = nnsight.str("I am a ").save()
  input_str.update(input_str + "Transformer")

print("Input: ", input_str)
```

This is also useful for calculating running sums and other statistics.

## Logging

We are probably all guilty, at least once, of trying to print an Intervention Proxy from within the tracing context to look at its value:

```python
with tiny_model.trace(input):
    print(tiny_model.layer1.output)
```

The reason this does not print any actual value is because the model is only executed upon exiting the `Tracer` context, and thus, the proxies' values have not been populated yet.

If you are still only interested in looking at some intermediate values without necessary saving them, you can call our logging feature which will be executed as an `nnsight` node during the model's execution and show you the actual values.

```python
import nnsight

with tiny_model.trace(input) as tracer:
  nnsight.log("Layer 1 - out: ", tiny_model.layer1.output)
```

## Tracing function calls

Everything within the tracing context operates on the intervention graph. Therefore for `nnsight` to trace a function it must also be a part of the intervention graph.

Out-of-the-box `nnsight` supports `Pytorch` functions and methods, all operators, as well the `einops` library. We don’t need to do anything special to use them.

For custom functions we can use `nnsight.apply()` to add them to the intervention graph.

```python
import nnsight
import torch

# We define a simple custom function that sums all the elements of a tensor
def tensor_sum(tensor):
    flat = tensor.flatten()
    total = 0
    for element in flat:
        total += element.item()

    return torch.tensor(total)

with lm.trace("The Eiffel Tower is in the city of") as tracer:

    # Specify the function name and its arguments (in a coma-separated form) to add to the intervention graph
    custom_sum = nnsight.apply(tensor_sum, lm.transformer.h[0].output[0]).save()
    sum = lm.transformer.h[0].output[0].sum().save()

print("PyTorch sum: ", sum)
print("Our sum: ", custom_sum)
```

## Early Stopping

If you are only interested in a model's intermediate computations, you can halt a forward pass run at any module level, reducing runtime and conserving computational resources. This is particularly useful if you are working with SAEs.

```python
with tiny_model.trace(input):
   l1_out = tiny_model.layer1.output.save()
   tiny_model.layer1.output.stop()

print("L1 - Output: ", l1_out)
```

Interventions within the `Tracer` context do not necessarily execute in the order they are defined. Instead, their execution is tied to the module they are associated with.

As a result, if the forward pass is terminated early any interventions linked to modules beyond that point will be skipped, even if they were defined earlier in the context.

In the example below, the output of layer 2 **CANNOT** be accessed since the model's execution was stopped at layer 1.

```python
with tiny_model.trace(input):
   l2_out = tiny_model.layer2.output.save()
   tiny_model.layer1.output.stop()

print("L2 - Output: ", l2_out)
```

## Conditional Interventions

You can make interventions conditional!

Create a Conditional context and pass it a value to be evaluated as a boolean. The context will wrap all the interventions that you wish to be dependent on the condition specified.

Let's take a look at how you can do that:

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,)).item()

  with tracer.cond(rand_int % 2 == 0):
    tracer.apply(print, "Random Integer ", rand_int, " is Even")

  with tracer.cond(rand_int % 2 == 1):
    tracer.apply(print, "Random Integer ", rand_int, " is Odd")
```

In the example above, we have two Conditional contexts with mutually exclusive conditions, mimicking a conventional `If`-`Else` statement.

The condition passed to the Conditional context is evaluated directly by calling `bool()` on the proxy value, so be mindful of how your Intervention Proxy condition evaluates to boolean.

```python
with tiny_model.trace(input) as tracer:
  l1_out = tiny_model.layer1.output
  with tracer.cond(l1_out != 1):
    tracer.apply(print, "Condition is True")
```

The code above throws the **ERROR**: `Boolean value of Tensor with more than one value is ambiguous`, because the condition specified in the Conditional context cannot be handled properly.

Instead, use something like this:

```python
with tiny_model.trace(input) as tracer:
  l1_out = tiny_model.layer1.output
  with tracer.cond(torch.all(l1_out != 1)):
    tracer.apply(print, "Condition is True")
```

Conditional contexts can also be nested, if you want your interventions to depend on more than one condition at a time.

```python
with tiny_model.trace(input) as tracer:
  rand_int = tracer.apply(int, 6)
  with tracer.cond(rand_int > 0):
    with tracer.cond(rand_int % 2 == 0):
      tracer.apply(print, "Rand Int ", rand_int, " is Positive and Even")
```

## Model Editing

You can alter a model by setting default edits and interventions in an editing context, applied before each forward pass. This can be used to attach additional modules like SAEs.

```python
with tiny_model.edit() as edited_model:
  tiny_model.layer1.output[0][:] = 0

with tiny_model.trace(input):
  l1_out = tiny_model.layer1.output.save()

with edited_model.trace(input):
  l1_out_edited = edited_model.layer1.output.save()

print("L1 - Out: ", l1_out)
print("L1 - Out [edited]: ", l1_out_edited)
```

Let's look at anotehr example

```python
from nnsight.util import WrapperModule

class ComplexModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.one = WrapperModule()

    def forward(self, x):
        return self.one(x)

l0 = lm.transformer.h[0]
l0.attachment = ComplexModule()

with lm.edit() as gpt2_edited:
    acts = l0.output[0]
    l0.output[0][:] = l0.attachment(acts, hook=True)

# Get values pre editing
with lm.trace("Madison Square Garden is located in the city of"):
    original = l0.output[0].clone().save()
    l0.output[0][:] *= 0.
    original_output = lm.output.logits.save()

with gpt2_edited.trace("Madison Square Garden is located in the city of"):
    one = l0.attachment.one.output.clone().save()
    l0.attachment.output *= 0.
    edited_output = lm.output.logits.save()

print("Original output: ", original_output)
print("Edited output: ", edited_output)
```

Your edit call can be customized by choosing to perform edits in-place on the model andgetting access to the editor context (`nnsight.context.Tracer`).

You can also choose to remove edits perfomerd on a model at a later stage.

```python
with tiny_model.edit(inplace=True, return_context=True) as editor:
  tiny_model.layer1.output[0][:] = 0

with tiny_model.trace(input):
  l1_out = tiny_model.layer1.output.save()

print("L1 - Out: ", l1_out)

tiny_model.clear_edits()

with tiny_model.trace(input):
  l1_out = tiny_model.layer1.output.save()

print("L1 - Out [unedited]: ", l1_out)
```

Note that setting new modules with remote execution is currently not supported!

## Session Context

`nnsight 0.3` focuses on enhancing the capabilities of our remote execution API, powered by the [NDIF](https://ndif.us) backend.

To achieve this, we introduce the **Session** context: an overarching structure for efficiently handling multi-tracing experiments. This means that, multiple `Tracer` contexts can be packaged together as part of one single request to the server.

The `Session` context can also be used entirely for local usage, as it enables useful functionalities and optimizes experiments.

```python
with llm.session(remote=True):
  with llm.trace("_") as t1:
    # define interventions here
    pass

  with llm.trace("_") as t2:
    # define interventions here
    pass

  with llm.trace("_") as t3:
    # define interventions here
    pass
```

All operations defined within a `Session` context are executed at the very end (upon exiting the overarching context) and it is conducted sequentially, strictly following the order of definition (`t2` being executed after `t1` and `t3` after `t2`).

In a `Session`, interventions defined at any early stage can be seamlessly referenced.

```python
with llm.session(remote=True) as session:
  with llm.trace("The Eiffel Tower is in the city of") as t1:
    hs_11 = llm.model.layers[-1].output[0][:, -1, :] # no .save()
    t1_tokens_out = llm.output.save()

  with llm.trace("Buckingham Palace is in the city of") as t2:
    llm.model.layers[-2].output[0][:, -1, :] = hs_11[:]
    t2_tokens_out = llm.output.save()

print("\nT1 - Prediction: ", llm.tokenizer.decode(t1_tokens_out["logits"].argmax(dim=-1)[0][-1]))
print("T2 - Prediction: ", llm.tokenizer.decode(t2_tokens_out["logits"].argmax(dim=-1)[0][-1]))
```

In the example above, we are interested in patching the hidden state of a later layer into an earlier one. This experiment can only be conducted with two `Tracer` contexts; since we are using a `Session`, it is not required to save the hidden state from Tracer 1 to reference it in Tracer 2.

The `Session` context can also be terminated early.

```python
import nnsight

with llm.session(remote=True) as session:
  l = nnsight.list().save()

  l.append(0)
  l.append(1)
  nnsight.log("-- Early Stop --")
  session.exit()
  l.append(2)

print("List: ", l)
```

## Iterator Context

We mention earlier that the `Session` context enables multi-tracing execution. But how can we optimize a process that would require running an intervention graph in a loop?

If you create a `for` loop with a `Tracer` context inside of it, this will result in creating a new intervention graph at each iteration, which is not scalable.

We solve this problem the `nnsight` way by introducing the **Iterator** context: an intervention loop that iteratively executes a single intervention graph with an updated parameter.

```python
import nnsight

with llm.session(remote=True) as session:

  prompts = nnsight.list(["This is nnsight 0.3",
                          "It works with NDIF",
                          "pip install it now!"])
  results = nnsight.list().save()
  with session.iter(prompts) as prompt:

    with llm.trace(prompt):
      results.append(llm.lm_head.output)
```

Use a `Session` to define the `Iterator` context and pass in a sequence of items that you want to loop over at each executed iteration.

The sequence must be iterable or be a Proxy with an iterable value.

The iterable's item can be referenced in the inner intervention body of the `Iterator`.

### loop

The `Iterator` context extends all the `nnsight` graph-based functionalities, but also closely mimics the conventional `for` loop statement in Python, which allows it to support all kind of iterative operations.

```python
import nnsight

with llm.session(remote=True) as session:
  l = nnsight.list()
  [l.append(num) for num in range(0, 3)] # adding 0, 1, 2 to l
  with session.iter(l) as item: # with session.iter([0, 1, 2]) also works!
    nnsight.log(item)
```

You can create nested `Iterator` contexts:

```python
import nnsight

with llm.session() as session:
  l = nnsight.list([[10]] * 5)

  l2 = nnsight.list().save()
  with session.iter(l) as item:
    with session.iter(item) as item_2:
      l2.append(item_2)

print("List: ", l2)
```

You can skip some iterations:

```python
import nnsight

with llm.session(remote=True) as session:

  with session.iter([0, 1, 2, 3], return_context=True) as (item, iterator):
    with iterator.cond(item % 2 == 0):
      nnsight.log(item)
```

Or, you can choose to `break` out of the Iteration loop early:

```python
import nnsight

with llm.session(remote=True) as session:

  with session.iter([0, 1, 2, 3], return_context=True) as (item, iterator):
      with iterator.cond(item == 2):
        iterator.exit()

      nnsight.log(item)
```

The `Iterator` context is a niece piece of functionality that allows you to define a bunch of basic code operations that can now be "traceable" by `nnsight`.

But in what kind of experimental scenario would someone even need to use it?

In the next section, we delve into a powerful use case of the `Iterator` context and see how it enables it!

---

# NNsight_v0_4_guide.ipynb

# `nnsight 0.4`: walkthrough
**We have many exciting new features in this update, including:**

*   Descriptive error messages
*   .all() for multiple token generation
*   vLLM Integration
*   Streaming remote execution to local machine
*   Support for traditional Python syntax for `if` statements and `for` loops on proxies within tracing contexts
*   Ability to rename model modules

...and more!

The following walkthrough guides you through how to access `nnsight 0.4` and use all of its individual features.

**Breaking Changes**
* The InterventionGraph now follows a <u>sequential execution order</u>. Module envoys are expected to be referenced following the model’s architecture hierarchy. This means that out-of-order in-place operations will not take effect.

* Saved node values are automatically injected into their proxy reference in the Python frames post graph execution. If you are calling `.value` in your code after tracing, this could lead to the wrong behavior.

# Access `nnsight` 0.4

```python
from IPython.display import clear_output
!pip install nnsight
clear_output()
```

Import `nnsight` and load the GPT-2 model.

```python
# import packages
import nnsight
from nnsight import LanguageModel
```

```python
model = LanguageModel('openai-community/gpt2', device_map='auto')
print(model)
```

# Improved Error Messaging

If you've been using `nnsight`, you are probably familiar with the following type of error message:

```
IndexError: Above exception when execution Node: 'setitem_0' in Graph: '6063279136'
```
It can be quite difficult to troubleshoot with these errors, so in `nnsight 0.4` we've now improved error messaging to be descriptive and line-specific! Let's check it out:

```python
prompt = 'The Eiffel Tower is in the city of'

with model.trace(prompt) as tracer:

    # try to access a layer of model that doesn't exist
    model.transformer.h[12].output[0][:] = 0
    output = model.lm_head.output.save()

print("lm_head output = ",output)
```

Great! Now we know that our list index was out of range within the tracing context, and if we expand to see the full message, we can tell that it's happening in line 7.

Let's try again, now using the correct index for the final layer:

```python
prompt = 'The Eiffel Tower is in the city of'

with model.trace(prompt) as tracer:

    # ablate last layer output
    model.transformer.h[11].output[0][:] = 0
    output = model.lm_head.output.save()

print("lm_head output = ",output)
```

The error messaging feature can be toggled using `nnsight.CONFIG.APP.DEBUG` which defaults to true.

```python
# Turn off debugging:
import nnsight

nnsight.CONFIG.APP.DEBUG = False
nnsight.CONFIG.save()
```

# .all()

Sometimes you may want to recursively apply interventions to a model (e.g., when generating many tokens or for models like RNNs, where modules are called multiple times).

*   Calling `.all()` on a model or its submodules will recursively apply its `.input` and `.output` across all iterations.
*   When generating multiple tokens with `.generate` (see: [Multiple Token Generation](https://nnsight.net/notebooks/features/multiple_token/)), using `.all()` before applying an intervention will ensure that the model undergoes the intervention for *all* new tokens generated, not just the first.

## About

## .all() now streamlines multiple token generation

With .all, applying interventions during multiple token generation becomes much easier. Let's test this out!

We can use `.all()` to streamline the multiple token generation process. We simply call `.all` on the module where we are applying the intervention (in this case GPT-2's layers), apply our intervention, and append our hidden states (stored in an `nnsight.list()` object).

```python
# New: using .all():
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() to apply intervention to each new token
    layers.all()

    # Apply intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(layers[-1].output) # no need to call .save
    # Don't need to loop or call .next()!

print("Hidden state length: ",len(hidden_states))
```

Easy! Note that because `.all()` is recursive, it will only work to append outputs called on children of the module that `.all()` was called on. See example below for more information. TL;DR: apply `.all()` on the highest-level accessed module if interventions and outputs have different hierarchies within model structure.

### Note: (Old method) Applying interventions during multiple token generation without .all()

Without `.all()`, we would need to loop across each new generated token, saving the intervention for every generated token and calling `.next()` to move forward.

```python
# Old approach:
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
hidden_states = []
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    for i in range(n_new_tokens):
        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output.save())

        # Move to next generated token
        layers[0].next()

print("Hidden state length: ",len(hidden_states))
```

### Note: .all() recursive properties

As mentioned, `.all()` is recursive and will work to append outputs called on children of the module that `.all` was called on. In this example, calling `.all()` on the model's layer modules will not recursively affect `model.lm_head.output` as it is not a child of layers.

```python
# A note on .all() recursive properties:
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on layers
    layers.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save, it's already initialized

print("Hidden state length: ",len(hidden_states)) # length is 1, meaning it only saved the first token generation
```

So, if you want to apply an intervention during multiple token generation while saving the state of a model component that isn't a child of that module, you can apply .`all()` to the full model.

```python
# Applying .all() to model fixes issue
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on model
    model.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save

print("Hidden state length: ",len(hidden_states)) # length is 3!
```

## Known Issues: `.all()`

* IteratorEnvoy contexts can produce undesired behavior for subsequent operations defined <u>below</u> it that are not dependent on InterventionProxys.

Example:
```
with lm.generate("Hello World!", max_new_tokens=10):
    hs_4 = nnsight.list().save()

    with lm.transformer.h[4].all():
        hs_4.append(lm.transformer.h[4].output)

    hs_4.append(433)

print(len(hs_4))
```
`>>> 20 # expected: 11`

# Syntax updates

With `nnsight 0.4`, we now support `if` statements and `for` loops applied to proxies with traditional Python syntax! We also remove the need to call `.value` on a proxy output.

## If Statements

Previously, we would need to use `.cond()` to create a conditional context that would only execute upon meeting the logical conditions inside the `.cond()`.

```python
import torch
# Old method
# model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace("The Eiffel Tower is in the city of") as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # To create the conditional context you need to put the
  # condition within tracer.cond()
  with tracer.cond(rand_int % 2 == 0):
    tracer.log("Random Integer ", rand_int, " is Even")

  with tracer.cond(rand_int % 2 == 1):
    tracer.log("Random Integer ", rand_int, " is Odd")
```

Now, we can use Python `if` statements within the tracing context to create a conditional context!

*Note: Colab may be a little strangely with this feature the first time you run it - expect some lagging and warnings.*

```python
with model.trace("The Eiffel Tower is in the city of") as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")

  if rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

Note: If the conditional statements are outside the tracing context, `if` operates as in base Python.

`elif` statements should also work as `if` statements within the tracing context:

```python
with model.trace("The Eiffel Tower is in the city of") as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")
  elif rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

## For Loops

With `nnsight 0.4`, you can now use `for` loops within a tracer context at scale. Previously, a `for` loop within a tracer context inside it resulted in creating intervention graphs over and over for each iteration - this is not scalable!

The `session.iter` context allows for scalable looping within sessions, but doesn't utilize traditional Python syntax:

```python
# Old Method
with model.session() as session:

  li = nnsight.list() # an NNsight built-in list object
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

Now, you can use simple `for` loops within a tracer context to run an intervention loop at scale.

*NOTE: inline for loops (i.e., `[x for x in <Proxy object>]`) are not currently supported.*

```python
# New: Using Python for loops for iterative interventions
with model.session() as session:

    li = nnsight.list()
    [li.append([num]) for num in range(0, 3)]
    li2 = nnsight.list().save()

    # Using regular for loops
    for item in li:
        for item_2 in item: # for loops can be nested!
            li2.append(item_2)

print("\nList: ", li2)
```

## `.value` injected into saved results

Previously, directly using non-traceable functions (i.e., tokenizers) on a proxy returned from a tracing context required calling `.value` to access the proxy's numerical value. Calling traceable functions (like `print()` or `.argmax()`) on such proxies automatically returned the `.value`, making it optional to call `.value` in certain cases.

```
input = "The Eiffel Tower is in the city of"
with model.trace(input):

    l2_input = model.transformer.h[2].input.save()

print(l2_input.value) # could optionally call .value
print(l2_input) # but not required for traceable functions
```

Now with `nnsight 0.4`, the proxy's value is automatically injected into the variable name, negating any needs to call `.value` on proxies. Proxy variables will automatically be populated with their value upon exiting the tracing context. This is a breaking change, and calling `.value` on a proxy will now throw an error.

```python
input = "The Eiffel Tower is in the city of"
with model.trace(input):

    l2_input = model.transformer.h[2].input.save()

print(l2_input) # no need to call .value
print(l2_input.value) # will throw an error
```

## Turning off syntactic changes

If you would like to turn off either the `if`/`for` functionality or the `.value` syntactic changes, you can apply the following changes to `nnsight.CONFIG`

```python
# Turn off if/for statements within tracing context:
import nnsight

nnsight.CONFIG.APP.CONTROL_FLOW_HANDLING = False
nnsight.CONFIG.save()
```

```python
# Turn off .value injection:
import nnsight

nnsight.CONFIG.APP.FRAME_INJECTION = False
nnsight.CONFIG.save()
```

## Known Issues: Syntax Update
* Colab behaves a little strangely with these features the first time you run it - expect some lagging and warnings.

* Inline Control Flow (for loops) are not supported.

Example:
```
with lm.trace("Hello World!"):
    foo = nnsight.list([0, 1, 2, 3]).save()
    [nnsight.log(item) for item in foo]

```
`>>> Error`

* Value Injection is not supported for proxies referenced within objects.

# vLLM Integration

Our new update includes support for vLLM models using `nnsight`. [vLLM](https://github.com/vllm-project/vllm) is a popular library used for fast inference. By leveraging PagedAttention, dynamic batching, and Hugging Face model integration, vLLM makes inference more efficient and scalable for real-world applications.

## Setup

You will need to install `nnsight 0.4`, `vllm`, and `triton 3.1.0` to use vLLM with NNsight.

```python
from IPython.display import clear_output
# install vllm
!pip install vllm==0.6.4.post1

# install triton 3.1.0
!pip install triton==3.1.0

clear_output()
```

**NOTE: you may need to restart your Colab session before the following step to properly load the `VLLM` model wrapper.**

 Next, let's load in our NNsight-supported vLLM model. You can find vLLM-supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html). For this exercise, we will use GPT-2.

```python
from nnsight.modeling.vllm import VLLM

# NNsight's VLLM wrapper currently supports "device = cuda" and device = "auto"
vllm = VLLM("gpt2", device = "auto", dispatch = True) # See supported models: https://docs.vllm.ai/en/latest/models/supported_models.html
print(vllm)
```

## Interventions on vLLM models
We now have a vLLM model that runs with `nnsight`. Let's try applying some interventions on it.

Note that vLLM takes in sampling parameters including `temperature` and `top_p`. These parameters can be included in the `.trace()` or `.invoke()` contexts. For default model behavior, set `temperature = 0` and `top_p = 1`. For more information about parameters, reference the [vLLM documentation](https://docs.vllm.ai/en/latest/dev/sampling_params.html).

```python
with vllm.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
  with tracer.invoke("The Eiffel Tower is located in the city of"):
    clean_logits = vllm.logits.output.save()

  with tracer.invoke("The Eiffel Tower is located in the city of"):
    vllm.transformer.h[-2].mlp.output[:] = 0
    corrupted_logits = vllm.logits.output.save()
```

```python
print("CLEAN - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(clean_logits.argmax(dim=-1)))
print("CORRUPTED - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(corrupted_logits.argmax(dim=-1)))
```

We've successfully performed an intervention on our vLLM model!

## Sampled Token Traceability
vLLM provides functionality to configure how each sequence samples its next token. Here's an example of how you can trace token sampling operations with the nnsight VLLM wrapper.

```python
import nnsight
with vllm.trace("Madison Square Garden is located in the city of", temperature=0.8, top_p=0.95, max_tokens=3) as tracer:
    samples = nnsight.list().save()
    logits = nnsight.list().save()

    for ii in range(3):
        samples.append(vllm.samples.output)
        vllm.samples.next()
        logits.append(vllm.logits.output)
        vllm.logits.next()

print("Samples: ", samples)
print("Logits: ", logits) # different than samples with current sampling parameters
```

## Note: gradients are not supported with vLLM
vLLM speeds up inference through its paged attention mechanism. This means that accessing gradients and backward passes are not supported for vLLM models. As such, calling gradient operations when using `nnsight` vLLM wrappers will throw an error.

## Known Issues: vLLM Integration
* The vllm.LLM engine performs max_tokens + 1 forward passes which can lead to undesired behavior if you are running interventions on all iterations of multi-token generation.

Example:
```
with vllm_gpt2("Hello World!", max_tokens=10):
    logits = nnsight.list().save()
    with vllm_gpt2.logits.all():
        logits.append(vllm_gpt2.logits.output)

print(len(logits))

```
`>>> 11 # expected: 10`

# Streaming

Streaming enables users apply functions and datasets locally during remote model execution. This allows users to stream results for immediate consumption (i.e., seeing tokens as they are generated) or applying non-whitelisted functions such as model tokenizers, large local datasets, and more!

*   `nnsight.local()` context sends values immediately to user's local machine from server
*   Intervention graph is executed locally on downstream nodes
*   Exiting local context uploads data back to server
*   `@nnsight.trace` function decorator enables custom functions to be added to intervention graph when using `nnsight.local()`

## `nnsight.local()`

You may sometimes want to locally access and manipulate values during remote execution. Using `.local()` on a proxy, you can send remote content to your local machine and apply local functions. The intervention graph is then executed locally on downstream nodes until you exit the local context.

There are a few use cases for streaming with `.local()`, including live chat generation and applying large datasets or non-whitelisted local functions to the intervention graph.

Now let's explore how streaming works. We'll start by grabbing some hidden states of the model and printing their value using `tracer.log()`. Without calling `nnsight.local()`, these operations will all occur remotely.

```python
from nnsight import LanguageModel

llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
```

```python
# This will give you a remote LOG response because it's coming from the remote server
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

Now, let's try the same operation using the `nnsight.local()` context. This will send the operations to get and print the hidden states to your local machine, changing how the logging message is formatted (local formatting instead of remote).

```python
# This will print locally because it's already local
with llama.trace("hello", remote=True) as tracer:

    with nnsight.local():
        hs = llama.model.layers[-1].output[0]
        tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

## `@nnsight.trace` function decorator

We can also use function decorators to create custom functions to be used during `.local` calls. This is a handy way to enable live streaming of a chat or to train probing classifiers on model hidden states.

Let's try out `@nnsight.trace` and `nnsight.local()` to access a custom function during remote execution.

```python
# first, let's define our function
@nnsight.trace # decorator that enables this function to be added to the intervention graph
def my_local_fn(value):
    return value * 0

# We use a local function to ablate some hidden states
# This downloads the data for the .local context, and then uploads it back to set the value.
with llama.generate("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    with nnsight.local():

        hs = my_local_fn(hs)

    llama.model.layers[-1].output[0][:] = hs

    out =  llama.lm_head.output.save()
```

Note that without calling `.local`, the remote API does not know about `my_local_fn` and will throw a whitelist error. A whitelist error occurs because you are being allowed access to the function.

```python
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    hs = my_local_fn(hs) # no .local - will cause an error

    llama.model.layers[-1].output[0][:] = hs * 2

    out =  llama.lm_head.output.save()

print(out)
```

## Example: Live-streaming remote chat

Now that we can access data within the tracing context on our local computer, we can apply non-whitelisted functions, such as the model's tokenizer, within our tracing context.

Let's build a decoding function that will decode tokens into words and print the result.

```python
@nnsight.trace
def my_decoding_function(tokens, model, max_length=80, state=None):
    # Initialize state if not provided
    if state is None:
        state = {'current_line': '', 'current_line_length': 0}

    token = tokens[-1] # only use last token

    # Decode the token
    decoded_token = llama.tokenizer.decode(token).encode("unicode_escape").decode()

    if decoded_token == '\\n':  # Handle explicit newline tokens
        # Print the current line and reset state
        print('',flush=True)
        state['current_line'] = ''
        state['current_line_length'] = 0
    else:
        # Check if adding the token would exceed the max length
        if state['current_line_length'] + len(decoded_token) > max_length:
            print('',flush=True)
            state['current_line'] = decoded_token  # Start a new line with the current token
            state['current_line_length'] = len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line
        else:
            # Add a space if the line isn't empty and append the token
            if state['current_line']:
                state['current_line'] += decoded_token
            else:
                state['current_line'] = decoded_token
            state['current_line_length'] += len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line

    return state
```

Now we can decode and print our model outputs throughout token generation by accessing our decoding function through `nnsight.local()`.

```python
import torch

nnsight.CONFIG.APP.REMOTE_LOGGING = False

prompt = "A press release is an official statement delivered to members of the news media for the purpose of"
# prompt = "Your favorite board game is"

print("Prompt: ",prompt,'\n', end ="")

# Initialize the state for decoding
state = {'current_line': '', 'current_line_length': 0}

with llama.generate(prompt, remote=True, max_new_tokens = 50) as generator:
    # Call .all() to apply to each new token
    llama.all()

    all_tokens = nnsight.list().save()

    # Access model output
    out = llama.lm_head.output.save()

    # Apply softmax to obtain probabilities and save the result
    probs = torch.nn.functional.softmax(out, dim=-1)
    max_probs = torch.max(probs, dim=-1)
    tokens = max_probs.indices.cpu().tolist()
    all_tokens.append(tokens[0]).save()

    with nnsight.local():
        state = my_decoding_function(tokens[0], llama, max_length=20, state=state)
```

# General Considerations
* `Tracer.cond(…)` and `Tracer.iter(…)` are still supported.

* vLLM <U>does not</u> come as a pre-installed dependency of `nnsight`.

* `nnsight` supports `vllm==0.6.4.post1`

* vLLM support only includes `cuda` and `auto` devices at the moment.

* vLLM models <u>do not</u> support gradients.

* The `@nnsight.trace` decorator does not enable user-defined operations to be executed remotely. Something coming soon for that...

---

# activation_patching.ipynb

# Activation Patching

📗 You can find an interactive Colab version of this tutorial [here](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/activation_patching.ipynb).

**Activation patching** is a technique used to understand how different parts of a model contribute to its behavior. In an activation patching experiment, we modify or "patch" the activations of certain model components and observe the impact on model output.

**Activation patching experiments typically follow these steps:**

1.   **Baseline Run:** Run the model and record original activations.
2.   **Corrupted Run:** Run the model with with a counterfactual (i.e., corrupted) prompt and record the difference in activations.
3.   **Patching:** Replace activations at the model component of interest with alternate activations (or zeros, which is sometimes referred to as ablation).

By systematically testing different components this way, researchers can determine how information flows through the model. One common use case is **circuit identification**, where a circuit is a subgraph of a full model that is responsible for a specific and human-interpretable task (e.g., detecting whether an input is in English). Activation patching can help identify which model components are essential for model performance on a given task.

In this tutorial, we use `nnsight` to perform a simple activation patching experiment using an indirect object identification (IOI) task.

## Note: IOI Task

Activation patching was used to find the [Indirect Object Identification](https://openreview.net/forum?id=NpsVSN6o4ul) (IOI) circuit in GPT-2 small. IOI is a natural language task in which a model predicts the indirect object in a sentence. IOI tasks typically involve identifying the indirect object from two names introduced in an initial dependent clause. One name (e.g. "Mary") is the subject (S1), and the other name (e.g. "John") is the indirect object (IO). In the main clause, a second occurrence of the subject (S2) typically performs an action involving the exchange of an item. The sentence always ends with the preposition "to," and the task is to correctly complete it by identifying the non-repeated name (IO).

In this exercise, we will use the following 'clean' prompt:

```
"After John and Mary went to the store, Mary gave a bottle of milk to"
```
This prompt's correct answer (and thus its indirect object) is: `" John"`

We will also use a corrupted prompt to test how activation patching works. This corrupted prompt will switch the identity of the indirect object, so we can test how the model responds to this change.
```
"After John and Mary went to the store, John gave a bottle of milk to"
```
This prompt's correct answer (and thus its indirect object) is: `" Mary"`

![](images/Activation_patching-figure1.png)

# Setup

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
```

```python
from IPython.display import clear_output
```

```python
import nnsight
from nnsight import CONFIG
```

Let's start with our imports:

```python
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab" if is_colab else "plotly_mimetype+notebook_connected+colab+notebook"
from nnsight import LanguageModel, util
```

```python
# Load gpt2
model = LanguageModel("openai-community/gpt2", device_map="auto")
clear_output()
```

```python
print(model)
```

Next up, we define our clean prompt and our corrupted prompt. As prompts may be associated with many different feature circuits (i.e., circuits responsible for IOI, deciding if the language is English, or prompt refusal), choosing a counterfactual prompt with only changes directly related your feature of interest is essential.

Here, we switch the name of the repeated subject, thus swapping out the indirect object for our IOI task:

```python
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)
```

We then use the tokenizer on the two words of interest (“John” and “Mary”) to find the token that represents them. That way we can grab the prediction for these two tokens and compare. Because our prompts don't end in a space, make sure to add a space before each word (i.e., the combined space + word token is what we're looking for).

```python
correct_index = model.tokenizer(" John")["input_ids"][0] # includes a space
incorrect_index = model.tokenizer(" Mary")["input_ids"][0] # includes a space

print(f"' John': {correct_index}")
print(f"' Mary': {incorrect_index}")
```

# Patching Experiment
Now we can run the actual patching intervention! What does this even mean?

We now have two prompts, a "clean" one and a "corrupted" one. Intuitively, the model output for each of these prompts should be different: we'd expect the model to answer "John" for the clean prompt and "Mary" for the corrupted prompt.

In this experiment, we run the model with the clean prompt as an input and then (1) get each layer's output value (i.e., residual stream) and (2) calculate the logit difference between the correct and incorrect answers for this run. Next, we calculate the logit difference between the correct and incorrect answers for the corrupted prompt.

## Step 1: Clean Run

First, we run the model with the **clean prompt**:

`"After John and Mary went to the store, Mary gave a bottle of milk to"`

During this clean run, we collect the final output of each layer. We also record the logit difference in the final model output between the correct answer token `" John"` and the incorrect token `" Mary"`.

```python
N_LAYERS = len(model.transformer.h)

# Clean run
with model.trace(clean_prompt) as tracer:
    clean_tokens = tracer.invoker.inputs[0][0]['input_ids'][0]

    # Get hidden states of all layers in the network.
    # We index the output at 0 because it's a tuple where the first index is the hidden state.

    clean_hs = [
        model.transformer.h[layer_idx].output[0].save()
        for layer_idx in range(N_LAYERS)
    ]

    # Get logits from the lm_head.
    clean_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
    clean_logit_diff = (
        clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
    ).save()
```

## Step 2: Corrupted Run

Next, we run the model using the **corrupted** input prompt:

 `"After John and Mary went to the store, John gave a bottle of milk to"`

 During this corrupted run, we collect the logit difference in the final model output between the correct and incorrect answer tokens

 Note: because we are testing changes induced by the corrupted prompt, the target answers remain the same as in the clean run. That is, the correct token is still `" John"` and the incorrect token is still `" Mary"`.

```python
# Corrupted run
with model.trace(corrupted_prompt) as tracer:
    corrupted_logits = model.lm_head.output

    # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
    corrupted_logit_diff = (
        corrupted_logits[0, -1, correct_index]
        - corrupted_logits[0, -1, incorrect_index]
    ).save()
```

## Step 3: Activation Patching Intervention

Finally, we perform our **activation patching** procedure. For each token position in the clean prompt, we loop through all layers of the model. Within each layer, we run a forward pass using the corrupted prompt, and patch in the corresponding activation from our clean run at the given token position. We then collect the final output difference between the correct and incorrect answer tokens for each patched activation.

```python
# Activation Patching Intervention
ioi_patching_results = []

# Iterate through all the layers
for layer_idx in range(len(model.transformer.h)):
    _ioi_patching_results = []

    # Iterate through all tokens
    for token_idx in range(len(clean_tokens)):
        # Patching corrupted run at given layer and token
        with model.trace(corrupted_prompt) as tracer:
            # Apply the patch from the clean hidden states to the corrupted hidden states.
            model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[layer_idx][:, token_idx, :]

            patched_logits = model.lm_head.output

            patched_logit_diff = (
                patched_logits[0, -1, correct_index]
                - patched_logits[0, -1, incorrect_index]
            )

            # Calculate the improvement in the correct token after patching.
            patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                clean_logit_diff - corrupted_logit_diff
            )

            _ioi_patching_results.append(patched_result.item().save())

    ioi_patching_results.append(_ioi_patching_results)
```

### Note: Optimize workflow with NNsight batching

Although we broke up the workflow for ease of understanding, we can use `nnsight` to further speed up computation.

Thanks to `nnsight`, the whole experiment can happen in one forward pass by breaking up inputs into multiple invocation calls and batching them.

```python
N_LAYERS = len(model.transformer.h)

# Enter nnsight tracing context
with model.trace() as tracer:

    # Clean run
    with tracer.invoke(clean_prompt) as invoker:
        clean_tokens = invoker.inputs[0][0]['input_ids'][0]

        # No need to call .save() as we don't need the values after the run, just within the experiment run.
        clean_hs = [
            model.transformer.h[layer_idx].output[0]
            for layer_idx in range(N_LAYERS)
        ]

        # Get logits from the lm_head.
        clean_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the clean run and save it.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()

    # Corrupted run
    with tracer.invoke(corrupted_prompt) as invoker:
        corrupted_logits = model.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index]
            - corrupted_logits[0, -1, incorrect_index]
        ).save()

    ioi_patching_results = []

    # Iterate through all the layers
    for layer_idx in range(len(model.transformer.h)):
        _ioi_patching_results = []

        # Iterate through all tokens
        for token_idx in range(len(clean_tokens)):
            # Patching corrupted run at given layer and token
            with tracer.invoke(corrupted_prompt) as invoker:
                # Apply the patch from the clean hidden states to the corrupted hidden states.
                model.transformer.h[layer_idx].output[0][:, token_idx, :] = clean_hs[layer_idx][:, token_idx, :]

                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.item().save())

        ioi_patching_results.append(_ioi_patching_results)
```

## Visualize Results

Let's define a function to plot our activation patching results.

```python
from nnsight.tracing.graph import Proxy

def plot_ioi_patching_results(model,
                              ioi_patching_results,
                              x_labels,
                              plot_title="Normalized Logit Difference After Patching Residual Stream on the IOI Task"):

    ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value, Proxy)

    fig = px.imshow(
        ioi_patching_results,
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": "Position", "y": "Layer","color":"Norm. Logit Diff"},
        x=x_labels,
        title=plot_title,
    )

    return fig
```

Let's see how the patching intervention changes the logit difference! Let's use a heatmap to examine how the logit difference changes after patching each layer's output across token positions.

```python

print(f"Clean logit difference: {clean_logit_diff:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff:.3f}")

clean_decoded_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_decoded_tokens)]

fig = plot_ioi_patching_results(model, ioi_patching_results,token_labels,"Patching GPT-2-small Residual Stream on IOI task")
fig.show()
```

In the above plot, we see that patching the clean residual stream into the corrupted model does not change much in the final token difference for input tokens 0-9. This is expected, as there is no difference in the clean vs. corrupted prompt for these tokens, so patching in the clean activations at this point shouldn't change the model prediction.

However, when we get to token #10, "Mary", where the subject is introduced for the second time, there is a sharp increase in output logit difference, indicating that the patch changes how the model predicts the outcome downstream, particularly for the earlier layers. There is also a transition in the middle layers of the network where the logit difference starts decreasing. We are thus seeing how the network is tracking information about the indirect object as the layers progress.

A similar but opposite effect is observed when the activations for the final prompt token are patched: the normalized logit difference increases after a transition period in the middle layers.

# Limitations

Although activation patching is an effective technique for circuit localization, it requires running a forward pass through the model for every patch, making it computationally expensive.

**Attribution patching** is an approximation of activation patching that helps scale the technique to larger experiments and models. See our attribution patching tutorial [here](https://nnsight.net/notebooks/tutorials/attribution_patching/) to try it out!

# Trying on a bigger model

Although the original IOI experiment was performed on GPT-2 small, NDIF allows researchers to explore similar problems on largescale models!

Let's see how the residual stream of Llama 3.1-8B contributes to the IOI task using activation patching with NDIF's remote infrastructure.

### NNsight Remote Setup
Make sure you have obtained your [NDIF API key](https://login.ndif.us/) and configured your workspace for [remote execution](https://nnsight.net/notebooks/features/remote_execution/).

```python
from nnsight import CONFIG

if is_colab:
    # include your HuggingFace Token and NNsight API key on Colab secrets
    from google.colab import userdata
    NDIF_API = userdata.get('NDIF_API')
    HF_TOKEN = userdata.get('HF_TOKEN')

    CONFIG.set_default_api_key(NDIF_API)
    !huggingface-cli login -token HF_TOKEN

clear_output()
```

```python
import torch
import nnsight
from nnsight import LanguageModel
```

```python
# Load model
llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
print(llm)
```

Define some IOI prompts. Each of these prompts can be used as a 'clean' and as a 'corrupted' prompt, as each prompt has a related corrupted version with the IO switched out.

```python
prompts = [
    "When Lisa and Sarah went to the cinema, Lisa gave the ticket to",
    "When Lisa and Sarah went to the cinema, Sarah gave the ticket to"
]
```

Define the answers to these prompts, formatted as `(correct, incorrect)`

```python
answers = [
    (" Sarah", " Lisa"),
    (" Lisa", " Sarah")
]
```

```python
# Tokenize clean and corrupted inputs:
clean_tokens = llm.tokenizer(prompts, return_tensors="pt")["input_ids"]
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

# Tokenize answers for each prompt:
answer_token_indices = [
        [llm.tokenizer(answers[i][j])["input_ids"][1] for j in range(2)]
        for i in range(len(answers))
]

print("answer_tokens = " , answer_token_indices)
```

### Patching Attention Heads

The residual stream isn't the only model component you can apply activation patching on: let's try patching Llama's attention heads to see how they influence the IOI task! Here, we apply our patching intervention on the attention output, `o_proj.output` in Llama models.

Because the multihead attention of Llama models are stored in a projection matrix containing all attention heads, we will need to resize the tensor to reveal individual attention head contributions. The `einops` library is a handy way to resize tensors.

```python
import einops
```

Okay, now let's apply our three activation patching steps to our attention heads during an IOI task.

```python
# Enter nnsight tracing context
N_LAYERS = len(llm.model.layers)
batch = 1
seq = len(prompts[0]) #15 length of input tokens
N_HEADS = 32 #32 attention heads
D_MODEL = int(4096) #4096 size of model hidden
D_HEADS = int(D_MODEL/N_HEADS) #128 size of attention head

ioi_patching_results_all = []
prompt_id = 0
corrupt_id = (prompt_id + 1 if prompt_id % 2 == 0 else prompt_id - 1)

with llm.trace(remote = True) as tracer:
    # STEP 1: Clean run, grab clean activations for each attention head
    with tracer.invoke(prompts[prompt_id]) as invoker:
        clean_tokens = invoker.inputs[0][0]['input_ids'][0]

        # Get clean attention output for later patching
        z_hs = {}
        for layer_idx in range(N_LAYERS):
            # attention output for llama models needs to be reshaped to look at individual heads
            z = llm.model.layers[layer_idx].self_attn.o_proj.input # dimensions [1x15x4096] [batch x seq x D_MODEL]
            z_reshaped = einops.rearrange(z, 'b s (nh dh) -> b s nh dh',nh=32)
            for head_idx in range(N_HEADS):
                z_hs[layer_idx,head_idx] = z_reshaped[:,:,head_idx,:]

        # Get logits from the lm_head.
        clean_logits = llm.lm_head.output
        clean_logit_diff = (
            clean_logits[0, -1, answer_token_indices[prompt_id][0]] - clean_logits[0, -1, answer_token_indices[prompt_id][1]]
        ).save()

    # STEP 2: Corrupted run, grab corrupted logits for later comparison.
    with tracer.invoke(prompts[corrupt_id]) as invoker:
        corrupted_tokens = invoker.inputs[0][0]['input_ids'][0]
        corrupted_logits = llm.lm_head.output

        # Calculate the difference between the correct answer and incorrect answer for the corrupted run and save it.
        corrupted_logit_diff = (
            corrupted_logits[0, -1, answer_token_indices[prompt_id][0]] - corrupted_logits[0, -1, answer_token_indices[prompt_id][1]]
        ).save()

    # STEP 3: Patching runs, apply 'clean' model state at each layer and head,
    ioi_patching_results = []

    # Patching: Iterate through all the layers
    for layer_idx in range(len(llm.model.layers)):
        _ioi_patching_results = []
        # Iterate through all attention heads
        for head_idx in range(N_HEADS):
            # Patching corrupted run at given layer and token
            with tracer.invoke(prompts[corrupt_id]) as invoker:
                # Apply the patch from the clean hidden states to the corrupted hidden state for given layer and head.
                z_corrupt = llm.model.layers[layer_idx].self_attn.o_proj.input
                z_corrupt = einops.rearrange(z_corrupt, 'b s (nh dh) -> b s nh dh',nh=32)
                z_corrupt[:,:,head_idx,:] = z_hs[layer_idx,head_idx]
                z_corrupt = einops.rearrange(z_corrupt, 'b s nh dh -> b s (nh dh)', nh=32)
                llm.model.layers[layer_idx].self_attn.o_proj.input = z_corrupt

                patched_logits = llm.lm_head.output
                patched_logit_diff = (
                    patched_logits[0, -1, answer_token_indices[prompt_id][0]]
                    - patched_logits[0, -1, answer_token_indices[prompt_id][1]]
                )

                # Calculate the improvement in the correct token after patching.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.item().save())

        ioi_patching_results.append(_ioi_patching_results)
```

### Visualize Results

Let's use the same plotting function from earlier to visualize how patching the Llama-3.1-8B attention heads influenced model output during the IOI task.

```python
print(f"Clean logit difference: {clean_logit_diff.value:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff.value:.3f}") # why do I still need .value?

print(ioi_patching_results)

x_labels = [f"Head {i}" for i in range(N_HEADS)]

fig2 = plot_ioi_patching_results(llm, ioi_patching_results,x_labels,"Patching Llama Attention Heads on IOI task")
fig2.show()
```

---

# attribution_patching.ipynb

# Attribution Patching

📗 This tutorial is adapted from Neel Nanda’s [blog post](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching).

Activation patching is a method to determine how model components influence model computations (see our activation patching tutorial for more information). Although activation patching is a useful tool for circuit identification, it requires a separate forward pass through the model for each patched activation, making it time- and resource-intensive.

**Attribution patching** uses gradients to take a linear approximation to activation patching and can be done simultaneously in two forward and one backward pass, making it much more scalable to large models.

You can find a colab version of the tutorial [here](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/attribution_patching.ipynb) or Neel’s version [here](https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Attribution_Patching_Demo.ipynb).

Read more about an application of Attribution Patching in [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348). 📙

## Setup

If you are using Colab or haven't yet installed NNsight, install the package:
```
!pip install -U nnsight
```

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
```

Import libraries

```python
from IPython.display import clear_output
import einops
import torch
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "colab" if is_colab else "plotly_mimetype+notebook_connected+notebook"

from nnsight import LanguageModel
```

```python
import nnsight
print(nnsight.__version__)
```

## 1️⃣ Indirect Object Identification (IOI) Patching

Indirect object identification (IOI) is the ability to infer the correct indirect object in a sentence, allowing one to complete sentences like "John and Mary went to the shops, John gave a bag to" with the correct answer " Mary". Understanding how language models like GPT-2 perform linguistic tasks like IOI helps us gain insights into their internal mechanisms and decision-making processes.

Here, we apply the [IOI task](https://arxiv.org/abs/2211.00593) to explore how GPT-2 small is performing IOI with attribution patching.

*📚 Note: For more detail on the IOI task, check out the [ARENA walkthrough](https://arena3-chapter1-transformer-interp.streamlit.app/[1.4.1]_Indirect_Object_Identification).*

```python
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
clear_output()
print(model)
```

Looking at the model architecture, we can see there are 12 layers, each with 12 GPT-2 Blocks. We will use attribution patching to approximate the contribution of each layer and each attention head for the IOI task.

We next define 8 IOI prompts, with each prompt having one related corrupted prompt variation (i.e., the indirect object is swapped out).

```python
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

# Answers are each formatted as (correct, incorrect):
answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]

# Tokenize clean and corrupted inputs:
clean_tokens = model.tokenizer(prompts, return_tensors="pt")["input_ids"]
# The associated corrupted input is the prompt after the current clean prompt
# for even indices, or the prompt prior to the current clean prompt for odd indices
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

# Tokenize answers for each prompt:
answer_token_indices = torch.tensor(
    [
        [model.tokenizer(answers[i][j])["input_ids"][0] for j in range(2)]
        for i in range(len(answers))
    ]
)

```

Next, we create a function to calculate the mean logit difference for the correct vs incorrect answer tokens.

```python
def get_logit_diff(logits, answer_token_indices=answer_token_indices):
    logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    return (correct_logits - incorrect_logits).mean()
```

We then calculate the logit difference for both the clean and the corrupted baselines.

```python
clean_logits = model.trace(clean_tokens, trace=False).logits.cpu()
corrupted_logits = model.trace(corrupted_tokens, trace=False).logits.cpu()

CLEAN_BASELINE = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {CLEAN_BASELINE:.4f}")

CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")
```

Now let's define an `ioi_metric` function to evaluate patched IOI changes normalized to our clean and corruped baselines.

```python
def ioi_metric(
    logits,
    answer_token_indices=answer_token_indices,
):
    return (get_logit_diff(logits, answer_token_indices) - CORRUPTED_BASELINE) / (
        CLEAN_BASELINE - CORRUPTED_BASELINE
    )

print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
```

## 2️⃣ Attribution Patching Over Components

Attribution patching is a technique that uses gradients to take a linear approximation to activation patching. The key assumption is that the corrupted run is a locally linear function of its activations.

We thus take the gradient of the patch metric (`ioi_metric`) with respect to its activations, where we consider a patch of activations to be applying `corrupted_x` to `corrupted_x + (clean_x - corrupted_x)`. Then, we compute the patch metric's change: `(corrupted_grad_x * (clean_x - corrupted_x)).sum()`. All we need to do is take a backwards pass on the corrupted prompt with respect to the patch metric and cache all gradients with respect to the activations.

Let’s see how this breaks down in NNsight!

**A note on c_proj:** *Most HuggingFace models don’t have nice individual attention head representations to hook. Instead, we have the linear layer `c_proj` which implicitly combines the “projection per attention head” and the “sum over attention head” operations. See [this snippet](https://arena3-chapter1-transformer-interp.streamlit.app/~/+/[1.4.2]_Function_Vectors_&_Model_Steering) from ARENA for more information.*

TL;DR: We will use the input to `c_proj` for causal interventions on a particular attention head.

```python
clean_out = []
corrupted_out = []
corrupted_grads = []

with model.trace() as tracer:
# Using nnsight's tracer.invoke context, we can batch the clean and the
# corrupted runs into the same tracing context, allowing us to access
# information generated within each of these runs within one forward pass

    with tracer.invoke(clean_tokens) as invoker_clean:
        # Gather each layer's attention
        for layer in model.transformer.h:
            # Get clean attention output for this layer
            # across all attention heads
            attn_out = layer.attn.c_proj.input
            clean_out.append(attn_out.save())

    with tracer.invoke(corrupted_tokens) as invoker_corrupted:
        # Gather each layer's attention and gradients
        for layer in model.transformer.h:
            # Get corrupted attention output for this layer
            # across all attention heads
            attn_out = layer.attn.c_proj.input
            corrupted_out.append(attn_out.save())
            # save corrupted gradients for attribution patching
            corrupted_grads.append(attn_out.grad.save())

        # Let's get the logits for the model's output
        # for the corrupted run
        logits = model.lm_head.output.save()

        # Our metric uses tensors saved on cpu, so we
        # need to move the logits to cpu.
        value = ioi_metric(logits.cpu())

        # We also need to run a backwards pass to
        # update gradient values
        value.backward()
```

Next, for a given activation we compute `(corrupted_grad_act * (clean_act - corrupted_act)).sum()`. We use `einops.reduce` to rearrange and sum activations over the correct dimension. In this case, we want to estimate the effect of specific attention heads, so we sum over heads rather than token position.

```python
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
):

    residual_attr = einops.reduce(
        corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
        "batch (head dim) -> head",
        "sum",
        head = 12,
        dim = 64,
    )

    patching_results.append(
        residual_attr.detach().cpu().numpy()
    )
```

```python
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Attention Heads",
    labels={"x": "Head", "y": "Layer","color":"Norm. Logit Diff"},

)

fig.show()
```

Here, we see that the early layer attention heads may not be important for IOI.

## 3️⃣ Attribution Patching Over Position

One benefit of attribution patching is efficiency. Activation patching requires a separate forward pass per activation patched while every attribution patch can be done simultaneously in two forward passes and one backward pass. Attribution patching makes patching much more scalable to large models and can serve as a useful heuristic to find the interesting activations to patch.

In practice, whie this approximation is decent when patching in “small” activations like head outputs, performance decreases significantly when patching in “big” activations like those found in the residual stream.

Using the same outputs we cached above, we can get the individual contributions at each token position simply by summing across token positions. Although this is messy, it's a quick approximation of the attention mechanism's contribution across token position.

*Note: in our specific case here, patching across positions does NOT reflect the entire residual stream, just the post-attention output (i.e., excludes MLPs).*

```python
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
):

    residual_attr = einops.reduce(
        corrupted_grad.value * (clean.value - corrupted.value),
        "batch pos dim -> pos",
        "sum",
    )

    patching_results.append(
        residual_attr.detach().cpu().numpy()
    )
```

```python
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Token Position",
    labels={"x": "Token Position", "y": "Layer","color":"Norm. Logit Diff"},

)

fig.show()
```

This result looks similar to our previous result using activation patching but is much less precise, as expected!

# Remote Attribution Patching

Now that we know how to run an attribution patching experiment in `nnsight`, let's go over how you can use NDIF's publicly-hosted models to further scale your research!

We're going to run the same experiment, but now using Llama 3.1 8B. Completing this section of the tutorial will require you to [configure NNsight for remote execution](https://nnsight.net/notebooks/features/remote_execution/) if you haven't already.

## Remote Setup

```python
from nnsight import CONFIG

if is_colab:
    # include your HuggingFace Token and NNsight API key on Colab secrets
    from google.colab import userdata
    NDIF_API = userdata.get('NDIF_API')
    HF_TOKEN = userdata.get('HF_TOKEN')

    CONFIG.set_default_api_key(NDIF_API)
    !huggingface-cli login -token HF_TOKEN

clear_output()
```

```python
import torch
import nnsight
from nnsight import LanguageModel
```

Next, let's load the Llama 3.1 8B model, once again using NNsight's `LanguageModel` wrapper. Because we'll be running the model on NDIF's remote servers, no need to specify a `device_map`!

```python
# Load model
llm = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
print(llm)
```

## IOI Task Setup

We've already defined some prompts in the above tutorial, but we'll have to re-tokenize them for Llama 8B.

```python
# Tokenize clean and corrupted inputs:
clean_tokens = llm.tokenizer(prompts, return_tensors="pt")["input_ids"]
# The associated corrupted input is the prompt after the current clean prompt
# for even indices, or the prompt prior to the current clean prompt for odd indices
corrupted_tokens = clean_tokens[
    [(i + 1 if i % 2 == 0 else i - 1) for i in range(len(clean_tokens))]
]

# Tokenize answers for each prompt:
answer_token_indices = torch.tensor(
    [
        [llm.tokenizer(answers[i][j])["input_ids"][1] for j in range(2)]
        for i in range(len(answers))
    ]
)
```

Next, we'll establish clean & corrupted baselines for our IOI metric, using the model's clean and corrupted logits and the `get_logit_diff` function defined earlier.

```python
clean_logits = llm.trace(clean_tokens, trace=False, remote=True)
corrupted_logits = llm.trace(corrupted_tokens, trace=False, remote=True)

clean_logits = clean_logits['logits']
corrupted_logits = corrupted_logits['logits']

CLEAN_BASELINE = get_logit_diff(clean_logits, answer_token_indices).item()
print(f"\n\nClean logit diff: {CLEAN_BASELINE:.4f}")

CORRUPTED_BASELINE = get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {CORRUPTED_BASELINE:.4f}")
```

We've also already defined our `ioi_metric` function. Let's plug in our logit values.

```python
print(f"Clean Baseline is 1: {ioi_metric(clean_logits).item():.4f}")
print(f"Corrupted Baseline is 0: {ioi_metric(corrupted_logits).item():.4f}")
```

## Remote Attribution Patching

Great! We have some baselines. Now, let's run the attribution patching pipeline on Llama 8B. We can't copy the code exactly, because Llama 8B has a different model structure than GPT-2, but we're following the same steps: a clean run and a corrupted run as invokes during one tracing context.

```python
clean_out = []
corrupted_out = []
corrupted_grads = []

with llm.trace(remote = True) as tracer:
# Using nnsight's tracer.invoke context, we can batch the clean and the
# corrupted runs into the same tracing context, allowing us to access
# information generated within each of these runs within one forward pass

    with tracer.invoke(clean_tokens) as invoker_clean:
      # need to set requires grad to true for remote
        llm.model.layers[0].self_attn.o_proj.input.requires_grad = True
        # Gather each layer's attention
        for layer in llm.model.layers:
            # Get clean attention output for this layer
            # across all attention heads
            attn_out = layer.self_attn.o_proj.input
            clean_out.append(attn_out.save())

    with tracer.invoke(corrupted_tokens) as invoker_corrupted:
        # Gather each layer's attention and gradients
        for layer in llm.model.layers:
            # Get corrupted attention output for this layer
            # across all attention heads
            attn_out = layer.self_attn.o_proj.input
            corrupted_out.append(attn_out.save())
            # save corrupted gradients for attribution patching
            corrupted_grads.append(attn_out.grad.save())

        # Let's get the logits for the model's output
        # for the corrupted run
        logits = llm.lm_head.output.save()

        # Our IOI metric uses tensors saved on cpu, so we
        # need to move the logits to cpu.
        value = ioi_metric(logits.cpu())

        # We also need to run a backwards pass to
        # update gradient values
        value.backward()
```

Awesome! Let's take a look at attention head contributions across layers.

```python
# format data for plotting across attention heads
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
):

    residual_attr = einops.reduce(
        corrupted_grad.value[:,-1,:] * (clean.value[:,-1,:] - corrupted.value[:,-1,:]),
        "batch (head dim) -> head",
        "sum",
        head = 32,
        dim = 128,
    )

    patching_results.append(
        (residual_attr.float()).detach().numpy()
    )
```

```python
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Attention Heads",
    labels={"x": "Head", "y": "Layer","color":"Norm. Logit Diff"},

)

fig.show()
```

Next, let's check out the contribution of the residual stream over token position across layers.

```python
# format data for plotting across input tokens
patching_results = []

for corrupted_grad, corrupted, clean, layer in zip(
    corrupted_grads, corrupted_out, clean_out, range(len(clean_out))
):

    residual_attr = einops.reduce(
        corrupted_grad.value * (clean.value - corrupted.value),
        "batch pos dim -> pos",
        "sum",
    )

    patching_results.append(
        (residual_attr.detach().cpu().float()).numpy()
    )
```

```python
fig = px.imshow(
    patching_results,
    color_continuous_scale="RdBu",
    color_continuous_midpoint=0.0,
    title="Attribution Patching Over Token Position",
    labels={"x": "Token Position", "y": "Layer","color":"Norm. Logit Diff"},

)

fig.show()
```

Great! We've now successfully performed an attribution patching experiment on GPT-2 and Llama 8b.

---

# boundless_DAS.ipynb


---

# conditionals.ipynb

# Conditional Interventions

Interventions can also be made conditional.

Inside the tracing context, we can specify a conditional context:

```
with tracer.cond(Boolean):
  pass
```

This context will only execute its contained interventions if the specified condition is met. Let's try an example!

```python
import torch
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace("The Eiffel Tower is in the city of") as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  with tracer.cond(rand_int % 2 == 0):
    tracer.log("Random Integer ", rand_int, " is Even")

  with tracer.cond(rand_int % 2 == 1):
    tracer.log("Random Integer ", rand_int, " is Odd")
```

In the example above, we have two conditional contexts with mutually exclusive conditions, just like a usual `If`-`Else` statement.

Conditional contexts can also be nested, if we want our interventions to depend on more than one condition at a time.

```python
with model.trace("The Eiffel Tower is in the city of") as tracer:

  non_rand_int = 8

  with tracer.cond(non_rand_int > 0):
    with tracer.cond(non_rand_int % 2 == 0):
      tracer.log("Rand Int ", non_rand_int, " is Positive and Even")
```

`nnsight 0.4` introduces support for native Python `if` statements within the tracing context! Simply create an `if` statement within a trace, and it should perform as `tracer.cond()`.

```python
with model.trace("The Eiffel Tower is in the city of") as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")

  if rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

## Considerations
If you would like to turn off NNsight's support of native `if` statements, you can apply the following changes to `nnsight.CONFIG`

This will not affect any of NNsight's `tracer.cond()` functionality.

```python
# Turn off support if/for statements within tracing context.
import nnsight

nnsight.CONFIG.APP.CONTROL_FLOW_HANDLING = False
nnsight.CONFIG.save()
```

---

# cross_prompt.ipynb

# Cross-Prompt Intervention

Intervention operations work cross prompt! Use two invocations within the same generation block and operations can work between them.

In this case, we grab the token embeddings coming from the first prompt, "Madison square garden is located in the city of New" and replace the embeddings of the second prompt with them.

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')
```

```python
with model.generate(max_new_tokens=3) as tracer:

    with tracer.invoke("Madison square garden is located in the city of New") as invoker:

        embeddings = model.transformer.wte.output
        original = model.generator.output.save()

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _") as invoker:

        model.transformer.wte.output = embeddings
        intervened = model.generator.output.save()

print(model.tokenizer.batch_decode(original))
print(model.tokenizer.batch_decode(intervened))
```

We also could have entered a pre-saved embedding tensor as shown here:

```python
with model.generate("Madison square garden is located in the city of New", max_new_tokens=3) as tracer:

    embeddings = model.transformer.wte.output.save()
    original = model.generator.output.save()

print(model.tokenizer.batch_decode(original))

with model.generate("_ _ _ _ _ _ _ _ _ _", max_new_tokens=3) as tracer:

    model.transformer.wte.output = embeddings
    intervened = model.generator.output.save()

print(model.tokenizer.batch_decode(intervened))
```

---

# custom_functions.ipynb

# Custom Functions

Everything within the tracing context operates on the intervention graph. Therefore for `nnsight` to trace a  function it must also be a part of the intervention graph.

Out-of-the-box `nnsight` supports Pytorch functions and methods, all operators, as well the `einops` library. We don't need to do anything special to use them.

For custom functions we can use `nnsight.apply()` to add them to the intervention graph:

```python
import nnsight
from nnsight import LanguageModel
import torch

model = LanguageModel('openai-community/gpt2', device_map='auto')

# We define a simple custom function that sums all the elements of a tensor
def tensor_sum(tensor):
    flat = tensor.flatten()
    total = 0
    for element in flat:
        total += element.item()

    return torch.tensor(total)

with model.trace("The Eiffel Tower is in the city of") as tracer:

    # Specify the function name and its arguments (in a coma-separated form) to add to the intervention graph
    custom_sum = nnsight.apply(tensor_sum, model.transformer.h[0].output[0]).save()
    sum = model.transformer.h[0].output[0].sum().save()

print("\nPyTorch sum: ", sum)
print("Our sum: ", custom_sum)
```

`nnsight.apply()` executes the function it wraps and returns its output as a Proxy object. We can then use this Proxy object as we would any other.

The applications of `nnsight.apply` are wide. It can be used to wrap any custom function or functions from libraries that `nnsight` does not support out-of-the-box.

---

# dict_learning.ipynb

# Dictionary Learning

📗 You can find an interactive Colab version of this tutorial [here](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/dict_learning.ipynb).

## Polysemanticity

The field of mechanistic interpretability focuses on understanding individual components of neural networks. However, many neurons in these networks respond to multiple (seemingly unrelated) inputs – a phenomenon called *polysemanticity*. That is, a single neuron might separately respond to images of car tires **and** rubber ducks.

Although polysemanticity may help networks fit as many features as possible into a given parameter space, it makes it more difficult for humans to interpret the network's actions. There are a few strategies for finding monosemantic features, but in this tutorial we will explore the use of sparse autoencoders. If you are interested in learning more, this idea is explored by Anthropic in [*Towards Monosemanticity*](https://transformer-circuits.pub/2023/monosemantic-features) and [*Scaling Monosemanticity*](https://transformer-circuits.pub/2024/scaling-monosemanticity/).

## Sparse Autoencoders

Sparse autoencoders (SAEs) are algorithms that can extract learned features from a trained model. SAEs are a form of dictionary learning algorithms, which find a sparse representation of input data in a high-dimensional space. These features can serve as a more focused and monosemantic unit of analysis than the model's individual neurons, helping address polysemanticity and enabling a clearer and more interpretable understanding of model behavior.

📚 This tutorial is adapted from work by Samuel Marks and Aaron Mueller (See their [GitHub Repository](https://github.com/saprmarks/dictionary_learning) and [Alignment Forum post](https://www.alignmentforum.org/posts/AaoWLcmpY3LKvtdyq/some-open-source-dictionaries-and-dictionary-learning)). They created this repository as a resource for dictionary learning via sparse autoencoders on neural network activations, using Anthropic's approach detailed [here](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder).

Here, we will use one of their pre-trained autoencoders to explore how it creates an easily-interpretable monosemantic relationship between tokens and feature activation.

# Setup

Install NNsight & Dictionary Learning libraries

```python
from IPython.display import clear_output
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
    !git clone https://github.com/saprmarks/dictionary_learning
    %cd dictionary_learning
    !pip install -r requirements.txt
clear_output()
```

```python
from nnsight import LanguageModel
from dictionary_learning.dictionary import AutoEncoder
import torch
```

```python
# Load pretrained autoencoder
!./pretrained_dictionary_downloader.sh
clear_output()

weights_path = "./dictionaries/pythia-70m-deduped/mlp_out_layer0/10_32768/ae.pt"
activation_dim = 512 # dimension of the NN's activations to be autoencoded
dictionary_size = 64 * activation_dim # number of features in the dictionary

ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(torch.load(weights_path,weights_only=True))
ae.cuda()
```

# Apply SAE

```python
model = LanguageModel("EleutherAI/pythia-70m-deduped", device_map="auto")
tokenizer = model.tokenizer

prompt = """
Call me Ishmael. Some years ago--never mind how long precisely--having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world. It is a way I have of driving off the spleen and regulating the circulation. Whenever I find myself growing grim about the mouth; whenever it is a damp, drizzly November in my soul; whenever I find myself involuntarily pausing before coffin warehouses, and bringing up the rear of every funeral I meet; and especially whenever my hypos get such an upper hand of me, that it requires a strong moral principle to prevent me from deliberately stepping into the street, and methodically knocking people's hats off--then, I account it high time to get to sea as soon as I can.
"""

# Extract layer 0 MLP output from base model
with model.trace(prompt) as tracer:
    mlp_0 = model.gpt_neox.layers[0].mlp.output.save()

# Use SAE to get features from activations
features = ae.encode(mlp_0)
```

```python
# Find top features using the autoencoder
summed_activations = features.abs().sum(dim=1) # Sort by max activations
top_activations_indices = summed_activations.topk(20).indices # Get indices of top 20

compounded = []
for i in top_activations_indices[0]:
    compounded.append(features[:,:,i.item()].cpu()[0])

compounded = torch.stack(compounded, dim=0)
```

## Visualization

### With Autoencoder

Now let's take a look at each of the top 20 most active features and what they respond to in our prompt. Note that each feature only responds to one token, making these features highly interpretable!

```python
from circuitsvis.tokens import colored_tokens_multi

tokens = tokenizer.encode(prompt)
str_tokens = [tokenizer.decode(t) for t in tokens]

# Visualize activations for top 20 most prominent features
colored_tokens_multi(str_tokens, compounded.T)
```

### Without Autoencoder (optional comparison)
Without the autoencoder, the top neurons are active (or negatively associated) for many tokens, demonstrating how individual neurons can be difficult to interpret.

```python
# Find top neurons using the MLP output
summed_activations_or = mlp_0.abs().sum(dim=1) # Sort by max activations
top_activations_indices_or = summed_activations_or.topk(20).indices # Get indices of top 20

compounded_orig = []
for i in top_activations_indices_or[0]:
    compounded_orig.append(mlp_0[:,:,i.item()].cpu()[0])

compounded_orig = torch.stack(compounded_orig, dim=0)
```

```python
from circuitsvis.tokens import colored_tokens_multi

tokens = tokenizer.encode(prompt)
str_tokens = [tokenizer.decode(t) for t in tokens]

# Visualize original activations for top 20 most prominent neurons
colored_tokens_multi(str_tokens, compounded_orig.T)
```

---

# diffusion_lens.ipynb

# Diffusion Lens
## Introduction

🔎 Diffusion Lens is a technique to observe the inner computations of Diffusion Models, developed by Michael Toker in his paper, *Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines* ([Project Website](https://tokeron.github.io/DiffusionLensWeb/), [ACL Paper](https://aclanthology.org/2024.acl-long.524/)).

> **Colab: [exercises](https://colab.research.google.com/github/ndif-team/nnsight/blob/docs/docs/source/notebooks/tutorials/diffusion_lens.ipynb)**

Diffusion models produce images from text by first encoding text into numerical embeddings, which then guide image generation through a diffusion denoising process. Text encoder models can be trained along with the diffusion model, or models may use pretrained text encoders like CLIP.

Diffusion Lens works by generating images from the text encoder's intermediate representations during text embedding, allowing us to visualize how the model  encodes text as its computations move throughout its layers. The original Diffusion Lens paper revealed some fascinating insights into the text encoding process, finding differences in encoding between types of text encoders and the process of encoding different complexities of prompts. For instance, the authors observed that text encoders tend to embed common knowledge at earlier layers than uncommon knowledge. Another key finding was that different text encoders can encode the same prompt in different orders. For compound prompts with two nouns, they found that T5 and CLIP text encoders approached the encoding process differently.

![Compound Prompt Example](https://tokeron.github.io/DiffusionLensWeb/static/images/encoders.png)
**Text encoders differ in computation process (Toker et al.):** Diffusion models prompted with a compound prompt tend to represent concepts individually before combining them in the final embedding, and that the order of concepts introduced can vary between text encoding models (T5 vs CLIP). While processing a text prompt, T5  tended to represent the second noun first, while CLIP tended to represent the first noun first.

Let's test to see if this holds using NNsight and Diffusion Lens! We will use the prompt `"A few people are in the ocean splashing around"` to see if we can replicate the results from the paper. We will use Stable Diffusion 1.5 which uses a CLIP encoder and Deep Floyd which uses a T5 encoder. We will also explore the behavior of a few other models that weren't investigated in the paper.

## Setup

If you are running in Colab, install NNsight and ensure you are connected to GPU. NOTE: Colab built-in T4 GPUs will only have enough GPU RAM to run one model at a time. You will need to disconnect and restart the session to run multiple models. After restarting the session, you can rerun this setup section and then skip ahead to the model that you'd like to run.

```python
from IPython.display import clear_output
import torch
try:
    import google.colab
    is_colab = True
    if torch.cuda.is_available():
        print("GPU is connected")
    else:
        print("GPU is not connected: Please restart session with a GPU")

except ImportError:
    is_colab = False

if is_colab:
    !pip install --no-deps nnsight
    !pip install msgspec python-socketio[client]
    !pip install ftfy

clear_output()
```

Let's do our imports. We will be using the `DiffusionModel` class of NNsight for this exercise.

```python
from nnsight.modeling.diffusion import DiffusionModel
import matplotlib.pyplot as plt
from math import ceil, sqrt
import PIL
import torch
```

## Stable Diffusion 1.5 (CLIP encoder)

### Load Model

We will start with the Stable Diffusion 1.5 model, which uses the CLIP text encoder. We're going to apply the diffusion lens technique to visualize how CLIP is processing the prompt across its layers.

Let's instantiate the model and define some parameters for the experiment, including our prompt. We can use NNsight's `DiffusionModel` wrapper to load in the model from HuggingFace, which we will send to the GPU using `dispatch = True`.

```python
model = DiffusionModel(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    dispatch=True
).to("cuda")
```

```python
SEED = 17 # random seed for image generation: play around with it and see if it changes results!
STEP_SIZE = 1 # number of steps between layers for our experiment
```

```python
prompt = "A few people are in the ocean splashing around"
```

### Run Diffusion Lens

Now we have the model ready for our experiment.

Diffusion Lens works by processing each of the intermediate text encoder layer outputs through the final layer norm to visualize how the model is progressively refining the text embedding for the diffusion process.

Let's try implementing this in `nnsight`.

```python
layers = range(0, model.text_encoder.config.num_hidden_layers, STEP_SIZE)
images = []

for layer in layers:
    print(f"Generating Diffusion Lens for skip_layers {model.text_encoder.config.num_hidden_layers - layer - 1}")

    # We will use NNsight's .generate() method for image generation.
    # We're specifying our prompt and the random generation seed.
    with model.generate(
        prompt,
        seed=SEED
        ):

        # replace the final_layer_norm input with the text_encoder's output for the layer.
        hidden_state = model.text_encoder.text_model.encoder.layers[layer].output[0]
        model.text_encoder.text_model.final_layer_norm.input = hidden_state

        # Save the generated image and add it to our collection
        image = model.output.images[0].save()
        images.append(image)

if not isinstance(images[0], PIL.Image.Image):
    images = [image.value for image in images]
```

### Visualize Results

Great, now our Diffusion Lens experiment has finished running! Let's plot the images and see how the CLIP text encoder is processing the input across layers.

```python
# Calculate grid dimensions
num_images = len(images)
grid_size = ceil(sqrt(num_images))
fig, axes = plt.subplots(ceil(num_images / grid_size), grid_size, figsize=(15, 15))
axes = axes.flatten()

# Add a main title to the figure
fig.suptitle(f"SD1.5 Diffusion Lens - {prompt}", fontsize=16)

# Display images in a grid
for i, (layer, image) in enumerate(zip(layers, images)):
    if i < len(axes):
        axes[i].imshow(image.resize((256, 256)))
        axes[i].set_title(f"Layer {layer}")
        axes[i].axis('off')

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Cool! As reported in the diffusion lens paper, the CLIP encoder started by representing people (the first noun in the prompt) and then added water/ocean (the second noun in the prompt). Let's next try with the T5  text encoder to see if things change.

## Deep Floyd (T5 encoder)

*NOTE: If running on Colab T4 GPUs, you will need to disconnect and restart the session to load in this model. There isn't enough GPU RAM to load both models.*

Once you restart the session, rerun the "Setup" section and then you can skip ahead to this section to get the Deep Floyd results.

### Load Model

```python
model = DiffusionModel(
    "DeepFloyd/IF-I-L-v1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    dispatch=True
).to("cuda")

print(model)
```

```python
prompt = "A few people are in the ocean splashing around"
```

```python
SEED = 128998123
STEP_SIZE = 2
```

### Run Diffusion Lens

Now that we have Deep Floyd loaded, let's set up the diffusion lens experiment again. This code is pretty similar

```python
import ftfy
layers = range(0, model.text_encoder.config.num_hidden_layers - 1, STEP_SIZE)
images = []

for layer in layers:
    print(f"Generating Diffusion Lens for skip_layers {model.text_encoder.config.num_hidden_layers - layer}")
    with torch.no_grad():
        with model.generate(
            prompt,
            seed=SEED
        ):
            hidden_states = model.text_encoder.encoder.block[layer].output[0]
            model.text_encoder.encoder.final_layer_norm.input = hidden_states

            image = model.output.images[0].save()
            images.append(image)

if not isinstance(images[0], PIL.Image.Image):
    images = [image.value for image in images]
```

### Visualize Results

```python
# Calculate grid dimensions
num_images = len(images)
grid_size = ceil(sqrt(num_images))
fig, axes = plt.subplots(ceil(num_images / grid_size), grid_size, figsize=(15, 15))
axes = axes.flatten()

# Add a main title to the figure
fig.suptitle(f"Deep Floyd Diffusion Lens - {prompt}", fontsize=16)

# Display images in a grid
for i, (layer, image) in enumerate(zip(layers, images)):
    if i < len(axes):
        axes[i].imshow(image.resize((256, 256)))
        axes[i].set_title(f"Layer {layer}")
        axes[i].axis('off')

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Interesting! The T5 encoder first started with ocean (the second noun) and then added people (the first noun), exhibiting how CLIP and T5 differ in their encoding processes.

## BONUS: Stable Diffusion XL (Two CLIP encoders)

### Load Model

We will start with the Stable Diffusion XL model, which uses two CLIP encoders. Let's define some parameters for the experiment and instantiate the model.

```python
model = DiffusionModel(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    dispatch=True
).to("cuda")
```

```python
prompt = "A few people are in the ocean splashing around"
```

```python
ADD_LAYER_NORM = True # SDXL doesn't automatically use the layer norm, so we have some logic to add it in here
SEED = 17
NUM_INFERENCE_STEPS = 100
STEP_SIZE = 2
```

### Run Diffusion Lens

Great, we have the SDXL model ready for our experiment. SDXL is a little weird, so the diffusion lens code is a little more complicated. We need to mask the first text encoder to isolate the effects of the second text encoder. We also need to manually add in the layer norm, because SDXL doesn't include it automatically. Try setting `LAYER_NORM = False` to see the effects of this!

```python
# Defines the hidden states to embed, we skip the last layer because SDXL ignores it
layers = range(0, model.text_encoder_2.config.num_hidden_layers - 1, STEP_SIZE)
images = []

# Create an empty prompt input for the first text encoder
# This will be used to mask out the original text input, allowing us to isolate
# the effect of injecting hidden states from the second text encoder
mask_input = model.tokenizer(
    '',  # Empty string as we want to mask out the original text
    padding="max_length",
    max_length=model.tokenizer.model_max_length,
    truncation=True,
    return_overflowing_tokens=False,
    return_length=False,
    return_tensors="pt"
).to(model.device)

for layer in layers:
    print(f"Generating Diffusion Lens for skip_layers {model.text_encoder_2.config.num_hidden_layers - layer}")
    with model.generate(
        prompt,
        num_inference_steps=40,
        seed=SEED
        ):

        # Replace the input to the first text encoder with our empty mask
        # This effectively nullifies the contribution of the first text encoder
        model.text_encoder.input = mask_input['input_ids']

        if ADD_LAYER_NORM:

            hidden_state = model.text_encoder_2.text_model.encoder.layers[layer].output[0]

            # SDXL grabs the penultimate hidden state from the text encoder
            model.text_encoder_2.text_model.encoder.layers[-2].output[0][:] = model.text_encoder_2.text_model.final_layer_norm(hidden_state)[0][:]
        else:
            # SDXL grabs the penultimate hidden state from the text encoder
            model.text_encoder_2.text_model.encoder.layers[-2].output[0][:] = model.text_encoder_2.text_model.encoder.layers[layer].output[0][:]

        # Save the generated image and add it to our collection
        image = model.output.images[0].save()
        images.append(image)

if not isinstance(images[0], PIL.Image.Image):
    images = [image.value for image in images]

```

### Visualize Results

Great, now let's visualize this by plotting the image created from the processed intermediate layers.

```python
# Calculate grid dimensions
num_images = len(images)
grid_size = ceil(sqrt(num_images))
fig, axes = plt.subplots(ceil(num_images / grid_size), grid_size, figsize=(15, 15))
axes = axes.flatten()

# Add a main title to the figure
fig.suptitle(f"SDXL Diffusion Lens - {prompt}", fontsize=16)

# Display images in a grid
for i, (layer, image) in enumerate(zip(layers, images)):
    if i < len(axes):
        axes[i].imshow(image.resize((512, 512)))
        axes[i].set_title(f"Layer {layer}")
        axes[i].axis('off')

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

Interesting! SDXL starts by making the ocean, then the people. This is in contrast to the other CLIP encoder that represented the first noun in the prompt first. Try playing around with the settings to see if you can change how the encoder is operating.

## BONUS: FLUX Schnell (CLIP and T5 XXL encoders)

### Load Model

Let's try implementing diffusion lens on the FLUX Schnell Model, which uses both CLIP and T5 XXL encoders. We'll once again define some parameters and load in the model.

*NOTE: FLUX is too large for Google Colab T4s, so you will need to run this locally on your own GPU or use a paid Colab plan to run this section.*

```python
SEED = 17
NUM_INFERENCE_STEPS = 1
STEP_SIZE = 2
GUIDANCE_SCALE = 0.0
HEIGHT = 512
WIDTH = 512
```

```python
model = DiffusionModel(
    "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16, dispatch=True
).to('cuda')
```

Let's use the example prompt from FLUX to see how it is encoded with diffusion lens.

```python
prompt = "Penguin playing chess at a wooden table in a snowy landscape."
```

### Run Diffusion Lens

Let's run the Diffusion Lens experiment again. CLIP is the first text encoder, while T5 is the second. We're going to mask out the effects of the CLIP encoder to isolate the T5 encoder. Let's see if the pattern that T5 represents the prompt's nouns in reverse order holds.

```python
layers = range(0, model.text_encoder_2.config.num_hidden_layers - 1, STEP_SIZE)
images = []

mask_input = model.tokenizer(
    '',
    padding="max_length",
    max_length=model.tokenizer.model_max_length,
    truncation=True,
    return_overflowing_tokens=False,
    return_length=False,
    return_tensors="pt"
).to(model.device)

for layer in layers:
    print(f"Generating Diffusion Lens for skip_layers {model.text_encoder_2.config.num_hidden_layers - layer}")
    with torch.no_grad():
        with model.generate(
            prompt,
            guidance_scale=0.0,
            height=512,
            width=512,
            num_inference_steps=1,
            seed=17
        ):
            model.text_encoder.input = mask_input['input_ids']

            model.text_encoder_2.encoder.final_layer_norm.input = model.text_encoder_2.encoder.block[layer].output[0]

            image = model.output.images[0].save()
            images.append(image)

if not isinstance(images[0], PIL.Image.Image):
    images = [image.value for image in images]
```

### Visualize Results
Let's see how FLUX Schnell processed the penguin prompt.

```python
# Calculate grid dimensions
num_images = len(images)
grid_size = ceil(sqrt(num_images))
fig, axes = plt.subplots(ceil(num_images / grid_size), grid_size, figsize=(15, 15))
axes = axes.flatten()

# Add a main title to the figure
fig.suptitle(f"Flux Schnell Diffusion Lens - {prompt}", fontsize=16)

# Display images in a grid
for i, (layer, image) in enumerate(zip(layers, images)):
    if i < len(axes):
        axes[i].imshow(image)
        axes[i].set_title(f"Layer {layer}")
        axes[i].axis('off')

# Hide any unused subplots
for i in range(num_images, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
# Adjust layout to make room for the title
# plt.subplots_adjust(top=0.9)
plt.show()
```

Fascinating! FLUX Schnell creates the snowy landscape first, then the table, chess board, and finally the penguins. This supports our hypothesis that T5 models represent nouns in compound prompts in reverse order.

---

# early_stopping.ipynb

# Early Stopping

If we are only interested in a model's intermediate computations, we can halt a forward pass run at any module level, reducing runtime and conserving compute resources. One examples where this could be particularly useful would if we are working with SAEs - we can train an SAE on one layer and then stop the execution.

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace("The Eiffel Tower is in the city of"):
   l1_out = model.transformer.h[0].output.save()
   model.transformer.h[0].output.stop()

# get the output of the first layer and stop tracing
print("L1 - Output: ", l1_out)
```

Interventions within the tracing context do not necessarily execute in the order they are defined. Instead, their execution is tied to the module they are associated with.

As a result, if the forward pass is terminated early any interventions linked to modules beyond that point will be skipped, even if they were defined earlier in the context.

In the example below, the output of layer 2 _**cannot**_ be accessed since the model's execution was stopped at layer 1.

```python
with model.trace("The Eiffel Tower is in the city of"):
   l2_out = model.transformer.h[1].output.save()
   model.transformer.h[0].output.stop()

print("L2 - Output: ", l2_out)
```

---

# editing_tutorial.ipynb

# Editing

The edit module sets default nodes on the intervention graph to be executed on every future trace. Let's start by loading and dispatching a LanguageModel.

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

Editing is useful for attaching default modules to the graph such as LoRAs or SAEs. We declare a toy, passthrough SAE class below.

```python
import torch

# Create a simple torch module
class SAE(torch.nn.Module):
    def __init__(self):
        super(SAE, self).__init__()

    def forward(self, x):
        return x
```

To attach a module to a model's tree, simply set it as an attribute on a desired module. Note that edits must be of type `torch.nn.Module` in order to be attached to the tree.

To set a default edit on a model's intervention graph, create an `edit` context and declare operations as usual.

```python
# Create a reference to the l0 Envoy
submodule = model.transformer.h[0]
# Set the SAE as a property on .sae
submodule.sae = SAE()

# Declare an edit context like you would a trace
with model.edit(""):
    acts = submodule.output[0]
    submodule.sae(acts)
```

Calling the `.sae` attribute in future `trace` contexts will return the `l0` output as expected.

```python
with model.trace("Hello, world!"):
    acts = submodule.sae.output.save()

print(acts.shape)
```

You can also hook into submodules of attached modules. Let's edit the `SAE` class to include a passthrough `encoder` and `decoder`.

```python
class Coder(torch.nn.Module):
    def __init__(self):
        super(Coder, self).__init__()

    def forward(self, x):
        return x

class SAE(torch.nn.Module):
    def __init__(self):
        super(SAE, self).__init__()
        self.encoder = Coder()
        self.decoder = Coder()

    def forward(self, x):
        return self.decoder(
            self.encoder(x)
        )
```

We make the edit just as before, this time setting the `hook` kwarg to `True`. This tells NNsight that we'd like to call the `forward` method on the `SAE` module, passing inputs through all subhooks.

```python
# Create a reference to the l0 Envoy
submodule = model.transformer.h[0]
# Set the SAE as a property on .sae
submodule.sae = SAE()

# Declare an edit context like you would a trace
with model.edit(""):
    acts = submodule.output[0]
    submodule.sae(acts, hook=True)

# Now we can call .encoder and other submodules!
with model.trace("Hello, world!"):
    acts = submodule.sae.encoder.output.save()

print(acts.shape)
```

---

# function_vectors.ipynb

# Function Vectors
**ARENA Function Vectors & Model Steering Tutorial**

This tutorial is adapted from the ARENA program material and serves as a fantastic introduction to running experiments in NNsight and working with function vectors and model steering. Thanks to Callum McDougall for writing this comprehensive tutorial and for allowing us to adapt the tutorial for NNsight users, and thanks to Eric Todd for writing the original function vector paper!

> **ARENA: [Streamlit Page](https://arena-chapter1-transformer-interp.streamlit.app/22_📚_[1.4.2]_Function_Vectors_&_Model_Steering)**
>
> **Colab: [exercises](https://colab.research.google.com/github/ndif-team/nnsight/blob/docs/docs/source/notebooks/tutorials/function_vectors.ipynb) | [solutions](https://colab.research.google.com/github/ndif-team/nnsight/blob/docs/function_vectors_solutions.ipynb)**

You can collapse each section so only the headers are visible, by clicking the arrow symbol on the left hand side of the markdown header cells.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-14-2.png" width="350">

# Introduction

These exercises serve as an exploration of the following question: ***can we steer a model to produce different outputs / have a different behaviour, by intervening on the model's forward pass using vectors found by non gradient descent-based methods?***

The majority of the exercises focus on [function vectors](https://functions.baulab.info/): vectors which are extracted from forward passes on in-context learning (ICL) tasks, and added to the residual stream in order to trigger the execution of this task from a zero-shot prompt. The diagram below illustrates this.

<img src="https://functions.baulab.info/images/Paper/fv-demonstrations.png" width="650">

The exercises also take you through use of the `nnsight` library, which is designed to support this kind of work (and other interpretability research) on very large language models - i.e. larger than models like GPT2-Small which you might be used to at this point in the course.

The final set of exercises look at Alex Turner et al's work on [steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector), which is conceptually related but has different aims and methodologies.

## Content & Learning Objectives

### 1️⃣ Introduction to `nnsight`

In this section, you'll learn the basics of how to use the `nnsight` library: running forward passes on your model, and saving the internal states. You'll also learn some basics of HuggingFace models which translate over into `nnsight` models (e.g. tokenization, and how to work with model output).

> ##### Learning Objectives
>
> * Learn the basics of the `nnsight` library, and what it can be useful for
> * Learn some basics of HuggingFace models (e.g. tokenization, model output)
> * Use it to extract & visualise GPT-J-6B's internal activations

### 2️⃣ Task-encoding hidden states

We'll begin with the following question, posed by the Function Vectors paper:

> *When a transformer processes an ICL (in-context-learning) prompt with exemplars demonstrating task $T$, do any hidden states encode the task itself?*

We'll prove that the answer is yes, by constructing a vector $h$ from a set of ICL prompts for the **antonym task**, and intervening with our vector to make our model produce antonyms on zero-shot prompts.

This will require you to learn how to perform causal interventions with `nnsight`, not just save activations.

(Note - this section structurally follows section 2.1 of the function vectors paper).

> ##### Learning Objectives
>
> * Understand how `nnsight` can be used to perform causal interventions, and perform some yourself
> * Reproduce the "h-vector results" from the function vectors paper; that the residual stream does contain a vector which encodes the task and can induce task behaviour on zero-shot prompts

### 3️⃣ Function Vectors

In this section, we'll replicate the crux of the paper's results, by identifying a set of attention heads whose outputs have a large effect on the model's ICL performance, and showing we can patch with these vectors to induce task-solving behaviour on randomly shuffled prompts.

We'll also learn how to use `nnsight` for multi-token generation, and steer the model's behaviour. There exist exercises where you can try this out for different tasks, e.g. the Country-Capitals task, where you'll be able to steer the model to complete prompts like `"When you think of Netherlands, you usually think of"` by talking about Amsterdam.

(Note - this section structurally follows sections 2.2, 2.3 and some of section 3 from the function vectors paper).

> ##### Learning Objectives
>
> * Define a metric to measure the causal effect of each attention head on the correct performance of the in-context learning task
> * Understand how to rearrange activations in a model during an `nnsight` forward pass, to extract activations corresponding to a particular attention head
> * Learn how to use `nnsight` for multi-token generation

### 4️⃣ Steering Vectors in GPT2-XL

Here, we discuss a different but related set of research: Alex Turner's work on steering vectors. This also falls under the umbrella of "interventions in the residual stream using vectors found with forward pass (non-SGD) based methods in order to alter behaviour", but it has a different setup, objectives, and approach.

> ##### Learning Objectives
>
> * Understand the goals & main results from Alex Turner et al's work on steering vectors
> * Reproduce the changes in behaviour described in their initial post

### ☆ Bonus

Lastly, we discuss some possible extensions of function vectors & steering vectors work, which is currently an exciting area of development (e.g. with a paper on steering Llama 2-13b coming out as recently as December 2023).

## Setup code

```python
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import nnsight
except:
    %pip install openai>=1.56.2 nnsight einops jaxtyping plotly transformer_lens==2.11.0 git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python gradio typing-extensions
    %pip install --upgrade pydantic

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
```

```python
! pip install circuitsvis
! pip install plotly
! pip install jaxtyping
! pip install nnsight
```

```python
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import CONFIG, LanguageModel
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from torch import Tensor

# Hide some info logging messages from nnsight
logging.disable(sys.maxsize)

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part42_function_vectors_and_model_steering"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part42_function_vectors_and_model_steering.solutions as solutions
import part42_function_vectors_and_model_steering.tests as tests
from plotly_utils import imshow

MAIN = __name__ == "__main__"
```

# 1️⃣ Introduction to `nnsight`

> ##### Learning Objectives
>
> * Learn the basics of the `nnsight` library, and what it can be useful for
> * Learn some basics of HuggingFace models (e.g. tokenization, model output)
> * Use it to extract & visualise GPT-J-6B's internal activations

## Remote execution

We'll start by discussing [remote execution]((https://nnsight.net/notebooks/features/remote_execution/)) - the ability `nnsight` has to run models on an external server, which is one of the major benefits of the library as a research tool. This helps you bypass the memory & computational limits you might be faced with on your own machine. For remote execution to work, you need 2 things:

1. An API key from the NDIF login page, which you can request [here](https://login.ndif.us/)
2. The model you're working with being live - you can see all live models in the status page [here](https://nnsight.net/status/)

Note that the status page sometimes takes ~5 minutes to load all live models - click the dropdown below to see an example of what the status page should look like once the models have loaded. If you can't see the model you're looking for in this list, then you should set `REMOTE=False` for these exercises, or else make a request on the NDIF Discord to get the model live.

<details>
<summary>Example status page</summary>

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/ndif-status.png" width="650">

</details>

## Important syntax

Here, we'll discuss some important syntax for interacting with `nnsight` models. Since these models are extensions of HuggingFace models, some of this information (e.g. tokenization) applies to plain HuggingFace models as well as `nnsight` models, and some of it (e.g. forward passes) is specific to `nnsight`, i.e. it would work differently if you just had a standard HuggingFace model. Make sure to keep this distinction in mind, otherwise syntax can get confusing!

### Model config

Each model comes with a `model.config`, which contains lots of useful information about the model (e.g. number of heads and layers, size of hidden layers, etc.). You can access this with `model.config`. Run the code below to see this in action, and to define some useful variables for later.

```python
model = LanguageModel("EleutherAI/gpt-j-6b", device_map="auto", torch_dtype=t.bfloat16)
tokenizer = model.tokenizer

N_HEADS = model.config.n_head
N_LAYERS = model.config.n_layer
D_MODEL = model.config.n_embd
D_HEAD = D_MODEL // N_HEADS

print(f"Number of heads: {N_HEADS}")
print(f"Number of layers: {N_LAYERS}")
print(f"Model dimension: {D_MODEL}")
print(f"Head dimension: {D_HEAD}\n")

print("Entire config: ", model.config)
```

### Tokenizers

A model comes with a tokenizer, accessable with `model.tokenizer` (just like TransformerLens). Unlike TransformerLens, we won't be using utility functions like `model.to_str_tokens`, instead we'll be using the tokenizer directly. Some important functions for today's exercises are:

* `tokenizer` (i.e. just calling it on some input)
    * This takes in a string (or list of strings) and returns the tokenized version.
    * It will return a dictionary, always containing `input_ids` (i.e. the actual tokens) but also other things which are specific to the transformer model (e.g. `attention_mask` - see dropdown).
    * Other useful arguments for this function:
        * `return_tensors` - if this is `"pt"`, you'll get results returned as PyTorch tensors, rather than lists (which is the default).
        * `padding` - if True (default is False), the tokenizer can accept sequences of variable length. The shorter sequences get padded at the beginning (see dropdown below for more).
* `tokenizer.decode`
    * This takes in tokens, and returns the decoded string.
    * If the input is an integer, it returns the corresponding string. If the input is a list / 1D array of integers, it returns all those strings concatenated (which can sometimes not be what you want).
* `tokenizer.batch_decode`
    * Equivalent to `tokenizer.decode`, but it doesn't concatenate.
    * If the input is a list / 1D integer array, it returns a list of strings. If the input is 2D, it will concatenate within each list.
* `tokenizer.tokenize`
    * Takes in a string, and returns a list of strings.

Run the code below to see some examples of these functions in action.

```python
# Calling tokenizer returns a dictionary, containing input ids & other data.
# If returned as a tensor, then by default it will have a batch dimension.
print(tokenizer("This must be Thursday", return_tensors="pt"))

# Decoding a list of integers, into a concatenated string.
print(tokenizer.decode([40, 1239, 714, 651, 262, 8181, 286, 48971, 12545, 13]))

# Using batch decode, on both 1D and 2D input.
print(tokenizer.batch_decode([4711, 2456, 481, 307, 6626, 510]))
print(tokenizer.batch_decode([[1212, 6827, 481, 307, 1978], [2396, 481, 428, 530]]))

# Split sentence into tokens (note we see the special Ġ character in place of prepended spaces).
print(tokenizer.tokenize("This sentence will be tokenized"))
```

<details>
<summary>Note on <code>attention_mask</code> (optional)</summary>

`attention_mask`, which is a series of 1s and 0s. We mask attention at all 0-positions (i.e. we don't allow these tokens to be attended to). This is useful when you have to do padding. For example:

```python
model.tokenizer(["Hello world", "Hello"], return_tensors="pt", padding=True)
```

will return:

```python
{
    'attention_mask': tensor([[1, 1], [0, 1]]),
    'input_ids': tensor([[15496,   995], [50256, 15496]])
}
```

We can see how the shorter sequence has been padded at the beginning, and attention to this token will be masked.

</details>

### Model outputs

At a high level, there are 2 ways to run our model: using the `trace` method (a single forward pass) and the `generate` method (generating multiple tokens). We'll focus on `trace` for now, and we'll discuss `generate` when it comes to multi-token generation later.

The default behaviour of forward passes in normal HuggingFace models is to return an object containing logits (and optionally a bunch of other things). The default behaviour of `trace` in `nnsight` is to not return anything, because anything that we choose to return is explicitly returned inside the context manager.

Below is the simplest example of code to run the model (and also access the internal states of the model). Run it and look at the output, then read the explanation below. Remember to obtain and set an API key first if you're using remote execution!

```python
REMOTE = True

if IN_COLAB:
    # include your HuggingFace Token and NNsight API key on Colab secrets
    from google.colab import userdata
    NDI  F_API = userdata.get('NDIF_API')
    CONFIG.set_default_api_key(NDIF_API)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt, remote=REMOTE):
    # Save the model's hidden states
    hidden_states = model.transformer.h[-1].output[0].save()

    # Save the model's logit output
    logits = model.lm_head.output[0, -1].save()

# Get the model's logit output, and it's next token prediction
print(f"logits.shape = {logits.shape} = (vocab_size,)")
print("Predicted token ID =", predicted_token_id := logits.argmax().item())
print(f"Predicted token = {tokenizer.decode(predicted_token_id)!r}")

# Print the shape of the model's residual stream
print(f"\nresid.shape = {hidden_states.shape} = (batch_size, seq_len, d_model)")
```

Lets go over this piece by piece.

**First, we create a context block** by calling `.trace(...)` on the model object. This denotes that we wish to generate tokens given some prompts.

```python
with model.trace(prompt, remote=REMOTE):
```

By default, running this will cause your model to be loaded & run locally, but by passing `remote=REMOTE`, it causes the model to be run on the server instead. This is very useful when working with models too large to fit on your machine (or even models which can fit on your machine, but run slowly due to their size, however if you're running this material on a sufficiently large GPU, you may prefer to set `REMOTE=False`).  The input argument can take a variety of formats: strings, lists of tokens, tensors of tokens, etc. Here, we've just used a string `prompt`.

The most interesting part of `nnsight` is the ability to access the model's internal states (like you might already have done with TransformerLens). Let's now see how this works!

```python
hidden_states = model.transformer.h[-1].output[0]
```

On this line we're saying: within our forward pass, access the last layer of the transformer `model.transformer.h[-1]`, access this layer's output `.output` (which is a tuple of tensors), index the first tensor in this tuple `.output[0]`.

Let's break down this line in a bit more detail:

* `model.transformer.h[-1]` is a module in our transformer.
    * If you `print(model)`, you'll see that it consists of `transformer` and `lm_head` (for "language modelling head"). The `transformer` module is made up of embeddings & dropout, a series of layers (called `.h`, for "hidden states"), and a final layernorm. So indexing `.h[-1]` gives you the final layer.
    * Note - it's often useful to visit the documentation page for whatever model you're working on, e.g. you can find GPT-J [here](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html). Not all models will have a nice uniform standardized architecture like you might be used to in TransformerLens!
* `.output[0]` gives you this module's output, as a **proxy**.
    * The output of a module is often a tuple (again, you can see on the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) what the output of each module is). In this case, it's a tuple of 2 tensors, the first of which is the actual layer output (the thing we want).
    * Doing operations on a proxy still returns a proxy - this is why we can index into the `output` proxy tuple and get a proxy tensor!

<details>
<summary>Optional exercise - we mentioned that <code>.output</code> returns a tuple of 2 tensors. Can you use the <a href="https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html">documentation page</a> what the second tensor in this tuple is?</summary>

The second output is also a tuple of tensors, of length 2. In the GPT-J source code, they are called `present`. They represent the keys and values which were calculated in this forward pass (as opposed to those that were calculated in an earlier forward pass, and cached by the model). Since we're only generating one new token, these are just the full keys and values.

</details>

<br>

The next command:

```python
logits = model.lm_head.output[0, -1]
```

can be understood in a very similar way. The only difference is that we're accessing the output of `lm_head`, the language modelling head (i.e. the unembedding at the very end), and the output is just a tensor of shape `(batch, seq, d_vocab)` rather than a tuple of tensors. Again, see the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) for this.

If you've worked with Hugging Face models then you might be used to getting logits directly from the model output, but here we generally extract logits from the model internals just like any other activation because this allows us to **control exactly what we return.** If we return lots of very large tensors, this can take quite a while to download from the server (remember that `d_vocab` is often very large for transformers, i.e. around 50k). See the "which objects to save" section below for more discussion on this.

### Output vs input

You can also extract a module's input using `.input` or `.inputs`. If a module's forward method is called as `module.forward(*args, **kwargs)` then `.inputs` returns a tuple of `(tuple_of_args, dict_of_kwargs)`. Alternatively, `.input` is an alias for `.inputs[0][0]`, in other words it returns the first arg from the module's forward method (which is usually the tensor we want).

Remember that if you're not sure then you can debug with `print(module.input.shape)` - even if `.inputs` is a tuple of inputs, this will work to recursively print the shape of all the tensors in the tuple, rather than causing an error.

### Which objects to save

Note that we saved `logits` above, which is a vector of length 50k. In general, it's best to save as small an object as possible, because this reduces the size of object you'll have to download from the server. For example, if you only want the next token completions, just argmax the logits and then save the result! All basic tensor operations can be performed within your context manager.

## Scan & Validate

A really cool feature in nnsight is the scan & validate mode, which allows you to efficiently debug without getting long uninterpretable error messages. For example, consider the code below, which tries to zero ablate one of the model's output tensors. Can you figure out what's wrong with it before running it?

```python
seq_len = len(model.tokenizer.encode(prompt))

try:
    with model.trace(prompt, remote=REMOTE):
        original_output = model.transformer.h[-1].output[0].clone()
        model.transformer.h[-1].output[0][:, seq_len] = 0
        modified_output = model.transformer.h[-1].output[0].save()

except Exception as e:
    print(f"Uninformative error message:\n  {e.__class__.__name__}: {e}")
```

If you guessed "we're indexing a tensor along a dimension of size `seq_len` with index `seq_len` which is an indexing error, you'd be correct! But the error message we get is pretty opaque. This is because of the way the objects in nnsight work: they're not tensors, they're tensor proxies, and can behave in funny ways sometimes.

If we want to debug, we should instead pass `scan=True` and `validate=True` into our `model.trace` call. `scan=True` means we run "fake inputs" through the model which incur no memory costs, and so can be done very quickly and cheaply to detect errors. `validate=True` will run tests during our forward pass that make our error messages more informative. When we use both, we get fast no-memory-cost operations with interpretable error messages!

```python
try:
    with model.trace(prompt, remote=REMOTE, scan=True, validate=True):
        original_output = model.transformer.h[-1].output[0].clone()
        print(f"{model.transformer.h[-1].output.shape=}\n")
        model.transformer.h[-1].output[0][:, seq_len] = 0
        modified_output = model.transformer.h[-1].output[0].save()

except Exception as e:
    print(f"Informative error message:\n  {e.__class__.__name__}: {e}")
```

It's possible to use `validate` without using `scan` (e.g. if you have any `assert proxy.shape == ...` then you must use `validate=True`), although we generally recommend using both when debugging, and then neither when you're finished debugging.

Also note that (as the example above shows) it's useful to use `scan=True, validate=True` when printing tensor shapes, at the initial exploration phase, if you're not exactly sure what the shape of a particular input / output will be. Even if your proxy objects are tuples of tensors, you can still call `.shape`, and it will return a tuple of the shapes of each tensor in the proxy!

## Putting this into practice

### Exercise - visualize attention heads

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

We just covered a lot of content, so lets put it into practice. Your first task is to extract the attention patterns from the zeroth layer of the transformer, and visualize them using circuitsvis. As a reminder, the syntax for circuitsvis is:

```python
cv.attention.attention_patterns(
    tokens=tokens,
    attention=attention,
)
```

where `tokens` is a list of strings, and `attention` is a tensor of shape `(num_heads, num_tokens, num_tokens)`.

If you're stuck, [here's a link](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) to the source code for GPT-J. Look for how the attention patterns are calculated, within the `GPTJAttention` block.

*Note - this model uses dropout on the attention probabilities, as you'll probably notice from looking at the source code in the link above. This won't affect the model's behaviour because dropout is disabled in inference mode (and using the `generate` method always puts a model in inference mode). But it is still a layer which exists in the model, so you can access its input or output just like any other module.*

<details>
<summary>Aside - inference mode</summary>

Dropout is one of the two main layers whose behaviour changes in inference mode (the other is BatchNorm).

If you want to run the model without inference mode, you can wrap your code in `with model.trace(inference=False):`. However, you don't need to worry about this for the purposes of these exercises.

</details>

If you're stuck on how to reference the right module, see the following hint:

<details>
<summary>Hint - what module you should get attention from</summary>

You want to extract attention from `model.transformer.h[0].attn.attn_dropout.input`. If you used `.output`, it would give you the same values (although they might differ by a dummy batch dimension). Both of these will return a single tensor, because dropout layers take just one input and return just one output.

</details>

<details>
<summary>Aside - GPT2 tokenizer uses special characters to represent space </summary>

GPT2 tokenizer uses "Ġ" to represent prepended space. So ["My", " name", " is", " James"] will be tokenized as ["My", "Ġname", "Ġis", "ĠJames"]. Make sure you replace "Ġ" with an actual space.

</details>

```python
# YOUR CODE HERE - extract and visualize attention
```

<details>
<summary>Solution (and explanation)</summary>

```python
with model.trace(prompt, remote=REMOTE):
    attn_patterns = model.transformer.h[0].attn.attn_dropout.input.save()

# Get string tokens (replacing special character for spaces)
str_tokens = model.tokenizer.tokenize(prompt)
str_tokens = [s.replace('Ġ', ' ') for s in str_tokens]

# Attention patterns (squeeze out the batch dimension)
attn_patterns_value = attn_patterns.squeeze(0)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=attn_patterns_value,
))
```

Explanation:

* Within the context managers:
    * We access the attention patterns by taking the input to the `attn_dropout`.
        * From the GPT-J source code, we can see that the attention weights are calculated by standard torch functions (and an unnamed `nn.Softmax` module) from the key and query vectors, and are then passed through the dropout layer before being used to calculate the attention layer output. So by accessing the input to the dropdown layer, we get the attention weights before dropout is applied.
        * Because of the previously discussed point about dropout not working in inference mode, we could also use the output of `attn_dropout`, and get the same values.
* Outside of the context managers:
    * We use the `tokenize` method to tokenize the prompt.

</details>

As an optional bonus exercise, you can verify for yourself that these are the correct attention patterns, by calculating them from scratch using the key and query vectors. Using `model.transformer.h[0].attn.q_proj.output` will give you the query vectors, and `k_proj` for the key vectors. However, one thing to be wary of is that GPT-J uses **rotary embeddings**, which makes the computation of attention patterns from keys and queries a bit harder than it would otherwise be. See [here](https://blog.eleuther.ai/rotary-embeddings/) for an in-depth discussion of rotary embeddings, and [here](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=bef36Bf9k7FYsCt1DpzCw6eV) for some rough intuitions.

# 2️⃣ Task-encoding hidden states

> ##### Learning Objectives
>
> * Understand how `nnsight` can be used to perform causal interventions, and perform some yourself
> * Reproduce the "h-vector results" from the function vectors paper; that the residual stream does contain a vector which encodes the task and can induce task behaviour on zero-shot prompts

We'll begin with the following question, posed by the Function Vectors paper:

> *When a transformer processes an ICL (in-context-learning) prompt with exemplars demonstrating task $T$, do any hidden states encode the task itself?*

We'll prove that the answer is yes, by constructing a vector $h$ from a set of ICL prompts for the **antonym task**, and intervening with our vector to make our model produce antonyms on zero-shot prompts.

This will require you to learn how to perform causal interventions with `nnsight`, not just save activations.

Note - this section structurally follows section 2.1 of the function vectors paper.

## ICL Task

### Exercise (optional) - generate your own antonym pairs

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
>
> If you choose to do this exercise, you should spend up to 10-30 minutes on it - depending on your familiarity with the OpenAI Python API.
> ```

We've provided you two options for the antonym dataset you'll use in these exercises.

1. Firstly, we've provided you a list of word pairs, in the file `data/antonym_pairs.txt`.
2. Secondly, if you want to run experiments like the ones in this paper, it can be good practice to learn how to generate prompts from GPT-4 or other models (this is how we generated the data for this exercise).

If you just want to use the provided list of words, skip this exercise and run the code below to load in the dataset from the text file. Alternatively, if you want to generate your own dataset, you can fill in the function `generate_dataset` below, which should query GPT-4 and get a list of antonym pairs.

See [here](https://platform.openai.com/docs/guides/gpt/chat-completions-api) for a guide to using the chat completions API, if you haven't already used it. Use the two dropdowns below (in order) for some guidance.

<details>
<summary>Getting started #1</summary>

Here is a recommended template:

```python
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": antonym_task},
        {"role": "assistant", "content": start_of_response},
    ]
)
```

where `antonym_task` explains the antonym task, and `start_of_respose` gives the model a prompt to start from (e.g. "Sure, here are some antonyms: ..."), to guide its subsequent behaviour.

</details>

<details>
<summary>Getting started #2</summary>

Here is an template you might want to use for the actual request:

```python
example_antonyms = "old: young, top: bottom, awake: asleep, future: past, "

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Give me {N} examples of antonym pairs. They should be obvious, i.e. each word should be associated with a single correct antonym."},
        {"role": "assistant", "content": f"Sure! Here are {N} pairs of antonyms satisfying this specification: {example_antonyms}"},
    ]
)
```

where `N` is the function argument. Note that we've provided a few example antonyms, and appended them to the start of GPT4's completion. This is a classic trick to guide the rest of the output (in fact, it's commonly used in adversarial attacks).

</details>

Note - it's possible that not all the antonyms returned will be solvable by GPT-J. In this section, we won't worry too much about this. When it comes to testing out our zero-shot intervention, we'll make sure to only use cases where GPT-J can actually solve it.

```python
def generate_antonym_dataset(N: int):
    """
    Generates 100 pairs of antonyms, in the form of a list of 2-tuples.
    """
    assert os.environ.get("OPENAI_API_KEY", None) is not None, "Please set your API key before running this function!"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Generate {N} pairs of antonyms in the form of a list of 2-tuples. For example, [['old', 'young'], ['top', bottom'], ['awake', 'asleep']...].",
            },
            {"role": "assistant", "content": "Sure, here is a list of 100 antonyms: "},
        ],
    )
    return response

if os.environ.get("OPENAI_API_KEY", None) is not None:
    ANTONYM_PAIRS = generate_antonym_dataset(100)
    # Save the word pairs in a text file
    with open(section_dir / "data" / "my_antonym_pairs.txt", "w") as f:
        for word_pair in ANTONYM_PAIRS:
            f.write(f"{word_pair[0]} {word_pair[1]}\n")

# Load the word pairs from the text file
with open(section_dir / "data" / "antonym_pairs.txt", "r") as f:
    ANTONYM_PAIRS = [line.split() for line in f.readlines()]

print(ANTONYM_PAIRS[:10])
```

## ICL Dataset

To handle this list of word pairs, we've given you some helpful classes.

Firstly, there's the `ICLSequence` class, which takes in a list of word pairs and contains methods for constructing a prompt (and completion) from these words. Run the code below to see how it works.

```python
class ICLSequence:
    """
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    """

    def __init__(self, word_pairs: list[list[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        """Returns the prompt, which contains all but the second element in the last word pair."""
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[: -len(self.completion())]

    def completion(self):
        """Returns the second element in the last word pair (with padded space)."""
        return " " + self.y[-1]

    def __str__(self):
        """Prints a readable string representation of the prompt & completion (indep of template)."""
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")

word_list = [["hot", "cold"], ["yes", "no"], ["in", "out"], ["up", "down"]]
seq = ICLSequence(word_list)

print("Tuple-representation of the sequence:")
print(seq)
print("\nActual prompt, which will be fed into the model:")
print(seq.prompt())
```

Secondly, we have the `ICLDataset` class. This is also fed a word pair list, and it has methods for generating batches of prompts and completions. It can generate both clean prompts (where each pair is actually an antonym pair) and corrupted prompts (where the answers for each pair are randomly chosen from the dataset).

```python
class ICLDataset:
    """
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    """

    def __init__(
        self,
        word_pairs: list[list[str]],
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
    ):
        assert n_prepended + 1 <= len(word_pairs), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended + 1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended + 1)
            if not (bidirectional):
                random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        """Creates a corrupted version of the dataset (with same random seed)."""
        return ICLDataset(
            self.word_pairs,
            self.size,
            self.n_prepended,
            self.bidirectional,
            corrupted=True,
            seed=self.seed,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]
```

You can see how this dataset works below. **Note that the correct completions have a prepended space**, because this is how the antonym prompts are structured - the answers are tokenized as `"A: answer" -> ["A", ":", " answer"]`. Forgetting prepended spaces is a classic mistake when working with transformers!

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=False)

table = Table("Prompt", "Correct completion")
for seq, completion in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completion))

rprint(table)
```

Compare this output to what it looks like when `corrupted=True`. Each of the pairs before the last one has their second element replaced with a random one (but the last pair is unchanged).

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=True)

table = Table("Prompt", "Correct completion")
for seq, completions in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completions))

rprint(table)
```

<details>
<summary>Aside - the <code>rich</code> library</summary>

The `rich` library is a helpful little library to display outputs more clearly in a Python notebook or terminal. It's not necessary for this workshop, but it's a nice little tool to have in your toolbox.

The most important function is `rich.print` (usually imported as `rprint`). This can print basic strings, but it also supports the following syntax for printing colors:

```python
rprint("[green]This is green text[/], this is default color")
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-1.png" width="350">

and for making text bold / underlined:

```python
rprint("[u dark_orange]This is underlined[/], and [b cyan]this is bold[/].")
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-2.png" width="350">

It can also print tables:

```python
from rich.table import Table

table = Table("Col1", "Col2", title="Title") # title is optional
table.add_row("A", "a")
table.add_row("B", "b")

rprint(table)
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-3.png" width="150">

The text formatting (bold, underlined, colors, etc) is also supported within table cells.

</details>

## Task-encoding vector

### Exercise - forward pass on antonym dataset

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

You should fill in the `calculate_h` function below. It should:

* Run a forward pass on the model with the dataset prompts (i.e. the `.prompts` attribute), using the `nnsight` syntax we've demonstrated previously,
* Return a tuple of the model's output (i.e. a list of its string-token completions, one for each prompt in the batch) and the residual stream value at the end of layer `layer` (e.g. if `layer = -1`, this means the final value of the residual stream before we convert into logits).

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-1.png" width="900">

You should only return the residual stream values for the very last sequence position in each prompt, i.e. the last `-1` token (where the model makes the antonym prediction), and same for the completions.

<details>
<summary> Help - I'm not sure how to run (and index into) a batch of inputs.</summary>

If we pass a list of strings to the `generator.invoke` function, this will be tokenized with padding automatically.

The type of padding which is applied is **left padding**, meaning if you index at sequence position `-1`, this will get the final token in the prompt for all prompts in the list, even if the prompts have different lengths.

</details>

```python
def calculate_h(model: LanguageModel, dataset: ICLDataset, layer: int = -1) -> tuple[list[str], Tensor]:
    """
    Averages over the model's hidden representations on each of the prompts in `dataset` at layer `layer`, to produce
    a single vector `h`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset whose prompts `dataset.prompts` you're extracting the activations from (at the last seq pos)
        layer: int
            the layer you're extracting activations from

    Returns:
        completions: list[str]
            list of the model's next-token predictions (i.e. the strings the model predicts to follow the last token)
        h: Tensor
            average hidden state tensor at final sequence position, of shape (d_model,)
    """
    raise NotImplementedError()

tests.test_calculate_h(calculate_h, model)
```

<details><summary>Solution</summary>

```python
def calculate_h(model: LanguageModel, dataset: ICLDataset, layer: int = -1) -> tuple[list[str], Tensor]:
    """
    Averages over the model's hidden representations on each of the prompts in `dataset` at layer `layer`, to produce
    a single vector `h`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset whose prompts `dataset.prompts` you're extracting the activations from (at the last seq pos)
        layer: int
            the layer you're extracting activations from

    Returns:
        completions: list[str]
            list of the model's next-token predictions (i.e. the strings the model predicts to follow the last token)
        h: Tensor
            average hidden state tensor at final sequence position, of shape (d_model,)
    """
    with model.trace(dataset.prompts, remote=REMOTE):
        h = model.transformer.h[layer].output[0][:, -1].mean(dim=0).save()
        logits = model.lm_head.output[:, -1]
        next_tok_id = logits.argmax(dim=-1).save()

    completions = model.tokenizer.batch_decode(next_tok_id)
    return completions, h
```
</details>

We've provided you with a helper function, which displays the model's output on the antonym dataset (and highlights the examples where the model's prediction is correct). Note, we're using the `repr` function, because a lot of the completions are line breaks, and this helps us see them more clearly!

If the antonyms dataset was constructed well, you should find that the model's completion is correct most of the time, and most of its mistakes are either copying (e.g. predicting `wet -> wet` rather than `wet -> dry`) or understandable completions which shouldn't really be considered mistakes (e.g. predicting `right -> left` rather than `right -> wrong`). If we were being rigorous, we'd want to filter this dataset to make sure it only contains examples where the model can correctly perform the task - but for these exercises, we won't worry about this.

```python
def display_model_completions_on_antonyms(
    model: LanguageModel,
    dataset: ICLDataset,
    completions: list[str],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Prompt (tuple representation)",
        "Model's completion\n(green=correct)",
        "Correct completion",
        title="Model's antonym completions",
    )

    for i in range(min(len(completions), num_to_display)):
        # Get model's completion, and correct completion
        completion = completions[i]
        correct_completion = dataset.completions[i]
        correct_completion_first_token = model.tokenizer.tokenize(correct_completion)[0].replace("Ġ", " ")
        seq = dataset.seqs[i]

        # Color code the completion based on whether it's correct
        is_correct = completion == correct_completion_first_token
        completion = f"[b green]{repr(completion)}[/]" if is_correct else repr(completion)

        table.add_row(str(seq), completion, repr(correct_completion))

    rprint(table)

# Get uncorrupted dataset
dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=2)

# Getting it from layer 12, as in the description in section 2.1 of paper
model_completions, h = calculate_h(model, dataset, layer=12)

# Displaying the output
display_model_completions_on_antonyms(model, dataset, model_completions)
```

### Using multiple invokes

Another cool feature of `nnsight` is the ability to run multiple different batches through the model at once (or the same batch multiple times) in a way which leads to very clean syntax for doing causal interventions. Rather than doing something like this:

```python
with model.trace(inputs, remote=REMOTE):
    # some causal interventions
```

we can write a double-nested context manager:

```python
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(inputs):
        # some causal interventions

    with tracer.invoke(other_inputs):
        # some other causal interventions
```

Both inputs will be run together in parallel, and proxies defined within one `tracer.invoke` block can be used in another. A common use-case is to have clean and corrupted inputs, so we can patch from one to the other and get both outputs all in a single forward pass:

```python
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(clean_inputs):
        # extract clean activations
        clean_activations = model.transformer.h[10].output[0]

    with tracer.invoke(corrupted_inputs):
        # patch clean into corrupted
        model.transformer.h[10].output[0][:] = clean_activations
```

You'll do something like this in a later exercise. However for your first exercise (immediately below), you'll only be intervening with vectors that are defined outside of your context manager.

**One important thing to watch out for** - make sure you're not using your proxy before it's being defined! For example, if you were extracting `clean_activations` from `model.transformer.h[10]` but then intervening with it on `model.transformer.h[9]`, this couldn't be done in parallel (you'd need to first extract the clean activations, *then* run the patched forward pass). Doing this should result in a warning message, but may pass silently in some cases - so you need to be extra vigilant!

### Exercise - intervene with $h$

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

You should fill in the function `intervene_with_h` below. This will involve:

* Run two forward passes (within the same context manager) on a zero-shot dataset:
    * One with no intervention (i.e. the residual stream is unchanged),
    * One with an intervention using `h` (i.e. `h` is added to the residual stream at the layer it was taken from).
* Return the completions for no intervention and intervention cases respectively (see docstring).

The diagram below shows how all of this should work, when combined with the `calculate_h` function.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-2.png" width="950">

Hint - you can use `tokenizer.batch_decode` to turn a list of tokens into a list of strings.

<details>
<summary>Help - I'm not sure how best to get both the no-intervention and intervention completions.</summary>

You can use `with tracer.invoke...` more than once within the same context manager, in order to add to your batch. This will eventually give you output of shape (2*N, seq_len), which can then be indexed and reshaped to get the completions in the no intervention & intervention cases respectively.

</details>

<details>
<summary>Help - I'm not sure how to intervene on the hidden state.</summary>

First, you can define the tensor of hidden states (i.e. using `.output[0]`, like you've done before).

Then, you can add to this tensor directly (or add to some indexed version of it). You can use inplace operations (i.e. `tensor += h`) or redefining the tensor (i.e. `tensor = tensor + h`); either work.

</details>

```python
def intervene_with_h(
    model: LanguageModel,
    zero_shot_dataset: ICLDataset,
    h: Tensor,
    layer: int,
    remote: bool = REMOTE,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h` using previously defined function, and intervenes by adding `h` to the
    residual stream of a set of generated zero-shot prompts.

    Inputs:
        model: the model we're using to generate completions
        zero_shot_dataset: the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        h: the `h`-vector we'll be adding to the residual stream
        layer: the layer we'll be extracting the `h`-vector from
        remote: whether to run the forward pass on the remote server (used for running test code)

    Returns:
        completions_zero_shot: list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list of string completions for the zero-shot prompts, with h-intervention
    """
    raise NotImplementedError()

tests.test_intervene_with_h(intervene_with_h, model, h, ANTONYM_PAIRS, REMOTE)
```

<details><summary>Solution</summary>

```python
def intervene_with_h(
    model: LanguageModel,
    zero_shot_dataset: ICLDataset,
    h: Tensor,
    layer: int,
    remote: bool = REMOTE,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h` using previously defined function, and intervenes by adding `h` to the
    residual stream of a set of generated zero-shot prompts.

    Inputs:
        model: the model we're using to generate completions
        zero_shot_dataset: the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        h: the `h`-vector we'll be adding to the residual stream
        layer: the layer we'll be extracting the `h`-vector from
        remote: whether to run the forward pass on the remote server (used for running test code)

    Returns:
        completions_zero_shot: list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list of string completions for the zero-shot prompts, with h-intervention
    """
    with model.trace(remote=remote) as tracer:
        # First, run a forward pass where we don't intervene, just save token id completions
        with tracer.invoke(zero_shot_dataset.prompts):
            token_completions_zero_shot = model.lm_head.output[:, -1].argmax(dim=-1).save()

        # Next, run a forward pass on the zero-shot prompts where we do intervene
        with tracer.invoke(zero_shot_dataset.prompts):
            # Add the h-vector to the residual stream, at the last sequence position
            hidden_states = model.transformer.h[layer].output[0]
            hidden_states[:, -1] += h
            # Also save completions
            token_completions_intervention = model.lm_head.output[:, -1].argmax(dim=-1).save()

    # Decode to get the string tokens
    completions_zero_shot = model.tokenizer.batch_decode(token_completions_zero_shot)
    completions_intervention = model.tokenizer.batch_decode(token_completions_intervention)

    return completions_zero_shot, completions_intervention
```
</details>

Run the code below to calculate completions for the function.

**Note, it's very important that we set a different random seed for the zero shot dataset, otherwise we'll be intervening on examples which were actually in the dataset we used to compute $h$!**

```python
layer = 12
dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

# Run previous function to get h-vector
h = calculate_h(model, dataset, layer=layer)[1]

# Run new function to intervene with h-vector
completions_zero_shot, completions_intervention = intervene_with_h(model, zero_shot_dataset, h, layer=layer)

print("Zero-shot completions: ", completions_zero_shot)
print("Completions with intervention: ", completions_intervention)
```

Next, run the code below to visualise the completions in a table. You should see:

* ~0% correct completions on the zero-shot prompt with no intervention, because the model usually just copies the first and only word in the prompt
* ~25% correct completions on the zero-shot prompt with intervention

```python
def display_model_completions_on_h_intervention(
    dataset: ICLDataset,
    completions: list[str],
    completions_intervention: list[str],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Prompt",
        "Model's completion\n(no intervention)",
        "Model's completion\n(intervention)",
        "Correct completion",
        title="Model's antonym completions",
    )

    for i in range(min(len(completions), num_to_display)):
        completion_ni = completions[i]
        completion_i = completions_intervention[i]
        correct_completion = dataset.completions[i]
        correct_completion_first_token = tokenizer.tokenize(correct_completion)[0].replace("Ġ", " ")
        seq = dataset.seqs[i]

        # Color code the completion based on whether it's correct
        is_correct = completion_i == correct_completion_first_token
        completion_i = f"[b green]{repr(completion_i)}[/]" if is_correct else repr(completion_i)

        table.add_row(str(seq), repr(completion_ni), completion_i, repr(correct_completion))

    rprint(table)

display_model_completions_on_h_intervention(zero_shot_dataset, completions_zero_shot, completions_intervention)
```

### Exercise - combine the last two functions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

One great feature of the `nnsight` library is its ability to parallelize forward passes and perform complex interventions within a single context manager.

In the code above, we had one function to extract the hidden states from the model, and another function where we intervened with those hidden states. But we can actually do both at once: we can compute $h$ within our forward pass, and then intervene with it on a different forward pass (using our zero-shot prompts), all within the same `model.trace` context manager. In other words, **we'll be using `with tracer.invoke...` three times** in this context manager.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-3.png" width="1000">

You should fill in the `calculate_h_and_intervene` function below, to do this. Mostly, this should involve combining your `calculate_h` and `intervene_with_h` functions, and wrapping the forward passes in the same context manager (plus a bit of code rewriting).

Your output should be exactly the same as before (since the `ICLDataset` class is deterministic), hence we've not provided test functions in this case - you can just compare the table you get to the one before! However, this time around your code should run twice as fast, because you're batching the operations of "compute $h$" and "intervene with $h$" together into a single forward pass.

<details>
<summary>Help - I'm not sure how to use the <code>h</code> vector inside the context manager.</summary>

You extract `h` the same way as before, but you don't need to save it. It is kept as a proxy. You can still use it later in the context manager, just like it actually was a tensor.

You shouldn't have to `.save()` anything inside your context manager, other than the token completions.

</details>
<details>
<summary>Help - If I want to add <code>x</code> vector to a slice of my hidden state tensor <code>h</code>, is <code>h[slice]+=x</code> the same as <code>h2 = h[slice], h2 += x</code>?</summary>

No, only `h[slice]+=x` does what you want. This is because when doing <code>h2 = h[slice], h2 += x</code>, the modification line <code>h2 += x</code> is no longer modifying the original tensor `h`, but a different tensor`h2`. In contrast, `h[slice]+=x` keeps the original tensor `h` in the modification line.

A good rule to keep in mind is: If you're trying to modify a tensor some in-place operation, make sure that tensor is in the actual modification line!

</details>

```python
def calculate_h_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the completions from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        completions_zero_shot: list[str]
            list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list[str]
            list of string completions for the zero-shot prompts, with h-intervention
    """
    raise NotImplementedError()

dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

completions_zero_shot, completions_intervention = calculate_h_and_intervene(
    model, dataset, zero_shot_dataset, layer=layer
)

display_model_completions_on_h_intervention(zero_shot_dataset, completions_zero_shot, completions_intervention)
```

<details><summary>Solution</summary>

```python
def calculate_h_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the completions from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        completions_zero_shot: list[str]
            list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list[str]
            list of string completions for the zero-shot prompts, with h-intervention
    """
    with model.trace(remote=REMOTE) as tracer:
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        with tracer.invoke(zero_shot_dataset.prompts):
            clean_tokens = model.lm_head.output[:, -1].argmax(dim=-1).save()

        with tracer.invoke(zero_shot_dataset.prompts):
            hidden = model.transformer.h[layer].output[0]
            hidden[:, -1] += h
            intervene_tokens = model.lm_head.output[:, -1].argmax(dim=-1).save()

    completions_zero_shot = tokenizer.batch_decode(clean_tokens)
    completions_intervention = tokenizer.batch_decode(intervene_tokens)
    return completions_zero_shot, completions_intervention
```
</details>

### Exercise - compute change in accuracy

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

So far, all we've done is look at the most likely completions, and see what fraction of the time these were correct. But our forward pass doesn't just give us token completions, it gives us logits too!

You should now rewrite the `calculate_h_and_intervene` function so that, rather than returning two lists of string completions, it returns two lists of floats containing the **logprobs assigned by the model to the correct antonym** in the no intervention / intervention cases respectively.

<details>
<summary>Help - I don't know how to get the correct logprobs from the logits.</summary>

First, apply log softmax to the logits, to get logprobs.

Second, you can use `tokenizer(dataset.completions)["input_ids"]` to get the token IDs of the correct completions. (Gotcha - some words might be tokenized into multiple tokens, so make sure you're just picking the first token ID for each completion.)

Note - we recommend doing all this inside the context manager, then saving and returning just the correct logprobs not all the logits (this means less to download from the server!).

</details>

```python
def calculate_h_and_intervene_logprobs(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[float], list[float]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the logprobs on correct tokens from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        correct_logprobs: list[float]
            list of correct-token logprobs for the zero-shot prompts, without intervention
        correct_logprobs_intervention: list[float]
            list of correct-token logprobs for the zero-shot prompts, with h-intervention
    """
    raise NotImplementedError()
```

<details><summary>Solution</summary>

```python
def calculate_h_and_intervene_logprobs(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[float], list[float]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the logprobs on correct tokens from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        correct_logprobs: list[float]
            list of correct-token logprobs for the zero-shot prompts, without intervention
        correct_logprobs_intervention: list[float]
            list of correct-token logprobs for the zero-shot prompts, with h-intervention
    """
    correct_completion_ids = [toks[0] for toks in tokenizer(zero_shot_dataset.completions)["input_ids"]]

    with model.trace(remote=REMOTE) as tracer:
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        with tracer.invoke(zero_shot_dataset.prompts):
            clean_logprobs = model.lm_head.output.log_softmax(dim=-1)[
                range(len(zero_shot_dataset)), -1, correct_completion_ids
            ].save()

        with tracer.invoke(zero_shot_dataset.prompts):
            hidden = model.transformer.h[layer].output[0]
            hidden[:, -1] += h
            intervene_logprobs = model.lm_head.output.log_softmax(dim=-1)[
                range(len(zero_shot_dataset)), -1, correct_completion_ids
            ].save()

    return clean_logprobs, intervene_logprobs
```
</details>

When you run the code below, it will display the log-probabilities (highlighting green when they increase from the zero-shot case). You should find that in every sequence, the logprobs on the correct token increase in the intervention. This helps make something clear - **even if the maximum-likelihood token doesn't change, this doesn't mean that the intervention isn't having a significant effect.**

```python
def display_model_logprobs_on_h_intervention(
    dataset: ICLDataset,
    correct_logprobs_zero_shot: list[float],
    correct_logprobs_intervention: list[float],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Zero-shot prompt",
        "Model's logprob\n(no intervention)",
        "Model's logprob\n(intervention)",
        "Change in logprob",
        title="Model's antonym logprobs, with zero-shot h-intervention\n(green = intervention improves accuracy)",
    )

    for i in range(min(len(correct_logprobs_zero_shot), num_to_display)):
        logprob_ni = correct_logprobs_zero_shot[i]
        logprob_i = correct_logprobs_intervention[i]
        delta_logprob = logprob_i - logprob_ni
        zero_shot_prompt = f"{dataset[i].x[0]:>8} -> {dataset[i].y[0]}"

        # Color code the logprob based on whether it's increased with this intervention
        is_improvement = delta_logprob >= 0
        delta_logprob = f"[b green]{delta_logprob:+.2f}[/]" if is_improvement else f"{delta_logprob:+.2f}"

        table.add_row(zero_shot_prompt, f"{logprob_ni:.2f}", f"{logprob_i:.2f}", delta_logprob)

    rprint(table)

dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

correct_logprobs_zero_shot, correct_logprobs_intervention = calculate_h_and_intervene_logprobs(
    model, dataset, zero_shot_dataset, layer=layer
)

display_model_logprobs_on_h_intervention(
    zero_shot_dataset, correct_logprobs_zero_shot, correct_logprobs_intervention
)
```

# 3️⃣ Function Vectors

> ##### Learning Objectives
>
> * Define a metric to measure the causal effect of each attention head on the correct performance of the in-context learning task
> * Understand how to rearrange activations in a model during an `nnsight` forward pass, to extract activations corresponding to a particular attention head
> * Learn how to use `nnsight` for multi-token generation

In this section, we'll replicate the crux of the paper's results, by identifying a set of attention heads whose outputs have a large effect on the model's ICL performance, and showing we can patch with these vectors to induce task-solving behaviour on randomly shuffled prompts.

We'll also learn how to use `nnsight` for multi-token generation, and steer the model's behaviour. There exist exercises where you can try this out for different tasks, e.g. the Country-Capitals task, where you'll be able to steer the model to complete prompts like `"When you think of Netherlands, you usually think of"` by talking about Amsterdam.

Note - this section structurally follows sections 2.2, 2.3 and some of section 3 from the function vectors paper.

Here, we'll move from thinking about residual stream states to thinking about the **output of specific attention heads.**

## Extracting & using FVs

### A note on `out_proj`

First, a bit of a technical complication. Most HuggingFace models don't have the nice attention head representations. What we have is the linear layer `out_proj` which implicitly combines the "projection per attention head" and the "sum over attention head" operations (if you can't see how this is possible, see the section "Attention Heads are Independent and Additive" from Anthropic's [Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html)).

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rearrange-output-2.png" width="950">

This presents some question for us, when it comes to causal interventions on attention heads. Use the dropdowns below to read them answer these questions (they'll be important for the coming exercises).

<br>

<details>
<summary>If we want to do a causal intervention on a particular head, should we intervene on <code>z</code> (the input of <code>out_proj</code>) or on <code>attn_output</code> (the output of <code>out_proj</code>) ?</summary>

We should intervene on `z`, because we can just rearrange the `z` tensor of shape `(batch, seq, d_model)` into `(batch, seq, n_heads, d_head)`, in other words separating out all the heads. On the other hand, we can't do this with the `attn_output` because it's *already* summed over heads and we can't separate them out.

</details>

<br>

<details>
<summary>How could we get the <code>attn_output</code> vector for a single head, if we had the ability to access model weights within our context managers?</summary>

We can take a slice of the `z` tensor corresponding to a single attention head:

```python
z.reshape(batch, seq, n_heads, d_head)[:, :, head_idx]
```

and we can take a slice of the `out_proj` weight matrix corresponding to a single attention head (remember that PyTorch stores linear layers in the shape `(out_feats, in_feats)`):

```python
out_proj.weight.rearrange(d_model, n_heads, d_head)[:, head_idx]
```

then finally we can multiply these together.

</details>

<br>

<details>
<summary>How could we get the <code>attn_output</code> vector for a single head, if we </b>didn't have</b> the ability to access model weights within our context managers? (This is currently the case for <code>nnsight</code>, since having access to the weights could allow users to change them!).</summary>

We can be a bit clever, and ablate certain heads in the `z` vector before passing it through the output projection:

```python
# ablate all heads except #2 (using a cloned activation)
heads_to_ablate = [0, 1, 3, 4, ...]
z_ablated = z.reshape(batch, seq, n_heads, d_head).clone()
z_ablated[:, :, heads_to_ablate] = 0

# save the output
attn_head_output = out_proj(z_ablated)
```

Illustration:

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rearrange-output-ablated-2.png" width="950">

Note - this would actually fail if `out_proj` had a bias, because we want to just get an attention head's output, not the bias term as well. But if you look at the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) you'll see that `out_proj` doesn't have a bias term, so we're all good!

</details>

### Exercise - implement `calculate_fn_vectors_and_intervene`

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 30-60 minutes on this exercise.
> ```

This is probably the most important function in today's exercises. Implementing it will be pretty similar to the previous function `calculate_h_and_intervene`, but:

* Rather than extracting the value of the residual stream `h` at some particular layer, you'll be extracting the output of the attention heads: iterating over each layer and each head in the model.
    * You'll only need to run one clean forward pass to compute all these values, but you'll need to run a separate corrupted forward pass for each head.
* Rather than your 2 different datasets being (dataset, zero-shot dataset), your two datasets will be (dataset, corrupted version of that same dataset).
    * You can use the method `create_corrupted_dataset` method of the `ICLDataset` class for this.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/cie-intervention.png" width="1200">

Before you actually start writing the code, it might be helpful to answer the following:

<details>
<summary>How many different <code>invoke</code> calls will you need in total?</summary>

You'll need `(N_LAYERS * N_HEADS) + 2`. To explain:

- One for the clean prompts, which you'll extract internal activations from and patch them into corrupted prompts,
- One for the corrupted prompts, which you don't intervene on,
- One for the corrupted prompts **for every attention head**, which you'll patch into using the clean run activations.

</details>

<details>
<summary>Which proxy outputs (if any) will you need to use <code>.save()</code> on, in this function?</summary>

You don't need to `.save()` the function vectors you're extracting from the model's internals, because these will only be used for causal interventions within the context manager.

The only thing you need to save is the correct token logprobs for (1) the corrupted forward pass where we don't intervene, and (2) each corrupted forward pass where we do intervene on one of the heads. In other words, you'll need to save `(N_LAYERS * N_HEADS) + 1` tensors in total.

</details>

A few other notes:

* We've added a `layers` argument, so you can iterate through different layers of the model (i.e. running the model with `layers = [3, 4, 5]` will only test the intervention on the attention heads in layers 3, 4 and 5). This is helpful if you're getting memory errors when trying to run all layers at once (remember we have 24 layers, 16 heads per layer, so even with few prompts per head this adds up fast!).
    * We've included code for you below showing how you can call the function multiple times, clearing memory between each run, then combine the results.
* When it comes to intervening, you can set the value of a reshaped tensor, i.e. `tensor.reshape(*new_shape)[index] = new_value` will change the values in `tensor` without actually reshaping it (for more on this, see the documentation for [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)).
* It's good practice to insert a lot of assert statements in your code, to check the shapes are what you expect.
* If you're confused about dimensions, use `einops.rearrange` rather than `.reshape` - this is a wonderful tool, it's like using code annotations within your actual code!

One last note - **if this function is proving impossible to run for computational reasons, you can skip the exercise and move on to the next ones. They don't rely on this function working.** However, you should definitely at least read & understand the solution.

```python
def calculate_fn_vectors_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    layers: list[int] | None = None,
) -> Float[Tensor, "layers heads"]:
    """
    Returns a tensor of shape (layers, heads), containing the CIE for each head.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a corrupted
            version of this dataset for interventions)
        layers: list[int] | None
            the layers which this function will calculate the score for (if None, we assume all layers)
    """
    raise NotImplementedError()
```

<details><summary>Solution</summary>

```python
def calculate_fn_vectors_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    layers: list[int] | None = None,
) -> Float[Tensor, "layers heads"]:
    """
    Returns a tensor of shape (layers, heads), containing the CIE for each head.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a corrupted
            version of this dataset for interventions)
        layers: list[int] | None
            the layers which this function will calculate the score for (if None, we assume all layers)
    """
    layers = range(model.config.n_layer) if (layers is None) else layers
    heads = range(model.config.n_head)

    # Get corrupted dataset
    corrupted_dataset = dataset.create_corrupted_dataset()
    N = len(dataset)

    # Get correct token ids, so we can get correct token logprobs
    correct_completion_ids = [toks[0] for toks in tokenizer(dataset.completions)["input_ids"]]

    with model.trace(remote=REMOTE) as tracer:
        # Run a forward pass on clean prompts, where we store attention head outputs
        z_dict = {}
        with tracer.invoke(dataset.prompts):
            for layer in layers:
                # Get hidden states, reshape to get head dimension, store the mean tensor
                z = model.transformer.h[layer].attn.out_proj.input[:, -1]
                z_reshaped = z.reshape(N, N_HEADS, D_HEAD).mean(dim=0)
                for head in heads:
                    z_dict[(layer, head)] = z_reshaped[head]

        # Run a forward pass on corrupted prompts, where we don't intervene or store activations (just so we can get the
        # correct-token logprobs to compare with our intervention)
        with tracer.invoke(corrupted_dataset.prompts):
            logits = model.lm_head.output[:, -1]
            correct_logprobs_corrupted = logits.log_softmax(dim=-1)[t.arange(N), correct_completion_ids].save()

        # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes, since
        # we're doing different interventions each time)
        correct_logprobs_dict = {}
        for layer in layers:
            for head in heads:
                with tracer.invoke(corrupted_dataset.prompts):
                    # Get hidden states, reshape to get head dimension, then set it to the a-vector
                    z = model.transformer.h[layer].attn.out_proj.input[:, -1]
                    z.reshape(N, N_HEADS, D_HEAD)[:, head] = z_dict[(layer, head)]
                    # Get logprobs at the end, which we'll compare with our corrupted logprobs
                    logits = model.lm_head.output[:, -1]
                    correct_logprobs_dict[(layer, head)] = logits.log_softmax(dim=-1)[
                        t.arange(N), correct_completion_ids
                    ].save()

    # Get difference between intervention logprobs and corrupted logprobs, and take mean over batch dim
    all_correct_logprobs_intervention = einops.rearrange(
        t.stack([v for v in correct_logprobs_dict.values()]),
        "(layers heads) batch -> layers heads batch",
        layers=len(layers),
    )
    logprobs_diff = all_correct_logprobs_intervention - correct_logprobs_corrupted  # shape [layers heads batch]

    # Return mean effect of intervention, over the batch dimension
    return logprobs_diff.mean(dim=-1)
```
</details>

As mentioned, the code below calls the function multiple times separately and combines the results.

When you run this code & plot the results, you should replicate Figure 3(a) in the Function Vectors paper (more or less). If the code is taking too long to run, we recommend just choosing a single layer to run, which has a distinctive pattern that can be compared to the paper's figure (e.g. layer 8, since head L8H1 has a much higher score than all the other heads in this layer).

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=8, n_prepended=2)

def batch_process_layers(n_layers, batch_size):
    for i in range(0, n_layers, batch_size):
        yield range(n_layers)[i : i + batch_size]

results = t.empty((0, N_HEADS), device=device)

# If this fails to run, reduce the batch size so the fwd passes are split up more, or reduce dataset size
for layers in batch_process_layers(N_LAYERS, batch_size=4):
    print(f"Computing layers in {layers} ...")
    t0 = time.time()
    results = t.concat([results, calculate_fn_vectors_and_intervene(model, dataset, layers).to(device)])
    print(f"... finished in {time.time()-t0:.2f} seconds.\n")
```

```python
imshow(
    results.T,
    title="Average indirect effect of function-vector intervention on antonym task",
    width=1000,
    height=600,
    labels={"x": "Layer", "y": "Head"},
    aspect="equal",
)
```

### Exercise - calculate the function vector

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 25-50 minutes on this exercise.
> ```

Your next task is to actually calculate and return the function vector, so we can do a few experiments with it. The function vector is the sum of the outputs of all the attention heads we found using the previous function (i.e. the sum of all of the vectors these heads write to the residual stream), averaged over the prompts in our dataset.

There's a difficulty here - rather than just getting the `z` vectors, we're actually trying to get the `attn_out` vectors, but *before* they're summed over heads. As we discussed previously, this is a bit tricky to do for the model we're working with, because the `out_proj` linear map actually does the "project up" and "sum over heads" operations simultaneously. It would be nice to just take a slice of the `out_proj` matrix and multiply it with a slice of the `z` vector, but the `nnsight` library doesn't yet allow users to access weights directly (for security reasons). To understand how we can extract the `attn_out` vector for a head separately without accessing the underlying weights, you should go back to read the subsection **A note on `out_proj`** at the start of this section.

```python
def calculate_fn_vector(
    model: LanguageModel,
    dataset: ICLDataset,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "d_model"]:
    """
    Returns a vector of length `d_model`, containing the sum of vectors written to the residual stream
    by the attention heads in `head_list`, averaged over all inputs in `dataset`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a
            corrupted version of this dataset for interventions)
        head_list: list[tuple[int, int]]
            list of attention heads we're calculating the function vector from
    """
    raise NotImplementedError()

tests.test_calculate_fn_vector(calculate_fn_vector, model)
```

<details><summary>Solution</summary>

```python
def calculate_fn_vector(
    model: LanguageModel,
    dataset: ICLDataset,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "d_model"]:
    """
    Returns a vector of length `d_model`, containing the sum of vectors written to the residual stream
    by the attention heads in `head_list`, averaged over all inputs in `dataset`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a
            corrupted version of this dataset for interventions)
        head_list: list[tuple[int, int]]
            list of attention heads we're calculating the function vector from
    """
    # Turn head_list into a dict of {layer: heads we need in this layer}
    head_dict = defaultdict(set)
    for layer, head in head_list:
        head_dict[layer].add(head)

    fn_vector_list = []

    with model.trace(dataset.prompts, remote=REMOTE):
        for layer, head_list in head_dict.items():
            # Get the output projection layer
            out_proj = model.transformer.h[layer].attn.out_proj

            # Get the mean output projection input (note, setting values of this tensor will not have
            # downstream effects on other tensors)
            hidden_states = out_proj.input[:, -1].mean(dim=0)

            # Zero-ablate all heads which aren't in our list, then get the output (which
            # will be the sum over the heads we actually do want!)
            heads_to_ablate = set(range(N_HEADS)) - head_dict[layer]
            for head in heads_to_ablate:
                hidden_states.reshape(N_HEADS, D_HEAD)[head] = 0.0

            # Now that we've zeroed all unimportant heads, get the output & add it to the list
            # (we need a single batch dimension so we can use `out_proj`)
            out_proj_output = out_proj(hidden_states.unsqueeze(0)).squeeze().save()
            fn_vector_list.append(out_proj_output)

    # We sum all attention head outputs to get our function vector
    fn_vector = sum([v for v in fn_vector_list])

    assert fn_vector.shape == (D_MODEL,)
    return fn_vector
```
</details>

## Multi-token generation

We're now going to replicate some of the results in Table 3, in the paper:

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/tab3.png" width="700">

This will involve doing something we haven't done before - **intervening on multi-token prompt generation**.

Most of the interpretability exercises in this chapter have just consisted of running single forward passes, rather than autoregressive text generation. But we're trying something different here: we're adding the function vector to the final sequence position at each forward pass during text generation, and seeing if we can get the model to output a sentence with a different meaning.

The results of Table 3 came from adding the function vector to the residual stream at the final sequence position of the original prompt, **and the final sequence position for each subsequent generation.** The reason we do this is to guide the model's behaviour over time. Our hypothesis is that the function vector induces "next-token antonym behaviour" (because it was calculated by averaging attention head outputs at the sequence position before the model made its antonym prediction in the ICL prompts).

### Using `nnsight` for multi-token generation

Previously, our context managers have looked like:

```python
# Single invoke
with model.trace(prompt, remote=REMOTE):
    ... # Intervene on fwd pass

# Multiple invokes
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(prompt):
        ... # Intervene on fwd pass
```

But for multi-token generation, we'll be using the `generate` method rather than `trace`. Our context managers will look like:

```python
# Single invoke
with model.generate(prompt, remote=REMOTE, max_new_tokens=max_new_tokens):
    with model.all(): # signals to NNsight that you want to run interventions performed on all generated tokens
        ... # Intervene on fwd pass for n-th token to be generated

# Multiple invokes
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with model.all():
        with generator.invoke(prompt):
            ... # Intervene on fwd pass for n-th token to be generated
        with generator.invoke(prompt2):
            ... # Intervene on fwd pass for n-th token to be generated
```

The line `with model.all():` denotes that the following interventions should be applied to the forward pass for all generated tokens.

Mostly, everything you learned during single-token generation generalizes to the multi-token case. For example, using `.save()` still saves proxies outside the context managers (although make sure that you don't use the same variable names over different generations, otherwise you'll overwrite them - it's easier to store your saved proxies in e.g. a list or dict).

Note that `model.generate` takes the same arguments as the normal [HuggingFace generate method](https://huggingface.co/docs/transformers/en/main_classes/text_generation). This means we can use arguments like `top_k`, `top_p`, or `repetition_penalty` to control generation behaviour. In the exercises below we use a repetition penalty (we choose a value of 1.2, in line with the [paper](https://arxiv.org/pdf/1909.05858) that suggested it) - this can avoid the model falling into loops of repeating the same sequence, which is especially common in steering when we're pushing the model OOD.

<!-- #### Optional questions - multi-token generation with NNsight

Here are a few quick optional questions to test your understanding of how multi-generation works with NNsight. These are non-essential, and only mentioned here as potentially helpful pointers.

<details>
<summary>How do I add vector <code>h</code> to all the tokens in the original prompt but not to the generated tokens? </summary>

```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):
        # Add vectors to the model's internals on the first forward pass
        model.transformer.h[layer].output[0][:, :seq_len] += h

```
You don't have to call `model.next()` because you're only adding the vector once to tokens in the original prompt. This will be cached when the model is subsequently generating tokens.

</details>

<details>
<summary>How do I intervene with vector <code>h</code> during the generation of the first k generated tokens? </summary>

To intervene during the generation of the first `k` generated tokens:
```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):

        for n in range(k+1):
            # Add vector to the model's internals, on the k-th forward pass
            model.transformer.h[layer].output[0] += h
            model.next()
```
When `n=0`, you are adding to tokens in the original prompt before a new token is a generated. After calling `model.next()`, you are accessing the hidden state of the last token that was generated (with seq_len=1).

</details>

</details>

<details>
<summary>How do I intervene with vector <code>h</code> only during the generation of the first k tokens, but not to tokens in the original prompt before the first generated token? </summary>

```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):

        for n in range(k+1):
            model.next()
            # Add vector AFTER calling model.next() to add to the token that just got generated
            model.transformer.h[layer].output[0] += h

```
By not adding things before `model.next()`, we never add to the original prompt but always after a new token has been generated.

</details>

</details>

<details>
<summary>What is the difference between adding vector <code>h</code> before and after vector <code>model.next()</code>? </summary>

As explained in Q3, adding vector before `model.next()` means the operation is always done to the current sequence **before** a new generated token is appended. Adding vector after `model.next()` means the operation is always done to the newly generated token.

</details> -->

### Key-Value Caching

TLDR - caching can make causal interventions inside `model.generate` more complicated, but if you only intervene on sequence positions other than the very last one. In our exercises, we'll only be intervening on the last seqpos so you don't need to worry about it, but it's still a useful topic to understand.

<details>
<summary>See this dropdown if you're curious for more details.</summary>

To speed up inference, transformer models perform **key-value caching** to speed up text generation. This means that the time taken to generate $n$ tokens is ***much*** less than $n$ times longer than generating a single token. See [this blog post](https://kipp.ly/transformer-inference-arithmetic/) for more on transformer inference arithmetic.

When caching takes place, and we're doing causal interventions, we have to be careful that the caching won't override our causal interventions. Sometimes caching has to be disabled to make sure that our causal intervention works correctly. For example, if we wanted to perform the intervention "add the function vector to *only* the final sequence position of the prompt for each token we generate" then we'd have to disable caching (since previous forward passes would contain cached values where we intervened on a sequence position which is no longer the final sequence position). However, here we're performing the intervention "add the function vector to the final token of the original prompt, and to *all subsequent sequence positions*", meaning enabling caching (the default behaviour) will give us the right causal intervention.

</details>

### Generator Output

The object `generator.output` is by default a tensor which contains the model's token ID completions (not the logits).

By default the `generate` method will generate tokens greedily, i.e. always taking the maximum-probability token at each step. For now, we don't need to worry about changing this behaviour. But in future exercises we'll experiment with different sampling methods than greedy sampling (which generate uses by default), so `generator.output` and argmaxing over logits will not be identical!

### Exercise - intervene with function vector, in multi-token generation

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-30 minutes on this exercise.
> ```

You should now fill in the function `intervene_with_fn_vector` below. This will take a function vector (calculated from the function you wrote above), as well as a few other arguments (see docstring), and return the model's string completion on the given prompt template.

We hope to observe results qualitatively like the ones in Table 3, i.e. having the model define a particular word as its antonym.

```python
def intervene_with_fn_vector(
    model: LanguageModel,
    word: str,
    layer: int,
    fn_vector: Float[Tensor, "d_model"],
    prompt_template='The word "{x}" means',
    n_tokens: int = 5,
) -> tuple[str, str]:
    """
    Intervenes with a function vector, by adding it at the last sequence position of a generated prompt.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        word: str
            The word which is substituted into the prompt template, via prompt_template.format(x=word)
        layer: int
            The layer we'll make the intervention (by adding the function vector)
        fn_vector: Float[Tensor, "d_model"]
            The vector we'll add to the final sequence position for each new token to be generated
        prompt_template:
            The template of the prompt we'll use to produce completions
        n_tokens: int
            The number of additional tokens we'll generate for our unsteered / steered completions

    Returns:
        completion: str
            The full completion (including original prompt) for the no-intervention case
        completion_intervention: str
            The full completion (including original prompt) for the intervention case
    """
    raise NotImplementedError()

```

<details><summary>Solution</summary>

```python
def intervene_with_fn_vector(
    model: LanguageModel,
    word: str,
    layer: int,
    fn_vector: Float[Tensor, "d_model"],
    prompt_template='The word "{x}" means',
    n_tokens: int = 5,
) -> tuple[str, str]:
    """
    Intervenes with a function vector, by adding it at the last sequence position of a generated prompt.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        word: str
            The word which is substituted into the prompt template, via prompt_template.format(x=word)
        layer: int
            The layer we'll make the intervention (by adding the function vector)
        fn_vector: Float[Tensor, "d_model"]
            The vector we'll add to the final sequence position for each new token to be generated
        prompt_template:
            The template of the prompt we'll use to produce completions
        n_tokens: int
            The number of additional tokens we'll generate for our unsteered / steered completions

    Returns:
        completion: str
            The full completion (including original prompt) for the no-intervention case
        completion_intervention: str
            The full completion (including original prompt) for the intervention case
    """
    prompt = prompt_template.format(x=word)

    with model.generate(remote=REMOTE, max_new_tokens=n_tokens, repetition_penalty=1.2) as generator:
        with model.all():

            with generator.invoke(prompt):
                tokens = model.generator.output.save()

            with generator.invoke(prompt):
                model.transformer.h[layer].output[0][0, -1] += fn_vector
                tokens_intervention = model.generator.output.save()

    completion, completion_intervention = tokenizer.batch_decode(
        [tokens.squeeze().tolist(), tokens_intervention.squeeze().tolist()]
    )
    return completion, completion_intervention
```
</details>

To test your function, run the code below. You should find that the first completion seems normal, but the second completion defines a word as its antonym (you might have to play around a bit with the scale factor of `fn_vector`, to balance between effectiveness and coherence of output). If this works, congratulations - **you've just successfully induced an OOD behavioural change in a 6b-parameter model!**

```python
# Remove word from our pairs, so it can be a holdout
word = "light"
_ANTONYM_PAIRS = [pair for pair in ANTONYM_PAIRS if word not in pair]

# Define our dataset, and the attention heads we'll use
dataset = ICLDataset(_ANTONYM_PAIRS, size=20, n_prepended=5)
head_list = [
    (8, 0),
    (8, 1),
    (9, 14),
    (11, 0),
    (12, 10),
    (13, 12),
    (13, 13),
    (14, 9),
    (15, 5),
    (16, 14),
]

# Extract the function vector
fn_vector = calculate_fn_vector(model, dataset, head_list)

# Intervene with the function vector
completion, completion_intervention = intervene_with_fn_vector(
    model,
    word=word,
    layer=9,
    fn_vector=0.1 * fn_vector,
    prompt_template='The word "{x}" means',
    n_tokens=40,
)

table = Table("No intervention", "intervention")
table.add_row(repr(completion), repr(completion_intervention))
rprint(table)
```

### Exercise - generalize results to another task (optional)

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 15-30 minutes on this exercise.
> ```

In this exercise, you get to pick a task different to the antonyms task, and see if the results still hold up (for the same set of attention heads).

We'll leave this exercise fairly open-ended, without any code templates for you to fill in. However, if you'd like some guidance you can use the dropdown below.

<details>
<summary>Guidance for exercise</summary>

Whatever your task, you'll want to generate a new set of words. You can repurpose the `generate_dataset` function from the antonyms task, by supplying a different prompt and initial set of examples (this will require generating & using an OpenAI api key, if you haven't already), or you can just find an appropriate dataset online.

When you define the `ICLDataset`, you might want to use `bidirectional=False`, if your task isn't symmetric. The antonym task is symmetric, but others (e.g. the Country-Capitals task) are not.

You'll need to supply a new prompt template for the `intervene_with_fn_vector` function, but otherwise most of your code should stay the same.

</details>

```python
with open(section_dir / "data/country_capital_pairs.txt", "r", encoding="utf-8") as f:
    COUNTRY_CAPITAL_PAIRS = [line.split() for line in f.readlines()]

country = "Netherlands"
_COUNTRY_CAPITAL_PAIRS = [pair for pair in COUNTRY_CAPITAL_PAIRS if pair[0] != country]

dataset = ICLDataset(_COUNTRY_CAPITAL_PAIRS, size=20, n_prepended=5, bidirectional=False)
head_list = [
    (8, 0),
    (8, 1),
    (9, 14),
    (11, 0),
    (12, 10),
    (13, 12),
    (13, 13),
    (14, 9),
    (15, 5),
    (16, 14),
]

fn_vector = calculate_fn_vector(model, dataset, head_list)

# Intervene with the function vector
completion, completion_intervention = intervene_with_fn_vector(
    model=model,
    word=country,
    layer=9,
    fn_vector=0.05 * fn_vector,
    prompt_template="When you think of {x},",
    n_tokens=40,
)

table = Table("No intervention", "intervention")
table.add_row(repr(completion), repr(completion_intervention))
rprint(table)
```

# 4️⃣ Steering Vectors in GPT2-XL

> ##### Learning Objectives
>
> * Understand the goals & main results from Alex Turner et al's work on steering vectors
> * Reproduce the changes in behaviour described in their initial post

**Note**: GPT2-XL is not hosted remotely by NNsight at the moment. If you use GPT2-XL, we recommend setting `REMOTE = False`. Otherwise, you can use one of the remotely hosted models (see [here](https://nnsight.net/status/)) and set `REMOTE = True`.

## Steering model behaviour

In the final non-bonus exercise of the previous section, we touched on the idea of using function vectors to induce behavioural changes in the model's completions, rather than specifically making it solve zero-shot or corrupted prompts with the right completion. In these next exercises, we'll explore this kind of work in more detail. We'll be primarily using Turner et al's work on [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector).

Summary of the way in which this work differs from the function vector work we've done so far:

* Function vectors focused on the model performing a particular function (e.g. mapping a word to its opposite), whereas this work focuses on behavioural changes (e.g. completing a prompt which has negative tone in a positive way).
* Function vectors work looked at very large models (our exercises used Pythia-7B, the smallest model which was examined in the function vectors paper). This particular steering vectors post focused on the smaller models GPT2-Small (85m) and GPT2-XL (1.5B). We'll be focusing on GPT2-XL.
* The second half of our function vectors work identified important attention heads and focused on their outputs, rather than just adding to the residual stream directly. In this steering vector setup, we'll go back to the simpler method of adding directly into the residual stream.

Despite these differences, much of the work which was done here overlaps with function vector work, since they both fall into the broader category of *"finding vectors using forward-pass-based methods (i.e. not with SGD) and using them to intervene on models during forward passes & change the model's output"*. This description would also include the following:

* [Inference-time intervention](https://www.lesswrong.com/posts/kuQfnotjkQA4Kkfou/inference-time-intervention-eliciting-truthful-answers-from), which focuses on inducing the behavioural change of "making the model tell the truth". It also looks at other non-forward-pass-based techniques for finding an intervention vector, e.g. CCS and linear probing, although it concludes that forward-pass-based methods similar to the ones we've been using so far work the best.
* [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), which can be thought of as an extension of the GPT2-XL steering vector work to larger models, specifically Llama 2 13B. It also takes more of a high-level evals framework; measuring the model's change in attributes such as sycophancy, myopia, and power-seeking (finding that these attributes can be increased or decreased by adding the appropriate vectors).

We'll discuss some of this work more in the bonus section, but for now, let's get on with the exercises!

First, we'll load in GPT2-XL, then we'll replicate some of the examples in the main post.

```python
gpt2_xl = LanguageModel("gpt2-xl", device_map="auto", torch_dtype=t.bfloat16)
tokenizer = gpt2_xl.tokenizer

REMOTE = False
# If you are using gpt2_xl, set REMOTE = False as gpt2_xl is not hosted remotely by nnsight. You can
# set REMOTE = True for a remotely hosted model here (https://nnsight.net/status/)
```

### Exercise - replicate the steering vector results

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 30-50 minutes on this exercise.
> ```

Replicate the results in the LessWrong post [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#fnrefcvnfx3e6sfu); specifically the "demonstrations of additions that work well" section.

Read the "How activation additions work" section of [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#How_activation_additions_work) to understand how vectors are extracted and added. We've provided a function template as well as some example code to run; your main job will be to fill in the function. This will be like a hybrid of several previous exercises (with most similarity to the function `calculate_and_intervene_with_h`), although there will be a few methodological differences.

This is the last exercise in this set, and hopefully it'll provide an opportunity to draw together all the threads of what you've learned so far!

### Caching

This is a different kind of causal intervention than we performed in previous sections. Rather than adding a single vector to the final sequence position at each token generation, we're adding a slice of vectors to the first sequence positions of the original prompt (see tables like in [this section](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#1__Love___Hate) for an illustration). How do you think this will affect our function? Should we still cache? Should we be using `.generate()` or `.trace()`? If using `.generate()`, do we need to call `model.next()` ?

<details>
<summary>Click this dropdown for answers to the questions above.</summary>

Rather than adding to each final sequence position for every token generated, we just add the vectors once, to the end of the prompt. This means that:

- We can still use caching (because the values we cache shouldn't be different in subsequent token generations),
- We should be using `.generate()` (because we're doing multi-token generation),
- We don't need to call `model.next()` (because we only intervene once, and our intervention will be cached & applied to all subsequent tokens which are generated).

Again, if any of this is confusing then please ask a TA or message in the Slack channel.

</details>

### Padding

The [tables](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#1__Love___Hate) show the activations being added on the left (i.e. the sequences are padded on the right), but by default padding is applied on the left. There are 2 possible ways you can get around this:

1. Right-pad the input sequences manually, i.e. use something like `len(tokenizer.tokenize(prompt))` to see how long each of the prompts is, and add copies of `tokenizer.pad_token` to the end of each sequence.
2. Don't manually pad the input sequences, instead slice the sequences you add to the original prompt from the right side of the activation addition sequences, rather than from the left side.

The solutions use (2), but you can use either of these methods.

### Sampling

Following the post, we'll use top-p sampling with probability 0.3 to generate our sequences. We'll also use a small frequency penalty to penalize repetition (so the model gets stuck in loops less). If you've done earlier exercises in this section then you might have implemented `freq_penalty` during sampling; this is supported by TransformerLens models, but HuggingFace uses the somewhat similar `repetition_penalty` (default value is 1.0 indicating no penalty, values higher than 1.0 apply a penalty to repeated tokens).

We apply these sampling methods by passing keyword arguments into the `generate` method:

```python
{
    "do_sample": True, # necessary whenever we're sampling rather than doing greedy decoding
    "top_p": 0.3,
    "repetition_penalty": 1.1,
}
```

Note that the sequences are generated stochastically rather than greedily - this means we'll get different results if we input multiple different copies of the same sequence. We've given you the `n_comparisons` argument in the function below, i.e. you should generate this many steered *and* this many unsteered completions.

### Other tips / notes

We recommend starting with example #9 (the "talking about weddings" one). It seems quite robust to the exact conditions of the forward pass, unlike the `Love - Hate` example. You can use any of the template cells we've given you below.

We've given you a `use_bos` argument; if this is True then you should append `tokenizer.bos_token` to the start of all the prompts. This is just to be true to the LessWrong post's implementation; it won't change behaviour much and you can probably ignore it and still get good results.

```python
SAMPLING_KWARGS = {
    "do_sample": True,
    "top_p": 0.3,
    "repetition_penalty": 1.2,
}

def calculate_and_apply_steering_vector(
    model: LanguageModel,
    prompt: str,
    activation_additions: list[tuple[int, float, str]],
    n_tokens: int,
    n_comparisons: int = 1,
    use_bos: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Performs the steering vector experiments described in the LessWrong post.

    Args:
        model: LanguageModel
            the transformer you're doing this computation with
        prompt: str
            The original prompt, which we'll be doing activation steering on.

        activation_additions: list[tuple[int, float, str]], each tuple contains:
            layer - the layer we're applying these steering vectors to
            coefficient - the value we're multiplying it by
            prompt - the prompt we're inputting
            e.g. activation_additions[0] = [6, 5.0, " Love"] means we add the " Love" vector at layer 6, scaled by 5x

        n_tokens: int
            Number of tokens which will be generated for each completion

        n_comparisons: int
            Number of sequences generated in this function (i.e. we generate `n_comparisons` which are unsteered, and
            the same number which are steered).

    Returns:
        unsteered_completions: list[str]
            List of length `n_comparisons`, containing all the unsteered completions.

        steered_completions: list[str]
            List of length `n_comparisons`, containing all the steered completions.
    """
    # Add the BOS token manually, if we're including it
    if use_bos:
        bos = model.tokenizer.bos_token
        prompt = bos + prompt
        activation_additions = [[layer, coeff, bos + p] for layer, coeff, p in activation_additions]

    raise NotImplementedError()

```

<details><summary>Solution</summary>

```python
SAMPLING_KWARGS = {
    "do_sample": True,
    "top_p": 0.3,
    "repetition_penalty": 1.2,
}

def calculate_and_apply_steering_vector(
    model: LanguageModel,
    prompt: str,
    activation_additions: list[tuple[int, float, str]],
    n_tokens: int,
    n_comparisons: int = 1,
    use_bos: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Performs the steering vector experiments described in the LessWrong post.

    Args:
        model: LanguageModel
            the transformer you're doing this computation with
        prompt: str
            The original prompt, which we'll be doing activation steering on.

        activation_additions: list[tuple[int, float, str]], each tuple contains:
            layer - the layer we're applying these steering vectors to
            coefficient - the value we're multiplying it by
            prompt - the prompt we're inputting
            e.g. activation_additions[0] = [6, 5.0, " Love"] means we add the " Love" vector at layer 6, scaled by 5x

        n_tokens: int
            Number of tokens which will be generated for each completion

        n_comparisons: int
            Number of sequences generated in this function (i.e. we generate `n_comparisons` which are unsteered, and
            the same number which are steered).

    Returns:
        unsteered_completions: list[str]
            List of length `n_comparisons`, containing all the unsteered completions.

        steered_completions: list[str]
            List of length `n_comparisons`, containing all the steered completions.
    """
    # Add the BOS token manually, if we're including it
    if use_bos:
        bos = model.tokenizer.bos_token
        prompt = bos + prompt
        activation_additions = [[layer, coeff, bos + p] for layer, coeff, p in activation_additions]

    # Get the (layers, coeffs, prompts) in an easier form to use, also calculate the prompt lengths & check they're all the same
    act_add_layers, act_add_coeffs, act_add_prompts = zip(*activation_additions)
    act_add_seq_lens = [len(tokenizer.tokenize(p)) for p in act_add_prompts]
    assert len(set(act_add_seq_lens)) == 1, "All activation addition prompts must be the same length."
    assert act_add_seq_lens[0] <= len(
        tokenizer.tokenize(prompt)
    ), "All act_add prompts should be shorter than original prompt."

    # Get the prompts we'll intervene on (unsteered and steered)
    steered_prompts = [prompt for _ in range(n_comparisons)]
    unsteered_prompts = [prompt for _ in range(n_comparisons)]

    with model.generate(max_new_tokens=n_tokens, remote=REMOTE, **SAMPLING_KWARGS) as generator:
        # Run the act_add prompts (i.e. the contrast pairs), and extract their activations
        with generator.invoke(act_add_prompts):
            # Get all the prompts from the activation additions, and put them in a list
            # (note, we slice from the end of the sequence because of left-padding)
            act_add_vectors = [
                model.transformer.h[layer].output[0][i, -seq_len:]
                for i, (layer, seq_len) in enumerate(zip(act_add_layers, act_add_seq_lens))
            ]

        # Forward pass on unsteered prompts (no intervention, no activations saved - we only need the completions)
        with generator.invoke(unsteered_prompts):
            unsteered_out = model.generator.output

        # Forward pass on steered prompts (we add in the results from the act_add prompts)
        with generator.invoke(steered_prompts):
            # For each act_add prompt, add the vector to residual stream, at the start of the sequence
            for i, (layer, coeff, seq_len) in enumerate(zip(act_add_layers, act_add_coeffs, act_add_seq_lens)):
                model.transformer.h[layer].output[0][:, :seq_len] += coeff * act_add_vectors[i]
            steered_out = model.generator.output

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions
```
</details>

To test your function, use any of the following code snippets (as mentioned, we recommend starting with the weddings example, since the results tend to be pretty robust).

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I hate you because",
    activation_additions=[(6, +5.0, "Love "), (6, -5.0, "Hate")],
    n_tokens=50,
    n_comparisons=3,
    use_bos=True,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I went up to my friend and said",
    activation_additions=[
        (20, +4.0, "I talk about weddings constantly  "),
        (20, -4.0, "I do not talk about weddings constantly"),
    ],
    n_tokens=50,
    n_comparisons=3,
    use_bos=False,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="To see the eiffel tower, people flock to",
    activation_additions=[
        (24, +10.0, "The Eiffel Tower is in Rome"),
        (24, -10.0, "The Eiffel Tower is in France"),
    ],
    n_tokens=50,
    n_comparisons=3,
    use_bos=False,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

# ☆ Bonus

## Extensions of the Function Vectors Paper

There are two other interesting results from the paper, although neither of them are as important as the ones we've covered so far. If you have time, you can try to reproduce these results yourself.

### The Decoded Vocabulary of Function Vectors (3.2)

In this section, the authors find the top words in the decoded vocabulary of the function vector (i.e. the words whose unembedding vectors have the highest dot product with the function vector), and show that these words seem conceptually related to the task. For example:

* For the antonyms task, the top words evoke the idea of antonyms, e.g. `" negate"`, `" counterpart"`, `" lesser"`.
* For the country-capitals task, the top words are actually the names of capitals, e.g. `" Moscow"`, `" Paris"`, `" Madrid"`.

Can you replicate these results, both with the antonyms task and with the task you chose in the previous section?

An interesting extension - what happens if you take a task like the Country-Capitals task (which is inherently asymmetric), and get your function vector from the symmetric version of the task (i.e. the one where each of your question-answer pairs might be flipped around)? Do you still get the same behavioural results, and how (if at all) do the decoded vocabulary results change?

```python
# YOUR CODE HERE - find the decoded vocabulary
```

<details>
<summary>My results for this (spoiler!)</summary>

In the Country-Capitals task, I found:

* The bidirectional task does still work to induce behavioural changes, although slightly less effectively than for the original task.
* The top decoded vocabulary items are a mix of country names and capital names, but mostly capitals.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Top logits:
' London'
' Moscow'
' Madrid'
' Budapest'
' Athens'
' Paris'
' Berlin'
' Bangkok'
' Istanbul'
' Montreal'
' Barcelona'
' Jerusalem'
' Seoul'
' Miami'
' Dublin'
' Atlanta'
' Copenhagen'
' Mumbai'
' Minneapolis'
' Beijing'</pre>

</details>

<details><summary>Solution</summary>

```python
# Code to calculate decoded vocabulary:
logits = model._model.lm_head(fn_vector)
max_logits = logits.topk(20).indices.tolist()
tokens = model.tokenizer.batch_decode(max_logits)
print("Top logits:\n" + "\n".join(map(repr, tokens)))
```
</details>

### Vector Algebra on Function Vectors (3.3)

In this section, the authors investigate whether function vectors can be composed. For instance, if we have three separate ICL tasks which in some sense compose to make a fourth task, can we add together the three function vectors of the first tasks, and use this as the function vector of the fourth task?

The authors test this on a variety of different tasks. They find that it's effective on some tasks (e.g. Country-Capitals, where it outperforms function vectors), but generally isn't as effective as function vectors. Do you get these same results?

## Extensions of the Steering Vectors Post

We only implemented one small subset of the results from the steering vectors post (and did it in a fairly slap-dash way). But there are many others you can play around with. For example:

* The authors note that they were unsuccessful in finding a "speak in French" vector. One of the top comments on the LessWrong post describes a process they used to create a French vector which happened to work (link to comment [here](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector?commentId=sqsS9QaDy2bG83XKP)). Can you replicate their results? (They also linked a Colab in this comment, which can help if you're stuck.)
* In a [later section](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#Perplexity_on_lots_of_sentences_about_weddings_or_about_shipping) of the paper, the authors extensively discuss perplexity (a measure which is related to entropy). They find that the "weddings" vector reduces perplexity on wedding-related sentences, and maintains perplexity on unrelated sentences. Can you replicate their results - in particular, their graph of perplexity ratios against injection layers for wedding and non-wedding-related sentences?
* The authors wrote up the post into a full paper, which you can find [here](https://arxiv.org/abs/2308.10248). Can you replicate some of the extra results in this paper?

## Suggested paper replications

### [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)

In this paper, the authors focus on inducing the behavioural change of "making the model tell the truth". They also look at other non-forward-pass-based techniques for finding an intervention vector, e.g. CCS and linear probing, although it concludes that forward-pass-based methods similar to the ones we've been using so far work the best.

This might be a good replication for you if:

* You enjoyed the exercises in this section, but are also interested in experimenting with techniques which weren't covered in this section (e.g. linear probing),
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in studying [model truthfulness](https://arxiv.org/abs/2109.07958).

### [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

This paper can be thought of as an extension of the GPT2-XL steering vector work to larger models, specifically Llama 2 13B. It also takes more of a high-level evals framework; measuring the model's change in attributes such as sycophancy, myopia, and power-seeking (finding that these attributes can be increased or decreased by adding the appropriate vectors).

This might be a good replication for you if:

* You enjoyed the exercises in this section, but want to apply these ideas in more of a behavioural context than a task-based context
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in [evaluating models](https://www.alignmentforum.org/posts/yRAo2KEGWenKYZG9K/discovering-language-model-behaviors-with-model-written) on traits like myopia, power seeking, etc,
* You're comfortable doing prompt-engineering, and working with large datasets (like the ones linked above).

*Update* - there is now a [LessWrong post](https://www.lesswrong.com/posts/v7f8ayBxLhmMFRzpa/steering-llama-2-with-contrastive-activation-additions) associated with this paper, which also briefly discusses related areas. We strongly recommend reading this post if you're interested in this replication, or any of the other suggested replications in this section.

### [Red-teaming language models via activation engineering](https://www.alignmentforum.org/posts/iHmsJdxgMEWmAfNne/red-teaming-language-models-via-activation-engineering)

This work, done by Nina Rimsky, extends the results from much of the work we've seen previously, but applied to the domain of **refusal** - what determines whether the LLM will refuse to answer your request, and how can you affect this behaviour? From her post:

> *Validating if finetuning and RLHF have robustly achieved the intended outcome is challenging ... We can try to trigger unwanted behaviors in models more efficiently by manipulating their internal states during inference rather than searching through many inputs. The idea is that if a behavior can be easily triggered through techniques such as activation engineering, it may also occur in deployment. The inability to elicit behaviors via small internal perturbations could serve as a stronger guarantee of safety.*

This might be a good replication for you if:

* You enjoyed the exercises in this section, but want to apply these ideas in more of a behavioural context than a task-based context,
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in RLHF, adversarial attacks and jailbreaking,
* You're comfortable doing prompt-engineering (although some of the data you'd need for this replication is available on Nina's [GitHub repo](https://github.com/nrimsky/LM-exp/tree/main)).

<br>

---

<br>

Note - for a week of work, we weakly suggest participants don't try one of these paper replications, because they're quite compute-heavy (even considering the fact that participants have the `nnsight` library at their disposal). There are many possible replications and extensions that can be done from the function vectors or GPT2-XL work, and this might be a better option for you if you enjoyed the exercises in this section and want to do more things like them.

However, if you do feel comfortable working with large models (e.g. you have some past experience of this) and you're interested in this work, then you're certainly welcome to try one of these replications!

---

# function_vectors_solutions.ipynb

# Function Vectors
**ARENA Function Vectors & Model Steering Tutorial**

This tutorial is adapted from the ARENA program material and serves as a fantastic introduction to running experiments in NNsight and working with function vectors and model steering. Thanks to Callum McDougall for writing this comprehensive tutorial and for allowing us to adapt the tutorial for NNsight users, and thanks to Eric Todd for writing the original function vector paper!

> **ARENA: [Streamlit Page](https://arena-chapter1-transformer-interp.streamlit.app/22_📚_[1.4.2]_Function_Vectors_&_Model_Steering)**
>
> **Colab: [exercises](https://colab.research.google.com/github/ndif-team/nnsight/blob/docs/docs/source/notebooks/tutorials/function_vectors_.ipynb) | [solutions](https://colab.research.google.com/github/ndif-team/nnsight/blob/docs/function_vectors_solutions.ipynb)**

You can collapse each section so only the headers are visible, by clicking the arrow symbol on the left hand side of the markdown header cells.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-14-2.png" width="350">

# Introduction

These exercises serve as an exploration of the following question: ***can we steer a model to produce different outputs / have a different behaviour, by intervening on the model's forward pass using vectors found by non gradient descent-based methods?***

The majority of the exercises focus on [function vectors](https://functions.baulab.info/): vectors which are extracted from forward passes on in-context learning (ICL) tasks, and added to the residual stream in order to trigger the execution of this task from a zero-shot prompt. The diagram below illustrates this.

<img src="https://functions.baulab.info/images/Paper/fv-demonstrations.png" width="650">

The exercises also take you through use of the `nnsight` library, which is designed to support this kind of work (and other interpretability research) on very large language models - i.e. larger than models like GPT2-Small which you might be used to at this point in the course.

The final set of exercises look at Alex Turner et al's work on [steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector), which is conceptually related but has different aims and methodologies.

## Content & Learning Objectives

### 1️⃣ Introduction to `nnsight`

In this section, you'll learn the basics of how to use the `nnsight` library: running forward passes on your model, and saving the internal states. You'll also learn some basics of HuggingFace models which translate over into `nnsight` models (e.g. tokenization, and how to work with model output).

> ##### Learning Objectives
>
> * Learn the basics of the `nnsight` library, and what it can be useful for
> * Learn some basics of HuggingFace models (e.g. tokenization, model output)
> * Use it to extract & visualise GPT-J-6B's internal activations

### 2️⃣ Task-encoding hidden states

We'll begin with the following question, posed by the Function Vectors paper:

> *When a transformer processes an ICL (in-context-learning) prompt with exemplars demonstrating task $T$, do any hidden states encode the task itself?*

We'll prove that the answer is yes, by constructing a vector $h$ from a set of ICL prompts for the **antonym task**, and intervening with our vector to make our model produce antonyms on zero-shot prompts.

This will require you to learn how to perform causal interventions with `nnsight`, not just save activations.

(Note - this section structurally follows section 2.1 of the function vectors paper).

> ##### Learning Objectives
>
> * Understand how `nnsight` can be used to perform causal interventions, and perform some yourself
> * Reproduce the "h-vector results" from the function vectors paper; that the residual stream does contain a vector which encodes the task and can induce task behaviour on zero-shot prompts

### 3️⃣ Function Vectors

In this section, we'll replicate the crux of the paper's results, by identifying a set of attention heads whose outputs have a large effect on the model's ICL performance, and showing we can patch with these vectors to induce task-solving behaviour on randomly shuffled prompts.

We'll also learn how to use `nnsight` for multi-token generation, and steer the model's behaviour. There exist exercises where you can try this out for different tasks, e.g. the Country-Capitals task, where you'll be able to steer the model to complete prompts like `"When you think of Netherlands, you usually think of"` by talking about Amsterdam.

(Note - this section structurally follows sections 2.2, 2.3 and some of section 3 from the function vectors paper).

> ##### Learning Objectives
>
> * Define a metric to measure the causal effect of each attention head on the correct performance of the in-context learning task
> * Understand how to rearrange activations in a model during an `nnsight` forward pass, to extract activations corresponding to a particular attention head
> * Learn how to use `nnsight` for multi-token generation

### 4️⃣ Steering Vectors in GPT2-XL

Here, we discuss a different but related set of research: Alex Turner's work on steering vectors. This also falls under the umbrella of "interventions in the residual stream using vectors found with forward pass (non-SGD) based methods in order to alter behaviour", but it has a different setup, objectives, and approach.

> ##### Learning Objectives
>
> * Understand the goals & main results from Alex Turner et al's work on steering vectors
> * Reproduce the changes in behaviour described in their initial post

### ☆ Bonus

Lastly, we discuss some possible extensions of function vectors & steering vectors work, which is currently an exciting area of development (e.g. with a paper on steering Llama 2-13b coming out as recently as December 2023).

## Setup code

```python
import os
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

chapter = "chapter1_transformer_interp"
repo = "ARENA_3.0"
branch = "main"

# Install dependencies
try:
    import nnsight
except:
    %pip install openai>=1.56.2 nnsight einops jaxtyping plotly transformer_lens==2.11.0 git+https://github.com/callummcdougall/CircuitsVis.git#subdirectory=python gradio typing-extensions
    %pip install --upgrade pydantic

# Get root directory, handling 3 different cases: (1) Colab, (2) notebook not in ARENA repo, (3) notebook in ARENA repo
root = (
    "/content"
    if IN_COLAB
    else "/root"
    if repo not in os.getcwd()
    else str(next(p for p in Path.cwd().parents if p.name == repo))
)

if Path(root).exists() and not Path(f"{root}/{chapter}").exists():
    if not IN_COLAB:
        !sudo apt-get install unzip
        %pip install jupyter ipython --upgrade

    if not os.path.exists(f"{root}/{chapter}"):
        !wget -P {root} https://github.com/callummcdougall/ARENA_3.0/archive/refs/heads/{branch}.zip
        !unzip {root}/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d {root}
        !mv {root}/{repo}-{branch}/{chapter} {root}/{chapter}
        !rm {root}/{branch}.zip
        !rmdir {root}/{repo}-{branch}

if f"{root}/{chapter}/exercises" not in sys.path:
    sys.path.append(f"{root}/{chapter}/exercises")

os.chdir(f"{root}/{chapter}/exercises")
```

```python
! pip install circuitsvis
! pip install plotly
! pip install jaxtyping
! pip install nnsight
```

```python
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import circuitsvis as cv
import einops
import numpy as np
import torch as t
from IPython.display import display
from jaxtyping import Float
from nnsight import CONFIG, LanguageModel
from openai import OpenAI
from rich import print as rprint
from rich.table import Table
from torch import Tensor

# Hide some info logging messages from nnsight
logging.disable(sys.maxsize)

t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part42_function_vectors_and_model_steering"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

import part42_function_vectors_and_model_steering.solutions as solutions
import part42_function_vectors_and_model_steering.tests as tests
from plotly_utils import imshow

MAIN = __name__ == "__main__"
```

# 1️⃣ Introduction to `nnsight`

> ##### Learning Objectives
>
> * Learn the basics of the `nnsight` library, and what it can be useful for
> * Learn some basics of HuggingFace models (e.g. tokenization, model output)
> * Use it to extract & visualise GPT-J-6B's internal activations

## Remote execution

We'll start by discussing [remote execution]((https://nnsight.net/notebooks/features/remote_execution/)) - the ability `nnsight` has to run models on an external server, which is one of the major benefits of the library as a research tool. This helps you bypass the memory & computational limits you might be faced with on your own machine. For remote execution to work, you need 2 things:

1. An API key from the NDIF login page, which you can request [here](https://login.ndif.us/)
2. The model you're working with being live - you can see all live models in the status page [here](https://nnsight.net/status/)

Note that the status page sometimes takes ~5 minutes to load all live models - click the dropdown below to see an example of what the status page should look like once the models have loaded. If you can't see the model you're looking for in this list, then you should set `REMOTE=False` for these exercises, or else make a request on the NDIF Discord to get the model live.

<details>
<summary>Example status page</summary>

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/ndif-status.png" width="650">

</details>

## Important syntax

Here, we'll discuss some important syntax for interacting with `nnsight` models. Since these models are extensions of HuggingFace models, some of this information (e.g. tokenization) applies to plain HuggingFace models as well as `nnsight` models, and some of it (e.g. forward passes) is specific to `nnsight`, i.e. it would work differently if you just had a standard HuggingFace model. Make sure to keep this distinction in mind, otherwise syntax can get confusing!

### Model config

Each model comes with a `model.config`, which contains lots of useful information about the model (e.g. number of heads and layers, size of hidden layers, etc.). You can access this with `model.config`. Run the code below to see this in action, and to define some useful variables for later.

```python
model = LanguageModel("EleutherAI/gpt-j-6b", device_map="auto", torch_dtype=t.bfloat16)
tokenizer = model.tokenizer

N_HEADS = model.config.n_head
N_LAYERS = model.config.n_layer
D_MODEL = model.config.n_embd
D_HEAD = D_MODEL // N_HEADS

print(f"Number of heads: {N_HEADS}")
print(f"Number of layers: {N_LAYERS}")
print(f"Model dimension: {D_MODEL}")
print(f"Head dimension: {D_HEAD}\n")

print("Entire config: ", model.config)
```

### Tokenizers

A model comes with a tokenizer, accessable with `model.tokenizer` (just like TransformerLens). Unlike TransformerLens, we won't be using utility functions like `model.to_str_tokens`, instead we'll be using the tokenizer directly. Some important functions for today's exercises are:

* `tokenizer` (i.e. just calling it on some input)
    * This takes in a string (or list of strings) and returns the tokenized version.
    * It will return a dictionary, always containing `input_ids` (i.e. the actual tokens) but also other things which are specific to the transformer model (e.g. `attention_mask` - see dropdown).
    * Other useful arguments for this function:
        * `return_tensors` - if this is `"pt"`, you'll get results returned as PyTorch tensors, rather than lists (which is the default).
        * `padding` - if True (default is False), the tokenizer can accept sequences of variable length. The shorter sequences get padded at the beginning (see dropdown below for more).
* `tokenizer.decode`
    * This takes in tokens, and returns the decoded string.
    * If the input is an integer, it returns the corresponding string. If the input is a list / 1D array of integers, it returns all those strings concatenated (which can sometimes not be what you want).
* `tokenizer.batch_decode`
    * Equivalent to `tokenizer.decode`, but it doesn't concatenate.
    * If the input is a list / 1D integer array, it returns a list of strings. If the input is 2D, it will concatenate within each list.
* `tokenizer.tokenize`
    * Takes in a string, and returns a list of strings.

Run the code below to see some examples of these functions in action.

```python
# Calling tokenizer returns a dictionary, containing input ids & other data.
# If returned as a tensor, then by default it will have a batch dimension.
print(tokenizer("This must be Thursday", return_tensors="pt"))

# Decoding a list of integers, into a concatenated string.
print(tokenizer.decode([40, 1239, 714, 651, 262, 8181, 286, 48971, 12545, 13]))

# Using batch decode, on both 1D and 2D input.
print(tokenizer.batch_decode([4711, 2456, 481, 307, 6626, 510]))
print(tokenizer.batch_decode([[1212, 6827, 481, 307, 1978], [2396, 481, 428, 530]]))

# Split sentence into tokens (note we see the special Ġ character in place of prepended spaces).
print(tokenizer.tokenize("This sentence will be tokenized"))
```

<details>
<summary>Note on <code>attention_mask</code> (optional)</summary>

`attention_mask`, which is a series of 1s and 0s. We mask attention at all 0-positions (i.e. we don't allow these tokens to be attended to). This is useful when you have to do padding. For example:

```python
model.tokenizer(["Hello world", "Hello"], return_tensors="pt", padding=True)
```

will return:

```python
{
    'attention_mask': tensor([[1, 1], [0, 1]]),
    'input_ids': tensor([[15496,   995], [50256, 15496]])
}
```

We can see how the shorter sequence has been padded at the beginning, and attention to this token will be masked.

</details>

### Model outputs

At a high level, there are 2 ways to run our model: using the `trace` method (a single forward pass) and the `generate` method (generating multiple tokens). We'll focus on `trace` for now, and we'll discuss `generate` when it comes to multi-token generation later.

The default behaviour of forward passes in normal HuggingFace models is to return an object containing logits (and optionally a bunch of other things). The default behaviour of `trace` in `nnsight` is to not return anything, because anything that we choose to return is explicitly returned inside the context manager.

Below is the simplest example of code to run the model (and also access the internal states of the model). Run it and look at the output, then read the explanation below. Remember to obtain and set an API key first if you're using remote execution!

```python
REMOTE = True

if IN_COLAB:
    # include your HuggingFace Token and NNsight API key on Colab secrets
    from google.colab import userdata
    NDIF_API = userdata.get('NDIF_API')
    CONFIG.set_default_api_key(NDIF_API)

prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt, remote=REMOTE):
    # Save the model's hidden states
    hidden_states = model.transformer.h[-1].output[0].save()

    # Save the model's logit output
    logits = model.lm_head.output[0, -1].save()

# Get the model's logit output, and it's next token prediction
print(f"logits.shape = {logits.shape} = (vocab_size,)")
print("Predicted token ID =", predicted_token_id := logits.argmax().item())
print(f"Predicted token = {tokenizer.decode(predicted_token_id)!r}")

# Print the shape of the model's residual stream
print(f"\nresid.shape = {hidden_states.shape} = (batch_size, seq_len, d_model)")
```

Lets go over this piece by piece.

**First, we create a context block** by calling `.trace(...)` on the model object. This denotes that we wish to generate tokens given some prompts.

```python
with model.trace(prompt, remote=REMOTE):
```

By default, running this will cause your model to be loaded & run locally, but by passing `remote=REMOTE`, it causes the model to be run on the server instead. This is very useful when working with models too large to fit on your machine (or even models which can fit on your machine, but run slowly due to their size, however if you're running this material on a sufficiently large GPU, you may prefer to set `REMOTE=False`).  The input argument can take a variety of formats: strings, lists of tokens, tensors of tokens, etc. Here, we've just used a string `prompt`.

The most interesting part of `nnsight` is the ability to access the model's internal states (like you might already have done with TransformerLens). Let's now see how this works!

```python
hidden_states = model.transformer.h[-1].output[0]
```

On this line we're saying: within our forward pass, access the last layer of the transformer `model.transformer.h[-1]`, access this layer's output `.output` (which is a tuple of tensors), index the first tensor in this tuple `.output[0]`.

Let's break down this line in a bit more detail:

* `model.transformer.h[-1]` is a module in our transformer.
    * If you `print(model)`, you'll see that it consists of `transformer` and `lm_head` (for "language modelling head"). The `transformer` module is made up of embeddings & dropout, a series of layers (called `.h`, for "hidden states"), and a final layernorm. So indexing `.h[-1]` gives you the final layer.
    * Note - it's often useful to visit the documentation page for whatever model you're working on, e.g. you can find GPT-J [here](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html). Not all models will have a nice uniform standardized architecture like you might be used to in TransformerLens!
* `.output[0]` gives you this module's output, as a **proxy**.
    * The output of a module is often a tuple (again, you can see on the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) what the output of each module is). In this case, it's a tuple of 2 tensors, the first of which is the actual layer output (the thing we want).
    * Doing operations on a proxy still returns a proxy - this is why we can index into the `output` proxy tuple and get a proxy tensor!

<details>
<summary>Optional exercise - we mentioned that <code>.output</code> returns a tuple of 2 tensors. Can you use the <a href="https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html">documentation page</a> what the second tensor in this tuple is?</summary>

The second output is also a tuple of tensors, of length 2. In the GPT-J source code, they are called `present`. They represent the keys and values which were calculated in this forward pass (as opposed to those that were calculated in an earlier forward pass, and cached by the model). Since we're only generating one new token, these are just the full keys and values.

</details>

<br>

The next command:

```python
logits = model.lm_head.output[0, -1]
```

can be understood in a very similar way. The only difference is that we're accessing the output of `lm_head`, the language modelling head (i.e. the unembedding at the very end), and the output is just a tensor of shape `(batch, seq, d_vocab)` rather than a tuple of tensors. Again, see the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) for this.

If you've worked with Hugging Face models then you might be used to getting logits directly from the model output, but here we generally extract logits from the model internals just like any other activation because this allows us to **control exactly what we return.** If we return lots of very large tensors, this can take quite a while to download from the server (remember that `d_vocab` is often very large for transformers, i.e. around 50k). See the "which objects to save" section below for more discussion on this.

### Output vs input

You can also extract a module's input using `.input` or `.inputs`. If a module's forward method is called as `module.forward(*args, **kwargs)` then `.inputs` returns a tuple of `(tuple_of_args, dict_of_kwargs)`. Alternatively, `.input` is an alias for `.inputs[0][0]`, in other words it returns the first arg from the module's forward method (which is usually the tensor we want).

Remember that if you're not sure then you can debug with `print(module.input.shape)` - even if `.inputs` is a tuple of inputs, this will work to recursively print the shape of all the tensors in the tuple, rather than causing an error.

### Which objects to save

Note that we saved `logits` above, which is a vector of length 50k. In general, it's best to save as small an object as possible, because this reduces the size of object you'll have to download from the server. For example, if you only want the next token completions, just argmax the logits and then save the result! All basic tensor operations can be performed within your context manager.

## Scan & Validate

A really cool feature in nnsight is the scan & validate mode, which allows you to efficiently debug without getting long uninterpretable error messages. For example, consider the code below, which tries to zero ablate one of the model's output tensors. Can you figure out what's wrong with it before running it?

```python
seq_len = len(model.tokenizer.encode(prompt))

try:
    with model.trace(prompt, remote=REMOTE):
        original_output = model.transformer.h[-1].output[0].clone()
        model.transformer.h[-1].output[0][:, seq_len] = 0
        modified_output = model.transformer.h[-1].output[0].save()

except Exception as e:
    print(f"Uninformative error message:\n  {e.__class__.__name__}: {e}")
```

If you guessed "we're indexing a tensor along a dimension of size `seq_len` with index `seq_len` which is an indexing error, you'd be correct! But the error message we get is pretty opaque. This is because of the way the objects in nnsight work: they're not tensors, they're tensor proxies, and can behave in funny ways sometimes.

If we want to debug, we should instead pass `scan=True` and `validate=True` into our `model.trace` call. `scan=True` means we run "fake inputs" through the model which incur no memory costs, and so can be done very quickly and cheaply to detect errors. `validate=True` will run tests during our forward pass that make our error messages more informative. When we use both, we get fast no-memory-cost operations with interpretable error messages!

```python
try:
    with model.trace(prompt, remote=REMOTE, scan=True, validate=True):
        original_output = model.transformer.h[-1].output[0].clone()
        print(f"{model.transformer.h[-1].output.shape=}\n")
        model.transformer.h[-1].output[0][:, seq_len] = 0
        modified_output = model.transformer.h[-1].output[0].save()

except Exception as e:
    print(f"Informative error message:\n  {e.__class__.__name__}: {e}")
```

It's possible to use `validate` without using `scan` (e.g. if you have any `assert proxy.shape == ...` then you must use `validate=True`), although we generally recommend using both when debugging, and then neither when you're finished debugging.

Also note that (as the example above shows) it's useful to use `scan=True, validate=True` when printing tensor shapes, at the initial exploration phase, if you're not exactly sure what the shape of a particular input / output will be. Even if your proxy objects are tuples of tensors, you can still call `.shape`, and it will return a tuple of the shapes of each tensor in the proxy!

## Putting this into practice

### Exercise - visualize attention heads

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

We just covered a lot of content, so lets put it into practice. Your first task is to extract the attention patterns from the zeroth layer of the transformer, and visualize them using circuitsvis. As a reminder, the syntax for circuitsvis is:

```python
cv.attention.attention_patterns(
    tokens=tokens,
    attention=attention,
)
```

where `tokens` is a list of strings, and `attention` is a tensor of shape `(num_heads, num_tokens, num_tokens)`.

If you're stuck, [here's a link](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) to the source code for GPT-J. Look for how the attention patterns are calculated, within the `GPTJAttention` block.

*Note - this model uses dropout on the attention probabilities, as you'll probably notice from looking at the source code in the link above. This won't affect the model's behaviour because dropout is disabled in inference mode (and using the `generate` method always puts a model in inference mode). But it is still a layer which exists in the model, so you can access its input or output just like any other module.*

<details>
<summary>Aside - inference mode</summary>

Dropout is one of the two main layers whose behaviour changes in inference mode (the other is BatchNorm).

If you want to run the model without inference mode, you can wrap your code in `with model.trace(inference=False):`. However, you don't need to worry about this for the purposes of these exercises.

</details>

If you're stuck on how to reference the right module, see the following hint:

<details>
<summary>Hint - what module you should get attention from</summary>

You want to extract attention from `model.transformer.h[0].attn.attn_dropout.input`. If you used `.output`, it would give you the same values (although they might differ by a dummy batch dimension). Both of these will return a single tensor, because dropout layers take just one input and return just one output.

</details>

<details>
<summary>Aside - GPT2 tokenizer uses special characters to represent space </summary>

GPT2 tokenizer uses "Ġ" to represent prepended space. So ["My", " name", " is", " James"] will be tokenized as ["My", "Ġname", "Ġis", "ĠJames"]. Make sure you replace "Ġ" with an actual space.

</details>

```python
with model.trace(prompt, remote=REMOTE):
    attn_patterns = model.transformer.h[0].attn.attn_dropout.input.save()

# Get string tokens (replacing special character for spaces)
str_tokens = model.tokenizer.tokenize(prompt)
str_tokens = [s.replace('Ġ', ' ') for s in str_tokens]

# Attention patterns (squeeze out the batch dimension)
attn_patterns_value = attn_patterns.squeeze(0)

print("Layer 0 Head Attention Patterns:")
display(cv.attention.attention_patterns(
    tokens=str_tokens,
    attention=attn_patterns_value,
))
```

<details>
<summary>Explanation</summary>

Explanation:

* Within the context managers:
    * We access the attention patterns by taking the input to the `attn_dropout`.
        * From the GPT-J source code, we can see that the attention weights are calculated by standard torch functions (and an unnamed `nn.Softmax` module) from the key and query vectors, and are then passed through the dropout layer before being used to calculate the attention layer output. So by accessing the input to the dropdown layer, we get the attention weights before dropout is applied.
        * Because of the previously discussed point about dropout not working in inference mode, we could also use the output of `attn_dropout`, and get the same values.
* Outside of the context managers:
    * We use the `tokenize` method to tokenize the prompt.

</details>

As an optional bonus exercise, you can verify for yourself that these are the correct attention patterns, by calculating them from scratch using the key and query vectors. Using `model.transformer.h[0].attn.q_proj.output` will give you the query vectors, and `k_proj` for the key vectors. However, one thing to be wary of is that GPT-J uses **rotary embeddings**, which makes the computation of attention patterns from keys and queries a bit harder than it would otherwise be. See [here](https://blog.eleuther.ai/rotary-embeddings/) for an in-depth discussion of rotary embeddings, and [here](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=bef36Bf9k7FYsCt1DpzCw6eV) for some rough intuitions.

# 2️⃣ Task-encoding hidden states

> ##### Learning Objectives
>
> * Understand how `nnsight` can be used to perform causal interventions, and perform some yourself
> * Reproduce the "h-vector results" from the function vectors paper; that the residual stream does contain a vector which encodes the task and can induce task behaviour on zero-shot prompts

We'll begin with the following question, posed by the Function Vectors paper:

> *When a transformer processes an ICL (in-context-learning) prompt with exemplars demonstrating task $T$, do any hidden states encode the task itself?*

We'll prove that the answer is yes, by constructing a vector $h$ from a set of ICL prompts for the **antonym task**, and intervening with our vector to make our model produce antonyms on zero-shot prompts.

This will require you to learn how to perform causal interventions with `nnsight`, not just save activations.

Note - this section structurally follows section 2.1 of the function vectors paper.

## ICL Task

### Exercise (optional) - generate your own antonym pairs

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵⚪⚪⚪
>
> If you choose to do this exercise, you should spend up to 10-30 minutes on it - depending on your familiarity with the OpenAI Python API.
> ```

We've provided you two options for the antonym dataset you'll use in these exercises.

1. Firstly, we've provided you a list of word pairs, in the file `data/antonym_pairs.txt`.
2. Secondly, if you want to run experiments like the ones in this paper, it can be good practice to learn how to generate prompts from GPT-4 or other models (this is how we generated the data for this exercise).

If you just want to use the provided list of words, skip this exercise and run the code below to load in the dataset from the text file. Alternatively, if you want to generate your own dataset, you can fill in the function `generate_dataset` below, which should query GPT-4 and get a list of antonym pairs.

See [here](https://platform.openai.com/docs/guides/gpt/chat-completions-api) for a guide to using the chat completions API, if you haven't already used it. Use the two dropdowns below (in order) for some guidance.

<details>
<summary>Getting started #1</summary>

Here is a recommended template:

```python
client = OpenAI(api_key=api_key)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": antonym_task},
        {"role": "assistant", "content": start_of_response},
    ]
)
```

where `antonym_task` explains the antonym task, and `start_of_respose` gives the model a prompt to start from (e.g. "Sure, here are some antonyms: ..."), to guide its subsequent behaviour.

</details>

<details>
<summary>Getting started #2</summary>

Here is an template you might want to use for the actual request:

```python
example_antonyms = "old: young, top: bottom, awake: asleep, future: past, "

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Give me {N} examples of antonym pairs. They should be obvious, i.e. each word should be associated with a single correct antonym."},
        {"role": "assistant", "content": f"Sure! Here are {N} pairs of antonyms satisfying this specification: {example_antonyms}"},
    ]
)
```

where `N` is the function argument. Note that we've provided a few example antonyms, and appended them to the start of GPT4's completion. This is a classic trick to guide the rest of the output (in fact, it's commonly used in adversarial attacks).

</details>

Note - it's possible that not all the antonyms returned will be solvable by GPT-J. In this section, we won't worry too much about this. When it comes to testing out our zero-shot intervention, we'll make sure to only use cases where GPT-J can actually solve it.

```python
def generate_antonym_dataset(N: int):
    """
    Generates 100 pairs of antonyms, in the form of a list of 2-tuples.
    """
    assert os.environ.get("OPENAI_API_KEY", None) is not None, "Please set your API key before running this function!"

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Generate {N} pairs of antonyms in the form of a list of 2-tuples. For example, [['old', 'young'], ['top', bottom'], ['awake', 'asleep']...].",
            },
            {"role": "assistant", "content": "Sure, here is a list of 100 antonyms: "},
        ],
    )
    return response

if os.environ.get("OPENAI_API_KEY", None) is not None:
    ANTONYM_PAIRS = generate_antonym_dataset(100)
    # Save the word pairs in a text file
    with open(section_dir / "data" / "my_antonym_pairs.txt", "w") as f:
        for word_pair in ANTONYM_PAIRS:
            f.write(f"{word_pair[0]} {word_pair[1]}\n")

# Load the word pairs from the text file
with open(section_dir / "data" / "antonym_pairs.txt", "r") as f:
    ANTONYM_PAIRS = [line.split() for line in f.readlines()]

print(ANTONYM_PAIRS[:10])
```

## ICL Dataset

To handle this list of word pairs, we've given you some helpful classes.

Firstly, there's the `ICLSequence` class, which takes in a list of word pairs and contains methods for constructing a prompt (and completion) from these words. Run the code below to see how it works.

```python
class ICLSequence:
    """
    Class to store a single antonym sequence.

    Uses the default template "Q: {x}\nA: {y}" (with separate pairs split by "\n\n").
    """

    def __init__(self, word_pairs: list[list[str]]):
        self.word_pairs = word_pairs
        self.x, self.y = zip(*word_pairs)

    def __len__(self):
        return len(self.word_pairs)

    def __getitem__(self, idx: int):
        return self.word_pairs[idx]

    def prompt(self):
        """Returns the prompt, which contains all but the second element in the last word pair."""
        p = "\n\n".join([f"Q: {x}\nA: {y}" for x, y in self.word_pairs])
        return p[: -len(self.completion())]

    def completion(self):
        """Returns the second element in the last word pair (with padded space)."""
        return " " + self.y[-1]

    def __str__(self):
        """Prints a readable string representation of the prompt & completion (indep of template)."""
        return f"{', '.join([f'({x}, {y})' for x, y in self[:-1]])}, {self.x[-1]} ->".strip(", ")

word_list = [["hot", "cold"], ["yes", "no"], ["in", "out"], ["up", "down"]]
seq = ICLSequence(word_list)

print("Tuple-representation of the sequence:")
print(seq)
print("\nActual prompt, which will be fed into the model:")
print(seq.prompt())
```

Secondly, we have the `ICLDataset` class. This is also fed a word pair list, and it has methods for generating batches of prompts and completions. It can generate both clean prompts (where each pair is actually an antonym pair) and corrupted prompts (where the answers for each pair are randomly chosen from the dataset).

```python
class ICLDataset:
    """
    Dataset to create antonym pair prompts, in ICL task format. We use random seeds for consistency
    between the corrupted and clean datasets.

    Inputs:
        word_pairs:
            list of ICL task, e.g. [["old", "young"], ["top", "bottom"], ...] for the antonym task
        size:
            number of prompts to generate
        n_prepended:
            number of antonym pairs before the single-word ICL task
        bidirectional:
            if True, then we also consider the reversed antonym pairs
        corrupted:
            if True, then the second word in each pair is replaced with a random word
        seed:
            random seed, for consistency & reproducibility
    """

    def __init__(
        self,
        word_pairs: list[list[str]],
        size: int,
        n_prepended: int,
        bidirectional: bool = True,
        seed: int = 0,
        corrupted: bool = False,
    ):
        assert n_prepended + 1 <= len(word_pairs), "Not enough antonym pairs in dataset to create prompt."

        self.word_pairs = word_pairs
        self.word_list = [word for word_pair in word_pairs for word in word_pair]
        self.size = size
        self.n_prepended = n_prepended
        self.bidirectional = bidirectional
        self.corrupted = corrupted
        self.seed = seed

        self.seqs = []
        self.prompts = []
        self.completions = []

        # Generate the dataset (by choosing random word pairs, and constructing `ICLSequence` objects)
        for n in range(size):
            np.random.seed(seed + n)
            random_pairs = np.random.choice(len(self.word_pairs), n_prepended + 1, replace=False)
            # Randomize the order of each word pair (x, y). If not bidirectional, we always have x -> y not y -> x
            random_orders = np.random.choice([1, -1], n_prepended + 1)
            if not (bidirectional):
                random_orders[:] = 1
            word_pairs = [self.word_pairs[pair][::order] for pair, order in zip(random_pairs, random_orders)]
            # If corrupted, then replace y with a random word in all (x, y) pairs except the last one
            if corrupted:
                for i in range(len(word_pairs) - 1):
                    word_pairs[i][1] = np.random.choice(self.word_list)
            seq = ICLSequence(word_pairs)

            self.seqs.append(seq)
            self.prompts.append(seq.prompt())
            self.completions.append(seq.completion())

    def create_corrupted_dataset(self):
        """Creates a corrupted version of the dataset (with same random seed)."""
        return ICLDataset(
            self.word_pairs,
            self.size,
            self.n_prepended,
            self.bidirectional,
            corrupted=True,
            seed=self.seed,
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int):
        return self.seqs[idx]
```

You can see how this dataset works below. **Note that the correct completions have a prepended space**, because this is how the antonym prompts are structured - the answers are tokenized as `"A: answer" -> ["A", ":", " answer"]`. Forgetting prepended spaces is a classic mistake when working with transformers!

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=False)

table = Table("Prompt", "Correct completion")
for seq, completion in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completion))

rprint(table)
```

Compare this output to what it looks like when `corrupted=True`. Each of the pairs before the last one has their second element replaced with a random one (but the last pair is unchanged).

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=10, n_prepended=2, corrupted=True)

table = Table("Prompt", "Correct completion")
for seq, completions in zip(dataset.seqs, dataset.completions):
    table.add_row(str(seq), repr(completions))

rprint(table)
```

<details>
<summary>Aside - the <code>rich</code> library</summary>

The `rich` library is a helpful little library to display outputs more clearly in a Python notebook or terminal. It's not necessary for this workshop, but it's a nice little tool to have in your toolbox.

The most important function is `rich.print` (usually imported as `rprint`). This can print basic strings, but it also supports the following syntax for printing colors:

```python
rprint("[green]This is green text[/], this is default color")
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-1.png" width="350">

and for making text bold / underlined:

```python
rprint("[u dark_orange]This is underlined[/], and [b cyan]this is bold[/].")
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-2.png" width="350">

It can also print tables:

```python
from rich.table import Table

table = Table("Col1", "Col2", title="Title") # title is optional
table.add_row("A", "a")
table.add_row("B", "b")

rprint(table)
```

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rprint-3.png" width="150">

The text formatting (bold, underlined, colors, etc) is also supported within table cells.

</details>

## Task-encoding vector

### Exercise - forward pass on antonym dataset

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

You should fill in the `calculate_h` function below. It should:

* Run a forward pass on the model with the dataset prompts (i.e. the `.prompts` attribute), using the `nnsight` syntax we've demonstrated previously,
* Return a tuple of the model's output (i.e. a list of its string-token completions, one for each prompt in the batch) and the residual stream value at the end of layer `layer` (e.g. if `layer = -1`, this means the final value of the residual stream before we convert into logits).

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-1.png" width="900">

You should only return the residual stream values for the very last sequence position in each prompt, i.e. the last `-1` token (where the model makes the antonym prediction), and same for the completions.

<details>
<summary> Help - I'm not sure how to run (and index into) a batch of inputs.</summary>

If we pass a list of strings to the `generator.invoke` function, this will be tokenized with padding automatically.

The type of padding which is applied is **left padding**, meaning if you index at sequence position `-1`, this will get the final token in the prompt for all prompts in the list, even if the prompts have different lengths.

</details>

```python
def calculate_h(model: LanguageModel, dataset: ICLDataset, layer: int = -1) -> tuple[list[str], Tensor]:
    """
    Averages over the model's hidden representations on each of the prompts in `dataset` at layer `layer`, to produce
    a single vector `h`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset whose prompts `dataset.prompts` you're extracting the activations from (at the last seq pos)
        layer: int
            the layer you're extracting activations from

    Returns:
        completions: list[str]
            list of the model's next-token predictions (i.e. the strings the model predicts to follow the last token)
        h: Tensor
            average hidden state tensor at final sequence position, of shape (d_model,)
    """
    with model.trace(dataset.prompts, remote=REMOTE):
        h = model.transformer.h[layer].output[0][:, -1].mean(dim=0).save()
        logits = model.lm_head.output[:, -1]
        next_tok_id = logits.argmax(dim=-1).save()

    completions = model.tokenizer.batch_decode(next_tok_id)
    return completions, h

tests.test_calculate_h(calculate_h, model)
```

We've provided you with a helper function, which displays the model's output on the antonym dataset (and highlights the examples where the model's prediction is correct). Note, we're using the `repr` function, because a lot of the completions are line breaks, and this helps us see them more clearly!

If the antonyms dataset was constructed well, you should find that the model's completion is correct most of the time, and most of its mistakes are either copying (e.g. predicting `wet -> wet` rather than `wet -> dry`) or understandable completions which shouldn't really be considered mistakes (e.g. predicting `right -> left` rather than `right -> wrong`). If we were being rigorous, we'd want to filter this dataset to make sure it only contains examples where the model can correctly perform the task - but for these exercises, we won't worry about this.

```python
def display_model_completions_on_antonyms(
    model: LanguageModel,
    dataset: ICLDataset,
    completions: list[str],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Prompt (tuple representation)",
        "Model's completion\n(green=correct)",
        "Correct completion",
        title="Model's antonym completions",
    )

    for i in range(min(len(completions), num_to_display)):
        # Get model's completion, and correct completion
        completion = completions[i]
        correct_completion = dataset.completions[i]
        correct_completion_first_token = model.tokenizer.tokenize(correct_completion)[0].replace("Ġ", " ")
        seq = dataset.seqs[i]

        # Color code the completion based on whether it's correct
        is_correct = completion == correct_completion_first_token
        completion = f"[b green]{repr(completion)}[/]" if is_correct else repr(completion)

        table.add_row(str(seq), completion, repr(correct_completion))

    rprint(table)

# Get uncorrupted dataset
dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=2)

# Getting it from layer 12, as in the description in section 2.1 of paper
model_completions, h = calculate_h(model, dataset, layer=12)

# Displaying the output
display_model_completions_on_antonyms(model, dataset, model_completions)
```

### Using multiple invokes

Another cool feature of `nnsight` is the ability to run multiple different batches through the model at once (or the same batch multiple times) in a way which leads to very clean syntax for doing causal interventions. Rather than doing something like this:

```python
with model.trace(inputs, remote=REMOTE):
    # some causal interventions
```

we can write a double-nested context manager:

```python
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(inputs):
        # some causal interventions

    with tracer.invoke(other_inputs):
        # some other causal interventions
```

Both inputs will be run together in parallel, and proxies defined within one `tracer.invoke` block can be used in another. A common use-case is to have clean and corrupted inputs, so we can patch from one to the other and get both outputs all in a single forward pass:

```python
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(clean_inputs):
        # extract clean activations
        clean_activations = model.transformer.h[10].output[0]

    with tracer.invoke(corrupted_inputs):
        # patch clean into corrupted
        model.transformer.h[10].output[0][:] = clean_activations
```

You'll do something like this in a later exercise. However for your first exercise (immediately below), you'll only be intervening with vectors that are defined outside of your context manager.

**One important thing to watch out for** - make sure you're not using your proxy before it's being defined! For example, if you were extracting `clean_activations` from `model.transformer.h[10]` but then intervening with it on `model.transformer.h[9]`, this couldn't be done in parallel (you'd need to first extract the clean activations, *then* run the patched forward pass). Doing this should result in a warning message, but may pass silently in some cases - so you need to be extra vigilant!

### Exercise - intervene with $h$

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

You should fill in the function `intervene_with_h` below. This will involve:

* Run two forward passes (within the same context manager) on a zero-shot dataset:
    * One with no intervention (i.e. the residual stream is unchanged),
    * One with an intervention using `h` (i.e. `h` is added to the residual stream at the layer it was taken from).
* Return the completions for no intervention and intervention cases respectively (see docstring).

The diagram below shows how all of this should work, when combined with the `calculate_h` function.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-2.png" width="950">

Hint - you can use `tokenizer.batch_decode` to turn a list of tokens into a list of strings.

<details>
<summary>Help - I'm not sure how best to get both the no-intervention and intervention completions.</summary>

You can use `with tracer.invoke...` more than once within the same context manager, in order to add to your batch. This will eventually give you output of shape (2*N, seq_len), which can then be indexed and reshaped to get the completions in the no intervention & intervention cases respectively.

</details>

<details>
<summary>Help - I'm not sure how to intervene on the hidden state.</summary>

First, you can define the tensor of hidden states (i.e. using `.output[0]`, like you've done before).

Then, you can add to this tensor directly (or add to some indexed version of it). You can use inplace operations (i.e. `tensor += h`) or redefining the tensor (i.e. `tensor = tensor + h`); either work.

</details>

```python
def intervene_with_h(
    model: LanguageModel,
    zero_shot_dataset: ICLDataset,
    h: Tensor,
    layer: int,
    remote: bool = REMOTE,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h` using previously defined function, and intervenes by adding `h` to the
    residual stream of a set of generated zero-shot prompts.

    Inputs:
        model: the model we're using to generate completions
        zero_shot_dataset: the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        h: the `h`-vector we'll be adding to the residual stream
        layer: the layer we'll be extracting the `h`-vector from
        remote: whether to run the forward pass on the remote server (used for running test code)

    Returns:
        completions_zero_shot: list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list of string completions for the zero-shot prompts, with h-intervention
    """
    with model.trace(remote=remote) as tracer:
        # First, run a forward pass where we don't intervene, just save token id completions
        with tracer.invoke(zero_shot_dataset.prompts):
            token_completions_zero_shot = model.lm_head.output[:, -1].argmax(dim=-1).save()

        # Next, run a forward pass on the zero-shot prompts where we do intervene
        with tracer.invoke(zero_shot_dataset.prompts):
            # Add the h-vector to the residual stream, at the last sequence position
            hidden_states = model.transformer.h[layer].output[0]
            hidden_states[:, -1] += h
            # Also save completions
            token_completions_intervention = model.lm_head.output[:, -1].argmax(dim=-1).save()

    # Decode to get the string tokens
    completions_zero_shot = model.tokenizer.batch_decode(token_completions_zero_shot)
    completions_intervention = model.tokenizer.batch_decode(token_completions_intervention)

    return completions_zero_shot, completions_intervention

tests.test_intervene_with_h(intervene_with_h, model, h, ANTONYM_PAIRS, REMOTE)
```

Run the code below to calculate completions for the function.

**Note, it's very important that we set a different random seed for the zero shot dataset, otherwise we'll be intervening on examples which were actually in the dataset we used to compute $h$!**

```python
layer = 12
dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

# Run previous function to get h-vector
h = calculate_h(model, dataset, layer=layer)[1]

# Run new function to intervene with h-vector
completions_zero_shot, completions_intervention = intervene_with_h(model, zero_shot_dataset, h, layer=layer)

print("Zero-shot completions: ", completions_zero_shot)
print("Completions with intervention: ", completions_intervention)
```

Next, run the code below to visualise the completions in a table. You should see:

* ~0% correct completions on the zero-shot prompt with no intervention, because the model usually just copies the first and only word in the prompt
* ~25% correct completions on the zero-shot prompt with intervention

```python
def display_model_completions_on_h_intervention(
    dataset: ICLDataset,
    completions: list[str],
    completions_intervention: list[str],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Prompt",
        "Model's completion\n(no intervention)",
        "Model's completion\n(intervention)",
        "Correct completion",
        title="Model's antonym completions",
    )

    for i in range(min(len(completions), num_to_display)):
        completion_ni = completions[i]
        completion_i = completions_intervention[i]
        correct_completion = dataset.completions[i]
        correct_completion_first_token = tokenizer.tokenize(correct_completion)[0].replace("Ġ", " ")
        seq = dataset.seqs[i]

        # Color code the completion based on whether it's correct
        is_correct = completion_i == correct_completion_first_token
        completion_i = f"[b green]{repr(completion_i)}[/]" if is_correct else repr(completion_i)

        table.add_row(str(seq), repr(completion_ni), completion_i, repr(correct_completion))

    rprint(table)

display_model_completions_on_h_intervention(zero_shot_dataset, completions_zero_shot, completions_intervention)
```

### Exercise - combine the last two functions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

One great feature of the `nnsight` library is its ability to parallelize forward passes and perform complex interventions within a single context manager.

In the code above, we had one function to extract the hidden states from the model, and another function where we intervened with those hidden states. But we can actually do both at once: we can compute $h$ within our forward pass, and then intervene with it on a different forward pass (using our zero-shot prompts), all within the same `model.trace` context manager. In other words, **we'll be using `with tracer.invoke...` three times** in this context manager.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/h-intervention-3.png" width="1000">

You should fill in the `calculate_h_and_intervene` function below, to do this. Mostly, this should involve combining your `calculate_h` and `intervene_with_h` functions, and wrapping the forward passes in the same context manager (plus a bit of code rewriting).

Your output should be exactly the same as before (since the `ICLDataset` class is deterministic), hence we've not provided test functions in this case - you can just compare the table you get to the one before! However, this time around your code should run twice as fast, because you're batching the operations of "compute $h$" and "intervene with $h$" together into a single forward pass.

<details>
<summary>Help - I'm not sure how to use the <code>h</code> vector inside the context manager.</summary>

You extract `h` the same way as before, but you don't need to save it. It is kept as a proxy. You can still use it later in the context manager, just like it actually was a tensor.

You shouldn't have to `.save()` anything inside your context manager, other than the token completions.

</details>
<details>
<summary>Help - If I want to add <code>x</code> vector to a slice of my hidden state tensor <code>h</code>, is <code>h[slice]+=x</code> the same as <code>h2 = h[slice], h2 += x</code>?</summary>

No, only `h[slice]+=x` does what you want. This is because when doing <code>h2 = h[slice], h2 += x</code>, the modification line <code>h2 += x</code> is no longer modifying the original tensor `h`, but a different tensor`h2`. In contrast, `h[slice]+=x` keeps the original tensor `h` in the modification line.

A good rule to keep in mind is: If you're trying to modify a tensor some in-place operation, make sure that tensor is in the actual modification line!

</details>

```python
def calculate_h_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[str], list[str]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the completions from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        completions_zero_shot: list[str]
            list of string completions for the zero-shot prompts, without intervention
        completions_intervention: list[str]
            list of string completions for the zero-shot prompts, with h-intervention
    """
    with model.trace(remote=REMOTE) as tracer:
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        with tracer.invoke(zero_shot_dataset.prompts):
            clean_tokens = model.lm_head.output[:, -1].argmax(dim=-1).save()

        with tracer.invoke(zero_shot_dataset.prompts):
            hidden = model.transformer.h[layer].output[0]
            hidden[:, -1] += h
            intervene_tokens = model.lm_head.output[:, -1].argmax(dim=-1).save()

    completions_zero_shot = tokenizer.batch_decode(clean_tokens)
    completions_intervention = tokenizer.batch_decode(intervene_tokens)
    return completions_zero_shot, completions_intervention

dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

completions_zero_shot, completions_intervention = calculate_h_and_intervene(
    model, dataset, zero_shot_dataset, layer=layer
)

display_model_completions_on_h_intervention(zero_shot_dataset, completions_zero_shot, completions_intervention)
```

### Exercise - compute change in accuracy

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-20 minutes on this exercise.
> ```

So far, all we've done is look at the most likely completions, and see what fraction of the time these were correct. But our forward pass doesn't just give us token completions, it gives us logits too!

You should now rewrite the `calculate_h_and_intervene` function so that, rather than returning two lists of string completions, it returns two lists of floats containing the **logprobs assigned by the model to the correct antonym** in the no intervention / intervention cases respectively.

<details>
<summary>Help - I don't know how to get the correct logprobs from the logits.</summary>

First, apply log softmax to the logits, to get logprobs.

Second, you can use `tokenizer(dataset.completions)["input_ids"]` to get the token IDs of the correct completions. (Gotcha - some words might be tokenized into multiple tokens, so make sure you're just picking the first token ID for each completion.)

Note - we recommend doing all this inside the context manager, then saving and returning just the correct logprobs not all the logits (this means less to download from the server!).

</details>

```python
def calculate_h_and_intervene_logprobs(
    model: LanguageModel,
    dataset: ICLDataset,
    zero_shot_dataset: ICLDataset,
    layer: int,
) -> tuple[list[float], list[float]]:
    """
    Extracts the vector `h`, intervenes by adding `h` to the residual stream of a set of generated zero-shot prompts,
    all within the same forward pass. Returns the logprobs on correct tokens from this intervention.

    Inputs:
        model: LanguageModel
            the model we're using to generate completions
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the `h`-vector
        zero_shot_dataset: ICLDataset
            the dataset of zero-shot prompts which we'll intervene on, using the `h`-vector
        layer: int
            the layer we'll be extracting the `h`-vector from

    Returns:
        correct_logprobs: list[float]
            list of correct-token logprobs for the zero-shot prompts, without intervention
        correct_logprobs_intervention: list[float]
            list of correct-token logprobs for the zero-shot prompts, with h-intervention
    """
    correct_completion_ids = [toks[0] for toks in tokenizer(zero_shot_dataset.completions)["input_ids"]]

    with model.trace(remote=REMOTE) as tracer:
        with tracer.invoke(dataset.prompts):
            h = model.transformer.h[layer].output[0][:, -1].mean(dim=0)

        with tracer.invoke(zero_shot_dataset.prompts):
            clean_logprobs = model.lm_head.output.log_softmax(dim=-1)[
                range(len(zero_shot_dataset)), -1, correct_completion_ids
            ].save()

        with tracer.invoke(zero_shot_dataset.prompts):
            hidden = model.transformer.h[layer].output[0]
            hidden[:, -1] += h
            intervene_logprobs = model.lm_head.output.log_softmax(dim=-1)[
                range(len(zero_shot_dataset)), -1, correct_completion_ids
            ].save()

    return clean_logprobs, intervene_logprobs
```

When you run the code below, it will display the log-probabilities (highlighting green when they increase from the zero-shot case). You should find that in every sequence, the logprobs on the correct token increase in the intervention. This helps make something clear - **even if the maximum-likelihood token doesn't change, this doesn't mean that the intervention isn't having a significant effect.**

```python
def display_model_logprobs_on_h_intervention(
    dataset: ICLDataset,
    correct_logprobs_zero_shot: list[float],
    correct_logprobs_intervention: list[float],
    num_to_display: int = 20,
) -> None:
    table = Table(
        "Zero-shot prompt",
        "Model's logprob\n(no intervention)",
        "Model's logprob\n(intervention)",
        "Change in logprob",
        title="Model's antonym logprobs, with zero-shot h-intervention\n(green = intervention improves accuracy)",
    )

    for i in range(min(len(correct_logprobs_zero_shot), num_to_display)):
        logprob_ni = correct_logprobs_zero_shot[i]
        logprob_i = correct_logprobs_intervention[i]
        delta_logprob = logprob_i - logprob_ni
        zero_shot_prompt = f"{dataset[i].x[0]:>8} -> {dataset[i].y[0]}"

        # Color code the logprob based on whether it's increased with this intervention
        is_improvement = delta_logprob >= 0
        delta_logprob = f"[b green]{delta_logprob:+.2f}[/]" if is_improvement else f"{delta_logprob:+.2f}"

        table.add_row(zero_shot_prompt, f"{logprob_ni:.2f}", f"{logprob_i:.2f}", delta_logprob)

    rprint(table)

dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=3, seed=0)
zero_shot_dataset = ICLDataset(ANTONYM_PAIRS, size=20, n_prepended=0, seed=1)

correct_logprobs_zero_shot, correct_logprobs_intervention = calculate_h_and_intervene_logprobs(
    model, dataset, zero_shot_dataset, layer=layer
)

display_model_logprobs_on_h_intervention(
    zero_shot_dataset, correct_logprobs_zero_shot, correct_logprobs_intervention
)
```

# 3️⃣ Function Vectors

> ##### Learning Objectives
>
> * Define a metric to measure the causal effect of each attention head on the correct performance of the in-context learning task
> * Understand how to rearrange activations in a model during an `nnsight` forward pass, to extract activations corresponding to a particular attention head
> * Learn how to use `nnsight` for multi-token generation

In this section, we'll replicate the crux of the paper's results, by identifying a set of attention heads whose outputs have a large effect on the model's ICL performance, and showing we can patch with these vectors to induce task-solving behaviour on randomly shuffled prompts.

We'll also learn how to use `nnsight` for multi-token generation, and steer the model's behaviour. There exist exercises where you can try this out for different tasks, e.g. the Country-Capitals task, where you'll be able to steer the model to complete prompts like `"When you think of Netherlands, you usually think of"` by talking about Amsterdam.

Note - this section structurally follows sections 2.2, 2.3 and some of section 3 from the function vectors paper.

Here, we'll move from thinking about residual stream states to thinking about the **output of specific attention heads.**

## Extracting & using FVs

### A note on `out_proj`

First, a bit of a technical complication. Most HuggingFace models don't have the nice attention head representations. What we have is the linear layer `out_proj` which implicitly combines the "projection per attention head" and the "sum over attention head" operations (if you can't see how this is possible, see the section "Attention Heads are Independent and Additive" from Anthropic's [Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html)).

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rearrange-output-2.png" width="950">

This presents some question for us, when it comes to causal interventions on attention heads. Use the dropdowns below to read them answer these questions (they'll be important for the coming exercises).

<br>

<details>
<summary>If we want to do a causal intervention on a particular head, should we intervene on <code>z</code> (the input of <code>out_proj</code>) or on <code>attn_output</code> (the output of <code>out_proj</code>) ?</summary>

We should intervene on `z`, because we can just rearrange the `z` tensor of shape `(batch, seq, d_model)` into `(batch, seq, n_heads, d_head)`, in other words separating out all the heads. On the other hand, we can't do this with the `attn_output` because it's *already* summed over heads and we can't separate them out.

</details>

<br>

<details>
<summary>How could we get the <code>attn_output</code> vector for a single head, if we had the ability to access model weights within our context managers?</summary>

We can take a slice of the `z` tensor corresponding to a single attention head:

```python
z.reshape(batch, seq, n_heads, d_head)[:, :, head_idx]
```

and we can take a slice of the `out_proj` weight matrix corresponding to a single attention head (remember that PyTorch stores linear layers in the shape `(out_feats, in_feats)`):

```python
out_proj.weight.rearrange(d_model, n_heads, d_head)[:, head_idx]
```

then finally we can multiply these together.

</details>

<br>

<details>
<summary>How could we get the <code>attn_output</code> vector for a single head, if we </b>didn't have</b> the ability to access model weights within our context managers? (This is currently the case for <code>nnsight</code>, since having access to the weights could allow users to change them!).</summary>

We can be a bit clever, and ablate certain heads in the `z` vector before passing it through the output projection:

```python
# ablate all heads except #2 (using a cloned activation)
heads_to_ablate = [0, 1, 3, 4, ...]
z_ablated = z.reshape(batch, seq, n_heads, d_head).clone()
z_ablated[:, :, heads_to_ablate] = 0

# save the output
attn_head_output = out_proj(z_ablated)
```

Illustration:

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/rearrange-output-ablated-2.png" width="950">

Note - this would actually fail if `out_proj` had a bias, because we want to just get an attention head's output, not the bias term as well. But if you look at the [documentation page](https://huggingface.co/transformers/v4.11.3/_modules/transformers/models/gptj/modeling_gptj.html) you'll see that `out_proj` doesn't have a bias term, so we're all good!

</details>

### Exercise - implement `calculate_fn_vectors_and_intervene`

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 30-60 minutes on this exercise.
> ```

This is probably the most important function in today's exercises. Implementing it will be pretty similar to the previous function `calculate_h_and_intervene`, but:

* Rather than extracting the value of the residual stream `h` at some particular layer, you'll be extracting the output of the attention heads: iterating over each layer and each head in the model.
    * You'll only need to run one clean forward pass to compute all these values, but you'll need to run a separate corrupted forward pass for each head.
* Rather than your 2 different datasets being (dataset, zero-shot dataset), your two datasets will be (dataset, corrupted version of that same dataset).
    * You can use the method `create_corrupted_dataset` method of the `ICLDataset` class for this.

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/cie-intervention.png" width="1200">

Before you actually start writing the code, it might be helpful to answer the following:

<details>
<summary>How many different <code>invoke</code> calls will you need in total?</summary>

You'll need `(N_LAYERS * N_HEADS) + 2`. To explain:

- One for the clean prompts, which you'll extract internal activations from and patch them into corrupted prompts,
- One for the corrupted prompts, which you don't intervene on,
- One for the corrupted prompts **for every attention head**, which you'll patch into using the clean run activations.

</details>

<details>
<summary>Which proxy outputs (if any) will you need to use <code>.save()</code> on, in this function?</summary>

You don't need to `.save()` the function vectors you're extracting from the model's internals, because these will only be used for causal interventions within the context manager.

The only thing you need to save is the correct token logprobs for (1) the corrupted forward pass where we don't intervene, and (2) each corrupted forward pass where we do intervene on one of the heads. In other words, you'll need to save `(N_LAYERS * N_HEADS) + 1` tensors in total.

</details>

A few other notes:

* We've added a `layers` argument, so you can iterate through different layers of the model (i.e. running the model with `layers = [3, 4, 5]` will only test the intervention on the attention heads in layers 3, 4 and 5). This is helpful if you're getting memory errors when trying to run all layers at once (remember we have 24 layers, 16 heads per layer, so even with few prompts per head this adds up fast!).
    * We've included code for you below showing how you can call the function multiple times, clearing memory between each run, then combine the results.
* When it comes to intervening, you can set the value of a reshaped tensor, i.e. `tensor.reshape(*new_shape)[index] = new_value` will change the values in `tensor` without actually reshaping it (for more on this, see the documentation for [`torch.Tensor.view`](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html)).
* It's good practice to insert a lot of assert statements in your code, to check the shapes are what you expect.
* If you're confused about dimensions, use `einops.rearrange` rather than `.reshape` - this is a wonderful tool, it's like using code annotations within your actual code!

One last note - **if this function is proving impossible to run for computational reasons, you can skip the exercise and move on to the next ones. They don't rely on this function working.** However, you should definitely at least read & understand the solution.

```python
def calculate_fn_vectors_and_intervene(
    model: LanguageModel,
    dataset: ICLDataset,
    layers: list[int] | None = None,
) -> Float[Tensor, "layers heads"]:
    """
    Returns a tensor of shape (layers, heads), containing the CIE for each head.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a corrupted
            version of this dataset for interventions)
        layers: list[int] | None
            the layers which this function will calculate the score for (if None, we assume all layers)
    """
    layers = range(model.config.n_layer) if (layers is None) else layers
    heads = range(model.config.n_head)

    # Get corrupted dataset
    corrupted_dataset = dataset.create_corrupted_dataset()
    N = len(dataset)

    # Get correct token ids, so we can get correct token logprobs
    correct_completion_ids = [toks[0] for toks in tokenizer(dataset.completions)["input_ids"]]

    with model.trace(remote=REMOTE) as tracer:
        # Run a forward pass on clean prompts, where we store attention head outputs
        z_dict = {}
        with tracer.invoke(dataset.prompts):
            for layer in layers:
                # Get hidden states, reshape to get head dimension, store the mean tensor
                z = model.transformer.h[layer].attn.out_proj.input[:, -1]
                z_reshaped = z.reshape(N, N_HEADS, D_HEAD).mean(dim=0)
                for head in heads:
                    z_dict[(layer, head)] = z_reshaped[head]

        # Run a forward pass on corrupted prompts, where we don't intervene or store activations (just so we can get the
        # correct-token logprobs to compare with our intervention)
        with tracer.invoke(corrupted_dataset.prompts):
            logits = model.lm_head.output[:, -1]
            correct_logprobs_corrupted = logits.log_softmax(dim=-1)[t.arange(N), correct_completion_ids].save()

        # For each head, run a forward pass on corrupted prompts (here we need multiple different forward passes, since
        # we're doing different interventions each time)
        correct_logprobs_dict = {}
        for layer in layers:
            for head in heads:
                with tracer.invoke(corrupted_dataset.prompts):
                    # Get hidden states, reshape to get head dimension, then set it to the a-vector
                    z = model.transformer.h[layer].attn.out_proj.input[:, -1]
                    z.reshape(N, N_HEADS, D_HEAD)[:, head] = z_dict[(layer, head)]
                    # Get logprobs at the end, which we'll compare with our corrupted logprobs
                    logits = model.lm_head.output[:, -1]
                    correct_logprobs_dict[(layer, head)] = logits.log_softmax(dim=-1)[
                        t.arange(N), correct_completion_ids
                    ].save()

    # Get difference between intervention logprobs and corrupted logprobs, and take mean over batch dim
    all_correct_logprobs_intervention = einops.rearrange(
        t.stack([v for v in correct_logprobs_dict.values()]),
        "(layers heads) batch -> layers heads batch",
        layers=len(layers),
    )
    logprobs_diff = all_correct_logprobs_intervention - correct_logprobs_corrupted  # shape [layers heads batch]

    # Return mean effect of intervention, over the batch dimension
    return logprobs_diff.mean(dim=-1)
```

As mentioned, the code below calls the function multiple times separately and combines the results.

When you run this code & plot the results, you should replicate Figure 3(a) in the Function Vectors paper (more or less). If the code is taking too long to run, we recommend just choosing a single layer to run, which has a distinctive pattern that can be compared to the paper's figure (e.g. layer 8, since head L8H1 has a much higher score than all the other heads in this layer).

```python
dataset = ICLDataset(ANTONYM_PAIRS, size=8, n_prepended=2)

def batch_process_layers(n_layers, batch_size):
    for i in range(0, n_layers, batch_size):
        yield range(n_layers)[i : i + batch_size]

results = t.empty((0, N_HEADS), device=device)

# If this fails to run, reduce the batch size so the fwd passes are split up more, or reduce dataset size
for layers in batch_process_layers(N_LAYERS, batch_size=4):
    print(f"Computing layers in {layers} ...")
    t0 = time.time()
    results = t.concat([results, calculate_fn_vectors_and_intervene(model, dataset, layers).to(device)])
    print(f"... finished in {time.time()-t0:.2f} seconds.\n")
```

```python
imshow(
    results.T,
    title="Average indirect effect of function-vector intervention on antonym task",
    width=1000,
    height=600,
    labels={"x": "Layer", "y": "Head"},
    aspect="equal",
)
```

### Exercise - calculate the function vector

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 25-50 minutes on this exercise.
> ```

Your next task is to actually calculate and return the function vector, so we can do a few experiments with it. The function vector is the sum of the outputs of all the attention heads we found using the previous function (i.e. the sum of all of the vectors these heads write to the residual stream), averaged over the prompts in our dataset.

There's a difficulty here - rather than just getting the `z` vectors, we're actually trying to get the `attn_out` vectors, but *before* they're summed over heads. As we discussed previously, this is a bit tricky to do for the model we're working with, because the `out_proj` linear map actually does the "project up" and "sum over heads" operations simultaneously. It would be nice to just take a slice of the `out_proj` matrix and multiply it with a slice of the `z` vector, but the `nnsight` library doesn't yet allow users to access weights directly (for security reasons). To understand how we can extract the `attn_out` vector for a head separately without accessing the underlying weights, you should go back to read the subsection **A note on `out_proj`** at the start of this section.

```python
def calculate_fn_vector(
    model: LanguageModel,
    dataset: ICLDataset,
    head_list: list[tuple[int, int]],
) -> Float[Tensor, "d_model"]:
    """
    Returns a vector of length `d_model`, containing the sum of vectors written to the residual stream
    by the attention heads in `head_list`, averaged over all inputs in `dataset`.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        dataset: ICLDataset
            the dataset of clean prompts from which we'll extract the function vector (we'll also create a
            corrupted version of this dataset for interventions)
        head_list: list[tuple[int, int]]
            list of attention heads we're calculating the function vector from
    """
    # Turn head_list into a dict of {layer: heads we need in this layer}
    head_dict = defaultdict(set)
    for layer, head in head_list:
        head_dict[layer].add(head)

    fn_vector_list = []

    with model.trace(dataset.prompts, remote=REMOTE):
        for layer, head_list in head_dict.items():
            # Get the output projection layer
            out_proj = model.transformer.h[layer].attn.out_proj

            # Get the mean output projection input (note, setting values of this tensor will not have
            # downstream effects on other tensors)
            hidden_states = out_proj.input[:, -1].mean(dim=0)

            # Zero-ablate all heads which aren't in our list, then get the output (which
            # will be the sum over the heads we actually do want!)
            heads_to_ablate = set(range(N_HEADS)) - head_dict[layer]
            for head in heads_to_ablate:
                hidden_states.reshape(N_HEADS, D_HEAD)[head] = 0.0

            # Now that we've zeroed all unimportant heads, get the output & add it to the list
            # (we need a single batch dimension so we can use `out_proj`)
            out_proj_output = out_proj(hidden_states.unsqueeze(0)).squeeze().save()
            fn_vector_list.append(out_proj_output)

    # We sum all attention head outputs to get our function vector
    fn_vector = sum([v for v in fn_vector_list])

    assert fn_vector.shape == (D_MODEL,)
    return fn_vector

tests.test_calculate_fn_vector(calculate_fn_vector, model)
```

## Multi-token generation

We're now going to replicate some of the results in Table 3, in the paper:

<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/tab3.png" width="700">

This will involve doing something we haven't done before - **intervening on multi-token prompt generation**.

Most of the interpretability exercises in this chapter have just consisted of running single forward passes, rather than autoregressive text generation. But we're trying something different here: we're adding the function vector to the final sequence position at each forward pass during text generation, and seeing if we can get the model to output a sentence with a different meaning.

The results of Table 3 came from adding the function vector to the residual stream at the final sequence position of the original prompt, **and the final sequence position for each subsequent generation.** The reason we do this is to guide the model's behaviour over time. Our hypothesis is that the function vector induces "next-token antonym behaviour" (because it was calculated by averaging attention head outputs at the sequence position before the model made its antonym prediction in the ICL prompts).

### Using `nnsight` for multi-token generation

Previously, our context managers have looked like:

```python
# Single invoke
with model.trace(prompt, remote=REMOTE):
    ... # Intervene on fwd pass

# Multiple invokes
with model.trace(remote=REMOTE) as tracer:
    with tracer.invoke(prompt):
        ... # Intervene on fwd pass
```

But for multi-token generation, we'll be using the `generate` method rather than `trace`. Our context managers will look like:

```python
# Single invoke
with model.generate(prompt, remote=REMOTE, max_new_tokens=max_new_tokens):
    with model.all(): # signals to NNsight that you want to run interventions performed on all generated tokens
        ... # Intervene on fwd pass for n-th token to be generated

# Multiple invokes
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with model.all():
        with generator.invoke(prompt):
            ... # Intervene on fwd pass for n-th token to be generated
        with generator.invoke(prompt2):
            ... # Intervene on fwd pass for n-th token to be generated
```

The line `with model.all():` denotes that the following interventions should be applied to the forward pass for all generated tokens.

Mostly, everything you learned during single-token generation generalizes to the multi-token case. For example, using `.save()` still saves proxies outside the context managers (although make sure that you don't use the same variable names over different generations, otherwise you'll overwrite them - it's easier to store your saved proxies in e.g. a list or dict).

Note that `model.generate` takes the same arguments as the normal [HuggingFace generate method](https://huggingface.co/docs/transformers/en/main_classes/text_generation). This means we can use arguments like `top_k`, `top_p`, or `repetition_penalty` to control generation behaviour. In the exercises below we use a repetition penalty (we choose a value of 1.2, in line with the [paper](https://arxiv.org/pdf/1909.05858) that suggested it) - this can avoid the model falling into loops of repeating the same sequence, which is especially common in steering when we're pushing the model OOD.

<!-- #### Optional questions - multi-token generation with NNsight

Here are a few quick optional questions to test your understanding of how multi-generation works with NNsight. These are non-essential, and only mentioned here as potentially helpful pointers.

<details>
<summary>How do I add vector <code>h</code> to all the tokens in the original prompt but not to the generated tokens? </summary>

```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):
        # Add vectors to the model's internals on the first forward pass
        model.transformer.h[layer].output[0][:, :seq_len] += h

```
You don't have to call `model.next()` because you're only adding the vector once to tokens in the original prompt. This will be cached when the model is subsequently generating tokens.

</details>

<details>
<summary>How do I intervene with vector <code>h</code> during the generation of the first k generated tokens? </summary>

To intervene during the generation of the first `k` generated tokens:
```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):

        for n in range(k+1):
            # Add vector to the model's internals, on the k-th forward pass
            model.transformer.h[layer].output[0] += h
            model.next()
```
When `n=0`, you are adding to tokens in the original prompt before a new token is a generated. After calling `model.next()`, you are accessing the hidden state of the last token that was generated (with seq_len=1).

</details>

</details>

<details>
<summary>How do I intervene with vector <code>h</code> only during the generation of the first k tokens, but not to tokens in the original prompt before the first generated token? </summary>

```python
with model.generate(max_new_tokens=max_new_tokens, remote=REMOTE) as generator:
    with generator.invoke(prompt):

        for n in range(k+1):
            model.next()
            # Add vector AFTER calling model.next() to add to the token that just got generated
            model.transformer.h[layer].output[0] += h

```
By not adding things before `model.next()`, we never add to the original prompt but always after a new token has been generated.

</details>

</details>

<details>
<summary>What is the difference between adding vector <code>h</code> before and after vector <code>model.next()</code>? </summary>

As explained in Q3, adding vector before `model.next()` means the operation is always done to the current sequence **before** a new generated token is appended. Adding vector after `model.next()` means the operation is always done to the newly generated token.

</details> -->

### Key-Value Caching

TLDR - caching can make causal interventions inside `model.generate` more complicated, but if you only intervene on sequence positions other than the very last one. In our exercises, we'll only be intervening on the last seqpos so you don't need to worry about it, but it's still a useful topic to understand.

<details>
<summary>See this dropdown if you're curious for more details.</summary>

To speed up inference, transformer models perform **key-value caching** to speed up text generation. This means that the time taken to generate $n$ tokens is ***much*** less than $n$ times longer than generating a single token. See [this blog post](https://kipp.ly/transformer-inference-arithmetic/) for more on transformer inference arithmetic.

When caching takes place, and we're doing causal interventions, we have to be careful that the caching won't override our causal interventions. Sometimes caching has to be disabled to make sure that our causal intervention works correctly. For example, if we wanted to perform the intervention "add the function vector to *only* the final sequence position of the prompt for each token we generate" then we'd have to disable caching (since previous forward passes would contain cached values where we intervened on a sequence position which is no longer the final sequence position). However, here we're performing the intervention "add the function vector to the final token of the original prompt, and to *all subsequent sequence positions*", meaning enabling caching (the default behaviour) will give us the right causal intervention.

</details>

### Generator Output

The object `generator.output` is by default a tensor which contains the model's token ID completions (not the logits).

By default the `generate` method will generate tokens greedily, i.e. always taking the maximum-probability token at each step. For now, we don't need to worry about changing this behaviour. But in future exercises we'll experiment with different sampling methods than greedy sampling (which generate uses by default), so `generator.output` and argmaxing over logits will not be identical!

### Exercise - intervene with function vector, in multi-token generation

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-30 minutes on this exercise.
> ```

You should now fill in the function `intervene_with_fn_vector` below. This will take a function vector (calculated from the function you wrote above), as well as a few other arguments (see docstring), and return the model's string completion on the given prompt template.

We hope to observe results qualitatively like the ones in Table 3, i.e. having the model define a particular word as its antonym.

```python
def intervene_with_fn_vector(
    model: LanguageModel,
    word: str,
    layer: int,
    fn_vector: Float[Tensor, "d_model"],
    prompt_template='The word "{x}" means',
    n_tokens: int = 5,
) -> tuple[str, str]:
    """
    Intervenes with a function vector, by adding it at the last sequence position of a generated prompt.

    Inputs:
        model: LanguageModel
            the transformer you're doing this computation with
        word: str
            The word which is substituted into the prompt template, via prompt_template.format(x=word)
        layer: int
            The layer we'll make the intervention (by adding the function vector)
        fn_vector: Float[Tensor, "d_model"]
            The vector we'll add to the final sequence position for each new token to be generated
        prompt_template:
            The template of the prompt we'll use to produce completions
        n_tokens: int
            The number of additional tokens we'll generate for our unsteered / steered completions

    Returns:
        completion: str
            The full completion (including original prompt) for the no-intervention case
        completion_intervention: str
            The full completion (including original prompt) for the intervention case
    """
    prompt = prompt_template.format(x=word)

    with model.generate(remote=REMOTE, max_new_tokens=n_tokens, repetition_penalty=1.2) as generator:
        with model.all():

            with generator.invoke(prompt):
                tokens = model.generator.output.save()

            with generator.invoke(prompt):
                model.transformer.h[layer].output[0][0, -1] += fn_vector
                tokens_intervention = model.generator.output.save()

    completion, completion_intervention = tokenizer.batch_decode(
        [tokens.squeeze().tolist(), tokens_intervention.squeeze().tolist()]
    )
    return completion, completion_intervention

```

To test your function, run the code below. You should find that the first completion seems normal, but the second completion defines a word as its antonym (you might have to play around a bit with the scale factor of `fn_vector`, to balance between effectiveness and coherence of output). If this works, congratulations - **you've just successfully induced an OOD behavioural change in a 6b-parameter model!**

```python
# Remove word from our pairs, so it can be a holdout
word = "light"
_ANTONYM_PAIRS = [pair for pair in ANTONYM_PAIRS if word not in pair]

# Define our dataset, and the attention heads we'll use
dataset = ICLDataset(_ANTONYM_PAIRS, size=20, n_prepended=5)
head_list = [
    (8, 0),
    (8, 1),
    (9, 14),
    (11, 0),
    (12, 10),
    (13, 12),
    (13, 13),
    (14, 9),
    (15, 5),
    (16, 14),
]

# Extract the function vector
fn_vector = calculate_fn_vector(model, dataset, head_list)

# Intervene with the function vector
completion, completion_intervention = intervene_with_fn_vector(
    model,
    word=word,
    layer=9,
    fn_vector=0.1 * fn_vector,
    prompt_template='The word "{x}" means',
    n_tokens=40,
)

table = Table("No intervention", "intervention")
table.add_row(repr(completion), repr(completion_intervention))
rprint(table)
```

### Exercise - generalize results to another task (optional)

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 15-30 minutes on this exercise.
> ```

In this exercise, you get to pick a task different to the antonyms task, and see if the results still hold up (for the same set of attention heads).

We'll leave this exercise fairly open-ended, without any code templates for you to fill in. However, if you'd like some guidance you can use the dropdown below.

<details>
<summary>Guidance for exercise</summary>

Whatever your task, you'll want to generate a new set of words. You can repurpose the `generate_dataset` function from the antonyms task, by supplying a different prompt and initial set of examples (this will require generating & using an OpenAI api key, if you haven't already), or you can just find an appropriate dataset online.

When you define the `ICLDataset`, you might want to use `bidirectional=False`, if your task isn't symmetric. The antonym task is symmetric, but others (e.g. the Country-Capitals task) are not.

You'll need to supply a new prompt template for the `intervene_with_fn_vector` function, but otherwise most of your code should stay the same.

</details>

```python
with open(section_dir / "data/country_capital_pairs.txt", "r", encoding="utf-8") as f:
    COUNTRY_CAPITAL_PAIRS = [line.split() for line in f.readlines()]

country = "Netherlands"
_COUNTRY_CAPITAL_PAIRS = [pair for pair in COUNTRY_CAPITAL_PAIRS if pair[0] != country]

dataset = ICLDataset(_COUNTRY_CAPITAL_PAIRS, size=20, n_prepended=5, bidirectional=False)
head_list = [
    (8, 0),
    (8, 1),
    (9, 14),
    (11, 0),
    (12, 10),
    (13, 12),
    (13, 13),
    (14, 9),
    (15, 5),
    (16, 14),
]

fn_vector = calculate_fn_vector(model, dataset, head_list)

# Intervene with the function vector
completion, completion_intervention = intervene_with_fn_vector(
    model=model,
    word=country,
    layer=9,
    fn_vector=0.05 * fn_vector,
    prompt_template="When you think of {x},",
    n_tokens=40,
)

table = Table("No intervention", "intervention")
table.add_row(repr(completion), repr(completion_intervention))
rprint(table)
```

# 4️⃣ Steering Vectors in GPT2-XL

> ##### Learning Objectives
>
> * Understand the goals & main results from Alex Turner et al's work on steering vectors
> * Reproduce the changes in behaviour described in their initial post

**Note**: GPT2-XL is not hosted remotely by NNsight at the moment. If you use GPT2-XL, we recommend setting `REMOTE = False`. Otherwise, you can use one of the remotely hosted models (see [here](https://nnsight.net/status/)) and set `REMOTE = True`.

## Steering model behaviour

In the final non-bonus exercise of the previous section, we touched on the idea of using function vectors to induce behavioural changes in the model's completions, rather than specifically making it solve zero-shot or corrupted prompts with the right completion. In these next exercises, we'll explore this kind of work in more detail. We'll be primarily using Turner et al's work on [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector).

Summary of the way in which this work differs from the function vector work we've done so far:

* Function vectors focused on the model performing a particular function (e.g. mapping a word to its opposite), whereas this work focuses on behavioural changes (e.g. completing a prompt which has negative tone in a positive way).
* Function vectors work looked at very large models (our exercises used Pythia-7B, the smallest model which was examined in the function vectors paper). This particular steering vectors post focused on the smaller models GPT2-Small (85m) and GPT2-XL (1.5B). We'll be focusing on GPT2-XL.
* The second half of our function vectors work identified important attention heads and focused on their outputs, rather than just adding to the residual stream directly. In this steering vector setup, we'll go back to the simpler method of adding directly into the residual stream.

Despite these differences, much of the work which was done here overlaps with function vector work, since they both fall into the broader category of *"finding vectors using forward-pass-based methods (i.e. not with SGD) and using them to intervene on models during forward passes & change the model's output"*. This description would also include the following:

* [Inference-time intervention](https://www.lesswrong.com/posts/kuQfnotjkQA4Kkfou/inference-time-intervention-eliciting-truthful-answers-from), which focuses on inducing the behavioural change of "making the model tell the truth". It also looks at other non-forward-pass-based techniques for finding an intervention vector, e.g. CCS and linear probing, although it concludes that forward-pass-based methods similar to the ones we've been using so far work the best.
* [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681), which can be thought of as an extension of the GPT2-XL steering vector work to larger models, specifically Llama 2 13B. It also takes more of a high-level evals framework; measuring the model's change in attributes such as sycophancy, myopia, and power-seeking (finding that these attributes can be increased or decreased by adding the appropriate vectors).

We'll discuss some of this work more in the bonus section, but for now, let's get on with the exercises!

First, we'll load in GPT2-XL, then we'll replicate some of the examples in the main post.

```python
gpt2_xl = LanguageModel("gpt2-xl", device_map="auto", torch_dtype=t.bfloat16)
tokenizer = gpt2_xl.tokenizer

REMOTE = False
# If you are using gpt2_xl, set REMOTE = False as gpt2_xl is not hosted remotely by nnsight. You can
# set REMOTE = True for a remotely hosted model here (https://nnsight.net/status/)
```

### Exercise - replicate the steering vector results

> ```yaml
> Difficulty: 🔴🔴🔴🔴🔴
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 30-50 minutes on this exercise.
> ```

Replicate the results in the LessWrong post [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#fnrefcvnfx3e6sfu); specifically the "demonstrations of additions that work well" section.

Read the "How activation additions work" section of [Steering GPT-2-XL by adding an activation vector](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#How_activation_additions_work) to understand how vectors are extracted and added. We've provided a function template as well as some example code to run; your main job will be to fill in the function. This will be like a hybrid of several previous exercises (with most similarity to the function `calculate_and_intervene_with_h`), although there will be a few methodological differences.

This is the last exercise in this set, and hopefully it'll provide an opportunity to draw together all the threads of what you've learned so far!

### Caching

This is a different kind of causal intervention than we performed in previous sections. Rather than adding a single vector to the final sequence position at each token generation, we're adding a slice of vectors to the first sequence positions of the original prompt (see tables like in [this section](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#1__Love___Hate) for an illustration). How do you think this will affect our function? Should we still cache? Should we be using `.generate()` or `.trace()`? If using `.generate()`, do we need to call `model.next()` ?

<details>
<summary>Click this dropdown for answers to the questions above.</summary>

Rather than adding to each final sequence position for every token generated, we just add the vectors once, to the end of the prompt. This means that:

- We can still use caching (because the values we cache shouldn't be different in subsequent token generations),
- We should be using `.generate()` (because we're doing multi-token generation),
- We don't need to call `model.next()` (because we only intervene once, and our intervention will be cached & applied to all subsequent tokens which are generated).

Again, if any of this is confusing then please ask a TA or message in the Slack channel.

</details>

### Padding

The [tables](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#1__Love___Hate) show the activations being added on the left (i.e. the sequences are padded on the right), but by default padding is applied on the left. There are 2 possible ways you can get around this:

1. Right-pad the input sequences manually, i.e. use something like `len(tokenizer.tokenize(prompt))` to see how long each of the prompts is, and add copies of `tokenizer.pad_token` to the end of each sequence.
2. Don't manually pad the input sequences, instead slice the sequences you add to the original prompt from the right side of the activation addition sequences, rather than from the left side.

The solutions use (2), but you can use either of these methods.

### Sampling

Following the post, we'll use top-p sampling with probability 0.3 to generate our sequences. We'll also use a small frequency penalty to penalize repetition (so the model gets stuck in loops less). If you've done earlier exercises in this section then you might have implemented `freq_penalty` during sampling; this is supported by TransformerLens models, but HuggingFace uses the somewhat similar `repetition_penalty` (default value is 1.0 indicating no penalty, values higher than 1.0 apply a penalty to repeated tokens).

We apply these sampling methods by passing keyword arguments into the `generate` method:

```python
{
    "do_sample": True, # necessary whenever we're sampling rather than doing greedy decoding
    "top_p": 0.3,
    "repetition_penalty": 1.1,
}
```

Note that the sequences are generated stochastically rather than greedily - this means we'll get different results if we input multiple different copies of the same sequence. We've given you the `n_comparisons` argument in the function below, i.e. you should generate this many steered *and* this many unsteered completions.

### Other tips / notes

We recommend starting with example #9 (the "talking about weddings" one). It seems quite robust to the exact conditions of the forward pass, unlike the `Love - Hate` example. You can use any of the template cells we've given you below.

We've given you a `use_bos` argument; if this is True then you should append `tokenizer.bos_token` to the start of all the prompts. This is just to be true to the LessWrong post's implementation; it won't change behaviour much and you can probably ignore it and still get good results.

```python
SAMPLING_KWARGS = {
    "do_sample": True,
    "top_p": 0.3,
    "repetition_penalty": 1.2,
}

def calculate_and_apply_steering_vector(
    model: LanguageModel,
    prompt: str,
    activation_additions: list[tuple[int, float, str]],
    n_tokens: int,
    n_comparisons: int = 1,
    use_bos: bool = True,
) -> tuple[list[str], list[str]]:
    """
    Performs the steering vector experiments described in the LessWrong post.

    Args:
        model: LanguageModel
            the transformer you're doing this computation with
        prompt: str
            The original prompt, which we'll be doing activation steering on.

        activation_additions: list[tuple[int, float, str]], each tuple contains:
            layer - the layer we're applying these steering vectors to
            coefficient - the value we're multiplying it by
            prompt - the prompt we're inputting
            e.g. activation_additions[0] = [6, 5.0, " Love"] means we add the " Love" vector at layer 6, scaled by 5x

        n_tokens: int
            Number of tokens which will be generated for each completion

        n_comparisons: int
            Number of sequences generated in this function (i.e. we generate `n_comparisons` which are unsteered, and
            the same number which are steered).

    Returns:
        unsteered_completions: list[str]
            List of length `n_comparisons`, containing all the unsteered completions.

        steered_completions: list[str]
            List of length `n_comparisons`, containing all the steered completions.
    """
    # Add the BOS token manually, if we're including it
    if use_bos:
        bos = model.tokenizer.bos_token
        prompt = bos + prompt
        activation_additions = [[layer, coeff, bos + p] for layer, coeff, p in activation_additions]

    # Get the (layers, coeffs, prompts) in an easier form to use, also calculate the prompt lengths & check they're all the same
    act_add_layers, act_add_coeffs, act_add_prompts = zip(*activation_additions)
    act_add_seq_lens = [len(tokenizer.tokenize(p)) for p in act_add_prompts]
    assert len(set(act_add_seq_lens)) == 1, "All activation addition prompts must be the same length."
    assert act_add_seq_lens[0] <= len(
        tokenizer.tokenize(prompt)
    ), "All act_add prompts should be shorter than original prompt."

    # Get the prompts we'll intervene on (unsteered and steered)
    steered_prompts = [prompt for _ in range(n_comparisons)]
    unsteered_prompts = [prompt for _ in range(n_comparisons)]

    with model.generate(max_new_tokens=n_tokens, remote=REMOTE, **SAMPLING_KWARGS) as generator:
        # Run the act_add prompts (i.e. the contrast pairs), and extract their activations
        with generator.invoke(act_add_prompts):
            # Get all the prompts from the activation additions, and put them in a list
            # (note, we slice from the end of the sequence because of left-padding)
            act_add_vectors = [
                model.transformer.h[layer].output[0][i, -seq_len:]
                for i, (layer, seq_len) in enumerate(zip(act_add_layers, act_add_seq_lens))
            ]

        # Forward pass on unsteered prompts (no intervention, no activations saved - we only need the completions)
        with generator.invoke(unsteered_prompts):
            unsteered_out = model.generator.output

        # Forward pass on steered prompts (we add in the results from the act_add prompts)
        with generator.invoke(steered_prompts):
            # For each act_add prompt, add the vector to residual stream, at the start of the sequence
            for i, (layer, coeff, seq_len) in enumerate(zip(act_add_layers, act_add_coeffs, act_add_seq_lens)):
                model.transformer.h[layer].output[0][:, :seq_len] += coeff * act_add_vectors[i]
            steered_out = model.generator.output

    # Decode steered & unsteered completions (discarding the sequences we only used for extracting activations) & return results
    unsteered_completions = tokenizer.batch_decode(unsteered_out[-n_comparisons:])
    steered_completions = tokenizer.batch_decode(steered_out[-n_comparisons:])

    return unsteered_completions, steered_completions

```

To test your function, use any of the following code snippets (as mentioned, we recommend starting with the weddings example, since the results tend to be pretty robust).

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I hate you because",
    activation_additions=[(6, +5.0, "Love "), (6, -5.0, "Hate")],
    n_tokens=50,
    n_comparisons=3,
    use_bos=True,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="I went up to my friend and said",
    activation_additions=[
        (20, +4.0, "I talk about weddings constantly  "),
        (20, -4.0, "I do not talk about weddings constantly"),
    ],
    n_tokens=50,
    n_comparisons=3,
    use_bos=False,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

```python
unsteered_completions, steered_completions = calculate_and_apply_steering_vector(
    gpt2_xl,
    prompt="To see the eiffel tower, people flock to",
    activation_additions=[
        (24, +10.0, "The Eiffel Tower is in Rome"),
        (24, -10.0, "The Eiffel Tower is in France"),
    ],
    n_tokens=50,
    n_comparisons=3,
    use_bos=False,
)

table = Table("Unsteered", "Steered", title="Completions", show_lines=True)
for usc, sc in zip(unsteered_completions, steered_completions):
    table.add_row(usc, sc)
rprint(table)
```

# ☆ Bonus

## Extensions of the Function Vectors Paper

There are two other interesting results from the paper, although neither of them are as important as the ones we've covered so far. If you have time, you can try to reproduce these results yourself.

### The Decoded Vocabulary of Function Vectors (3.2)

In this section, the authors find the top words in the decoded vocabulary of the function vector (i.e. the words whose unembedding vectors have the highest dot product with the function vector), and show that these words seem conceptually related to the task. For example:

* For the antonyms task, the top words evoke the idea of antonyms, e.g. `" negate"`, `" counterpart"`, `" lesser"`.
* For the country-capitals task, the top words are actually the names of capitals, e.g. `" Moscow"`, `" Paris"`, `" Madrid"`.

Can you replicate these results, both with the antonyms task and with the task you chose in the previous section?

An interesting extension - what happens if you take a task like the Country-Capitals task (which is inherently asymmetric), and get your function vector from the symmetric version of the task (i.e. the one where each of your question-answer pairs might be flipped around)? Do you still get the same behavioural results, and how (if at all) do the decoded vocabulary results change?

```python
# YOUR CODE HERE - find the decoded vocabulary
```

<details>
<summary>My results for this (spoiler!)</summary>

In the Country-Capitals task, I found:

* The bidirectional task does still work to induce behavioural changes, although slightly less effectively than for the original task.
* The top decoded vocabulary items are a mix of country names and capital names, but mostly capitals.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Top logits:
' London'
' Moscow'
' Madrid'
' Budapest'
' Athens'
' Paris'
' Berlin'
' Bangkok'
' Istanbul'
' Montreal'
' Barcelona'
' Jerusalem'
' Seoul'
' Miami'
' Dublin'
' Atlanta'
' Copenhagen'
' Mumbai'
' Minneapolis'
' Beijing'</pre>

</details>

<details><summary>Solution</summary>

```python
# Code to calculate decoded vocabulary:
logits = model._model.lm_head(fn_vector)
max_logits = logits.topk(20).indices.tolist()
tokens = model.tokenizer.batch_decode(max_logits)
print("Top logits:\n" + "\n".join(map(repr, tokens)))
```
</details>

### Vector Algebra on Function Vectors (3.3)

In this section, the authors investigate whether function vectors can be composed. For instance, if we have three separate ICL tasks which in some sense compose to make a fourth task, can we add together the three function vectors of the first tasks, and use this as the function vector of the fourth task?

The authors test this on a variety of different tasks. They find that it's effective on some tasks (e.g. Country-Capitals, where it outperforms function vectors), but generally isn't as effective as function vectors. Do you get these same results?

## Extensions of the Steering Vectors Post

We only implemented one small subset of the results from the steering vectors post (and did it in a fairly slap-dash way). But there are many others you can play around with. For example:

* The authors note that they were unsuccessful in finding a "speak in French" vector. One of the top comments on the LessWrong post describes a process they used to create a French vector which happened to work (link to comment [here](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector?commentId=sqsS9QaDy2bG83XKP)). Can you replicate their results? (They also linked a Colab in this comment, which can help if you're stuck.)
* In a [later section](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector#Perplexity_on_lots_of_sentences_about_weddings_or_about_shipping) of the paper, the authors extensively discuss perplexity (a measure which is related to entropy). They find that the "weddings" vector reduces perplexity on wedding-related sentences, and maintains perplexity on unrelated sentences. Can you replicate their results - in particular, their graph of perplexity ratios against injection layers for wedding and non-wedding-related sentences?
* The authors wrote up the post into a full paper, which you can find [here](https://arxiv.org/abs/2308.10248). Can you replicate some of the extra results in this paper?

## Suggested paper replications

### [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341)

In this paper, the authors focus on inducing the behavioural change of "making the model tell the truth". They also look at other non-forward-pass-based techniques for finding an intervention vector, e.g. CCS and linear probing, although it concludes that forward-pass-based methods similar to the ones we've been using so far work the best.

This might be a good replication for you if:

* You enjoyed the exercises in this section, but are also interested in experimenting with techniques which weren't covered in this section (e.g. linear probing),
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in studying [model truthfulness](https://arxiv.org/abs/2109.07958).

### [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

This paper can be thought of as an extension of the GPT2-XL steering vector work to larger models, specifically Llama 2 13B. It also takes more of a high-level evals framework; measuring the model's change in attributes such as sycophancy, myopia, and power-seeking (finding that these attributes can be increased or decreased by adding the appropriate vectors).

This might be a good replication for you if:

* You enjoyed the exercises in this section, but want to apply these ideas in more of a behavioural context than a task-based context
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in [evaluating models](https://www.alignmentforum.org/posts/yRAo2KEGWenKYZG9K/discovering-language-model-behaviors-with-model-written) on traits like myopia, power seeking, etc,
* You're comfortable doing prompt-engineering, and working with large datasets (like the ones linked above).

*Update* - there is now a [LessWrong post](https://www.lesswrong.com/posts/v7f8ayBxLhmMFRzpa/steering-llama-2-with-contrastive-activation-additions) associated with this paper, which also briefly discusses related areas. We strongly recommend reading this post if you're interested in this replication, or any of the other suggested replications in this section.

### [Red-teaming language models via activation engineering](https://www.alignmentforum.org/posts/iHmsJdxgMEWmAfNne/red-teaming-language-models-via-activation-engineering)

This work, done by Nina Rimsky, extends the results from much of the work we've seen previously, but applied to the domain of **refusal** - what determines whether the LLM will refuse to answer your request, and how can you affect this behaviour? From her post:

> *Validating if finetuning and RLHF have robustly achieved the intended outcome is challenging ... We can try to trigger unwanted behaviors in models more efficiently by manipulating their internal states during inference rather than searching through many inputs. The idea is that if a behavior can be easily triggered through techniques such as activation engineering, it may also occur in deployment. The inability to elicit behaviors via small internal perturbations could serve as a stronger guarantee of safety.*

This might be a good replication for you if:

* You enjoyed the exercises in this section, but want to apply these ideas in more of a behavioural context than a task-based context,
* You're comfortable working with very large models, possibly via the `nnsight` library,
* You're interested in RLHF, adversarial attacks and jailbreaking,
* You're comfortable doing prompt-engineering (although some of the data you'd need for this replication is available on Nina's [GitHub repo](https://github.com/nrimsky/LM-exp/tree/main)).

<br>

---

<br>

---

# getting.ipynb

# Getting Values

Hidden states are exposed by accessing the desired module and calling its `.input` or `.output` attributes.

Once accessed, you call `.save()` on it so it's value is populated and not deleted after.

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace("The Eiffel Tower is in the city of") as tracer:

    hidden_states = model.transformer.h[-1].output[0].save()
```

After exiting the tracing context, the `.value` attribute of the `hidden_states` object will be populated.

```python
print(hidden_states)
```

---

# gradients.ipynb

# Gradients

There are a couple of ways we can interact with the gradients during and after a backward pass.

In the following example, we save the hidden states of the last layer and do a backward pass on the sum of the logits.

Note two things:

1. `requires_grad=True` by default.
2. We can all `.backward()` on a value within the tracing context just like you normally would.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel("openai-community/gpt2", device_map="auto")
```

```python
with model.trace("Hello World") as tracer:

    hidden_states = model.transformer.h[-1].output[0].save()

    logits = model.output.logits

    logits.sum().backward()

print(hidden_states)
```

If we wanted to see the gradients for the hidden_states, we can call `.retain_grad()` on it and access the `.grad` attribute after execution.

```python
with model.trace("Hello World") as tracer:

    hidden_states = model.transformer.h[-1].output[0].save()
    hidden_states_grad = model.transformer.h[-1].output[0].grad.save()

    model.output.logits.sum().backward()

print(hidden_states)
print(hidden_states_grad)
```

Even better, `nnsight` also provides proxy access into the backward process via the `.grad` attribute on proxies. This works just like  `.input` and `.output` where operations , including getting and setting, are traced and performed on the model at runtime. (assuming it's a proxy of a Tensor, as this calls `.register_hook(...)` on it!)

The following examples demonstrate ablating (setting to zero) the gradients for a hidden state in GPT-2. The first example is an in-place operation and the second swaps the gradient out for a new tensor of zeroes.

```python
with model.trace("Hello World") as tracer:
    hidden_states = model.transformer.h[-1].output[0].save()

    hidden_states_grad_before = hidden_states.grad.clone().save()
    hidden_states.grad[:] = 0
    hidden_states_grad_after = hidden_states.grad.save()

    logits = model.output.logits

    logits.sum().backward()

print("Before", hidden_states_grad_before)
print("After", hidden_states_grad_after)
```

```python
with model.trace("Hello World") as tracer:
    hidden_states = model.transformer.h[-1].output[0].save()

    hidden_states_grad_before = hidden_states.grad.clone().save()
    hidden_states.grad = torch.zeros(hidden_states.grad.shape).to(hidden_states.grad.device)
    hidden_states_grad_after = hidden_states.grad.save()

    logits = model.output.logits

    logits.sum().backward()

print("Before", hidden_states_grad_before)
print("After", hidden_states_grad_after)
```

---

# iterator.ipynb

# Iterative Interventions

NNsight's <b> iterator context </b> allows us to run an intervention loop at scale. It iteratively executes and updates a single intervention graph.

Use a `session` to define the Iterator context and pass in a sequence of items that you want to loop over at each iteration:

```python
import nnsight
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.session() as session:

  with session.iter([0, 1, 2]) as item:
    # define intervention body here ...

    with model.trace("_"):
      # define interventions here ...
      pass

    with model.trace("_"):
      # define interventions here ...
      pass
```

The Iterator context extends all the nnsight graph-based functionalities, but also closely mimics the conventional `for` loop statement in Python, which allows it to support all kind of iterative operations with a use of `as item` syntax.

Beyond specifying iteration indices, you can also loop across an NNsight list object (`nnsight.list()`).

```python
with model.session() as session:

  li = nnsight.list() # an NNsight built-in list object
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

`nnsight 0.4` introduces support for native Python for loops within a tracer context at scale!

*NOTE: inline for loops (i.e., `[x for x in <Proxy object>]`) are not currently supported.*

```python
# New: Using Python for loops for iterative interventions
with model.session() as session:

    li = nnsight.list()
    [li.append([num]) for num in range(0, 3)]
    li2 = nnsight.list().save()

    # Using regular for loops
    for item in li:
        for item_2 in item: # for loops can be nested!
            li2.append(item_2)

print("\nList: ", li2)
```

## Considerations

If you would like to turn off NNsight's support of native `for` loops, you can apply the following changes to `nnsight.CONFIG`

This will not affect any of NNsight's `.iter()` functionality.

```python
# Turn off support if/for statements within tracing context.
import nnsight

nnsight.CONFIG.APP.CONTROL_FLOW_HANDLING = False
nnsight.CONFIG.save()
```

---

# logit_lens.ipynb

# Logit Lens

## Introduction

🔍 Logit Lens is a powerful tool that grants us a simplified (yet insightful) understanding of the inner workings of transformer models.

We can estimate the model's guess for the output after each computational step by applying a softmax function to each layer's output. Unlike traditional approaches focusing on *how* beliefs are updated within a step, with Logit Lens we gain a glimpse into *what* output the model is predicting at each processing step.

📗 Read more about Logit Lens from nostalgebraist’s blog post on LessWrong, [here](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

💻 You can find a Colab version of our tutorial [here](https://colab.research.google.com/github/ndif-team/nnsight/blob/main/docs/source/notebooks/tutorials/logit_lens.ipynb), or nostalgebraist’s original code [here](https://colab.research.google.com/drive/1-nOE-Qyia3ElM17qrdoHAtGmLCPUZijg?usp=sharing)

## Setup

If using Colab, install NNsight:
```
!pip install -U nnsight
```

```python
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
```

Import libraries and load GPT-2 model.

```python
# Import libraries
from IPython.display import clear_output
from nnsight import LanguageModel
from typing import List, Callable
import torch
import numpy as np
from IPython.display import clear_output

clear_output()
```

```python
# Load gpt2
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

## GPT-2 Model Architecture

Let's take a look at GPT-2's architecture. GPT-2 has 12 layers, accessed as `model.transformer.h`.

```python
print(model)
```

## Apply Logit Lens

To apply logit lens, we collect activations at each layer's output, apply layer normalization (`model.transformer.ln_f`), and then process through the model's head (`model.lm_head`) to get the logits. Next, we apply the softmax to the logits to obtain output token probabilities.

By observing different layers' output token probabilities, logit lens provides insights into the model's confidence throughout processing steps.

```python
prompt= "The Eiffel Tower is in the city of"
layers = model.transformer.h
probs_layers = []

with model.trace() as tracer:
    with tracer.invoke(prompt) as invoker:
        for layer_idx, layer in enumerate(layers):
            # Process layer output through the model's head and layer normalization
            layer_output = model.lm_head(model.transformer.ln_f(layer.output[0]))

            # Apply softmax to obtain probabilities and save the result
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            probs_layers.append(probs)

probs = torch.cat([probs.value for probs in probs_layers])

# Find the maximum probability and corresponding tokens for each position
max_probs, tokens = probs.max(dim=-1)

# Decode token IDs to words for each layer
words = [[model.tokenizer.decode(t.cpu()).encode("unicode_escape").decode() for t in layer_tokens]
    for layer_tokens in tokens]

# Access the 'input_ids' attribute of the invoker object to get the input words
input_words = [model.tokenizer.decode(t) for t in invoker.inputs[0][0]["input_ids"][0]]
```

## Visualizing GPT-2 Layer Interpretations

Now we will visualize the prediction of the GPT-2 model while processing the string *`'The Eiffel Tower is in the city of'`* and we’ll explore the interpretations of each layer within the GPT2Block, gaining insights into what each layer believes could be the next word for every input word.

```python
import plotly.express as px
import plotly.io as pio

if is_colab:
    pio.renderers.default = "colab"
else:
    pio.renderers.default = "plotly_mimetype+notebook_connected+notebook"

fig = px.imshow(
    max_probs.detach().cpu().numpy(),
    x=input_words,
    y=list(range(len(words))),
    color_continuous_scale=px.colors.diverging.RdYlBu_r,
    color_continuous_midpoint=0.50,
    text_auto=True,
    labels=dict(x="Input Tokens", y="Layers", color="Probability")
)

fig.update_layout(
    title='Logit Lens Visualization',
    xaxis_tickangle=0
)

fig.update_traces(text=words, texttemplate="%{text}")
fig.show()
```

The vertical axis indexes the layers, zero-indexed from 0 to 11. The top guess for each token, according to the model’s activations at a given layer, is printed in each cell. The colors show the probability associated with the top guess.

---

# lora_training.ipynb


---

# model_editing.ipynb

# Model Editing

NNsight's model editing feature allows you to create persistently modified versions of a model with a use of `.edit()`. Unlike interventions in a tracing context, which are temporary, the **Editor** context enables you to make lasting changes to a model instance.

This feature is useful for:
* Creating modified model variants without altering the original
* Applying changes that persist across multiple forward passes
* Comparing interventions between original and edited models

Let's explore how to use the **Editor** context to make a simple persistent change to a model:

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

# we take the hidden states with the expected output "Paris"
with model.trace("The Eiffel Tower is located in the city of") as tracer:
    hs11 = model.transformer.h[11].output[0][:, -1, :].save()

# the edited model will now always predict "Paris" as the next token
with model.edit() as model_edited:
    model.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with model.trace("Vatican is located in the city of") as tracer:
    original_tokens = model.lm_head.output.argmax(dim=-1).save()

# ...with the output of the edited model
with model_edited.trace("Vatican is located in the city of") as tracer:
    modified_tokens = model.lm_head.output.argmax(dim=-1).save()

print("\nOriginal Prediction: ", model.tokenizer.decode(original_tokens[0][-1]))
print("Modified Prediction: ", model.tokenizer.decode(modified_tokens[0][-1]))
```

Edits defined within an **Editor** context create a new, modified version of the model by default, preserving the original. This allows for safe experimentation with model changes. If you wish to modify the original model directly, you can set `inplace=True` when calling `.edit()`.

Use this option cautiously, as in-place edits alter the base model for all the consequent model calls.

```python
# we use the hidden state we saved above (hs11)
with model.edit(inplace=True) as model_edited:
    model.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with model.trace("Vatican is located in the city of") as tracer:
    modified_tokens = model.lm_head.output.argmax(dim=-1).save()

print("Modified In-place: ", model.tokenizer.decode(modified_tokens[0][-1]))
```

If you've made in-place edits to your model and need to revert these changes, `.clear_edits()` can help. This method removes all edits applied to the model, effectively restoring it to its original state.

```python
model.clear_edits()

with model.trace("Vatican is located in the city of"):
    modified_tokens = model.lm_head.output.argmax(dim=-1).save()

print("Edits cleared: ", model.tokenizer.decode(modified_tokens[0][-1]))
```

---

# modules.ipynb

# Modules

We can also apply modules in the model's module tree at any point during computation, even if they are out of order.

Here, we get the hidden states of the last layer like usual. We also chain apply `model.transformer.ln_f` and `model.lm_head` in order to "decode" the hidden states into the vocabulary space. Applying softmax and then argmax then transformz the vocabulary space hidden states into tokens that we can decode with the tokenizer.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel("openai-community/gpt2", device_map='auto')

with model.trace('The Eiffel Tower is in the city of') as tracer:

    hidden_states = model.transformer.h[-1].output[0]

    hidden_states = model.lm_head(model.transformer.ln_f(hidden_states)).save()

    tokens = torch.softmax(hidden_states, dim=2).argmax(dim=2).save()
```

The output looks like:

```python
print(hidden_states)
print(tokens)
print(model.tokenizer.decode(tokens[0]))
```

---

# multiple_token.ipynb

# Multiple Token Generation

When generating more than one token, use `<module>.next()` to denote following interventions should be applied to the subsequent generations for that module.

Here we generate three tokens and save the hidden states of the last layer for each one:

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')
```

## `.generate()`

NNsight's `LanguageModel` class supports multiple token generation with `.generate()`. You can control the number of new tokens generated by setting `max_new_tokens = N` within your call to `.generate()`.

```python
prompt = 'The Eiffel Tower is in the city of'
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    out = model.generator.output.save()

decoded_prompt = model.tokenizer.decode(out[0][0:-n_new_tokens].cpu())
decoded_answer = model.tokenizer.decode(out[0][-n_new_tokens:].cpu())

print("Prompt: ", decoded_prompt)
print("Generated Answer: ", decoded_answer)
```

## `.next()`

When generating more than one token, use `<module>.next()` to denote following interventions should be applied to the subsequent generations for that module.

Here we generate three tokens and save the hidden states of the last layer for each one:

```python
n_new_tokens = 3
with model.generate('The Eiffel Tower is in the city of', max_new_tokens=n_new_tokens) as tracer:

    hidden_states1 = model.transformer.h[-1].output[0].save()

    hidden_states2 = model.transformer.h[-1].next().output[0].save()

    hidden_states3 = model.transformer.h[-1].next().output[0].save()

    out = model.generator.output.save()
```

Note how calling save before `tracer.next()` returns the hidden state across the initial prompt while calling save after returns the hidden state of each subsequent generated token.

```python
print(hidden_states1.shape) # hidden states across prompt & first generated token
print(hidden_states2.shape) # only hidden states across next token
print(hidden_states3.shape) # only hidden states across next token
print(out) # model output tokens, including prompt
```

Great, we've now successfully stored hidden states across multiple different rounds of token generation! However, if you're generating many tokens while applying interventions, using `.next()` requires you to set a loop within the tracing context, which can be clunky:

```python
# Old approach:
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 50
hidden_states = []
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    for i in range(n_new_tokens):
        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output.save())

        # Move to next generated token
        layers[0].next()

print("Hidden state length: ",len(hidden_states))
```

## `.all()` streamlines interventions on many generated tokens

With `nnsight 0.4` you can use `.all()` to recursively apply interventions to a model. Calling `.all()` on a module within a model will recursively apply its `.input` and `.output` across all iterations. Previously, we'd need to loop across each new generated token, saving the intervention for every generated token and calling `.next()` to move forward, as demonstrated in the previous section.

Let's try using `.all()` to streamline the multiple token generation process. We simply call `.all()` on the module where we are applying the intervention (in this case GPT-2's layers), apply our intervention, and append our hidden states (stored in an `nnsight.list()` object).

```python
import nnsight
# using .all():
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 50
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() to apply intervention to each new token
    with layers.all():

        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output) # no need to call .save
        # Don't need to loop or call .next()!

print("Hidden state length: ",len(hidden_states))
```

Easy! Note that because `.all()` is recursive, it will only work to append outputs called on children of the module that `.all()` was called on. See example below for more information. TL;DR: apply `.all()` on the highest-level accessed module if interventions and outputs have different hierarchies within model structure.

<details>
<summary>Recursive properties of .all()</summary>

`.all()` recursively acts on model components. In the below code example, only the first token generation is saved, because `.all()` applied to `layers`, while the saved variable `hidden_states` is produced from `model.lm_head`, which is not a child of `layers`.

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on layers
    with layers.all():

        # Apply same intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(model.lm_head.output) # no need to call .save, it's already initialized

print("Hidden state length: ",len(hidden_states)) # length is 1, meaning it only saved the first token generation
```

If you want to apply an intervention during multiple token generation while saving the state of a model component that isn't a child of that module, you can instead apply `.all()` to the full model:

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on model
    with model.all():

        # Apply same intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(model.lm_head.output) # no need to call .save

print("Hidden state length: ",len(hidden_states)) # length is 3, as expected!
```

</details>

---

# operations.ipynb

# Operations

Most basic operations and torch operations work on proxies and are added to the computation graph.

In this example we get the sum of the hidden states and add them to the hidden_states themselves (for whatever reason). By saving the various steps, we can see how the values change.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace('The Eiffel Tower is in the city of') as tracer:

    hidden_states_pre = model.transformer.h[-1].output[0].save()

    hs_sum = torch.sum(hidden_states_pre).save()

    hs_edited = hidden_states_pre + hs_sum

    hs_edited = hs_edited.save()
```

```python
print(hidden_states_pre)
print(hs_sum)
print(hs_edited)
```

---

# remote_execution.ipynb

# Remote Execution

To access remote models, `NDIF` requires you to receive an API key. To get one, simply
go to https://login.ndif.us and sign up.

With a valid API key, you then can configure `nnsight` by doing the following:

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")
```

This only needs to be run once as it will save this api key as the default in a
config file along with the `nnsight` installation.

Let's demonstrate using `nnsight`'s tracing
context with one of the larger open source language models, `Llama-3.1-70b`!

```python
import os

# llama3.1 70b is a gated model and you need access via your huggingface token
os.environ['HF_TOKEN'] = "YOUR_HUGGING_FACE_TOKEN"
```

```python
from nnsight import LanguageModel
# We'll never actually load the parameters so no need to specify a device_map.
llama = LanguageModel("meta-llama/Meta-Llama-3.1-70B")

# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:

    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])
```

It really is as simple as `remote=True`! All of the techniques available in NNsight locally work just the same when running remotely.

# Remote Model Considerations & System Limits
To view currently hosted models, please visit https://nnsight.net/status/. All models except for `meta-llama/Meta-Llama-3.1-405B` and `meta-llama/Meta-Llama-3.1-405B-Instruct` are currently available for public access. If you are interested in running an experiment on Llama 405b, please reach out to us at [info@ndif.us](mailto:info@ndif.us)
.

Our system is currently actively in development, so please be prepared for system outages, updates, and wait times. NDIF is running on [DeltaAI](https://delta.ncsa.illinois.edu/deltaai-allocations/), so our services will be down during any of their planned and unplanned outages.

We currently have some rate-limiting and timeouts in place to ensure equitable model access between users.

- Maximum Request Rate: 2 requests/minute
- Maximum Job Run Time: 1 hour

Jobs violating these parameters will be automatically denied or aborted. Please plan your experiments accordingly. You can also reach out to our team at [info@ndif.us](mailto:info@ndif.us) if you have a special research case and would like to request any changes!

---

# scan_validate.ipynb

# Scan and Validate

Have you encountered a situation where you are changing the tensor values in the intervention code and getting an error message that is not very helpful?

This is where "Scanning" and "Validating" can help. As the name suggests, these features help you scan the shapes of the tensors throughout the model and validate that the current tensor shapes are compatible with the model.

We can enable these helpful tools by setting the `scan=True` and `validate=True` flags in the `trace` method.

Here is an example that demonstrates how **Scan** and **Validate** can help us debug the model:

```python
from nnsight import LanguageModel

model = LanguageModel('openai-community/gpt2', device_map='auto')

input = "The Eiffel Tower is in the city of"
number_of_tokens = len(model.tokenizer.encode(input))

# turn on scan and validate
with model.trace(input, scan=True, validate=True):

    original_output = model.transformer.h[11].output[0].clone().save()

    # we want to modify the hidden states for the last token
    model.transformer.h[11].output[0][:, number_of_tokens, :] = 0

    modified_output = model.transformer.h[11].output[0].save()

print("\nOriginal Output: ", original_output[0][-1])
print("Modified Output: ", modified_output[0][-1])
```

Ah of course, we needed to index at `number_of_tokens - 1` not `number_of_tokens`.

How was `nnsight` able to catch this error?

If `scan` and `validate` are enabled, our input is run though the model, but under its own "fake" context. This means the input makes its way through all of the model operations, allowing `nnsight` to record the shapes and data types of module inputs and outputs! The operations are never executed using tensors with real values so it doesn't incur any memory costs. Then, when creating proxy requests like the setting one above, `nnsight` also attempts to execute the request on the "fake" values we recorded.

"Scanning" is what we call running "fake" inputs throught the model to collect
information like shapes and types. "Validating" is what we call trying to
execute the intervention proxies with "fake" inputs to see if they work.
"Validating" is dependent on "Scanning" to work correctly, so we need to run the scan of the model at least once to debug with validate.

<details>
<summary>A word of caution</summary>

---

Some pytorch operations and related libraries don't work well with fake tensors

If you are doing anything in a loop where efficiency is important, you should keep scanning and validating off. It's best to use them only when debugging or when you are unsure if your intervention will work.

---

</details>

Let's try again with the correct indexing, and view the shape of the output
before leaving the tracing context:

```python
with model.trace(input, scan=True, validate=True):

    original_output = model.transformer.h[11].output[0].clone().save()

    # we want to modify the hidden states for the last token
    model.transformer.h[11].output[0][:, number_of_tokens-1, :] = 0

    modified_output = model.transformer.h[11].output[0].save()

print("\nOriginal Output: ", original_output[0][-1])
print("Modified Output: ", modified_output[0][-1])
```

We can also just replace proxy inputs and outputs with tensors of the same shape
and type. Let's use the shape information we have at our disposal to add noise
to the output, and replace it with this new noised tensor:

```python
with model.scan(input):

    dim = model.transformer.h[11].output[0].shape[-1]

print(dim)
```

---

# sessions.ipynb

# Sessions

NDIF uses a queue to handle concurrent requests from multiple users. To optimize the execution of our experiments we can use the `session` context to efficiently package multiple interventions together as one single request to the server.

This offers the following benefits:
1) All interventions within a session will be executed one after another without additional wait in the queue
2) All intermediate outputs of each intervention are stored on the server and can be accessed by other interventions in the same session without moving the data back and forth between NDIF and the local machine.

Let's take a look:

```python
from nnsight import CONFIG
import os

# we are using Llama model remotely hosted on NDIF servers
CONFIG.set_default_api_key("YOUR_API_KEY")
os.environ['HF_TOKEN'] = "YOUR_HUGGING_FACE_TOKEN"
```

```python
from nnsight import LanguageModel
model = LanguageModel("meta-llama/Meta-Llama-3.1-70B")
```

```python
with model.session(remote=True) as session:

  with model.trace("The Eiffel Tower is in the city of") as t1:
    # capture the hidden state from layer 11 at the last token
    hs_79 = model.model.layers[79].output[0][:, -1, :] # no .save()
    t1_tokens_out = model.lm_head.output.argmax(dim=-1).save()

  with model.trace("Buckingham Palace is in the city of") as t2:
    model.model.layers[1].output[0][:, -1, :] = hs_79[:]
    t2_tokens_out = model.lm_head.output.argmax(dim=-1).save()

print("\nT1 - Original Prediction: ", model.tokenizer.decode(t1_tokens_out[0][-1]))
print("T2 - Modified Prediction: ", model.tokenizer.decode(t2_tokens_out[0][-1]))
```

In the example above, we are interested in replacing the hidden state of a later layer with an earlier one. Since we are using a `session`, we don't have to save the hidden state from Tracer 1 to reference it in Tracer 2.

It is important to note that all the traces defined within the `session` context are executed sequentially, strictly following the order of definition (i.e. `t2` being executed after `t1` and `t3` after `t2` etc.).

The `session` context object has its own methods to log values and be terminated early.

```python
import nnsight
with model.session(remote=True) as session:

  nnsight.log("-- Early Stop --")
  nnsight.stop

```

In addition to the benefits mentioned above, the `session` context also enables interesting experiments not possible with other `nnsight` tools - since every trace is run on its own model, it means that within one session we can run interventions between different models – for example, we can swap activations between vanilla and instruct versions of the Llama model and compare the outputs. And `session` can also be used to run experiments entirely locally!

---

# setting.ipynb

# Setting Values

We often not only want to see whats happening during computation, but intervene and edit the flow of information.

In this example, we create a tensor of noise to add to the hidden states. We then add it, use the assigment `=` operator to update the tensors of `.output[0][:]` with these new noised values.

```python
from nnsight import LanguageModel
import torch

model = LanguageModel('openai-community/gpt2', device_map='auto')

with model.trace('The Eiffel Tower is in the city of') as tracer:

    hidden_states_pre = model.transformer.h[-1].output[0].clone().save()

    noise = (0.001**0.5)*torch.randn(hidden_states_pre.shape)

    # model.transformer.h[-1].output = (hidden_states_pre + noise, model.transformer.h[-1].output[1])
    model.transformer.h[-1].output[0][:] = hidden_states_pre + noise

    hidden_states_post = model.transformer.h[-1].output[0].save()
```

We can see the change in the results:

```python
print(hidden_states_pre)
print(hidden_states_post)
```

---

# start_remote_access.ipynb

# Access LLMs with NDIF and NNsight

* [NDIF](https://ndif.us/) is an inference service hosting large open-weight LLMs for use by researchers.
* [NNsight](https://nnsight.net/) is a package for interpreting and manipulating internals of deep learning models.

Together, NDIF and NNsight work hand in hand to let researchers run complex experiments on huge open models easily with full transparent access.

[Run an interactive version of this walkthrough in Google Colab](https://colab.research.google.com/github/ndif-team/ndif-website/blob/onboarding-fixes/public/notebooks/NDIFGetStarted.ipynb)

# Install NNsight

To start using NNsight, you can install it via `pip`.

```python
!pip install nnsight

from IPython.display import clear_output
clear_output()
```

# Sign up for NDIF remote model access

In order to remotely access LLMs through NDIF, users must sign up for an NDIF API key.

## **[Register here](https://login.ndif.us/) for a free API key!**

Once you have a valid NDIF API key, you then can configure `nnsight` by doing the following:

```python
from nnsight import CONFIG

CONFIG.API.APIKEY = input("Enter your API key: ")
clear_output()
```

<details>
<summary>
More about API key configuration
</summary>

The above code saves your API key as the default in a config file along with the `nnsight` installation. If you're running this walkthrough using a local Python installation, this only needs to be run once. If you're using Colab, we recommend saving your API key as a Colab Secret, and configuring it as follows in your notebooks:

```
from nnsight import CONFIG

if is_colab:
    # include your NNsight API key on Colab secrets
    from google.colab import userdata
    NDIF_API = userdata.get('NDIF_API')
    CONFIG.set_default_api_key(NDIF_API)
```

</details>

# Choose a Model

NDIF hosts multiple LLMs, including various sizes of the Llama 3.1 models and DeepSeek-R1 models. **You can view the full list of hosted models on [our status page](https://nnsight.net/status/).** All of our models are open for public use, except you need to apply for access to the Llama-3.1-405B models.

<details>
<summary>
Apply for 405B access
</summary>

If you have a clear research need for Llama-3.1-405B and would like more details about applying for access, please refer to [this page](https://ndif.us/405b.html)!

</details>

For these exercises, we will explore how we can access and modify the Llama-3.1-70B model's internal states. This 70-billion-parameter model is about the maximum size that you could run on a single A100 GPU with 80GB of VRAM, but we are going to access it remotely on NDIF resources, so you can run it on Colab or your laptop computer!

<details>
<summary>
Note: Llama models are gated on HuggingFace
</summary>

Llama models are gated and require you to register for access via HuggingFace. [Check out their website for more information about registration with Meta](https://huggingface.co/meta-llama/Llama-3.1-70B).

If you are using a local Python installation, you can activate your HuggingFace token using the terminal:

`huggingface-cli login -token YOUR_HF_TOKEN`

If you are using Colab, you can add your HuggingFace token to your Secrets.

</details>

We will be using the `LanguageModel` subclass of NNsight to load in the Llama-3.1-70B model and access its internal states.

<details>
<summary>
About NNsight LanguageModel
</summary>

The `LanguageModel` subclass of NNsight is a wrapper that includes special support for HuggingFace language models, including automatically loading models from a HuggingFace ID together with the appropriate tokenizer.

This way there's no need to pretokenize your input, and instead you can just pass a string as an input!

*Note: `LanguageModel` models also accept tokenized inputs, including [chat templates](https://huggingface.co/docs/transformers/main/en/chat_templating).*
</details>

```python
# instantiate the model using the LanguageModel class
from nnsight import LanguageModel

# don't worry, this won't load locally!
llm = LanguageModel("meta-llama/Meta-Llama-3.1-70B", device_map="auto")

print(llm)
```

# Access model internals

Now that we've installed `nnsight`, configured our API key, and instantiated a model, we can run an experiment.

For this experiment, let's try grabbing some of the LLM's hidden states using `nnsight`'s tracing context, `.trace()`.

Entering the tracing context allows us to customize how a neural network runs. By calling `.trace()`, we are telling the model to run with a given input and to collect and/or modify the internal model states based on user-defined code within the tracing context. We can also specify that we want to use an NDIF-hosted model instead of executing locally by setting `remote=True`.

To get started, let's ask NNsight to collect the layer output (known as "logits") at the final layer, along with the overall model output. NNsight needs to know what specific parts of the model we're interested in accessing later, so we need to specify which elements we'd like to save after exiting the tracing context using `.save()`.

*Note: You will not be able to access any values defined within a `.trace()` that aren't saved with `.save()` after exiting the tracing context!*

```python
# remote = True means the model will execute on NDIF's shared resources
with llm.trace("The Eiffel Tower is in the city of", remote=True):

    # user-defined code to access internal model components
    hidden_states = llm.model.layers[-1].output[0].save()
    output = llm.output.save()

# after exiting the tracing context, we can access any values that were saved
print("Hidden State Logits: ",hidden_states[0])

output_logits = output["logits"]
print("Model Output Logits: ",output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])
```

What are we seeing here? NNsight tells you if your job is recieved, approved, running, or completed via logs.

<details>
<summary>
Disabling remote logging notifications
</summary>
If you prefer, you can disable NNsight remote logging notifications with the following code, although they can help troubleshoot any network issues.

```
from nnsight import CONFIG
CONFIG.APP.REMOTE_LOGGING = False
```

If you'd like to turn them back on, just set `REMOTE_LOGGING = True`:
```
from nnsight import CONFIG
CONFIG.APP.REMOTE_LOGGING = True
```
</details>

We are also seeing our printed results. After exiting the tracing context, NNsight downloads the saved results, which we can perform operations on using Python code. Pretty simple!

# Alter model internals

Now that we've accessed the internal layers of the model, let's try modifying them and see how it affects the output!

We can do this using in-place operations in NNsight, which alter the model's state during execution. Let's try changing the output of layer 8 to be equal to 4.

```python
# remote = True means the model will execute on NDIF's shared resources
with llm.trace("The Eiffel Tower is in the city of", remote=True):

    # user-defined code to access internal model components
    llm.model.layers[7].output[0][:] = 4 # in-place operation to change a single layer's output values
    output = llm.output.save()

# after exiting the tracing context, we can access any values that were saved

output_logits = output["logits"]
print("Model Output Logits: ",output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])
```

Okay! The output for "The Eiffel Tower is in the city of" is now "Bounty". Looks like our intervention on the hidden 8th layer worked to change the model output!

Are you ready for something a little more complicated? Let's take the model's state when answering the city that the London Bridge is in, and swap that into the model's final layer when answering the Eiffel Tower question! We can do this using NNsight's invoking contexts, which batch different inputs into the same run through the model.

We can access values defined in invoking contexts throughout the other invoke context, allowing us to do something like swapping model states for different inputs. Let's try it out!

```python
import nnsight
# remote = True means the model will execute on NDIF's shared resources
with llm.trace(remote=True) as tracer:

    with tracer.invoke("The London Bridge is in the city of"):
        hidden_states = llm.model.layers[-1].output[0] # no .save()

    with tracer.invoke("The Eiffel Tower is in the city of"):
        # user-defined code to access internal model components
        llm.model.layers[-1].output[0][:] = hidden_states # can be accessed without .save()!
        output = llm.output.save()

output_logits = output["logits"]
print("Model Output Logits: ",output_logits[0])

# decode the final model output from output logits
max_probs, tokens = output_logits[0].max(dim=-1)
word = [llm.tokenizer.decode(tokens.cpu()[-1])]
print("Model Output: ", word[0])
```

Awesome, looks like it worked! The model output London instead of Paris when asked about the location of the Eiffel Tower.

# Next steps: Run your own experiment with NDIF and NNsight

This is just a quick overview of some of NNsight's functionality when working with remote models, so to learn more we recommend taking a deeper dive into these resources:

*   📚 Get a comprehensive overview of the library with the [NNsight Walkthrough](https://nnsight.net/notebooks/tutorials/walkthrough/)
*   🔎 Check out some NNsight implementations of common [LLM interpretability techniques](https://nnsight.net/tutorials/)
*   💬 Join the conversation with the NDIF [Discord](https://discord.com/invite/6uFJmCSwW7) community
*   💟 Follow us on [GitHub](https://github.com/ndif-team/nnsight), [Bluesky](https://bsky.app/profile/ndif-team.bsky.social), [X](https://x.com/ndif_team), and [LinkedIn](https://www.linkedin.com/company/national-deep-inference-fabric/)

**Want to scale up your research? [Apply for access to Llama-3.1-405B](https://ndif.us/405b.html)!**

<br>

<img src="https://ndif.us/images//NDIF_Acr_color.png" alt="drawing" width="400"/>

---

# streaming.ipynb


---

# vllm_support.ipynb

# vLLM Support

[vLLM](https://github.com/vllm-project/vllm) is a popular library used for fast inference. By leveraging PagedAttention, dynamic batching, and Hugging Face model integration, vLLM makes inference more efficient and scalable for real-world applications.

Starting with `NNsight 0.4`, NNsight includes support for internal investigations of vLLM models.

## Setup

You will need to install `nnsight 0.4`, `vllm==0.6.4.post1`, and `triton 3.1.0` to use vLLM with NNsight.

Please note that the current version of `vllm` isn't supported with NNsight, so you will need to specifically install the supported version: `vllm==0.6.4.post1`.

```python
from IPython.display import clear_output
try:
    import google.colab
    is_colab = True
except ImportError:
    is_colab = False

if is_colab:
    !pip install -U nnsight
clear_output()
```

```python
# install vllm
!pip install vllm==0.6.4.post1

# install triton 3.1.0
!pip install triton==3.1.0

clear_output()
```

 Next, let's load in our NNsight-supported vLLM model. You can find vLLM-supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html). For this exercise, we will use GPT-2.

 Please note that vLLM models require a GPU to run.

```python
from IPython.display import clear_output
from nnsight.modeling.vllm import VLLM

# NNsight's VLLM wrapper currently supports "device = cuda" and device = "auto"
vllm = VLLM("gpt2", device = "auto", dispatch = True) # See supported models: https://docs.vllm.ai/en/v0.6.4.post1/models/supported_models.html

clear_output()
print(vllm)
```

## Interventions on vLLM models
We now have a vLLM model that runs with `nnsight`. Let's try applying some interventions on it.

Note that vLLM takes in sampling parameters including `temperature` and `top_p`. These parameters can be included in the `.trace()` or `.invoke()` contexts. For default model behavior, set `temperature = 0` and `top_p = 1`. For more information about parameters, reference the [vLLM documentation](https://docs.vllm.ai/en/latest/dev/sampling_params.html).

```python
with vllm.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
  with tracer.invoke("The Eiffel Tower is located in the city of"):
    clean_logits = vllm.logits.output.save()

  with tracer.invoke("The Eiffel Tower is located in the city of"):
    vllm.transformer.h[-2].mlp.output[:] = 0
    corrupted_logits = vllm.logits.output.save()
```

```python
print("\nCLEAN - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(clean_logits.argmax(dim=-1)))
print("\nCORRUPTED - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(corrupted_logits.argmax(dim=-1)))
```

We've successfully performed an intervention on our vLLM model!

## Sampled Token Traceability
vLLM provides functionality to configure how each sequence samples its next token. Here's an example of how you can trace token sampling operations with the nnsight VLLM wrapper.

```python
import nnsight
with vllm.trace("Madison Square Garden is located in the city of", temperature=0.8, top_p=0.95, max_tokens=3) as tracer:
    samples = nnsight.list().save()
    logits = nnsight.list().save()

    for ii in range(3):
        samples.append(vllm.samples.output)
        vllm.samples.next()
        logits.append(vllm.logits.output)
        vllm.logits.next()

print("Samples: ", samples)
print("Logits: ", logits) # different than samples with current sampling parameters
```

<details>
<summary>
Note: gradients are not supported with vLLM
</summary>

vLLM speeds up inference through its paged attention mechanism. This means that accessing gradients and backward passes are not supported for vLLM models. As such, calling gradient operations when using `nnsight` vLLM wrappers will throw an error.
</details>

## Known Issues
* The vllm.LLM engine performs max_tokens + 1 forward passes which can lead to undesired behavior if you are running interventions on all iterations of multi-token generation.

Example:
```
with vllm_gpt2("Hello World!", max_tokens=10):
    logits = nnsight.list().save()
    with vllm_gpt2.logits.all():
        logits.append(vllm_gpt2.logits.output)

print(len(logits))

```
`>>> 11 # expected: 10`

```python
with vllm.trace(temperature=0.0, top_p=1.0, max_tokens=1) as tracer:
  with tracer.invoke("The Eiffel Tower is located in the city of"):
    clean_logits = vllm.logits.output.save()

  with tracer.invoke("The Eiffel Tower is located in the city of"):
    vllm.language_model.model.layers[-2].mlp.output[:] = 0
    corrupted_logits = vllm.logits.output.save()
```

```python
print("\nCLEAN - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(clean_logits.argmax(dim=-1)))
print("\nCORRUPTED - The Eiffel Tower is located in the city of", vllm.tokenizer.decode(corrupted_logits.argmax(dim=-1)))
```

---

# walkthrough.ipynb

# Walkthrough

## The API for a transparent science on black-box AI

In this era of large-scale deep learning, the most interesting AI models are
massive black boxes that are hard to run. Ordinary commercial inference service
APIs let us interact with huge models, but they do not let us access model
internals.

The `nnsight` library is different: it provides full access to all neural
network internals. When using `nnsight` together with a remote service like the
[National Deep Inference Fabric](https://www.ndif.us)
(NDIF), it is possible to run complex experiments on huge open models easily
with fully transparent access.

Through NDIF and NNsight, our team wants to enable entire labs and independent researchers alike, as we
believe a large, passionate, and collaborative community will produce the next
big insights on this profoundly important field.

# 1 First, let's start small

[Run an interactive version of this walkthrough in Google Colab](https://colab.research.google.com/github/ndif-team/nnsight/blob/new-new-tutorials/docs/source/notebooks/tutorials/walkthrough.ipynb)

## Setup

Install NNsight:
```
pip install nnsight
```

## Tracing Context

To demonstrate the core functionality and syntax of nnsight, we'll define and
use a tiny two layer neural network.

Our little model here is composed of two submodules – linear layers `layer1` and `layer2`. We specify the sizes of each of these modules and create
some complementary example input.

```python
from collections import OrderedDict
import torch

input_size = 5
hidden_dims = 10
output_size = 2

net = torch.nn.Sequential(
    OrderedDict(
        [
            ("layer1", torch.nn.Linear(input_size, hidden_dims)),
            ("layer2", torch.nn.Linear(hidden_dims, output_size)),
        ]
    )
).requires_grad_(False)
```

The core object of the NNsight package is `NNsight`. This wraps around a given
PyTorch model to enable investigation of its internal parameters.

```python
import nnsight
from nnsight import NNsight

tiny_model = NNsight(net)
```

Printing a PyTorch model shows a named hierarchy of modules which is very useful
when accessing sub-components directly. NNsight reflect the same hierarchy and can be similarly printed.

```python
print(tiny_model)
```

Before we actually get to using the model we just created, let's talk about
Python contexts.

Python contexts define a scope using the `with` statement and are often used to
create some object, or initiate some logic, that you later want to destroy or
conclude.

The most common application is opening files as in the following example:

```python
with open('myfile.txt', 'r') as file:
  text = file.read()
```

Python uses the `with` keyword to enter a context-like object. This object
defines logic to be run at the start of the `with` block, as well as logic to be
run when exiting. When using `with` for a file, entering the context opens the
file and exiting the context closes it. Being within the context means we can
read from the file.

Simple enough! Now we can discuss how `nnsight` uses
contexts to enable intuitive access into the internals of a neural network.

The main tool with `nnsight` is a context for tracing.

We enter the tracing context by calling `model.trace(<input>)` on an `NNsight`
model, which defines how we want to run the model. Inside the context, we will
be able to customize how the neural network runs. The model is actually run upon
exiting the tracing context.

```python
# random input
input = torch.rand((1, input_size))

with tiny_model.trace(input) as tracer:
    pass
```

But where's the output? To get that, we'll have to learn how to request it from
within the tracing context.

## Getting

Earlier, we wrapped our little neural net with the `NNsight` class. This
added a couple properties to each module in the model (including the root model
itself). The two most important ones are `.input` and `.output`.

```python
model.input
model.output
```

The names are self explanatory. They correspond to the inputs and outputs of
their respective modules during a forward pass of the model. We can use these
attributes inside the `with` block.

However, it is important to understand that the model is not executed until the
end of the tracing context. How can we access inputs and outputs before the
model is run? The trick is deferred execution.

`.input` and `.output` are Proxies for the eventual inputs and outputs of a
module. In other words, when we access `model.output` what we are
communicating to `nnsight` is, "When you compute the output of `model`, please
grab it for me and put the value into its corresponding Proxy object. Let's try it:

```python
with tiny_model.trace(input) as tracer:

    output = tiny_model.output

print(output)
```

Oh no an error! "Accessing value before it's been set."

Why doesn't our `output` have a `value`?

Proxy objects will only have their value at the end of a context if we call
`.save()` on them. This helps to reduce memory costs. Adding `.save()` fixes the
error:

```python
with tiny_model.trace(input) as tracer:

    output = tiny_model.output.save()

print(output)
```

Success! We now have the model output. We just completed out first
intervention using `nnsight`.

Each time we access a module's input or output, we create an _intervention_ in
the neural network's forward pass. Collectively these requests form the
_intervention graph_. We call the process of executing it alongside the model's
normal computation graph, _interleaving_.

<details>
<summary>On Model output</summary>

---

If we don't need to access anything other than the model's final output (i.e., the model's predicted next token), we can
call the tracing context with `trace=False` and not use it as a context. This could be useful for simple inference using NNsight.

```python
  output = model.trace(<inputs>, trace=False)
```

---

</details>

Just like we saved the output of the model as a whole, we can save the output of
any of its submodules. We use normal Python attribute syntax. We can discover
how to access them by name by printing out the model:

```python
print(tiny_model)
```

Let's access the output of the first layer (which we've named `layer1`):

```python
with tiny_model.trace(input) as tracer:

    l1_output = tiny_model.layer1.output.save()

print(l1_output)
```

Let's do the same for the input of `layer2`.

Because we aren't accessing the `tracer` object within these tracing contexts, we can also drop `as tracer`.

```python
with tiny_model.trace(input):

    l2_input = tiny_model.layer2.input.save()

print(l2_input)
```

<details>
  <summary>On module inputs</summary>

---

Notice how the value for `l2_input` is just a single tensor. By default, the `.input` attribute of a module will return the **first** tensor input to the module.

We can also access the full input to a module by using the `.inputs` attribute, which will return the values in the form of:

      tuple(tuple(args), dictionary(kwargs))

Where the first index of the tuple is itself a tuple of all positional
arguments, and the second index is a dictionary of the keyword arguments.

---

</details>

Until now we were saving the output of the model and its submodules within the `Trace` context to then print it after exiting the context. We will continuing doing this in the rest of the tutorial since it's a good practice to save the computation results for later analysis.

However, we can also log the outputs of the model and its submodules within the `Trace` context. This is useful for debugging and understanding the model's behavior while saving memory.

Let's see how to do this:

```python
with tiny_model.trace(input) as tracer:
  tracer.log("Layer 1 - out: ", tiny_model.layer1.output)
```

## Functions, Methods, and Operations

Now that we can access activations, we also want to do some post-processing on
it. Let's find out which dimension of layer1's output has the highest value.

We could do this by calling `torch.argmax(...)` after the tracing context or we
can just leverage the fact that `nnsight` handles Pytorch functions and methods within
the tracing context, by creating a Proxy request for it:

```python
with tiny_model.trace(input):

    # Note we don't need to call .save() on the output,
    # as we're only using its value within the tracing context.
    l1_output = tiny_model.layer1.output

    # We do need to save the argmax tensor however,
    # as we're using it outside the tracing context.
    l1_amax = torch.argmax(l1_output, dim=1).save()

print(l1_amax[0])
```

Nice! That worked seamlessly, but hold on, how come we didn't need to call
`.value[0]` on the result? In previous sections, we were just being explicit to
get an understanding of Proxies and their value. In practice, however, `nnsight`
knows that when outside of the tracing context we only care about the actual
value, and so printing, indexing, and applying functions all immediately return
and reflect the data in `.value`. So for the rest of the tutorial we won't use
it.

The same principles work for Pytorch methods and all operators as well:

```python
with tiny_model.trace(input):

    value = (tiny_model.layer1.output.sum() + tiny_model.layer2.output.sum()).save()

print(value)
```

The code block above is saying to `nnsight`, "Run the model with
the given `input`. When the output of `tiny_model.layer1` is computed, take its sum. Then do
the same for `tiny_model.layer2`. Now that both of those are computed, add them and make sure
not to delete this value as I wish to use it outside of the tracing context."

## Custom Functions

Everything within the tracing context operates on the intervention graph. Therefore, for `nnsight` to trace a  function it must also be a part of the intervention graph.

Out-of-the-box `nnsight` supports PyTorch functions and methods, all operators, as well the `einops` library. We don't need to do anything special to use them. But what do we do if we want to use custom functions? How do we add them to the intervention graph?

Enter `nnsight.apply()`. It allows us to add new functions to the intervention graph. Let's see how it works:

```python
# Take a tensor and return the sum of its elements
def tensor_sum(tensor):
    flat = tensor.flatten()
    total = 0
    for element in flat:
        total += element.item()

    return torch.tensor(total)

with tiny_model.trace(input) as tracer:

    # Specify the function name and its arguments (in a comma-separated form) to add to the intervention graph
    custom_sum = nnsight.apply(tensor_sum, tiny_model.layer1.output).save()
    sum = tiny_model.layer1.output.sum()
    sum.save()

print(custom_sum, sum)
```

`nnsight.apply()` executes the function it wraps and returns its output as a Proxy object. We can then use this Proxy object as we would any other.

The applications of `nnsight.apply` are wide: it can be used to wrap any custom function or functions from libraries that `nnsight` does not support out-of-the-box.

## Setting

Getting and analyzing the activations from various points in a model can be
really insightful, and a number of ML techniques do exactly that. However, often we not only want to view the computation of a model, but also to influence it.

To demonstrate the effect of editing the flow of information through the model,
let's set the first dimension of the first layer's output to 0. `NNsight` makes
this really easy using the '=' operator:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the 0th index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, 0] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

Seems our change was reflected. Now let's do the same for the last dimension:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    # Access the last index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, hidden_dims] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

Oh no, we are getting an error! Ah of course, we needed to index at `hidden_dims - 1` not `hidden_dims`.

If you've been using `nnsight`, you are probably familiar with error messages that can be quite difficult to troubleshoot. In `nnsight 0.4` we've now improved error messaging to be descriptive and line-specific, as you should see in the above example!

<details>

<summary>
Old NNsight error messaging
</summary>

If you've been using NNsight prior to the NNsight 0.4 release, you will be familiar with the following non-descriptive error messaging. If you choose to turn off NNsight 0.4's new error messaging feature, this is how errors within the tracing context will appear.

```
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in execute(self)
    379                 # Call the target to get value.
--> 380                 output = self.target(*args, **kwargs)
    381

IndexError: index 10 is out of bounds for dimension 1 with size 10

The above exception was the direct cause of the following exception:

IndexError                                Traceback (most recent call last)
20 frames
<ipython-input-16-5c81de91fb1f> in <cell line: 0>()
----> 1 with tiny_model.trace(input):
      2
      3     # Save the output before the edit to compare.
      4     # Notice we apply .clone() before saving as the setting operation is in-place.
      5     l1_output_before = tiny_model.layer1.output.clone().save()

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/Tracer.py in __exit__(self, exc_type, exc_val, exc_tb)
    100
    101
--> 102         super().__exit__(exc_type, exc_val, exc_tb)
    103
    104     def invoke(self, *inputs: Any, **kwargs) -> Invoker:

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/GraphBasedContext.py in __exit__(self, exc_type, exc_val, exc_tb)
    215             raise exc_val
    216
--> 217         self.backend(self)
    218
    219     ### BACKENDS ########

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/backends/LocalBackend.py in __call__(self, obj)
     25     def __call__(self, obj: LocalMixin):
     26
---> 27         obj.local_backend_execute()

/usr/local/lib/python3.11/dist-packages/nnsight/contexts/Tracer.py in local_backend_execute(self)
    144         self.graph.execute()
    145
--> 146         self.model.interleave(
    147             self.model._execute,
    148             self.graph,

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in interleave(self, fn, intervention_graph, *inputs, **kwargs)
    467         module_paths = InterventionProtocol.get_interventions(intervention_graph).keys()
    468
--> 469         with HookHandler(
    470             self._model,
    471             list(module_paths),

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in __exit__(self, exc_type, exc_val, exc_tb)
    579
    580         if isinstance(exc_val, Exception):
--> 581             raise exc_val
    582
    583

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in interleave(self, fn, intervention_graph, *inputs, **kwargs)
    478         ):
    479             try:
--> 480                 fn(*inputs, **kwargs)
    481             except protocols.EarlyStopProtocol.EarlyStopException:
    482                 # TODO: Log.

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in _execute(self, *prepared_inputs, **kwargs)
    585             pass
    586
--> 587         return self._model(
    588             *prepared_inputs,
    589             **kwargs,

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1734             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735         else:
-> 1736             return self._call_impl(*args, **kwargs)
   1737
   1738     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1842
   1843         try:
-> 1844             return inner()
   1845         except Exception:
   1846             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1788                 args = bw_hook.setup_input_hook(args)
   1789
-> 1790             result = forward_call(*args, **kwargs)
   1791             if _global_forward_hooks or self._forward_hooks:
   1792                 for hook_id, hook in (

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1734             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1735         else:
-> 1736             return self._call_impl(*args, **kwargs)
   1737
   1738     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1842
   1843         try:
-> 1844             return inner()
   1845         except Exception:
   1846             # run always called hooks if they have not already been run

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in inner()
   1801                         hook_result = hook(self, args, kwargs, result)
   1802                     else:
-> 1803                         hook_result = hook(self, args, result)
   1804
   1805                     if hook_result is not None:

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in output_hook(module, input, output, module_path)
    564
    565                 def output_hook(module, input, output, module_path=module_path):
--> 566                     return self.output_hook(output, module_path)
    567
    568                 self.handles.append(

/usr/local/lib/python3.11/dist-packages/nnsight/models/NNsightModel.py in <lambda>(activations, module_path)
    473                 activations, module_path, "input", intervention_handler
    474             ),
--> 475             output_hook=lambda activations, module_path: InterventionProtocol.intervene(
    476                 activations, module_path, "output", intervention_handler
    477             ),

/usr/local/lib/python3.11/dist-packages/nnsight/intervention.py in intervene(cls, activations, module_path, key, intervention_handler)
    454
    455                 # Value injection.
--> 456                 node.set_value(value)
    457
    458                 # Check if through the previous value injection, there was a 'swap' intervention.

/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in set_value(self, value)
    408
    409             if listener.fulfilled() and not self.graph.sequential:
--> 410                 listener.execute()
    411
    412         for dependency in self.arg_dependencies:

/usr/local/lib/python3.11/dist-packages/nnsight/tracing/Node.py in execute(self)
    385         except Exception as e:
    386
--> 387             raise type(e)(
    388                 f"Above exception when execution Node: '{self.name}' in Graph: '{self.graph.id}'"
    389             ) from e

IndexError: Above exception when execution Node: 'setitem_0' in Graph: '132147685816016'

```

</details>

The error messaging feature can be toggled using `nnsight.CONFIG.APP.DEBUG` which defaults to true.

<details>

<summary>
Toggle Error Messaging
</summary>

Turn off debugging:
```
import nnsight

nnsight.CONFIG.APP.DEBUG = False
nnsight.CONFIG.save()
```

Turn on debugging:
```
import nnsight

nnsight.CONFIG.APP.DEBUG = True
nnsight.CONFIG.save()
```
</details>

Now that we know more about NNsight's error messaging, let's try our setting operation again with the correct indexing and view the shape of the output
before leaving the tracing context:

```python
with tiny_model.trace(input):

    # Save the output before the edit to compare.
    # Notice we apply .clone() before saving as the setting operation is in-place.
    l1_output_before = tiny_model.layer1.output.clone().save()

    print(f"Layer 1 output shape: {tiny_model.layer1.output.shape}")

    # Access the last index of the hidden state dimension and set it to 0.
    tiny_model.layer1.output[:, hidden_dims - 1] = 0

    # Save the output after to see our edit.
    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

## Scan and Validate
Error codes are helpful, but sometimes you may want to quickly troubleshoot your code without actually running it.

Enter "Scanning" and "Validating"! We can enable this features by setting the `scan=True` and `validate=True` flag in the `trace` method.

"Scanning" runs "fake" inputs throught the model to collect information like shapes and types (i.e., scanning will populate all called `.inputs` and `.outputs`).

"Validating" attempts to execute the intervention proxies with "fake" inputs to check if they work (i.e., executes all interventions in your code with fake tensors).

"Validating" is dependent on "Scanning" to work correctly, so we need to run the scan of the model at least once to debug with validate. Let's try it out on our example above.

```python
# turn on scan and validate
with tiny_model.trace(input, scan=True, validate=True):

    l1_output_before = tiny_model.layer1.output.clone().save()

    # the error is happening here
    tiny_model.layer1.output[:, hidden_dims] = 0

    l1_output_after = tiny_model.layer1.output.save()

print("Before:", l1_output_before)
print("After:", l1_output_after)
```

The operations are never executed using tensors with real values so it doesn't incur any memory costs. Then, when creating proxy requests like the setting one above, `nnsight` also attempts to execute the request on the "fake" values we recorded. Hence, it lets us know if our request is feasible before even running the model. [Here](https://nnsight.net/notebooks/features/scan_validate/) is a more detailed example of scan and validate in action!

<details>
<summary>A word of caution</summary>

---

Some pytorch operations and related libraries don't work well with fake tensors

If you are doing anything in a loop where efficiency is important, you should keep scanning and validating off. It's best to use them only when debugging or when you are unsure if your intervention will work.

---

</details>

We can also use the `.scan()` method to get the shape of a module without having to fully run the model. If scan  is enabled, our input is run though the model under its own "fake" context. This means the input makes its way through all of the model operations, allowing `nnsight` to record the shapes and data types of module inputs and outputs!

```python
with tiny_model.scan(input):

    dim = tiny_model.layer1.output.shape[-1]

print(dim)
```

## Gradients

`NNsight` also lets us apply backpropagation and access gradients with respect to a
loss. Like `.input` and `.output` on modules, `nnsight` exposes `.grad` on
Proxies themselves (assuming they are proxies of tensors):

```python
with tiny_model.trace(input):

    # We need to explicitly have the tensor require grad
    # as the model we defined earlier turned off requiring grad.
    tiny_model.layer1.output.requires_grad = True

    # We call .grad on a tensor Proxy to communicate we want to store its gradient.
    # We need to call .save() since .grad is its own Proxy.
    layer1_output_grad = tiny_model.layer1.output.grad.save()
    layer2_output_grad = tiny_model.layer2.output.grad.save()

    # Need a loss to propagate through the later modules in order to have a grad.
    loss = tiny_model.output.sum()
    loss.backward()

print("Layer 1 output gradient:", layer1_output_grad)
print("Layer 2 output gradient:", layer2_output_grad)
```

All of the features we learned previously, also apply to `.grad`. In other
words, we can apply operations to and edit the gradients. Let's zero the grad of
`layer1` and double the grad of `layer2`.

```python
with tiny_model.trace(input):

    # We need to explicitly have the tensor require grad
    # as the model we defined earlier turned off requiring grad.
    tiny_model.layer1.output.requires_grad = True

    tiny_model.layer1.output.grad[:] = 0
    tiny_model.layer2.output.grad = tiny_model.layer2.output.grad * 2

    layer1_output_grad = tiny_model.layer1.output.grad.save()
    layer2_output_grad = tiny_model.layer2.output.grad.save()

    # Need a loss to propagate through the later modules in order to have a grad.
    loss = tiny_model.output.sum()
    loss.backward()

print("Layer 1 output gradient:", layer1_output_grad)
print("Layer 2 output gradient:", layer2_output_grad)
```

## Early Stopping

If we are only interested in a model's intermediate computations, we can halt a forward pass run at any module level, reducing runtime and conserving compute resources. One examples where this could be particularly useful would if we are working with SAEs - we can train an SAE on one layer and then stop the execution.

```python
with tiny_model.trace(input):
   l1_out = tiny_model.layer1.output.save()
   tiny_model.layer1.output.stop()

# get the output of the first layer and stop tracing
print("L1 - Output: ", l1_out)
```

Interventions within the tracing context do not necessarily execute in the order they are defined. Instead, their execution is tied to the module they are associated with.

As a result, if the forward pass is terminated early any interventions linked to modules beyond that point will be skipped, even if they were defined earlier in the context.

In the example below, the output of layer 2 _**cannot**_ be accessed since the model's execution was stopped at layer 1.

```python
with tiny_model.trace(input):
   l2_out = tiny_model.layer2.output.save()
   tiny_model.layer1.output.stop()

print("L2 - Output: ", l2_out)
```

## Conditional Interventions

Interventions can also be made conditional.

Inside the tracing context we can specify a new - conditional - context. This context will only execute the interventions within it if the condition is met.

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  with tracer.cond(rand_int % 2 == 0):
    tracer.log("Random Integer ", rand_int, " is Even")

  with tracer.cond(rand_int % 2 == 1):
    tracer.log("Random Integer ", rand_int, " is Odd")
```

Conditional contexts can also be nested, if we want our interventions to depend on more than one condition at a time.

```python
with tiny_model.trace(input) as tracer:

  non_rand_int = 8

  with tracer.cond(non_rand_int > 0):
    with tracer.cond(non_rand_int % 2 == 0):
      tracer.log("Rand Int ", non_rand_int, " is Positive and Even")
```

With `nnsight 0.4` we can now also use Python `if` statements within the tracing context to create a conditional context!

*Note: Colab behaves a little strangely with this feature the first time you run it - expect some lagging and warnings*

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")

  if rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

`elif` statements should also work as `if` statements within the tracing context:

```python
with tiny_model.trace(input) as tracer:

  rand_int = torch.randint(low=-10, high=10, size=(1,))

  # Since this if statement is inside the tracing context the if will
  # create a conditional context and will only execute the intervention
  # if this condition is met
  if rand_int % 2 == 0:
    tracer.log("Random Integer ", rand_int, " is Even")
  elif rand_int % 2 == 1:
    tracer.log("Random Integer ", rand_int, " is Odd")
```

## Iterative Interventions

With the iterator context, you can now run an intervention loop at scale. It iteratively executes and updates a single intervention graph. Use a `.session()` to define the Iterator context and pass in a sequence of items that you want to loop over at each iteration

```python
with tiny_model.session() as session:

  li = nnsight.list() # an NNsight built-in list object
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

With `nnsight 0.4` we can now also use Python `for` loops within a tracer context at scale.

*NOTE: inline for loops (i.e., `[x for x in <Proxy object>`]) are not currently supported.*

```python
# New: Using Python for loops for iterative interventions
with tiny_model.session() as session:

    li = nnsight.list()
    [li.append([num]) for num in range(0, 3)]
    li2 = nnsight.list().save()

    # Using regular for loops
    for item in li:
        for item_2 in item: # for loops can be nested!
            li2.append(item_2)

print("\nList: ", li2)
```

# 2️ Bigger

Now that we have the basics of `nnsight` under our belt, we can scale our model
up and combine the techniques we've learned into more interesting experiments.

The `NNsight` class is very bare bones. It wraps a pre-defined model and does no
pre-processing on the inputs we enter. It's designed to be extended with more
complex and powerful types of models, and we're excited to see what can be done
to leverage its features!

However, if you'd like to load a Language Model from HuggingFace with its tokenizer, the`LanguageModel` subclass greatly simplifies this process.

## LanguageModel

`LanguageModel` is a subclass of `NNsight`. While we could define and create a
model to pass in directly, `LanguageModel` includes special support for
Huggingface language models, including automatically loading models from a
Huggingface ID, and loading the model together with the appropriate tokenizer.

Here is how we can use `LanguageModel` to load `GPT-2`:

```python
from nnsight import LanguageModel

llm = LanguageModel("openai-community/gpt2", device_map="auto")

print(llm)
```

When we initialize `LanguageModel`, we aren't yet loading the parameters of the
model into memory. We are actually loading a 'meta' version of the model which
doesn't take up any memory, but still allows us to view and trace actions on it.
After exiting the first tracing context, the model is then fully loaded into
memory. To load into memory on initialization, you can pass `dispatch=True` into
`LanguageModel` like
`LanguageModel('openai-community/gpt2', device_map="auto", dispatch=True)`.

<details>
<summary>On Model Initialization</summary>

---

A few important things to note:

Keyword arguments passed to the initialization of `LanguageModel` is forwarded
to HuggingFace specific loading logic. In this case, `device_map` specifies
which devices to use and its value `auto` indicates to evenly distribute it to
all available GPUs (and CPU if no GPUs available). Other arguments can be found
here:
https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForCausalLM

---

</details>

Let's now apply some of the features that we used on the small model to `GPT-2`. Unlike `NNsight`, `LanguageModel` does define logic to pre-process
inputs upon entering the tracing context. This makes interacting with the model
simpler (i.e., you can send prompts to the model without having to directly access the tokenizer).

In the following example, we ablate the value coming from the last layer's MLP
module and decode the logits to see what token the model predicts without
influence from that particular module:

```python
with llm.trace("The Eiffel Tower is in the city of"):

    # Access the last layer using h[-1] as it's a ModuleList
    # Access the first index of .output as that's where the hidden states are.
    llm.transformer.h[-1].mlp.output[0][:] = 0

    # Logits come out of model.lm_head and we apply argmax to get the predicted token ids.
    token_ids = llm.lm_head.output.argmax(dim=-1).save()

print("\nToken IDs:", token_ids)

# Apply the tokenizer to decode the ids into words after the tracing context.
print("Prediction:", llm.tokenizer.decode(token_ids[0][-1]))
```

We just ran a little intervention on a much more complex model with many more
parameters! However, we're missing an important piece of information: what the
prediction would have looked like without our ablation.

We could just run two tracing contexts and compare the outputs. However, this would require two forward passes through the model. `NNsight` can do
better than that with batching.

## Batching

Batching is a way to process multiple inputs in one forward pass. To better understand how batching works, we're going to bring back the `Tracer` object that we dropped before.

When we call `.trace(...)`, it's actually creating two different contexts behind the scenes. The first one is the tracing context that we've discussed previously, and the second one is the invoker context. The invoker context defines the values of the `.input` and `.output` Proxies.

If we call `.trace(...)` with some input, the input is passed on to the invoker. As there is only one input, only one invoker context is created.

If we call `.trace()` without an input, then we can call `tracer.invoke(input1)` to manually create the invoker context with an input, `input1`. We can also repeatedly call `tracer.invoke(...)` to create the invoker context for additional inputs. Every subsequent time we call
`.invoke(...)`, interventions within its context will only refer to the input in that particular invoke statement.

When exiting the tracing context, the inputs from all of the invokers will be batched together, and they will be executed in one forward pass! To test this out, let's do the same ablation experiment, but also add a 'control' output for comparison:

<details>
<summary>More on the invoker context</summary>

---

Note that when injecting data to only the relevant invoker interventions, `nnsight` tries, but can't guarantee, to narrow the data into the right
batch indices. Thus, there are cases
where all invokes will get all of the data. Specifically, if the input or output data is stored
as an object that is not an arbitrary collection of tensors, it will be broadcasted to all invokes.

Just like `.trace(...)` created a `Tracer` object, `.invoke(...)` creates an `Invoker` object. For `LanguageModel` models, the `Invoker` prepares the input by running a tokenizer on it.
`Invoker` stores pre-processed inputs at `invoker.inputs`, which can be accessed to see information about our inputs.
In a case where we pass a single input to `.trace(...)` directly, we can still access the invoker
object at `tracer.invoker` without having to call `tracer.invoke(...)`.

Keyword arguments given to `.invoke(..)` make their way to the input pre-processing.
`LanguageModel` has keyword arguments `max_length` and `truncation` used for tokenization which can be
passed to the invoker. If we want to pass keyword arguments to the invoker for a single-input `.trace(...)`, we can pass `invoker_args` as a dictionary of invoker keyword arguments.

Here is an example to demonstrate everything we've described:

**This snippet**

```
with llm.trace("hello", invoker_args={"max_length":10}) as tracer:
  invoker = tracer.invoker

```
  **does the same as**

```
with llm.trace() as tracer:
  with tracer.invoke("hello", max_length=10) as invoker:
    invoker = invoker
```

---

</details>

```python
with llm.trace() as tracer:

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Ablate the last MLP for only this batch.
        llm.transformer.h[-1].mlp.output[0][:] = 0

        # Get the output for only the intervened on batch.
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("The Eiffel Tower is in the city of"):

        # Get the output for only the original batch.
        token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("Original token IDs:", token_ids_original)
print("Modified token IDs:", token_ids_intervention)

print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))
```

Based on our control results, our ablation did end up affecting what the model predicted. That's pretty neat!

Another cool thing with multiple invokes is that Proxies can interact between them.

Here, we transfer the token embeddings from a real prompt into another placeholder prompt. Therefore the latter prompt produces the output of the former prompt:

```python
with llm.trace() as tracer:

    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = llm.transformer.wte.output

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
        llm.transformer.wte.output = embeddings
        token_ids_intervention = llm.lm_head.output.argmax(dim=-1).save()

    with tracer.invoke("_ _ _ _ _ _ _ _ _ _"):
      token_ids_original = llm.lm_head.output.argmax(dim=-1).save()

print("original prediction shape", token_ids_original[0][-1].shape)
print("Original prediction:", llm.tokenizer.decode(token_ids_original[0][-1]))

print("modified prediction shape", token_ids_intervention[0][-1].shape)
print("Modified prediction:", llm.tokenizer.decode(token_ids_intervention[0][-1]))
```

For larger batch sizes, you can also iteratate across multiple invoke contexts.

## Multiple Token Generation

### .next()

Some HuggingFace models define methods to generate multiple outputs at a time.
`LanguageModel` wraps that functionality to provide the same tracing features by
using `.generate(...)` instead of `.trace(...)`. This calls the underlying
model's `.generate` method. It passes the output through a `.generator`
module that we've added onto the model, allowing us to get the generate output
at `.generator.output`.

In a case like this, the underlying model is called more than once; the modules
of said model produce more than one output. Which iteration should a given
`module.output` refer to? That's where `Module.next()` comes in!

Each module has a call index associated with it and `.next()` simply increments
that attribute. At the time of execution, data is injected into the intervention
graph only at the iteration that matches the call index.

```python
with llm.generate('The Eiffel Tower is in the city of', max_new_tokens=3) as tracer:

    hidden_states1 = llm.transformer.h[-1].output[0].save()

    # use module.next() to access the next intervention
    hidden_states2 = llm.transformer.h[-1].next().output[0].save()

    # saving the output allows you to save the hidden state across the initial prompt
    out = llm.generator.output.save()

print(hidden_states1.shape)
print(hidden_states2.shape)
print(out)
```

### using .all()

With `nnsight 0.4` you can now use `.all()` to recursively apply interventions to a model. Calling `.all()` on a module within a model will recursively apply its `.input` and `.output` across all iterations. Previously, we'd need to loop across each new generated token, saving the intervention for every generated token and calling `.next()` to move forward.

```python
# Old approach:
prompt = 'The Eiffel Tower is in the city of'
layers = llm.transformer.h
n_new_tokens = 3
hidden_states = []
with llm.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    for i in range(n_new_tokens):
        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output.save())

        # Move to next generated token
        layers[0].next()

print("Hidden state length: ",len(hidden_states))
```

We can use also `.all()` to streamline the multiple token generation process. We simply call `.all` on the module where we are applying the intervention (in this case GPT-2's layers), apply our intervention, and append our hidden states (stored in an `nnsight.list()` object).
<br> <br>

Let's test this out for the multiple token generation case:

```python
# using .all():
prompt = 'The Eiffel Tower is in the city of'
layers = llm.transformer.h
n_new_tokens = 3
with llm.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() to apply intervention to each new token
    with layers.all():

        # Apply intervention - set first layer output to zero
        layers[0].output[0][:] = 0

        # Append desired hidden state post-intervention
        hidden_states.append(layers[-1].output) # no need to call .save
        # Don't need to loop or call .next()!

print("Hidden state length: ",len(hidden_states))
```

Easy! Note that because `.all()` is recursive, it will only work to append outputs called on children of the module that `.all()` was called on. See example below for more information. TL;DR: apply `.all()` on the highest-level accessed module if interventions and outputs have different hierarchies within model structure.

<details>
<summary>Recursive properties of .all()</summary>

`.all()` recursively acts on model components. In the below code example, only the first token generation is saved, because `.all()` applied to `layers`, while the saved variable `hidden_states` is produced from `model.lm_head`, which is not a child of `layers`.

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on layers
    layers.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save, it's already initialized

print("Hidden state length: ",len(hidden_states)) # length is 1, meaning it only saved the first token generation
```

If you want to apply an intervention during multiple token generation while saving the state of a model component that isn't a child of that module, you can instead apply `.all()` to the full model:

```
prompt = 'The Eiffel Tower is in the city of'
layers = model.transformer.h
n_new_tokens = 3
with model.generate(prompt, max_new_tokens=n_new_tokens) as tracer:
    hidden_states = nnsight.list().save() # Initialize & .save() nnsight list

    # Call .all() on model
    model.all()

    # Apply same intervention - set first layer output to zero
    layers[0].output[0][:] = 0

    # Append desired hidden state post-intervention
    hidden_states.append(model.lm_head.output) # no need to call .save

print("Hidden state length: ",len(hidden_states)) # length is 3, as expected!
```

</details>

## Model Editing

NNsight's model editing feature allows you to create persistently modified versions of a model with a use of `.edit()`. Unlike interventions in a tracing context, which are temporary, the **Editor** context enables you to make lasting changes to a model instance.

This feature is useful for:
* Creating modified model variants without altering the original
* Applying changes that persist across multiple forward passes
* Comparing interventions between original and edited models

Let's explore how to use the **Editor** context to make a simple persistent change to a model:

```python
# we take the hidden states with the expected output "Paris"
with llm.trace("The Eiffel Tower is located in the city of") as tracer:
    hs11 = llm.transformer.h[11].output[0][:, -1, :].save()

# the edited model will now always predict "Paris" as the next token
with llm.edit() as llm_edited:
    llm.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with llm.trace("Vatican is located in the city of") as tracer:
    original_tokens = llm.lm_head.output.argmax(dim=-1).save()

# ...with the output of the edited model
with llm_edited.trace("Vatican is located in the city of") as tracer:
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("\nOriginal Prediction: ", llm.tokenizer.decode(original_tokens[0][-1]))
print("Modified Prediction: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

Edits defined within an **Editor** context create a new, modified version of the model by default, preserving the original. This allows for safe experimentation with model changes. If you wish to modify the original model directly, you can set `inplace=True` when calling `.edit()`.

Use this option cautiously, as in-place edits alter the base model for all the consequent model calls.

```python
# we use the hidden state we saved above (hs11)
with llm.edit(inplace=True) as llm_edited:
    llm.transformer.h[11].output[0][:, -1, :] = hs11

# we demonstrate this by comparing the output of an unmodified model...
with llm.trace("Vatican is located in the city of") as tracer:
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("Modified In-place: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

If you've made in-place edits to your model and need to revert these changes, you can apply `.clear_edits()`. This method removes all edits applied to the model, effectively restoring it to its original state.

```python
llm.clear_edits()

with llm.trace("Vatican is located in the city of"):
    modified_tokens = llm.lm_head.output.argmax(dim=-1).save()

print("Edits cleared: ", llm.tokenizer.decode(modified_tokens[0][-1]))
```

# 3 I thought you said huge models?

`NNsight` is only one part of our project to democratize access to AI internals. The other half is the National Deep Inference Fabric, or `NDIF`. `NDIF` hosts large models for shared access using `NNsight`, so you don't have to worry about any of the headaches of hosting large models yourself!

The interaction between `NDIF` and `NNsight` is fairly straightforward. The
**intervention graph** we create via the tracing context can be encoded into a
custom json format and sent via an http request to the `NDIF` servers. `NDIF`
then decodes the **intervention graph** and **interleaves** it alongside the
specified model.

To see which models are currently being hosted, check out the following status
page: https://nnsight.net/status/

## Remote execution

In its current state, `NDIF` requires you to receive an API key. Therefore, to
run the rest of this walkthrough, you need one of your own. To get one, simply
register at https://login.ndif.us.

With a valid API key, you then can configure `nnsight` as follows:

```python
from nnsight import CONFIG

CONFIG.set_default_api_key("YOUR_API_KEY")
```

If you're running in a local IDE, this only needs to be run once as it will save the API key as the default in a
.config file along with your `nnsight` installation. You can also add your API key to Google Colab secrets.

To amp things up a few levels, let's demonstrate using `nnsight`'s tracing
context with `Llama-3.1-8b`!

```python
import os

# Llama 3.1 8b is a gated model, so you need to apply for access on HuggingFace and include your token.
os.environ['HF_TOKEN'] = "YOUR_HUGGING_FACE_TOKEN"
```

```python
from nnsight import LanguageModel

# We'll never actually load the parameters locally, so no need to specify a device_map.
llama = LanguageModel("meta-llama/Meta-Llama-3.1-8B")
# All we need to specify using NDIF vs executing locally is remote=True.
with llama.trace("The Eiffel Tower is in the city of", remote=True) as runner:

    hidden_states = llama.model.layers[-1].output.save()

    output = llama.output.save()

print(hidden_states)

print(output["logits"])
```

It really is as simple as `remote=True`. All of the techniques we went through
in earlier sections work just the same when running locally or remotely.

## Sessions

NDIF uses a queue to handle concurrent requests from multiple users. To optimize the execution of our experiments we can use the `session` context to efficiently package multiple interventions together as one single request to the server.

This offers the following benefits:
1.   All interventions within a session will be executed one after another without additional wait in the NDIF queue
2.   All intermediate outputs for each intervention are stored on the server and can be accessed by other interventions in the same session without moving the data back and forth between NDIF and the local machine

Let's take a look:

```python
with llama.session(remote=True) as session:

  with llama.trace("The Eiffel Tower is in the city of") as t1:
    # capture the hidden state from layer 32 at the last token
    hs_31 = llama.model.layers[31].output[0][:, -1, :] # no .save()
    t1_tokens_out = llama.lm_head.output.argmax(dim=-1).save()

  with llama.trace("Buckingham Palace is in the city of") as t2:
    llama.model.layers[1].output[0][:, -1, :] = hs_31[:]
    t2_tokens_out = llama.lm_head.output.argmax(dim=-1).save()

print("\nT1 - Original Prediction: ", llama.tokenizer.decode(t1_tokens_out[0][-1]))
print("T2 - Modified Prediction: ", llama.tokenizer.decode(t2_tokens_out[0][-1]))
```

In the example above, we are interested in replacing the hidden state of a later layer with an earlier one. Since we are using a `session`, we don't have to save the hidden state from Tracer 1 to reference it in Tracer 2.

It is important to note that all the traces defined within the `session` context are executed sequentially, strictly following the order of definition (i.e. `t2` being executed after `t1` and `t3` after `t2` etc.).

The `session` context object has its own methods to log values and be terminated early.

```python
with llama.session(remote=True) as session:
  session.log("-- Early Stop --")
  nnsight.stop
```

In addition to the benefits mentioned above, the `session` context also enables interesting experiments not possible with other `nnsight` tools — since every trace is run on its own model, it means that within one session we can run interventions between different models — for example, we could swap activations between base and instruct versions of the Llama model and compare their outputs. And `session` can also be used to run similar experiments entirely locally!

## Streaming

Streaming enables users apply functions and datasets locally during remote model execution. This allows users to stream results for immediate consumption (i.e., seeing tokens as they are generated) or applying non-whitelisted functions such as model tokenizers, large local datasets, and more!

*   `nnsight.local()` context sends values immediately to user's local machine from server
*   Intervention graph is executed locally on downstream nodes
*   Exiting local context uploads data back to server
*   `@nnsight.trace` function decorator enables custom functions to be added to intervention graph when using `nnsight.local()`

## `nnsight.local()`

You may sometimes want to locally access and manipulate values during remote execution. Using `.local()` on a proxy, you can send remote content to your local machine and apply local functions. The intervention graph is then executed locally on downstream nodes (until you send execution back to the remote server by exiting the `.local()` context).

There are a few use cases for streaming with `.local()`, including live chat generation and applying large datasets or non-whitelisted local functions to the intervention graph.

Now let's explore how streaming works. We'll start by grabbing some hidden states of the model and printing their value using `tracer.log()`. Without calling `nnsight.local()`, these operations will all occur remotely.

```python
# This will give you a remote LOG response because it's coming from the remote server
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

Now, let's try the same operation using the `nnsight.local()` context. This will send the operations to get and print the hidden states to your local machine, changing how the logging message is formatted (local formatting instead of remote).

```python
# This will print locally because it's already local
with llama.trace("hello", remote=True) as tracer:

    with nnsight.local():
        hs = llama.model.layers[-1].output[0]
        tracer.log(hs[0,0,0])

    out =  llama.lm_head.output.save()

print(out)
```

## `@nnsight.trace` function decorator

We can also use function decorators to create custom functions to be used during `.local` calls. This is a handy way to enable live streaming of a chat or to train probing classifiers on model hidden states.

Let's try out `@nnsight.trace` and `nnsight.local()` to access a custom function during remote execution.

```python
# first, let's define our function
@nnsight.trace # decorator that enables this function to be added to the intervention graph
def my_local_fn(value):
    return value * 0

# We use a local function to ablate some hidden states
# This downloads the data for the .local context, and then uploads it back to set the value.
with llama.generate("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    with nnsight.local():

        hs = my_local_fn(hs)

    llama.model.layers[-1].output[0][:] = hs

    out =  llama.lm_head.output.save()
```

Note that without calling `.local`, the remote API does not know about `my_local_fn` and will throw a whitelist error. A whitelist error occurs because you are being allowed access to the function.

```python
with llama.trace("hello", remote=True) as tracer:

    hs = llama.model.layers[-1].output[0]

    hs = my_local_fn(hs) # no .local - will cause an error

    llama.model.layers[-1].output[0][:] = hs * 2

    out =  llama.lm_head.output.save()

print(out)
```

## Example: Live-streaming remote chat

Now that we can access data within the tracing context on our local computer, we can apply non-whitelisted functions, such as the model's tokenizer, within our tracing context.

Let's build a decoding function that will decode tokens into words and print the result.

```python
@nnsight.trace
def my_decoding_function(tokens, model, max_length=80, state=None):
    # Initialize state if not provided
    if state is None:
        state = {'current_line': '', 'current_line_length': 0}

    token = tokens[-1] # only use last token

    # Decode the token
    decoded_token = llama.tokenizer.decode(token).encode("unicode_escape").decode()

    if decoded_token == '\\n':  # Handle explicit newline tokens
        # Print the current line and reset state
        print('',flush=True)
        state['current_line'] = ''
        state['current_line_length'] = 0
    else:
        # Check if adding the token would exceed the max length
        if state['current_line_length'] + len(decoded_token) > max_length:
            print('',flush=True)
            state['current_line'] = decoded_token  # Start a new line with the current token
            state['current_line_length'] = len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line
        else:
            # Add a space if the line isn't empty and append the token
            if state['current_line']:
                state['current_line'] += decoded_token
            else:
                state['current_line'] = decoded_token
            state['current_line_length'] += len(decoded_token)
            print(state['current_line'], flush=True, end="")  # Print the current line

    return state
```

Now we can decode and print our model outputs throughout token generation by accessing our decoding function through `nnsight.local()`.

```python
import torch

nnsight.CONFIG.APP.REMOTE_LOGGING = False

prompt = "A press release is an official statement delivered to members of the news media for the purpose of"
# prompt = "Your favorite board game is"

print("Prompt: ",prompt,'\n', end ="")

# Initialize the state for decoding
state = {'current_line': '', 'current_line_length': 0}

with llama.generate(prompt, remote=True, max_new_tokens = 50) as generator:
    # Call .all() to apply to each new token
    llama.all()

    all_tokens = nnsight.list().save()

    # Access model output
    out = llama.lm_head.output.save()

    # Apply softmax to obtain probabilities and save the result
    probs = torch.nn.functional.softmax(out, dim=-1)
    max_probs = torch.max(probs, dim=-1)
    tokens = max_probs.indices.cpu().tolist()
    all_tokens.append(tokens[0]).save()

    with nnsight.local():
        state = my_decoding_function(tokens[0], llama, max_length=20, state=state)
```

## Looping across sessions

We mention earlier that the `session` context enables multi-tracing execution. But how can we optimize a process that would require running an intervention graph in a loop? If we create a simple `for` loop with a **Tracer context** inside, this will result in creating a new intervention graph at each iteration, which is not scalable.

We solve this problem the `nnsight` way via the **Iterator context**: an intervention loop that iteratively executes and updates a single intervention graph.

Use a `session` to define the **Iterator context** and pass in a sequence of items that you want to loop over at each iteration:

```python
with llama.session(remote=True) as session:

  with session.iter([0, 1, 2]) as item:
    # define intervention body here ...

    with llama.trace("_"):
      # define interventions here ...
      pass

    with llama.trace("_"):
      # define interventions here ...
      pass
```

The `Iterator` context extends all the `nnsight` graph-based functionalities, but also closely mimics the conventional `for` loop statement in Python, which allows it to support all kind of iterative operations with a use of `as item` syntax:

```python
with llama.session(remote=True) as session:

  li = nnsight.list()
  [li.append([num]) for num in range(0, 3)] # adding [0], [1], [2] to the list
  li2 = nnsight.list().save()

  # You can create nested Iterator contexts
  with session.iter(li) as item:
    with session.iter(item) as item_2:
      li2.append(item_2)

print("\nList: ", li2)
```

Notice how we used the `nnsight.list()` method to create a list of lists to loop over. This type of method is what we call an **NNsight Built-in**. It is a special type of methods that serve as a wrapper around `nnsight.apply()` to provide a more user-friendly interface for adding common datatypes to the Intervention Graph.

<details>
<summary>A full list of NNsight Built-ins</summary>

`nnsight.bool()` creates a traceable Boolean

`nnsight.bytes()` creates a traceable Bytes

`nnsight.int()` creates a traceable Integer

`nnsight.float()` creates a traceable Float

`nnsight.str()` creates a traceable String

`nnsight.comples()` creates a traceable Complex number

`nnsight.bytearray()` creates a traceable Bytearray

`nnsight.tuple()` creates a traceable Tuple

`nnsight.list()` creates a traceable List

`nnsight.set()` creates a traceable Set

`nnsight.dict()` creates a traceable Dictionary

</details>

We can also expose the `iterator` context object via a `return_context` flag. You can then use it to `exit` out of the Iteration loop early and log the intermediate outputs within the loop:

```python
with llama.session(remote=True) as session:

  # with session.iter([0, 1, 2, 3], return_context=True) as (item, iterator):
  with session.iter([0, 1, 2, 3]) as item:

      nnsight.log(item)

      with nnsight.cond(item == 2):
        nnsight.stop()
```

The **Iterator** context is a niece piece of functionality that allows you to define a bunch of basic code operations that can now be "traceable" by `nnsight`.

But in what kind of experimental scenario would someone need iterators?

In the next section, we delve into a powerful use case of the `Iterator` context and see how it enables it!

## Training a LoRA

Here is an example of a task that uses everything we have covered in the last section - remote execution, **Session** context and iterative interventions. Using session and iterator contexts, we're going apply a very simple fine-tuning approach called low-rank adaptation (LoRA).

Let's try training a LoRA that, when applied, makes our model always predict "Paris" no matter what.

```python
import torch
import torch.nn as nn
import nnsight
# from nnsight.envoy import Envoy # this moved in 0.4
from nnsight import Envoy

# We will define a LORA class.
# The LORA class call method operations are simply traced like you would normally do in a .trace.
class LORA(nn.Module):
    def __init__(self, module: Envoy, dim: int, r: int) -> None:
        """Init.

        Args:
            module (Envoy): Which model Module we are adding the LORA to.
            dim (int): Dimension of the layer we are adding to (This could potentially be auto populated if the user scanned first so we know the shape)
            r (int): Inner dimension of the LORA
        """
        super(LORA, self).__init__()
        self.r = r
        self.module = module
        self.WA = torch.nn.Parameter(torch.randn(dim, self.r), requires_grad=True).save()
        self.WB = torch.nn.Parameter(torch.zeros(self.r, dim), requires_grad=True).save()

    # The Call method defines how to actually apply the LORA.
    def __call__(self, alpha: float = 1.0):
        """Call.

        Args:
            alpha (float, optional): How much to apply the LORA. Can be altered after training for inference. Defaults to 1.0.
        """

        # We apply WA to the first positional arg (the hidden states)
        A_x = torch.matmul(self.module.input[0][0], self.WA)
        BA_x = torch.matmul(A_x, self.WB)

        # LORA is additive
        h = BA_x + self.module.output

        # Replace the output with our new one * alpha
        # Could also have been self.module.output[:] = h * alpha, for in-place
        self.module.output = h * alpha

    def parameters(self):
        # Some way to get all the parameters.
        return [self.WA, self.WB]
```

Let's define all the variables to use in LoRA training.

```python
# We need the token id of the correct answer.
answer = " Paris"
answer_token = llama.tokenizer.encode(answer)[1]
# Inner LORA dimension
lora_dim = 4
# Module to train LORA on
module = llama.model.layers[-1].mlp
```

We can use the `.scan()` method to get the shape of the module without having to fully run the model.

```python
with llama.scan(" "):
    dim = module.output.shape[-1]

print(dim)
```

It's time to run the LORA training loop! We using the **Session** and the **Iterator** contexts to achieve this.

```python
from torch.utils.data import DataLoader

# The LORA object itself isn't transmitted to the server. Only the forward / call method.
# The parameters are created remotely and never sent only retrieved
with llama.session(remote=True) as session:

    # Create dataset of 100 pairs of a blank prompt and the " Paris " id
    dataset = [["_", answer_token]] * 100

    # Create a dataloader from it.
    dataloader = DataLoader(dataset, batch_size=10)

    # Create our LORA on the last mlp
    lora = LORA(module, dim, lora_dim)

    # Create an optimizer. Use the parameters from LORA
    optimizer = torch.optim.AdamW(lora.parameters(), lr=3)

    # Iterate over dataloader using .iter.
    with session.iter(dataloader) as batch:

        prompt = batch[0]
        correct_token = batch[1]

        # Run .trace with prompt
        with llama.trace(prompt) as tracer:

            # Apply LORA to intervention graph just by calling it with .trace
            lora()

            # Get logits
            logits = llama.lm_head.output

            # Do cross entropy on last predicted token and correct_token
            loss = torch.nn.functional.cross_entropy(logits[:, -1], batch[1])
            # Call backward
            loss.backward()

        # Call methods on optimizer. Graphs that arent from .trace (so in this case session and iterator both have their own graph) are executed sequentially.
        # The Graph of Iterator here will be:
        # 1.) Index batch at 0 for prompt
        # 2.) Index batch at 1 for correct_token
        # 3.) Execute the .trace using the prompt
        # 4.) Call .step() on optimizer
        optimizer.step()
        # 5.) Call .zero_grad() in optimizer
        optimizer.zero_grad()
        # 6.) Print out the lora WA weights to show they are indeed changing
        nnsight.log(lora.WA)

```

Now `WA` and `WB` are optimized! So we generate with the LoRA just by calling `lora()` in the `.generate` and save the output to then de-tokenize it.

```python
# With lora. Should produce "Hello Paris"
with llama.generate("Hello", remote=True) as generator:

    lora()

    out = llama.generator.output.save()

print(llama.tokenizer.batch_decode(out.value))

# Then without. Should produce "Hello,"
with llama.generate("Hello", remote=True) as generator:

    out = llama.generator.output.save()

print(llama.tokenizer.batch_decode(out.value))

```

# Next Steps
Check out [nnsight.net/tutorials](https://nnsight.net/tutorials) for more walkthroughs implementating classic interpretability techniques using `nnsight`.

## Getting Involved!

Note that both `nnsight` and `NDIF` are in active development, so changes may be made and errors may arise during use. If you’re interested in following updates to `nnsight`, contributing, giving feedback, or finding collaborators, please join the [NDIF discord](https://discord.gg/6uFJmCSwW7). We’d love to hear about your work using nnsight!

You can also follow us on [LinkedIn](https://www.linkedin.com/company/national-deep-inference-fabric/), Bluesky: [@ndif-team.bsky.social](https://bsky.app/profile/ndif-team.bsky.social), and X: [@ndif_team](https://x.com/ndif_team).

💟


---

