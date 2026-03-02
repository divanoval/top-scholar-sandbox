# Project: Evaluating activation oracles against sparse autoencoders

Activation oracles (https://arxiv.org/abs/2512.15674) are a new approach to interpreting language model activations with natural language descriptions. Oracles are trained on a few different tasks for recovering model activations Sparse autoencoders (SAEs; https://transformer-circuits.pub/2023/monosemantic-features) are another interpretability approach to understanding activations. 

Your task is to understand how well activation oracles recover the information contained in model activations by comparing their outputs to SAEs. Your work should focus on activation oracles and SAEs trained on medium-sized open-source models such as Gemma 3 or Qwen 3. You should approach this question across a variety of circumstances and situations. 

To get started, here are some questions that you might think about for approaching this task. You do not need to answer all of these questions and you should think critically about additional research questions that might be interesting.
- Are the outputs of activation oracles consistent with the outputs of SAEs? How can we think about this in a quantifiable and testable way?
- When are the interpretations provided by activation oracles and the interpretations provided by SAEs similar and when are they different? Are there patterns that emerge here to expose the benefits or gaps in each of these methods?
- Are activation oracles able to provide better granularity or qualitatively better answers than SAEs or vice versa?
