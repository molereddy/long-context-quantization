## `Paper`: Does quantization affect models’ performance on long-context tasks?

`Authors`: Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, Mohit Iyyer

`Abstract`: Large language models (LLMs) now support context windows exceeding 128K tokens, but this comes with significant memory requirements and high inference latency. Quantization can mitigate these costs, but may degrade performance. We present the first large-scale evaluation of quantized LLMs on tasks with long-inputs (≥64K tokens) and long-form outputs. Our evaluation spans 9.7K examples, five quantization methods (FP8, GPTQ-int8, AWQint4, GPTQ-int4, BNB-nf4), and five models (Llama-3.1 8B and 70B; Qwen-2.5 7B, 32B, and 72B). Results show that, on average, 8bit quantization preserves accuracy (≤0.8% drop), whereas 4-bit methods incur substantial losses up to 59% on long context tasks. Performance degradation from quantization is more pronounced in long-input tasks than in long-form generations. These drops are further amplified in a multilingual setup. Furthermore, the impact of quantization varies across models. While Qwen-2.5 72B remains robust under BNB-nf4, Llama-3.1 70B suffers a 32% performance drop. These findings underscore the importance of rigorous evaluation before deploying quantized LLMs, especially in long -context and multilingual settings.

Results and code soon.
