# Introduction to Transformers
November 9th, 2023
[[2304.10557.pdf]]

The paper aims to provide a clear and intuitive description of transformer architecture, addressing the lack of precise mathematical explanations in many existing introductions.

### Key Concepts
1. **Input Data Format**: Transformers process sequences of tokens, each represented by vectors. This tokenization allows for versatile application across different data types, such as text or images.
2. **Goal**: The primary goal of a transformer is to transform the input data into another sequence of vectors, each representing the input sequence's features at specific token positions. These representations are adaptable for various tasks like auto-regressive prediction, global classification, or even image-to-image prediction.

### Transformer Block Structure
- **Two-Stage Processing**: A transformer block consists of two main stages: one operating across the sequence (horizontal processing) and another across features (vertical processing). This design enables each token's representation to be influenced by other tokens and their features.
- **Self-Attention Mechanism**: The first stage involves self-attention, where an output vector is computed as a weighted average of input features, with weights determined by an attention matrix. This matrix is dynamically generated based on the input sequence itself, allowing the model to focus on different parts of the input data as needed【25†source】.
- **Multi-Head Self-Attention**: To increase the model's capacity, transformers use multiple sets of self-attention (heads) in parallel, each with its own set of parameters. This design allows the model to capture various aspects of the input data simultaneously.

### Position Encoding
- Transformers inherently do not consider the order of input tokens, necessitating position encoding to incorporate sequence order information. Position information can be added directly to the token embeddings or learned as part of the model. This inclusion is crucial for maintaining the sequence's meaning and structure.

### Application-Specific Variants
1. **Auto-regressive Language Modelling**: Transformers can be modified for tasks like predicting the next word in a sequence. Modifications include auto-regressive masking for efficient training and incremental updates to process sequences efficiently.
2. **Image Classification**: For image classification, transformers process tokenized image patches. An alternative approach introduces a new token at the sequence start to maintain and refine a global representation appropriate for classification.
3. **Complex Systems**: Transformers are also integral in more complex architectures, such as encoder-decoder models for translation and auto-encoders for self-supervised vision systems.

### Personal Notes
- [ ] Review softmax functions
- [ ] Understand Multi-layer perceptrons

# LoRA
November 11th, 2023
[[2106.09685.pdf]]

LoRA presents an efficient and effective method for adapting large language models for specific tasks while maintaining model quality and reducing computational costs. This method has broad implications for the practical deployment of large-scale NLP models in various applications.

### Key Concepts and Methodology

1. **Problem with Traditional Fine-Tuning**: Conventional fine-tuning of LLMs, where all model parameters are updated, is increasingly impractical for models like GPT-3 with 175 billion parameters due to their size and associated computational costs.

2. **Low-Rank Adaptation (LoRA)**: To address this, LoRA is introduced, a method that freezes the pre-trained model weights and introduces trainable rank decomposition matrices into each layer of the Transformer architecture. This significantly reduces the number of trainable parameters for downstream tasks without compromising model quality.

3. **Design of LoRA**: LoRA hypothesizes that updates to the weights during model adaptation have a low "intrinsic rank". It involves adding low-rank matrices (represented as the product of two smaller matrices, B and A) to the pre-trained weight matrices. This method keeps the core model weights frozen and only updates the added low-rank matrices during training.

4. **Benefits of LoRA**: LoRA reduces the number of trainable parameters by up to 10,000 times and GPU memory requirements by 3 times compared to full fine-tuning. Additionally, it maintains or improves model quality on various NLP tasks and models (RoBERTa, DeBERTa, GPT-2, GPT-3) and introduces no additional inference latency.

### Empirical Experiments

1. **Performance Evaluation**: The authors tested LoRA's effectiveness on models like RoBERTa, DeBERTa, GPT-2, and GPT-3, using tasks from the GLUE benchmark, WikiSQL, and SAMSum datasets. These tasks cover a range of NLP applications from natural language understanding (NLU) to generation (NLG).

2. **Scaling to GPT-3 175B**: LoRA was scaled up to GPT-3 with 175 billion parameters. It matched or exceeded the fine-tuning baseline on all datasets. However, performance drops were observed with an excessive number of trainable parameters in some other tuning methods.

### Related Works and Comparison

1. **Context in NLP**: Transformer-based language models have led the field of NLP, especially for tasks involving large amounts of text. Fine-tuning on task-specific data post pre-training offers significant performance gains.

2. **Other Parameter-Efficient Adaptation Methods**: Prior methods include inserting adapter layers in neural networks and optimizing input word embeddings. Unlike these methods, LoRA's learned weights can be merged with main weights during inference, avoiding additional latency.

3. **Investigation into Low-Rank Updates**: The paper also explores which weight matrices should be adapted for best performance and the nature of the "optimal" adaptation matrix. It concludes that adapting both query (Wq) and value (Wv) attention weights generally yields the best results, even with a relatively low rank.

### Personal Notes
- [ ] Look up SVD Decomposition (difference between PDP, Jordan Decomposition ,etc. ?)
- [ ] Look more into Catastrophic Forgetting
- [ ] Try implementation of LoRA (even if I don't understand every mathematical concept behind it)

# QLoRA
November 17th, 2023
[[2305.14314.pdf]]

QLORA represents a significant advancement in the finetuning of large-scale LLMs, offering a way to efficiently train large models on limited hardware without compromising performance. This methodology opens new possibilities for research and application development using state-of-the-art LLMs.

1. **QLORA Approach**: QLORA finetunes a 4-bit quantized pretrained model with Low Rank Adapters (LoRA), preserving 16-bit finetuning task performance and enabling efficient use of memory.
2. **Memory Efficiency Innovations**:
   - **4-bit NormalFloat (NF4)**: A data type optimal for normally distributed weights, yielding better results than 4-bit integers and floats.
   - **Double Quantization**: Reduces the memory footprint by quantizing the quantization constants.
   - **Paged Optimizers**: Manage memory spikes using NVIDIA's unified memory to avoid out-of-memory errors during gradient checkpointing.

### QLORA Finetuning Process
- QLORA achieves 4-bit finetuning through NF4 quantization and Double Quantization. It uses a low-precision storage data type (usually 4-bit) and a computation data type (typically BFloat16).

### Performance Analysis
1. **Comparison to Standard Finetuning**: QLORA matches the performance of 16-bit full finetuning and LoRA across different model scales and datasets. It outperforms previous methods in terms of memory efficiency and finetuning capabilities for larger models.
2. **Data Quality Over Size**: The study finds that data quality is more critical than dataset size for chatbot performance, with a smaller high-quality dataset outperforming larger datasets.

