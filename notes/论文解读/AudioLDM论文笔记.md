# AudioLDM 论文笔记

## 模型概述

受 [Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) 的启发，AudioLDM 2 是一种文生音频的 _ 隐扩散模型 (latent diffusion model，LDM)_，其可以将文本嵌入映射成连续的音频表征。

大体的生成流程总结如下:

1. 给定输入文本 $\boldsymbol{x}$，使用两个文本编码器模型来计算文本嵌入: [CLAP](https://huggingface.co/docs/transformers/main/en/model_doc/clap) 的文本分支，以及 [Flan-T5](https://huggingface.co/docs/transformers/main/en/model_doc/flan-t5) 的文本编码器。

   $$\boldsymbol{E} *{1} = \text{CLAP}\left(\boldsymbol{x} \right); \quad \boldsymbol{E}* {2} = \text{T5}\left(\boldsymbol{x}\right) $$

   CLAP 文本嵌入经过训练，可以与对应的音频嵌入对齐，而 Flan-T5 嵌入可以更好地表征文本的语义。

2. 这些文本嵌入通过各自的线性层投影到同一个嵌入空间:

   $$\boldsymbol{P} *{1} = \boldsymbol{W}* {\text{CLAP}} \boldsymbol{E} *{1}; \quad \boldsymbol{P}* {2} = \boldsymbol{W} *{\text{T5}}\boldsymbol{E}* {2} $$

   在 `diffusers` 实现中，这些投影由 [AudioLDM2ProjectionModel](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2/AudioLDM2ProjectionModel) 定义

3. 使用 [GPT2](https://huggingface.co/docs/transformers/main/en/model_doc/gpt2) 语言模型 (LM) 基于 CLAP 和 Flan-T5 嵌入自回归地生成一个含有 $N$ 个嵌入向量的新序列:

   $$\tilde{\boldsymbol{E}} *{i} = \text{GPT2}\left(\boldsymbol{P}* {1}, \boldsymbol{P} *{2}, \tilde{\boldsymbol{E}}* {1:i-1}\right) \qquad \text{for } i=1,\dots,N$$

4. 以生成的嵌入向量 $\tilde{\boldsymbol{E}} *{1:N}$ 和 Flan-T5 文本嵌入 $\boldsymbol{E}* {2}$ 为条件，通过 LDM 的反向扩散过程对随机隐变量进行 *去噪* 。LDM 在反向扩散过程中运行 $T$ 个步推理:

   $$\boldsymbol{z}_{t} = \text{LDM}\left(\boldsymbol{z}_{t-1} | \tilde{\boldsymbol{E}}_{1:N}, \boldsymbol{E}_{2}\right) \qquad \text{for } t = 1, \dots, T$$

   其中初始隐变量 $\boldsymbol{z}_{0}$ 是从正态分布 $\mathcal{N} \left(\boldsymbol{0}, \boldsymbol{I} \right )$ 中采样而得。 LDM 的 [UNet](https://huggingface.co/docs/diffusers/api/pipelines/audioldm2/AudioLDM2UNet2DConditionModel) 的独特之处在于它需要 **两组** 交叉注意力嵌入，来自 GPT2 语言模型的 $\tilde{\boldsymbol{E}}* {1:N}$ 和来自 Flan-T5 的 $\boldsymbol{E}_{2}$，而其他大多数 LDM 只有一个交叉注意力条件。

5. 把最终去噪后的隐变量 $\boldsymbol{z}_{T}$ 传给 VAE 解码器以恢复梅尔谱图 $\boldsymbol{s}$:

​	$$ \boldsymbol{s} = \text{VAE} *{\text{dec}} \left(\boldsymbol{z}* {T}\right) $$

6. 梅尔谱图被传给声码器 (vocoder) 以获得输出音频波形 $\mathbf{y}$:

   $$ \boldsymbol{y} = \text{Vocoder}\left(\boldsymbol{s}\right) $$