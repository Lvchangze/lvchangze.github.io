---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I'm currently a Ph.D. student (from fall, 2022) at the [School of Computer Science](https://cs.fudan.edu.cn/) of [Fudan University](https://www.fudan.edu.cn/) and a member of the [FudanNLP Lab](https://nlp.fudan.edu.cn/), advised by A.P. [Xiaoqing Zheng (郑骁庆)](https://faculty.fudan.edu.cn/zhengxq/zh_CN/) and Prof. [Xuanjing Huang (黄萱菁)](https://xuanjing-huang.github.io/).

My research interests cover Brain-inspired Computing, Large Language Models, and Multi-Modal LLMs.
I am currently working on **Spiking Neural Networks for Sequential Tasks** and **Biologically-Plausible Learning Algorithms**.

First-author Publications: ICLR 2023, ICML 2024, NeurIPS 2024.

Co-author Publications: ACL, EMNLP, COLING.

I serve as the reviewer for conferences (ICML 2025, ICLR 2025, NeurIPS 2024) and journals (Neural Networks).

# 🔥 News
- *2024.12*: &nbsp;🎉🎉 One paper on Jailbreak of LLMs was accepted by **COLING-2025**!
- *2024.09*: &nbsp;🎉🎉 One paper on positional encoding for SNNs was accepted by **NeurIPS-2024-Spotlight**!
- *2024.09*: &nbsp;🎉🎉 Two papers on RAG and LLM safety were accepted by **EMNLP-2024-Main/Findings**!
- *2024.05*: &nbsp;🎉🎉 Two papers on LLM alignment and PEFT were accepted by **ACL-2024-Main**!
- *2024.05*: &nbsp;🎉🎉 One paper on time-series forecasting with spiking neural networks was accepted by **ICML-2024**!
- *2023.10*: &nbsp;🎉🎉 One paper on parameter-efficient-fine-tuning was accepted by **EMNLP-2023-Findings**!
- *2023.01*: &nbsp;🎉🎉 One paper on spiking neural networks for text classification was accepted by **ICLR-2023**!

# 📝 Publications

## Spiking Neural Networks

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2024 (Spotlight)</div><img src='../images/cpg.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Advancing Spiking Neural Networks for Sequential Modeling with Central Pattern Generators \\
**Changze Lv**, Dongqi Han, Yansen Wang, et al.
- A bio-inspired novel positional encoding method for spiking neural networks.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2405.14362) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/microsoft/SeqSNN) 
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML-2024 (Poster)</div><img src='../images/snn_ts.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Efficient and Effective Time-Series Forecasting with Spiking Neural Networks \\
**Changze Lv**, Yansen Wang, Dongqi Han, et al.
- A framework for spiking neural networks in time-series forecasting tasks.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2402.01533.pdf) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/microsoft/SeqSNN) 

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR-2023 (Poster)</div><img src='../images/spikingTextCNN.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Spiking Convolutional Neural Networks for Text Classification \\
**Changze Lv**, Jianhan Xu, Xiaoqing Zheng
- A "conversion+ fine-tuning" two-step method for training SNNs for text classification.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://openreview.net/forum?id=pgU3k7QXuz0) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/snn) 
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv(2308.15122)</div><img src='../images/spikebert.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

SpikeBERT: A Language Spikformer Trained with Two-stage Knowledge Distillation from BERT \\
**Changze Lv**, Tianlong Li, Jianhan Xu, et al.
- A spiking language model for language understanding based on Spikformer.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2308.15122.pdf) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/SpikeBERT) 
</div>
</div>

## Brain-Inspired Learning

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv(2501.09976)</div><img src='../images/DLL.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Dendritic Localized Learning: Toward Biologically Plausible Algorithm \\
**Changze Lv**\*, Jingwen Xu\*, Yiyang Lu\*, et al.
- A novel biologically plausible training method.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2501.09976)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv(2406.16062)</div><img src='../images/bio_survey.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Towards Biologically Plausible Computing: A Comprehensive Comparison \\
**Changze Lv**, Yufei Gu, Zhengkang Guo, et al.
- A comprehensive comparison of various brain-inspired training methods.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2406.16062)
</div>
</div>

## Others

- ![](https://img.shields.io/badge/COLING--2025-darkblue) Revisiting Jailbreaking for Large Language Models: A Representation Engineering Perspective
<br> Tianlong Li, Zhenghua Wang, Wenhao Liu, Muling Wu, Shihan Dou, **Changze Lv**, Xiaohua Wang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2401.06824)

- ![](https://img.shields.io/badge/EMNLP--2024--Main-darkblue) Searching for Best Practices in Retrieval-Augmented Generation
<br> Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng Yin, **Changze Lv**, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2407.01219) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/FudanDNN-NLP/RAG)

- ![](https://img.shields.io/badge/EMNLP--2024--Findings-darkblue) Promoting Data and Model Privacy in Federated Learning through Quantized LoRA
<br> JianHao Zhu, **Changze Lv**, Xiaohua Wang, Muling Wu, Wenhao Liu, Tianlong Li, Zixuan Ling, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2406.10976)

- ![](https://img.shields.io/badge/ACL--2024--Main-darkblue) Aligning Large Language Models with Human Preferences through Representation Engineering
<br> Wenhao Liu, Xiaohua Wang, Muling Wu, Tianlong Li, **Changze Lv**, Zixuan Ling, Jianhao Zhu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2312.15997) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/LiuAmber/RAHF)

- ![](https://img.shields.io/badge/ACL--2024--Main-darkblue) Advancing Parameter Efficiency in Fine-tuning via Representation Editing
<br> Muling Wu, Wenhao Liu, Xiaohua Wang, Tianlong Li, **Changze Lv**, Zixuan Ling, Jianhao Zhu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2402.15179) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/mlwu22/RED)

- ![](https://img.shields.io/badge/EMNLP--2023--Findings-darkblue) Parameter Efficient Multi-task Fine-tuning by Learning to Transfer Token-wise Prompts
<br> Muling Wu, Wenhao Liu, Jianhan Xu, **Changze Lv**, Zixuan Ling, Tianlong Li, Longtao Huang, Xiaoqing Zheng, Xuan-Jing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://aclanthology.org/2023.findings-emnlp.584.pdf) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/mlwu22/TPT)

- ![](https://img.shields.io/badge/Arxiv--yellow) Tailoring Personality Traits in Large Language Models via Unsupervisedly-Built Personalized Lexicons
<br> Tianlong Li, Shihan Dou, **Changze Lv**, Wenhao Liu, Jianhan Xu, Muling Wu, Zixuan Ling, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2310.16582)

- ![](https://img.shields.io/badge/Arxiv--yellow) SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network
<br> Tianlong Li, Wenhao Liu, **Changze Lv**, Jianhan Xu, Cenyuan Zhang, Muling Wu, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2310.06488)

- ![](https://img.shields.io/badge/Bioxiv--yellow) Decoding Continuous Character-based Language from Non-invasive Brain Recordings
<br> Cenyuan Zhang, Xiaoqing Zheng, Ruicheng Yin, Shujie Geng, Jianhan Xu, Xuan Gao, **Changze Lv**, Zixuan Ling, Xuanjing Huang, Miao Cao, Jianfeng Feng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://www.biorxiv.org/content/biorxiv/early/2024/03/20/2024.03.19.585656.full.pdf)

- ![](https://img.shields.io/badge/Arxiv--yellow) Multi-Programming Language Sandbox for LLMs
<br> Shihan Dou, Jiazheng Zhang, Jianxiang Zang, Yunbo Tao, Haoxiang Jia, Shichun Liu, Yuming Yang, Shenxi Wu, Shaoqing Zhang, Muling Wu, **Changze Lv**, Limao Xiong, Wenyu Zhan, Lin Zhang, Rongxiang Weng, Jingang Wang, Xunliang Cai, Yueming Wu, Ming Wen, Rui Zheng, Tao Ji, Yixin Cao, Tao Gui, Xipeng Qiu, Qi Zhang, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2410.23074)

# 🎖 Honors and Awards
- *2024* Outstanding Student Leader of Fudan University (复旦大学优秀学生干部)
- *2023* Outstanding Student of Fudan University (复旦大学优秀学生)
- *2022* Excellent graduates of Fudan University (复旦大学优秀毕业生)
- *2021* Shanghai Scholarship (上海市奖学金)
- *2021* First Prize for Outstanding Undergraduate Student Scholarship, Fudan University (复旦大学一等奖学金)
- *2020* Meritorious Prize in the Mathematical Contest in Modeling/Interdisciplinary Contest In Modeling (美国数学建模大赛M奖)
- *2019* Third Prize in the National College Student Mathematics Competition (全国大学生数学竞赛三等奖)

# 📖 Educations
- *2022.09 - Current*, Ph.D. Student in Computer Science (School of Computer Science, Fudan University).
- *2019.09 - 2022.06*, Bachelor in Economics (School of Economics and Management, Fudan University, Second Degree)
- *2018.09 - 2022.06*, Bachelor in Software Engineering (School of Software, Fudan University)

# 💻 Internships
- *2023.11 - Current*, Microsoft Research Asia, Artificial Intelligence & Machine Learning Group, [MSRA](https://www.msra.cn/).
