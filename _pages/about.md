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

I'm currently a Ph.D. student (from fall 2022) at the [School of Computer Science](https://cs.fudan.edu.cn/) of [Fudan University](https://www.fudan.edu.cn/) and a member of the [FudanNLP Lab](https://nlp.fudan.edu.cn/), supervised by A.P. [Xiaoqing Zheng (郑骁庆)](https://faculty.fudan.edu.cn/zhengxq/zh_CN/) and Prof. [Xuanjing Huang (黄萱菁)](https://xuanjing-huang.github.io/).

My research interests include Brain-inspired Computing, Agentic RL, LLM Reasoning, Time Series Analysis, and AI for Science.
I am currently working on Agentic Reinforcement Learning, Generative Protein Foundation Models, and Biologically-Plausible Learning Algorithms.

I have led first-author research published at ICML, NeurIPS, and ICLR, and have also contributed as a co-author to works appearing at ACL, EMNLP, WWW, COLING, InfoCom, and IJCAI.

I serve as the reviewer for conferences (ICML, ICLR, NeurIPS, ACL, EMNLP, ICCV, IJCAI, AAAI, ACM MM, COLING) and journals (IEEE Transactions on Affective Computing, IEEE Transactions on Neural Networks and Learning Systems, IEEE Transactions on Cognitive and Developmental Systems, Neural Networks).

Phone/Wechat/Telegram: (+86) 13967492189. Please feel free to reach out to me.

# 📖 Educations
- *2018.09 - 2022.06*, Bachelor in Software Engineering (School of Software, Fudan University)
- *2019.09 - 2022.06*, Bachelor in Economics (School of Economics and Management, Fudan University, Second Degree)
- *2022.09 - 2027.06 (Expected)*, Ph.D. Student in Computer Science (School of Computer Science, Fudan University).

# 💻 Internships
- *2023.11 - 2025.02*, [Microsoft Research Asia](https://www.msra.cn/). Artificial Intelligence & Machine Learning Group. Research on time-series forecasting. Supervised by [Yansen Wang](https://scholar.google.com/citations?user=Hvbzb1kAAAAJ&hl=en), [Dongqi Han](https://scholar.google.com.hk/citations?user=3V_9fRUAAAAJ&hl=zh-CN), and [Dongsheng Li](https://scholar.google.com/citations?user=VNg5rA8AAAAJ&hl=zh-CN).
- *2025.03 - 2025.09*, [Shanghai AI Lab](https://www.shlab.org.cn/). AI for Science Group. Research on protein foundation models. Supervised by [Lihao Wang](https://github.com/leowang1217), [Jiangtao Feng](https://scholar.google.com/citations?user=7ufSFeIAAAAJ&hl=en), and [Hao Zhou](https://zhouh.github.io/).
- *2025.09 - Current*, [Tencent](https://www.tencent.com/zh-cn/index.html). Qingyun Plan (青云计划) at WeChat AI Group. Research on agentic reinforcement learning. Supervised by [Le Tian](https://openreview.net/profile?id=%7ELe_Tian1).

# 🔥 News
- *2026.01*: &nbsp;🎉🎉 One paper on Neuromorphic Edge Sensing was accepted by **WWW-2026**!
- *2025.12*: &nbsp;🎉🎉 One paper on Pretrained Spiking Language Model SpikeBERT was accepted by **Neural-Networks**!
- *2025.12*: &nbsp;🎉🎉 One paper on SNN Time Parallelism and Asynchronous Pipeline was accepted by **InfoCom-2026**!
- *2025.11*: &nbsp;🎉🎉 One paper on AIGC detection was accepted by **AAAI-2026**!
- *2025.09*: &nbsp;🎉🎉 One paper on Relative Positional Encoding for SNNs was accepted by **NeurIPS-2025-Spotlight**!
- *2025.09*: &nbsp;🎉🎉 Two papers on LLMs safety and personality were accepted by **EMNLP-2025-Findings**!
- *2025.05*: &nbsp;🎉🎉 Five papers on LLMs were accepted by **ACL-2025-Main/Findings/Demo**!
- *2025.05*: &nbsp;🎉🎉 One paper on Biologically-Plausible Learning Algorithm was accepted by **ICML-2025**!
- *2025.04*: &nbsp;🎉🎉 Two papers on On-device SNNs was accepted by **IJCAI-2025**!
- *2025.04*: &nbsp;🎉🎉 One paper on Multi-Modal SpikeCLIP was accepted by **Neural-Networks**!
- *2024.12*: &nbsp;🎉🎉 One paper on Jailbreak of LLMs was accepted by **COLING-2025**!
- *2024.09*: &nbsp;🎉🎉 One paper on Positional Encoding Analysis for SNNs was accepted by **NeurIPS-2024-Spotlight**!
- *2024.09*: &nbsp;🎉🎉 Two papers on RAG and LLM safety were accepted by **EMNLP-2024-Main/Findings**!
- *2024.05*: &nbsp;🎉🎉 Two papers on LLM alignment and PEFT were accepted by **ACL-2024-Main**!
- *2024.05*: &nbsp;🎉🎉 One paper on Time-Series Forecasting with SNNs was accepted by **ICML-2024**!
- *2023.10*: &nbsp;🎉🎉 One paper on Parameter-Efficient-Fine-Tuning of LLMs was accepted by **EMNLP-2023-Findings**!
- *2023.01*: &nbsp;🎉🎉 One paper on SNNs for Text Classification was accepted by **ICLR-2023**!

# 📚 Technical Reports

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">AMix-1</div><img src='../images/AMix1.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

AMix-1: A Pathway to Test-Time Scalable Protein Foundation Model
- Shanghai AI Lab & Tsinghua AIR GenSI Lab
- A powerful protein foundation model built on Bayesian Flow Networks and engined by a systematic training methodology, encompassing pretraining scaling laws, emergent capability analysis, in-context learning strategy, and test-time scaling algorithm. I was responsible for pretraining, scaling laws, and emergent ability.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2507.08920) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/GenSI-THUAIR/AMix-1) 
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">RAG</div><img src='../images/rag.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Searching for Best Practices in Retrieval-Augmented Generation
- FudanNLP Group
- This study investigates the contribution of each component and provides insights into optimal RAG practices through extensive experimentation.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2407.01219) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/FudanDNN-NLP/RAG) 
</div>
</div>

# 📝 Publications

## Brain-Inspired Computing

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">NeurIPS 2025 (Spotlight)</div><img src='../images/REP_SNN.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
  
Toward Relative Positional Encoding in Spiking Transformers \\
**Changze Lv**, Yansen Wang, Dongqi Han, et al.
- A relative positional encoding for spiking Transformers based on Gray Code and logarithm.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2501.16745) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/microsoft/SeqSNN)  
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICML-2025 (Poster)</div><img src='../images/DLL.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
  
Dendritic Localized Learning: Toward Biologically Plausible Algorithm \\
**Changze Lv**\*, Jingwen Xu\*, Yiyang Lu\*, et al.
- A new biologically plausible training method.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2501.09976) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/Dendritic-Localized-Learning) 
</div>
</div>

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
- A "conversion + fine-tuning" two-step method for training SNNs for text classification.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://openreview.net/forum?id=pgU3k7QXuz0) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/snn) 
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Neural Networks</div><img src='../images/spikeclip.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
  
SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network \\
**Changze Lv**\*, Tianlong Li\*, Xiaoqing Zheng, et al
- A method for training language-image multimodal SNNs.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://www.sciencedirect.com/science/article/abs/pii/S0893608025003545) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/SpikeCLIP)
</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Neural Networks</div><img src='../images/spikebert.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
  
SpikeBERT: A Language Spikformer Trained with Two-stage Knowledge Distillation from BERT \\
**Changze Lv**, Tianlong Li, Jianhan Xu, et al.
- A spiking language model for language understanding based on Spikformer.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2308.15122.pdf) \| [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/Lvchangze/SpikeBERT) 
</div>
</div>


<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ArXiv (2509.20284)</div><img src='../images/bsd.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">
  
Biologically Plausible Learning via Bidirectional Spike-Based Distillation \\
**Changze Lv**\*, Yifei Wang\*, Yanxun Zhang\*, et al.
- Bidirectional Spike-based Distillation (BSD), a novel learning algorithm that jointly trains a feedforward and a backward SNN.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2509.20284)
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

## Large Language Models

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">Arxiv(2503.04355)</div><img src='../images/layer-wise-PE.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

Layer-Specific Scaling of Positional Encodings for Superior Long-Context Modeling \\
Zhenghua Wang\*, Yiran Ding\*, **Changze Lv**\*, et al.
- A layer-specific positional encoding scaling method.
- [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2503.04355)
</div>
</div>

## Others

- ![](https://img.shields.io/badge/AAAI--2026-darkblue) Explainable Synthetic Image Detection through Diffusion Timestep Ensembling
<br> Yixin Wu, Feiran Zhang, Tianyuan Shi, Ruicheng Yin, Zhenghua Wang, Zhenliang Gan, Xiaohua Wang, **Changze Lv**, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2503.06201)

- ![](https://img.shields.io/badge/EMNLP--2025--Findings-darkblue) Enhancing Model Privacy in Federated Learning with Random Masking and Quantization
<br> Zhibo Xu, JianHao Zhu, Jingwen Xu, **Changze Lv**, Zhenghua Wang, Zisu Huang, Xiaohua Wang, Muling Wu, Qi Qian, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2508.18911) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/zb2313/FedQSN)

- ![](https://img.shields.io/badge/EMNLP--2025--Findings-darkblue) UPLex: Fine-Grained Personality Control in Large Language Models via Unsupervised Lexical Modulation
<br> Tianlong, Wenhao Liu, Muling Wu, Shihan Dou, Zhenghua Wang, **Changze Lv**, Xiaohua Wang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2310.16582v3)

- ![](https://img.shields.io/badge/ACL--2025--Main-darkblue) Beyond Single Labels: Improving Conversational Recommendation through LLM-Powered Data Augmentation
<br> Haozhe Xu, Xiaohua Wang, **Changze Lv**, Xiaoqing Zheng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2508.05657) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/xu1110/FNSCRS)

- ![](https://img.shields.io/badge/ACL--2025--Findings-darkblue) TripTailor: A Real-World Benchmark for Personalized Travel Planning
<br> Kaimin Wang, Yuanzhe Shen, **Changze Lv**, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://aclanthology.org/2025.findings-acl.503.pdf) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/swxkfm/TripTailor)

- ![](https://img.shields.io/badge/ACL--2025--Findings-darkblue) 	
Tell Me What You Don’t Know: Enhancing Refusal Capabilities of Role-Playing Agents via Representation Space Analysis and Editing
<br> Wenhao Liu, Siyu An, Junru Lu, Muling Wu, Tianlong Li, Xiaohua Wang, **Changze Lv**, Xiaoqing Zheng, Di Yin, Xing Sun, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://aclanthology.org/2025.findings-acl.311.pdf)

- ![](https://img.shields.io/badge/ACL--2025--Findings-darkblue) Improving Continual Pre-training Through Seamless Data Packing
<br> Ruicheng Yin, Xuan Gao, **Changze Lv**, Xiaohua Wang, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2505.22018) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/InfernusWIND/Seamless-Packing)

- ![](https://img.shields.io/badge/ACL--2025--Demo-darkblue) Multi-Programming Language Sandbox for LLMs
<br> Shihan Dou, Jiazheng Zhang, Jianxiang Zang, Yunbo Tao, Haoxiang Jia, Shichun Liu, Yuming Yang, Shenxi Wu, Shaoqing Zhang, Muling Wu, **Changze Lv**, Limao Xiong, Wenyu Zhan, Lin Zhang, Rongxiang Weng, Jingang Wang, Xunliang Cai, Yueming Wu, Ming Wen, Rui Zheng, Tao Ji, Yixin Cao, Tao Gui, Xipeng Qiu, Qi Zhang, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2410.23074)

- ![](https://img.shields.io/badge/IJCAI--2025-darkblue) ECC-SNN: Cost-Effective Edge-Cloud Collaboration for Spiking Neural Networks
<br> Di Yu, **Changze Lv**, Xin Du, Linshan Jiang, Wentao Tong, Zhenyu Liao, Xiaoqing Zheng, Shuiguang Deng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://www.ijcai.org/proceedings/2025/0768.pdf) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/AmazingDD/ECC-SNN)

- ![](https://img.shields.io/badge/IJCAI--2025-darkblue) Cost-Effective On-Device Sequential Recommendation with Spiking Neural Networks
<br> Di Yu, **Changze Lv**, Xin Du, Linshan Jiang, Qin Yin, Wentao Tong, Xiaoqing Zheng, Shuiguang Deng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://www.ijcai.org/proceedings/2025/0398.pdf) | [![](https://img.shields.io/badge/Code-fff?logo=github&logoColor=000)](https://github.com/AmazingDD/serenRec)

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

- ![](https://img.shields.io/badge/Arxiv--yellow) SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration
<br> Bang Hu, **Changze Lv**, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Fan Zhang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2508.02069)

- ![](https://img.shields.io/badge/Arxiv--yellow) STF: Shallow-Level Temporal Feedback to Enhance Spiking Transformers
<br> Zeqi Zheng, Zizheng Zhu, Yingchao Yu, Yanchen Huang, **Changze Lv**, Junfeng Tang, Zhaofei Yu, Yaochu Jin
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2508.00387?)

- ![](https://img.shields.io/badge/Arxiv--yellow) Edge Intelligence with Spiking Neural Networks
<br> Shuiguang Deng, Di Yu, **Changze Lv**, Xin Du, Linshan Jiang, Xiaofan Zhao, Wentao Tong, Xiaoqing Zheng, Weijia Fang, Peng Zhao, Gang Pan, Schahram Dustdar, Albert Y Zomaya
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2507.14069)

- ![](https://img.shields.io/badge/Arxiv--yellow) Progressive Mastery: Customized Curriculum Learning with Guided Prompting for Mathematical Reasoning
<br> Muling Wu, Qi Qian, Wenhao Liu, Xiaohua Wang, Zisu Huang, Di Liang, LI Miao, Shihan Dou, **Changze Lv**, Zhenghua Wang, Zhibo Xu, Lina Chen, Tianlong Li, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2506.04065)

- ![](https://img.shields.io/badge/Bioxiv--yellow) Decoding Continuous Character-based Language from Non-invasive Brain Recordings
<br> Cenyuan Zhang, Xiaoqing Zheng, Ruicheng Yin, Shujie Geng, Jianhan Xu, Xuan Gao, **Changze Lv**, Zixuan Ling, Xuanjing Huang, Miao Cao, Jianfeng Feng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://www.biorxiv.org/content/biorxiv/early/2024/03/20/2024.03.19.585656.full.pdf)

- ![](https://img.shields.io/badge/Arxiv--yellow) RECAST: Strengthening LLMs' Complex Instruction Following with Constraint-Verifiable Data
<br> Wenhao Liu, Zhengkang Guo, Mingchen Xie, Jingwen Xu, Zisu Huang, Muzhao Tian, Jianhan Xu, Muling Wu, Xiaohua Wang, **Changze Lv**, He-Da Wang, Hu Yao, Xiaoqing Zheng, Xuanjing Huang
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2505.19030)

- ![](https://img.shields.io/badge/Arxiv--yellow) SAFA-SNN: Sparsity-Aware On-Device Few-Shot Class-Incremental Learning with Fast-Adaptive Structure of Spiking Neural Network
<br> Huijing Zhang, Muyang Cao, Linshan Jiang, Xin Du, Di Yu, **Changze Lv**, Shuiguang Deng
<br> [![](https://img.shields.io/badge/Paper-fff?logo=readthedocs&logoColor=000)](https://arxiv.org/pdf/2510.03648?)

# 🎖 Honors and Awards
- *2025* National Scholarship (国家奖学金)
- *2025* Stars of Tomorrow Internship Program of Microsoft Research Asia (微软亚洲研究院“明日之星”)
- *2024* Outstanding Student Leader of Fudan University (复旦大学优秀学生干部)
- *2022* Excellent graduates of Fudan University (复旦大学优秀毕业生)
- *2021* Shanghai Scholarship (上海市奖学金)
- *2020* Meritorious Prize in the Mathematical Contest in Modeling/Interdisciplinary Contest in Modeling (美国数学建模大赛M奖)

# Welcome
<script type='text/javascript' id='clustrmaps' src='//cdn.clustrmaps.com/map_v2.js?cl=ffffff&w=300&t=tt&d=w2UTaZyU2VVylIDbKR9XlKrB0nuQxq0EeuTgqkW9LDg'></script>
