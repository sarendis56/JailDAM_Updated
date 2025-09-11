# <img src="asset/shield.png" width="50">JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model
Yi Nian*, Shenzhe Zhu*, Yuehan Qin, Shawn Li, Ziyi Wang,
[Chaowei Xiao](https://xiaocw11.github.io/), [Yue Zhao†](https://viterbi-web.usc.edu/~yzhao010/)

(*Contribute equally, †Corresponding author)

[**📝 arxiv**](https://arxiv.org/abs/2504.03770)

**<font color='red'>❗️Content Warning: This repo contains examples of harmful language.</font>**
## NEWS!!:
- 2025/09/03 We just released our code, thank you for your patient
- 2025/07/08 Our paper was accepted by COLM 2025!!!
## TODO List:
- [x] 2025/04/01 Create repo 
- [x] 2025/04/09 Release paper on Arxiv
- [x] 2025/05/25 Release data
- [x] 2025/09/03 Release Code

<figure style="margin: 0 auto; text-align: center; width: fit-content;">
  <img src="asset/JailDam.jpg" style="width: 600px;">
</figure>

## Quick Start

1. **Download the dataset** from [Google Drive](https://drive.google.com/drive/folders/16Ge5BKIbj6bbD0fJOPYT9i8o5lr7HNgc).  
2. **Place the data** in the project root under a folder named `./data`.  
3. **Set up the environment**:  
   ```bash
   conda env create -f environment.yml
   conda activate llava
4. **Run the demo** by opening `demo.ipynb`

## 💡 Abstract
Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where
the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce
a test-time adaptive framework called JAILDAM . Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework
improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.


## ⚡ Detection Framework
JAILDAM overview. (A) Training: We encode safe text and images with CLIP, computing attention scores against a policy-driven unsafe memory bank. An autoencoder learns to reconstruct these features, linking benign inputs to unsafe concepts without explicit harmful data. (B) Inference: For each new input, we compute attention scores and measure the autoencoder’s reconstruction error; high error indicates potential harm. If similarity to the memory bank is low, JAILDAM updates the least-used concept with a residual representation, adapting to new attacks over time.
<div style="text-align: center;">
<img src="asset/training_inference_stage.jpg"/>
</div>

## 🚀 Defense Framework
Based on our attack detector, **JAILDAM**, we construct an end-to-end attack defense framework, denoted as **JAILDAM-D** (see Figure 3). This framework implements a two-stage defense approach:

1. **Attack Detection**: The incoming instruction \( S \) is first analyzed by **JAILDAM**, which evaluates whether the input contains potential attack patterns or harmful content.  
2. **Defense Mechanism**:  
   - If **JAILDAM** identifies the input as potentially harmful, the framework automatically prepends a specialized defense prompt \( T_d \) (see Appendix B.2) before the original query. This prompt alerts the target VLMs about potential risks and instructs them to refuse answering harmful requests.  
   - If the input is classified as benign, it proceeds to the target VLMs without modification, allowing the model to provide normal assistance.  
<div style="text-align: center;">
<img src="asset/shield_pipe.jpg"/>
</div>

## 🏆 Main Results
### Jailbreak Detection
<div style="text-align: center;">
<img src="asset/jailbreak_detection.png"/>
</div>

### Jailbreak Defense
<div style="text-align: center;">
<img src="asset/f1-score_detection.jpg"/>
</div>

## 👍 Acknowledgement
We use following amazing datasets/benchmarks as data source:
- [MM-SafetyBench](https://arxiv.org/abs/2311.17600)
- [FigStep](https://arxiv.org/abs/2311.05608)
- [JailbreakV-28k](https://arxiv.org/abs/2404.03027)
- [MM-Vet](https://arxiv.org/abs/2308.02490)

## ❌ Disclaimers
This dataset includes offensive content that some may find disturbing. It is intended solely for educational and research use.

## Contact
- Yi Nian: yn2336@columbia.edu
- Shenzhe Zhu: cho.zhu@mail.utoronto.ca

## 📖 BibTeX:
```
@article{nian2025jaildam,
  title={JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model},
  author={Nian, Yi and Zhu, Shenzhe and Qin, Yuehan and Li, Li and Wang, Ziyi and Xiao, Chaowei and Zhao, Yue},
  journal={arXiv preprint arXiv:2504.03770},
  year={2025}
}
```
