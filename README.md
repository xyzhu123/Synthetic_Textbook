<br />
<p align="center">
  <h1 align="center">LLM Unlearning Without an Expert Curated Dataset</h1>
  <p align="center">
    <br />
    <a href="https://www.linkedin.com/in/xiaoyuan-zhu-38005a224"><strong>Xiaoyuan Zhu</strong></a>
    ·
    <a href="https://nanami18.github.io/"><strong>Muru Zhang</strong></a>
    ·
    <a href="https://ollieliu.com/"><strong>Ollie Liurobinr</strong></a>
    ·
    <a href="https://robinjia.github.io/"><strong>Robin Jia</strong></a>
    .
    <a href="https://willieneis.github.io/"><strong>Willie Neiswanger</strong></a>
  </p>

  <p align="center">
    <a href='https://arxiv.org/abs/2508.06595'>
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='arXiv PDF'>
    </a>
    <a href='https://github.com/xyzhu123/Synthetic_Textbook'>
      <img src='https://img.shields.io/badge/Github-Code-blue?style=flat&logo=Github' alt='Code'>
    </a>
    <a href='https://huggingface.co/collections/WhyTheMoon/synthetic-textbook-68e72400ca8b4228b720ec60'>
      <img src='https://img.shields.io/badge/HuggingFace-Collection-yellow?style=flat&logo=huggingface' alt='Hugging Face Collection'>
    </a>
  </p>
<br />

## 🎉 News

- [2025-7-7] Our work is accepted to [COLM 2025](https://colmweb.org/)!

## 🗂️ Datasets
We provide the synthetic-textbook baseline forget sets for different domains. 

- **Biosecurity**: [🤗 Textbook-Bio](https://huggingface.co/datasets/WhyTheMoon/textbook_bio) · [Keyword-Bio](https://huggingface.co/datasets/WhyTheMoon/keyword_bio) · [Filter-Bio](https://huggingface.co/datasets/WhyTheMoon/filter_bio)

- **Cybersecurity**: [🤗 Textbook-Cyber](https://huggingface.co/datasets/WhyTheMoon/textbook_cyber) · [Keyword-Cyber](https://huggingface.co/datasets/WhyTheMoon/keyword_cyber) · [Filter-Cyber](https://huggingface.co/datasets/WhyTheMoon/filter_cyber)

- **Harry Potter**: [🤗 Textbook-HP](https://huggingface.co/datasets/WhyTheMoon/textbook_hp)

## 🧩 Models
We provide the best unlearning checkpoints for RMU and RR methods.

---

### Mistral-7B-Instruct-v0.3

| Domain | RMU  | RR  |
|:-------|:----------------|:----------------|
| **Biosecurity** | [🤗 Textbook-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Textbook-Bio) · [Keyword-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Keyword-Bio) · [Filter-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Filter-Bio) | [🤗 Textbook-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Textbook-Bio) · [Keyword-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Keyword-Bio) · [Filter-Bio](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Filter-Bio) |
| **Cybersecurity** | [🤗 Textbook-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Textbook-Cyber) · [Keyword-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Keyword-Cyber) · [Filter-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Filter-Cyber) | [🤗 Textbook-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Textbook-Cyber) · [Keyword-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Keyword-Cyber) · [Filter-Cyber](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Filter-Cyber) |
| **Harry Potter** | [🤗 Textbook-HP](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Textbook-HP) · [Textbook-HP-Simplest](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RMU_Textbook-HP-Simplest) | [🤗 Textbook-HP](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Textbook-HP) · [Textbook-HP-Simplest](https://huggingface.co/WhyTheMoon/Mistral-7B-Instruct-v0.3_RR_Textbook-HP-Simplest) |

---

### Llama-3-8B-Instruct

| Domain | RMU  | RR  |
|:-------|:----------------|:----------------|
| **Biosecurity** | [🤗 Textbook-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Textbook-Bio) · [Keyword-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Keyword-Bio) · [Filter-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Filter-Bio) | [🤗 Textbook-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Textbook-Bio) · [Keyword-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Keyword-Bio) · [Filter-Bio](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Filter-Bio) |
| **Cybersecurity** | [🤗 Textbook-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Textbook-Cyber) · [Keyword-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Keyword-Cyber) · [Filter-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Filter-Cyber) | [🤗 Textbook-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Textbook-Cyber) · [Keyword-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Keyword-Cyber) · [Filter-Cyber](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Filter-Cyber) |
| **Harry Potter** | [🤗 Textbook-HP](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Textbook-HP) · [Textbook-HP-Simplest](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RMU_Textbook-HP-Simplest) | [🤗 Textbook-HP](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Textbook-HP) · [Textbook-HP-Simplest](https://huggingface.co/WhyTheMoon/Llama-3-8B-Instruct_RR_Textbook-HP-Simplest) |


## 🔧 Quick Start
### Installation
``` bash
# Clone and navigate to repository
git clone https://github.com/xyzhu123/Synthetic_Textbook.git
cd Synthetic_Textbook

# Set up environment
conda create -n synthetic_textbook python=3.10
conda activate synthetic_textbook
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API key (OPENAI_API_KEY or TOGETHER_API_KEY)
```
### Generate Synthetic Textbook Dataset
#### Option 1: Quick Test
Generate a small test dataset to verify setup:
``` bash
# Test with OpenAI (requires OPENAI_API_KEY in .env)
python scripts/generate_textbook.py \
    --provider openai \
    --keyword "biosecurity" \
    --stages all \
    --num-subfields 2 \
    --num-chapters-per-bp 1 \
    --audiences "undergraduate student"

# Output: data/biosecurity_gpt-4o-mini-2024-07-18_textbook_processed.csv
```
#### Option 2: Full Dataset Generation
Generate a complete synthetic textbook dataset:
```bash
# Generate with OpenAI
python scripts/generate_textbook.py \
    --provider openai \
    --keyword "biosecurity" \
    --model-name gpt-4o-mini-2024-07-18 \
    --stages all \
    --num-subfields 10 \
    --num-chapters-per-bp 5

# Generate with Together AI (open-sourced models)
python scripts/generate_textbook.py \
    --provider together \
    --keyword "biosecurity" \
    --model-name mistralai/Mistral-7B-Instruct-v0.3 \
    --stages all \
    --num-subfields 10 \
    --num-chapters-per-bp 5
```
#### Output Files
After generation, you'll find these files in `data/`:
``` bash
data/
├── {keyword}_{model}_subfields.json          # (1). Generated subfields
├── {keyword}_{model}_bps.json                # (2). Bullet points for all subfield-audience combinations
├── {keyword}_{model}_textbook_generated.json # (3). Generated chapters
└── {keyword}_{model}_textbook_processed.csv  # (4). Final dataset (use for unlearning)
```
The final CSV file (`{keyword}_{model}_textbook_processed.csv`) is ready to use as a forget set for unlearning. 

## ✏️ Citation
If you find this useful in your research, please consider citing our paper:
```
@misc{zhu2025llmunlearningexpertcurated,
      title={LLM Unlearning Without an Expert Curated Dataset}, 
      author={Xiaoyuan Zhu and Muru Zhang and Ollie Liu and Robin Jia and Willie Neiswanger},
      year={2025},
      eprint={2508.06595},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.06595}, 
}
```
