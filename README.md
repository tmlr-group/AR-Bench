<div align="center">

<h1>AR-Bench</h1>

<h3> <a href="https://arxiv.org/abs/2506.08295">From Passive to Active Reasoning:<br> Can Large Language Models Ask the Right Questions under Incomplete Information?</a> </h3>

<h3> Abstract </h3>
<p align="left">While existing benchmarks probe the reasoning abilities of large language models (LLMs) across diverse domains, they predominantly assess passive reasoning, providing models with all the information needed to reach a solution. By contrast, active reasoning-where an LLM must interact with external systems to acquire missing evidence or data-has received little systematic attention. To address this shortfall, we present AR-Bench, a novel benchmark designed explicitly to evaluate an LLM's active reasoning skills. AR-Bench comprises three task families-detective cases, situation puzzles, and guessing numbers-that together simulate real-world, agentic scenarios and measure performance across commonsense, logical, and symbolic reasoning. <br><br>Empirical evaluation on AR-Bench demonstrates that contemporary LLMs exhibit pronounced difficulties with active reasoning: they frequently fail to acquire or leverage the information needed to solve tasks. This gap highlights a stark divergence between their passive and active reasoning abilities. Moreover, ablation studies indicate that even advanced strategies, such as tree-based searching or post-training approaches, yield only modest gains and fall short of the levels required for real-world deployment. Collectively, these findings highlight the critical need to advance methodology for active reasoning, e.g., incorporating interactive learning, real-time feedback loops, and environment-aware objectives for training.</p>

</div>

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data Generation](#data-generation)
- [Evaluation](#evaluation)
- [Examples](#examples)
- [Citation](#citation)
- [Contributing](#contributing)
- [Contact](#contact)


<a id="features"></a>
## ✨ Features

- **Three Diverse Task Families**: Detective cases, situation puzzles, and guessing numbers
- **Comprehensive Reasoning Assessment**: Tests commonsense, logical, and symbolic reasoning
- **Real-world Simulation**: Agentic scenarios requiring information acquisition
- **Flexible Model Support**: Compatible with both remote APIs and local models
- **Extensible Framework**: Easy to add new tasks and evaluation methods

<a id="installation"></a>
## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (for local model serving)
- Sufficient disk space for model downloads (if using local models)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tmlr-group/AR-Bench.git
   cd AR-Bench
   ```

2. **Install the package:**
   ```bash
   pip install -e .
   ```

<a id="configuration"></a>
## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with your API configurations.

#### Option 1: Remote APIs (OpenAI, TogetherAI, etc.)

```bash
# API Configuration for Data Generation
API_KEY=your_api_key
BASE_URL=your_base_url

# API Configuration for Evaluation
# Policy model configuration
POLICY_API_KEY=your_policy_api_key
POLICY_BASE_URL=your_policy_base_url

# Response model configuration  
RESPONSE_API_KEY=your_response_api_key
RESPONSE_BASE_URL=your_response_base_url
```

#### Option 2: Local Models with vLLM

1. **Start your models:**
   ```bash
   # Policy model
   vllm serve /path/to/your/policy_model \
       --api-key 123456 \
       --port 8000 \
       --gpu-memory-utilization 0.9 \
       --served-model-name your_policy_model_name \
       --trust-remote-code

   # Response model
   vllm serve /path/to/your/response_model \
       --api-key 123456 \
       --port 8001 \
       --gpu-memory-utilization 0.9 \
       --served-model-name your_response_model_name \
       --trust-remote-code
   ```

2. **Configure environment variables:**
   ```bash
   API_KEY=123456
   BASE_URL=http://localhost:8000
   POLICY_API_KEY=123456
   POLICY_BASE_URL=http://localhost:8000
   RESPONSE_API_KEY=123456
   RESPONSE_BASE_URL=http://localhost:8001
   ```

<a id="data-generation"></a>
## 📊 Data Generation

Generate custom datasets for evaluation using the following commands:

### Situation Puzzle Game

```bash
python3 -m arbench.data_generation.sp.sp_generator \
    --save_path="data/sp_stories.json" \
    --cnt="1" \
    --model="Qwen2.5-32B-Instruct" \
    --supernatural="false" \
    --lethal="true" \
    --depth="3" \
    --temperature="0.7" \
    --top_p="0.7"
```

**Parameters:**
- `--cnt`: Number of stories to generate
- `--supernatural`: Include supernatural elements (true/false)
- `--lethal`: Include lethal scenarios (true/false)
- `--depth`: Story complexity depth

### Detective Cases Game

```bash
python3 -m arbench.data_generation.dc.dc_generator \
    --save_path="data/dc_custom_cases.json" \
    --cnt="1" \
    --model="Qwen2.5-32B-Instruct" \
    --temperature="0.7" \
    --top_p="0.7"
```

### Quick Start Scripts

Use the provided scripts for rapid data generation:

```bash
# Generate Situation Puzzle data
bash script/generate/sp_generation.sh

# Generate Detective Cases data
bash script/generate/dc_generation.sh
```

> **Note:** Guessing Numbers data is pre-generated due to the limited number of 4-unique-digit combinations and is available in the `data/gn` folder.

<a id="evaluation"></a>
## 🧪 Evaluation

### Detective Cases

```bash
python3 -m arbench.reasoner.dc.dc_evaluator \
    --method="zero_shot" \
    --data_path="data/dc/test.json" \
    --output_path="./zero_shot_dc.json" \
    --policy_model="Qwen2.5-32B-Instruct" \
    --response_model="Qwen2.5-32B-Instruct" \
    --branch="3" \
    --max_turn="25" \
    --policy_temperature="0.7" \
    --policy_top_p="0.7" \
    --response_temperature="0.7" \
    --response_top_p="0.7"
```

### Situation Puzzles

```bash
python3 -m arbench.reasoner.sp.sp_evaluator \
    --method="zero_shot" \
    --data_path="data/sp/test.json" \
    --output_path="./zero_shot_sp.json" \
    --policy_model="Qwen2.5-32B-Instruct" \
    --response_model="Qwen2.5-32B-Instruct" \
    --branch="3" \
    --max_turn="25" \
    --policy_temperature="0.7" \
    --policy_top_p="0.7" \
    --response_temperature="0.7" \
    --response_top_p="0.7"
```

### Guessing Numbers

```bash
python3 -m arbench.reasoner.gn.gn_evaluator \
    --model="Qwen2.5-32B-Instruct" \
    --method="greedy" \
    --data_path="data/gn/test.json" \
    --output_path="./greedy_gn.json" \
    --max_turn="25"
```

### Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--method` | Reasoning method (zero_shot, few_shot, etc.) | - |
| `--branch` | Number of reasoning branches in ToT | 3 |
| `--max_turn` | Maximum interaction turns | 25 |
| `--temperature` | Sampling temperature | 0.7 |
| `--top_p` | Nucleus sampling parameter | 0.7 |
| `--simulate_depth` | The rounds of simulation for estimate the quality of questions in UoT | 1 |

<a id="examples"></a>
## 📋 Examples

### Quick Evaluation Scripts

For convenience, use the provided evaluation scripts:

```bash
# Evaluate Detective Cases
bash script/evaluation/dc_evaluation.sh

# Evaluate Situation Puzzles  
bash script/evaluation/sp_evaluation.sh

# Evaluate Guessing Numbers
bash script/evaluation/gn_evaluation.sh
```

### Custom Evaluation

You can modify the evaluation parameters by editing the scripts or running the Python commands directly with your preferred configurations.

<a id="citation"></a>
## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{
    zhou2025from,
    title={From Passive to Active Reasoning: Can Large Language Models Ask the Right Questions under Incomplete Information?},
    author={Zhanke Zhou and Xiao Feng and Zhaocheng Zhu and Jiangchao Yao and Sanmi Koyejo and Bo Han},
    booktitle={ICML},
    year={2025},
}
```

<a id="contributing"></a>
## 🤝 Contributing

We welcome contributions to AR-Bench! Please feel free to:

- Report bugs and issues
- Suggest new features or improvements
- Submit pull requests
- Share your evaluation results

<a id="contact"></a>
## 📞 Contact

For questions, technical support, or collaboration inquiries:

- **Email**: [cszkzhou@comp.hkbu.edu.hk](mailto:cszkzhou@comp.hkbu.edu.hk) \& [xiaofeng@comp.hkbu.edu.hk](mailto:xiaofeng@comp.hkbu.edu.hk)
- **Issues**: [GitHub Issues](https://github.com/tmlr-group/AR-Bench/issues)
