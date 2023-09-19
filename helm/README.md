
# HELM-Extended Local 

<img src="https://github.com/stanford-crfm/helm/raw/main/src/helm/benchmark/static/images/helm-logo.png" alt=""  width="800"/>

HELM is one of the most popular evaluation framework for large language models.
However, it's annoying when we want to evaluate local model directly.
Because as is in the [here](https://github.com/stanford-crfm/helm/issues/1794), the suggesting huggingface-related workflow is :
- make a HTTP client and set `model=neurips/local` option, which is very unfriendly with slow local internet.
- huggingface hub and set `model=huggingface/xx` option, which needs extra upload process. 

based on [this PR](https://github.com/stanford-crfm/helm/pull/1505) which add local huggingface model support, I further develop it to:

1. add support for gptq, bitsandbytes, peft, tensorparallel
2. hack the [llama tokenization bug on `\n`](https://github.com/stanford-crfm/helm/issues/1782)

TODO:

1. support for multi-gpu data parallel
2. simplify the evaluation process to get excel. 


## Pre

```bash
conda create -n crfm-helm python=3.8
pip install -r requirements.txt
```

## Usage

run `tensor_parallel` `gptq` `peft` freely in the local!
```bash
# 1. compute eval output in benchmarking_output/runs/v1
python -m helm.benchmark.run \
--run-specs "truthful_qa:task=mc_single,model=huggingface/llama-2-13b-hf" \
--enable-local-huggingface-models path_to_your_Llama-2-13b-hf \
--model-args="{\"tensor_parallel\":true,\"dtype\":\"bfloat16\"}" \
--suite v1 \
--output-path outs/ \
--max-eval-instances 10

python -m helm.benchmark.run \
--conf-paths run.conf \
--enable-local-huggingface-models path_to_your_Llama-2-13B-GPTQ \
--suite v1 \
--max-eval-instances 4 \
--name 13b-gptq-v1 \
--model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\", \"peft\": path_to_your_peft_adapter }" \
--num-threads 1

# 2. Summarize benchmark in benchmarking_output/runs/v1 and Start a web server to display benchmark results
python -m helm.benchmark.presentation.summarize --suite v1
python -m helm.benchmark.server
```


# Holistic Evaluation of Language Models

Welcome! The **`crfm-helm`** Python package contains code used in the **Holistic Evaluation of Language Models** project ([paper](https://arxiv.org/abs/2211.09110), [website](https://crfm.stanford.edu/helm/latest/)) by [Stanford CRFM](https://crfm.stanford.edu/). This package includes the following features:

- Collection of datasets in a standard format (e.g., NaturalQuestions)
- Collection of models accessible via a unified API (e.g., GPT-3, MT-NLG, OPT, BLOOM)
- Collection of metrics beyond accuracy (efficiency, bias, toxicity, etc.)
- Collection of perturbations for evaluating robustness and fairness (e.g., typos, dialect)
- Modular framework for constructing prompts from datasets
- Proxy server for managing accounts and providing unified interface to access models
<!--intro-end-->

To get started, refer to [the documentation on Read the Docs](https://crfm-helm.readthedocs.io/) for how to install and run the package.