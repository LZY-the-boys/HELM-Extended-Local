
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
3. fix name problems: set display model name as `model-args['name_ext']` 


## Pre

```bash
conda create -n crfm-helm python=3.8
conda activate crfm-helm
pip install -r requirements.txt
```

## Usage

run `tensor_parallel` `gptq` `peft` freely in the local!
```bash
# 1. use default model name `path_to_your_Llama-2-13b-hf`.split('/')[-1]
python -m helm.benchmark.run \
--run-specs "truthful_qa:task=mc_single,model=huggingface/llama-2-13b-hf" \
--enable-local-huggingface-models path_to_your_Llama-2-13b-hf \
--model-args="{\"tensor_parallel\":true,\"dtype\":\"bfloat16\"}" \
--suite v1 \
--output-path outs/ \
--max-eval-instances 10
# or. set model name in run.conf files, must be `path_to_your_Llama-2-13b-hf`.split('/')[-1]
python -m helm.benchmark.run \
--conf-paths run.conf \
--enable-local-huggingface-models path_to_your_Llama-2-13B-GPTQ \
--suite v1 \
--max-eval-instances 4 \
--name 13b-gptq-v1 \
--model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\", \"peft\": path_to_your_peft_adapter }" \
--num-threads 1
# or. control your model name freely in `model-args['name_ext']`
python -m helm.benchmark.run \
--conf-paths run_test.conf \
--enable-local-huggingface-models /model/Llama-2-13B-GPTQ \
--suite v1 \
--max-eval-instances 4 \
--model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\",\"peft\":\"path_to_sft_adapter\",\"name_ext\":\"llama2-13b-sft_peft\"}" \
--num-threads 1

# 2. Summarize benchmark in benchmarking_output/runs/v1 and Start a web server to display benchmark results
python -m helm.benchmark.presentation.summarize --suite v1
python -m helm.benchmark.server
```

MMLU result (max-eval-instances=1000, 57 test sets)

| Model/adapter                       |                 EM |        ECE (10-bin) |     EM (Robustness) |      EM (Fairness) |             # eval | # train | truncated |   # prompt tokens | # output tokens | # trials |
|-------------------------------------|-------------------:|--------------------:|--------------------:|-------------------:|-------------------:|--------:|----------:|------------------:|----------------:|---------:|
| huggingface/Llama-2-70B-hf-4bit     | 0.6860255837649295 | 0.38117816485499945 |  0.6456410872600922 | 0.6552815245194281 | 235.28070175438597 |     5.0 |           | 706.7065997805888 |             1.0 |      1.0 |
| huggingface/Llama-2-70B-GPTQ        |  0.680822607304811 | 0.38476204408972614 |  0.6383747379915452 | 0.6442603542776053 | 235.28070175438597 |     5.0 |           | 706.7065997805888 |             1.0 |      1.0 |
| huggingface/Llama-2-13b-hf          |  0.548905959955963 |  0.4969764635935246 |   0.502294480135353 | 0.5085932646205037 | 235.28070175438597 |     5.0 |           | 706.7065997805888 |             1.0 |      1.0 |
| huggingface/Llama-2-13B-GPTQ        |  0.543006579002395 |  0.4701847938389892 | 0.49442028682068406 | 0.5045743983655909 | 235.28070175438597 |     5.0 |           | 706.7065997805888 |             1.0 |      1.0 |

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