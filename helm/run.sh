
# Run benchmark
# --suite specifies a subdirectory under the output directory in which all the output will be placed.
# --max-eval-instances limits evaluation to only the first N inputs (i.e. instances) from the benchmark.
# https://github.com/stanford-crfm/helm/issues/1794
# https://github.com/stanford-crfm/helm/pull/1505/files
# => benchmarking_output/runs/v1

source activate crfm-helm

# For local models
# export model_args=
# python -m helm.benchmark.run \
# --run-specs "truthful_qa:task=mc_single,model=huggingface/llama-2-13b-hf" \
# --enable-local-huggingface-models /model/Llama-2-13b-hf \
# --model-args="{\"tensor_parallel\":true,\"dtype\":\"bfloat16\"}" \
# --suite nips \
# --output-path outs/ \
# --max-eval-instances 10


python -m helm.benchmark.run \
--conf-paths run.conf \
--enable-local-huggingface-models /model/Llama-2-13B-GPTQ \
--suite nips-gpq-lora \
--max-eval-instances 4 \
--model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\",\"peft\":\"/home/lzy/nips2023_llm_challenge/sft/outputs/sft_gptq-b2-a4-c1.0-v1-lr4e5-flashattn-samplepacking\"}" \
--num-threads 1

python -m helm.benchmark.run \
--conf-paths run.conf \
--enable-local-huggingface-models /model/Llama-2-13B-GPTQ \
--suite nips \
--max-eval-instances 4 \
--name 13b-gptq-v1 \
--model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\"}" \
--num-threads 1
# For hub model
# helm-run \
#     --run-specs boolq:model=chainyo/alpaca-lora-7b@main  \
#     --enable-huggingface-models chainyo/alpaca-lora-7b@main \
#     --local \
#     --suite v1 \
#     --max-eval-instances 10

# Summarize benchmark results
python -m helm.benchmark.presentation.summarize --suite nips
# Start a web server to display benchmark results
python -m helm.benchmark.server -p 8001

# helm-run = helm.benchmark.run:main
# helm-summarize = helm.benchmark.presentation.summarize:main
# helm-server = helm.benchmark.server:main
# helm-create-plots = helm.benchmark.presentation.create_plots:main
# crfm-proxy-server = helm.proxy.server:main
# crfm-proxy-cli = helm.proxy.cli:main


# 0. 限制太多，model无法直接加载，无法直接送入model_args
# 1. 是按model=huggingface/Llama这种区分的，peft很麻烦，而且有cache会直接跳过