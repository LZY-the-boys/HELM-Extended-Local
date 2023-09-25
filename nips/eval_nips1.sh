set -e pipefail
source activate crfm-helm
cd helm
# CUDA_VISIBLE_DEVICES=0 python -m helm.benchmark.run \
#     --conf-paths run_specs.conf \
#     --enable-local-huggingface-models /model/Llama-2-13B-GPTQ \
#     --suite nips \
#     --max-eval-instances 200 \
#     --model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\",\"name_ext\":\"Llama-2-13B-GPTQ\"}" \
#     --num-threads 1 &

# CUDA_VISIBLE_DEVICES=1 python -m helm.benchmark.run \
#     --conf-paths run_specs.conf \
#     --enable-local-huggingface-models /model/Llama-2-13b-hf \
#     --suite nips \
#     --max-eval-instances 200 \
#     --model-args="{\"quantization_config\":{\"load_in_4bit\":true,\"bnb_4bit_use_double_quant\":true,\"bnb_4bit_quant_type\":\"nf4\",\"llm_int8_has_fp16_weight\":true,\"quant_method\":\"bitsandbytes\"},\"dtype\":\"float16\",\"name_ext\":\"Llama-2-13b-hf\"}" \
#     --num-threads 1 

for peft_name in sft_gptq-b2-a64-c1.0-lr4e-5-flashattn-samplepacking sft_gptq-b2-a64-c1.0-lr4e-5-v2_4090-flashattn-samplepacking ; do
    CUDA_VISIBLE_DEVICES=0 python -m helm.benchmark.run \
    --conf-paths run_test.conf \
    --enable-local-huggingface-models /model/Llama-2-13B-GPTQ \
    --suite nips \
    --max-eval-instances 200 \
    --model-args="{\"quantization_config\":{\"bits\": 4, \"disable_exllama\":false,\"quant_method\":\"gptq\",\"use_cuda_fp16\":false},\"dtype\":\"float16\",\"peft\":\"/home/lzy/nips2023_llm_challenge/sft/outputs/$peft_name\",\"name_ext\":\"$peft_name\"}" \
    --num-threads 1 
done



# Summarize benchmark results: Symlinking benchmark_output/runs/v2 to latest.
# python -m helm.benchmark.presentation.summarize --suite v2
# Start a web server to display benchmark results
# python -m helm.benchmark.server -p 8001
cd ..