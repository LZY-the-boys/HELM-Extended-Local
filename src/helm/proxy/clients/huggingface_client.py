from copy import deepcopy
import torch
from dataclasses import asdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List

from helm.common.cache import Cache, CacheConfig
from helm.common.hierarchical_logger import htrack_block, hlog
from helm.common.request import EMBEDDING_UNAVAILABLE_REQUEST_RESULT, Request, RequestResult, Sequence, Token
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
    DecodeRequest,
    DecodeRequestResult,
    TokenizationToken,
)
import os
from typing import Optional, Union
from .client import Client, wrap_request_time, truncate_sequence, cleanup_tokens
from .huggingface_tokenizer import HuggingFaceTokenizers
from helm.proxy.clients.huggingface_model_registry import (
    get_huggingface_model_config,
    HuggingFaceModelConfig,
    HuggingFaceHubModelConfig,
    HuggingFaceLocalModelConfig,
)
from transformers import GPTQConfig,BitsAndBytesConfig,LlamaTokenizerFast,LlamaTokenizer
from threading import Lock
from accelerate import Accelerator, DistributedType
from peft import PeftModelForCausalLM
# Map of HELM model name to Hugging Face Hub model name where they differ.
_KNOWN_MODEL_ALIASES: Dict[str, str] = {
    "huggingface/gpt2": "gpt2",
    "huggingface/starcoder": "bigcode/starcoder",
}

def _get_dtype(
    dtype: Union[str, torch.dtype]
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible. Does not use an instantiated HF AutoConfig"""
    if isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceServer:

    def __init__(self, model_config: HuggingFaceModelConfig):
        if torch.cuda.is_available():
            hlog("CUDA is available, initializing with a GPU...")
            self.device: str = "cuda:0"
        else:
            self.device = "cpu"
        model_kwargs = {}
        # If the HuggingFace model is stored locally, it will have a path defined and we should load it from there.
        # Otherwise, download it from the HuggingFace hub by passing in its identifier.
        if isinstance(model_config, HuggingFaceLocalModelConfig):
            model_name = model_config.path
        elif isinstance(model_config, HuggingFaceHubModelConfig):
            model_name = model_config.model_id
            if model_config.revision:
                model_kwargs["revision"] = model_config.revision
        else:
            raise Exception(f"Unknown type of model_config: {model_config}")
        # support for multi gpu.
        gpus = torch.cuda.device_count()
        accelerator = Accelerator()
        model_kwargs = {}
        self.rank = 0
        self.world_size = 1
        data_parallel = False
        tensor_parallel = model_config.model_args.get('tensor_parallel',False)
        quantization_config = model_config.model_args.get('quantization_config',None)
        dtype = model_config.model_args.get('dtype', 'float16')
        peft = model_config.model_args.get('peft', None)

        if not (tensor_parallel or accelerator.num_processes > 1):
            # single gpu
            model_kwargs.update({'device_map': {'':0}})  
        elif gpus > 1:
            assert not (accelerator.num_processes > 1 and tensor_parallel), (
                    "Attempted to use both a HF Accelerate `device_map` and to launch via `accelerate launch`. If this is the case, please either remove `parallelize=True` from --model_args or launch outside of the Accelerate launcher."
                )
            # multi gpu
            if tensor_parallel:
                # tensor parallel
                model_kwargs.update({'device_map':'auto'})
            else:
                assert not gpus > accelerator.num_processes, (
                    "set CUDA_VISIBLE_DEVICES"
                )   
                data_parallel = True
                model_kwargs.update({'device_map': {'':accelerator.local_process_index}})

        if quantization_config:
            # NOTICE: model.config.quantization_config > input quantization_config
            if quantization_config['quant_method'] == 'gptq':
                quantization_config = GPTQConfig.from_dict(quantization_config)
            elif quantization_config['quant_method'] == 'bitsandbytes':
                quantization_config['bnb_4bit_compute_dtype'] = _get_dtype(dtype)
                if self.rank == 0:
                    print(f'>>> set bnb_4bit_compute_dtype to {_get_dtype(dtype)}')
                quantization_config = BitsAndBytesConfig.from_dict(quantization_config)  
            else:
                raise Exception('wrong quantization_config')      
            model_kwargs.update({'quantization_config': quantization_config})

        with htrack_block(f"Loading Hugging Face model for config {model_config}"):
            # WARNING this may fail if your GPU does not have enough memory
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                trust_remote_code=True, 
                low_cpu_mem_usage=True,
                **model_kwargs
            ).eval()
        with htrack_block(f"Loading Hugging Face tokenizer model for config {model_config}"):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                use_fast=True,
                **model_kwargs
            )

        if peft:
            self.model = PeftModelForCausalLM.from_pretrained(
                self.model, peft
            )
        if self.rank == 0:
            print(self.model)

        if data_parallel:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self.gpt2 = accelerator.prepare(self.gpt2)
            else:
                self.gpt2 = accelerator.prepare_model(self.gpt2, evaluation_mode=True)
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.accelerator = accelerator
            self.rank = self.accelerator.local_process_index
            self.world_size = self.accelerator.num_processes

    def serve_request(self, raw_request: Dict[str, Any]):
        encoded_input = self.tokenizer(raw_request["prompt"], return_tensors="pt", return_token_type_ids=False).to(
            self.device
        )
        raw_request = deepcopy(raw_request)
        raw_request["do_sample"] = True
        raw_request["return_dict_in_generate"] = True
        raw_request["output_scores"] = True
        top_k_per_token: int = raw_request["top_k_per_token"]
        del raw_request["top_k_per_token"]
        if len(raw_request["stop_sequences"]) > 0:

            stop_sequence_ids = self.tokenizer(
                raw_request["stop_sequences"], return_token_type_ids=False, add_special_tokens=False
            )
            if isinstance(self.tokenizer,LlamaTokenizerFast) or isinstance(self.tokenizer, LlamaTokenizer):
                # llama2 tokenize \n as [1, 29871, 13], 
                # 29871 = ''
                stop_sequence_ids.input_ids=[[13]]
            assert len(stop_sequence_ids.input_ids) == 1, "Total number of stop words should be 1."
            assert len(stop_sequence_ids.input_ids[0]) == 1, "Total number of tokens in each stop word should be 1."
            del raw_request["stop_sequences"]
            raw_request["eos_token_id"] = stop_sequence_ids.input_ids[0][0]

        # Strip out irrelevant parameters
        relevant_raw_request = {
            key: raw_request[key]
            for key in raw_request
            if key not in ["engine", "prompt", "echo_prompt", "stop_sequences"]
        }

        # Use HuggingFace's `generate` method.
        output = self.model.generate(**encoded_input, **relevant_raw_request)
        sequences = output.sequences
        scores = output.scores

        # Compute logprobs for each completed sequence.
        all_logprobs_of_chosen_tokens = []
        all_top_logprobs_dicts = []
        for completion_id in range(raw_request["num_return_sequences"]):
            logprobs_of_chosen_tokens = []
            top_logprobs_dicts = []
            for i in range(len(sequences[completion_id]) - len(encoded_input.input_ids[0])):
                logprobs = torch.nn.functional.log_softmax(scores[i][completion_id], dim=0)

                # Get top tokens in terms of log probability.
                topk_logprobs = torch.topk(logprobs, k=top_k_per_token)
                top_logprobs_dicts.append(
                    {
                        self.tokenizer.convert_ids_to_tokens(k.item()): v.item()
                        for (k, v) in zip(topk_logprobs.indices, topk_logprobs.values)
                    }
                )

                # Get log probability of chosen token.
                j = i + len(encoded_input.input_ids[0])
                logprobs_of_chosen_tokens.append(logprobs[sequences[completion_id][j]].item())
            all_logprobs_of_chosen_tokens.append(logprobs_of_chosen_tokens)
            all_top_logprobs_dicts.append(top_logprobs_dicts)

        # Remove prompt from the start of each sequence if echo_prompt is False.
        if not raw_request["echo_prompt"]:
            sequences = [sequence[len(encoded_input.input_ids[0]) :] for sequence in sequences]

        all_tokens = [[self.tokenizer.decode(token) for token in sequence_tokens] for sequence_tokens in sequences]
        all_decoded_text = self.tokenizer.batch_decode(sequences)

        completions = []
        for decoded_text, tokens, logprobs_of_chosen_tokens, top_logprobs_dicts in zip(
            all_decoded_text, all_tokens, all_logprobs_of_chosen_tokens, all_top_logprobs_dicts
        ):
            completions.append(
                {
                    "text": decoded_text,
                    "tokens": tokens,
                    "logprobs": logprobs_of_chosen_tokens,
                    "top_logprobs_dicts": top_logprobs_dicts,
                }
            )

        return {"completions": completions, "input_length": len(encoded_input.input_ids[0])}


_servers_lock: Lock = Lock()
_servers: Dict[str, HuggingFaceServer] = {}


def _get_singleton_server(model_config: HuggingFaceModelConfig) -> HuggingFaceServer:
    """Lookup or create a new HuggingFaceServer that will be shared among all threads.

    When --num-threads > 1, multiple threads will attempt to instantiate
    `HuggingFaceServer`s simultaneously. Since we have limited GPU memory, we want to
    just share a single copy of each model we are using. So, this function uses a lock
    to make sure that for each model, only one thread creates a HuggingFaceServer.
    The other threads can share that same server in the global _servers dictionary."""
    global _servers_lock
    global _servers
    with _servers_lock:
        if model_config.model_id not in _servers:
            _servers[model_config.model_id] = HuggingFaceServer(model_config)
    return _servers[model_config.model_id]


class HuggingFaceClient(Client):
    def __init__(self, cache_config: CacheConfig):
        self.cache = Cache(cache_config)
        self.model_server_instances: Dict[str, HuggingFaceServer] = {}

    def get_model_server_instance(self, model: str) -> HuggingFaceServer:
        model_config = get_huggingface_model_config(model)
        # Special-case some models in so that users don't have to enable them with --enable-huggingface-models
        if not model_config:
            if model in _KNOWN_MODEL_ALIASES:
                model_config = HuggingFaceHubModelConfig.from_string(_KNOWN_MODEL_ALIASES[model])
            else:
                model_config = HuggingFaceHubModelConfig.from_string(model)
        return _get_singleton_server(model_config)

    def make_request(self, request: Request) -> RequestResult:
        # Embedding not supported for this model
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        # Only a single stop sequence is supported as we can only pass in a single value for `eos_token_id`
        if len(request.stop_sequences) > 1:
            raise ValueError("More than one stop sequence is not supported.")

        raw_request = {
            "engine": request.model_engine,
            "prompt": request.prompt,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "num_return_sequences": request.num_completions,
            "max_new_tokens": request.max_tokens,
            "top_p": request.top_p,
            "echo_prompt": request.echo_prompt,
            "top_k_per_token": request.top_k_per_token,
            "stop_sequences": request.stop_sequences,
        }

        # Get cached model server instance if possible (to save on model and tokenizer
        # loading times).
        model_server_instance: HuggingFaceServer = self.get_model_server_instance(request.model)

        # try:

        def do_it():
            return model_server_instance.serve_request(raw_request)

        cache_key = Client.make_cache_key(raw_request, request)
        # 调用cache返回结果
        if os.environ.get('use_cache', False):
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        else:
            response = wrap_request_time(do_it)()
            cached = False
        # except Exception as e:  # Do something if error is encountered.
        #     error: str = f"HuggingFace error: {e}"
        #     return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions = []
        for raw_completion in response["completions"]:
            sequence_logprob: float = 0
            tokens: List[Token] = []

            if request.echo_prompt:
                # Add prompt to list of generated tokens.
                generated_tokens = raw_completion["tokens"][response["input_length"] :]
                for token_text in raw_completion["tokens"][: response["input_length"]]:
                    tokens.append(Token(text=token_text, logprob=0.0, top_logprobs={}))
            else:
                generated_tokens = raw_completion["tokens"]

            # Compute logprob for the entire sequence.
            for token_text, logprob, top_logprobs_dict in zip(
                generated_tokens, raw_completion["logprobs"], raw_completion["top_logprobs_dicts"]
            ):
                tokens.append(Token(text=token_text, logprob=logprob, top_logprobs=top_logprobs_dict))
                sequence_logprob += logprob

            completion = Sequence(text=raw_completion["text"], logprob=sequence_logprob, tokens=tokens)
            completion = truncate_sequence(completion, request)
            completions.append(completion)

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def tokenize(self, request: TokenizationRequest) -> TokenizationRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                if request.encode:
                    if request.truncation:
                        tokens = tokenizer.encode(
                            request.text,
                            truncation=request.truncation,
                            max_length=request.max_length,
                            add_special_tokens=False,
                        )
                    else:
                        tokens = tokenizer.encode(request.text, add_special_tokens=False)
                else:
                    if "gpt" in request.tokenizer or request.tokenizer in [
                        "bigscience/bloom",
                        "Writer/palmyra-base",
                        "facebook/opt-66b",
                    ]:
                        # These models already handle the "▁" character correctly with the
                        # convert_tokens_to_string method. We prefer to use this method instead
                        # of the hacky cleanup_tokens method below as it might handle cases
                        # we haven't thought of in cleanup_tokens.
                        tokens = [
                            tokenizer.convert_tokens_to_string([token]) for token in tokenizer.tokenize(request.text)
                        ]
                    else:
                        # Tokenizes the text and returns the tokens as a list of strings,
                        # not a list of token objects (otherwise "Hello world" would be"
                        # ["Hello", "▁world"] and not ["Hello", " world"])
                        # We could do this with a simple replace like this:
                        # tokens = [tokenizer.convert_tokens_to_string([i]) for i in tokenizer.tokenize(request.text)]
                        # But this replaces all the "▁" characters by "", which is not what we want.
                        # This would be problematic as tokenize(" Hello", encode=False) would return ["Hello"]
                        # Just like tokenize("Hello", encode=False) would return ["Hello"].
                        tokens = tokenizer.tokenize(request.text)
                        tokens = cleanup_tokens(tokens, request.tokenizer)
                return {"tokens": tokens}

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return TokenizationRequestResult(success=False, cached=False, error=error, text="", tokens=[])

        return TokenizationRequestResult(
            success=True,
            cached=cached,
            text=request.text,
            tokens=[TokenizationToken(value) for value in result["tokens"]],
            request_time=result["request_time"],
        )

    def decode(self, request: DecodeRequest) -> DecodeRequestResult:
        tokenizer = HuggingFaceTokenizers.get_tokenizer(request.tokenizer)
        cache_key = asdict(request)

        try:

            def do_it():
                return {
                    "text": tokenizer.decode(
                        request.tokens, clean_up_tokenization_spaces=request.clean_up_tokenization_spaces
                    )
                }

            result, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except Exception as e:
            error: str = f"HuggingFace error: {e}"
            return DecodeRequestResult(success=False, cached=False, error=error, text="")

        return DecodeRequestResult(
            success=True, cached=cached, text=result["text"], request_time=result["request_time"]
        )
