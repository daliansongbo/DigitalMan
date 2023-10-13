import gc
import json
import os
import re
import time
import traceback
from queue import Queue
from threading import Thread
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
import torch
import transformers
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,AutoTokenizer, LlamaTokenizer)
import configs.config as config
from peft import PeftModel
import numpy as np
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
 
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


class _StopEverythingStoppingCriteria(transformers.StoppingCriteria):
    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        return False

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False



class ChatLLM:
    def __init__(self, params: dict = None):
        """
        模型初始化
        :param params:
        """
        self.lora_names=[]
        self.model_name = config.LLM_MODEL
        self.model_param = config.LLM_MODEL_DICT[self.model_name]
        print("self.model_param",self.model_param)
        self.local_model_path =self.model_param['local_model_path']  
        self.provides =self.model_param['provides']
        self.mdoel_param =self.model_param['mdoel_param'] 
        
        self.device = config.LLM_DEVICE
        self.default_model_param = config.LLM_MODEL_PARAMS
        self.default_generate_params = config.LLM_MODEL_GENERATION_PARAMS
        self.model_half =  config.LLM_MODEL_HALF
        self.model_quantized =  config.LLM_MODEL_QUANTIZE
        self.is_llamacpp =  config.LLM_LLAMACPP
        self.is_chatgmlcpp  =  config.LLM_CHATGLMCPP
        self.lora_dir = config.LORA_DIR
        self.is_lora = config.ISLORA
        self.lora =  config.LORALIST
        self.isstream =  config.ISSTREAM
    def load_model(self):
        """
        加载自定义位置的model
        :return:
        """
        if self.local_model_path :
            self.local_model_path = f'{self.local_model_path}/{self.model_name}'
            self.local_model_path = re.sub("\s", "", self.local_model_path)
            checkpoint = Path(f'{self.local_model_path}')
        
        print(f"Loading {checkpoint}...",checkpoint)
        if not os.path.exists(checkpoint):
            os.mkdir(checkpoint)
            
        if not self.provides:
            print("check your config 'provides' ")
        if not self.mdoel_param:
            self.mdoel_param = self.default_model_param
        print(self.mdoel_param)    
        if self.provides.lower() == "gptq":
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path, use_fast=True)
            self.model = AutoGPTQForCausalLM.from_quantized(self.local_model_path, **self.mdoel_param )

        elif self.provides.lower() == "auto" or 'chatglm' in self.model_name.lower() or "chatyuan" in self.model_name.lower():        
            self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            self.model = AutoModel.from_pretrained(self.local_model_path,**self.mdoel_param)
        if self.is_lora:
            self.load_lora_model([self.lora])
        elif self.provides.lower() == "llama":
            if self.is_llamacpp:    
                try:
                    from llama_cpp import Llama
                except ImportError as exc:
                    raise ValueError(
                        "Could not import depend python package "
                        "Please install it with `pip install llama-cpp-python`."
                    ) from exc

                model_file = list(self.local_model_path.glob('*.bin'))[0]
                print(f"llama.cpp weights detected: {model_file}\n")

                self.model = Llama(model_path=model_file._str)
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                self.model = AutoModelForCausalLM.from_pretrained(self.local_model_path,**self.mdoel_param)
        if self.model_half:
            self.model = self.model.half()
        if self.model_quantized :
            self.model = self.model.quantize(self.model_quantized)
        if not self.is_llamacpp and not self.is_chatgmlcpp:
            self.model = self.model.eval()
            
        config.LLM_ChatBot_Model = self.model
        config.LLM_ChatBot_Tokenizer = self.tokenizer
        return self.model,self.tokenizer
    def load_lora_model(self, lora_names):
        # 目前加载的lora
        prior_set = set(self.lora_names)
        # 需要加载的
        added_set = set(lora_names) - prior_set
        # 删除的lora
        removed_set = prior_set - set(lora_names)
        self.lora_names = list(lora_names)

        # Nothing to do = skip.
        if len(added_set) == 0 and len(removed_set) == 0:
            return

        # Only adding, and already peft? Do it the easy way.
        if len(removed_set) == 0 and len(prior_set) > 0:
            for lora in added_set:
                self.model.load_adapter(Path(f"{self.lora_dir}/{lora}"), lora)
            return

        # If removing anything, disable all and re-add.
        if len(removed_set) > 0:
            self.model.disable_adapter()

        if len(lora_names) > 0:
            params = {}
            # if self.device.lower() != "cpu":
                # params['dtype'] = self.model.dtype
                # if hasattr(self.model, "hf_device_map"):
                #     params['device_map'] = {"base_model.model." + k: v for k, v in self.model.hf_device_map.items()}
                # elif self.load_in_8bit:
                #     params['device_map'] = {'': 0}
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model = PeftModel.from_pretrained(self.model, Path(f"{self.lora_dir}/{lora_names[0]}"))
            for lora in lora_names[1:]:
                self.model.load_adapter(Path(f"{self.lora_dir}/{lora}"), lora)
            # if not self.load_in_8bit and self.llm_device.lower() != "cpu":

            #     if not hasattr(self.model, "hf_device_map"):
            #         if torch.has_mps:
            #             device = torch.device('mps')
            #             self.model = self.model.to(device)
            #         else:
            #             self.model = self.model.cuda()
    def reload_model(self):
        self.model, self.tokenizer = self._load_model()
        
        if self.is_lora:
            self.load_lora_model([self.lora])
        if not self.is_llamacpp and not self.is_chatgmlcpp:
            self.model = self.model.eval()
    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        if self.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel']:
            input_ids = self.tokenizer.encode(str(prompt))
            input_ids = np.array(input_ids).reshape(1, len(input_ids))
            return input_ids
        else:
            input_ids = self.tokenizer.encode(str(prompt), return_tensors='pt', add_special_tokens=add_special_tokens)
            # This is a hack for making replies more creative.
            if not add_bos_token and input_ids[0][0] == self.tokenizer.bos_token_id:
                input_ids = input_ids[:, 1:]
        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]
        if self.model.__class__.__name__ in ['LlamaCppModel', 'RWKVModel', 'ExllamaModel']:
            return input_ids
        return input_ids.cuda()
    def get_reply_from_output_ids(self,output_ids, input_ids, original_question, state, is_chat=False):

        new_tokens = len(output_ids) - len(input_ids[0])
        reply = self.tokenizer.decode(output_ids[-new_tokens:], skip_special_tokens = True)
        # Prevent LlamaTokenizer from skipping a space
        if type(self.tokenizer) in [transformers.LlamaTokenizer, transformers.LlamaTokenizerFast] and len(output_ids) > 0:
            if self.tokenizer.convert_ids_to_tokens(int(output_ids[-new_tokens])).startswith('▁'):
                reply = ' ' + reply

        return reply
    def chatbot_generate(self,question, seed=-1,  stopping_strings=None, is_chat=False):
        generate_params = {}
        for k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'typical_p', 'repetition_penalty', 'encoder_repetition_penalty', 'top_k', 'min_length', 'no_repeat_ngram_size', 'num_beams', 'penalty_alpha', 'length_penalty', 'early_stopping']:
            generate_params[k] = self.default_generate_params[k]
        for k in ['epsilon_cutoff', 'eta_cutoff']:
            if self.default_generate_params[k] > 0:
                generate_params[k] = self.default_generate_params[k] * 1e-4
        if self.default_generate_params['ban_eos_token']:
            generate_params['suppress_tokens'] = [self.tokenizer.eos_token_id]
        if not config.CACHE:
            generate_params.update({'use_cache': False})
        if config.DEEPSPEED:
            generate_params.update({'synced_gpus': True})
        # Encode the input
        input_ids = self.encode(question)
        output = input_ids[0]

        # Add the encoded tokens to generate_params
        inputs_embeds = None
        original_input_ids = input_ids
        generate_params.update({'inputs': input_ids})
        if inputs_embeds is not None:
            generate_params.update({'inputs_embeds': inputs_embeds})

        # Stopping criteria / eos token
        eos_token_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id is not None else []
        generate_params['eos_token_id'] = eos_token_ids
        generate_params['stopping_criteria'] = transformers.StoppingCriteriaList()
        generate_params['stopping_criteria'].append(_StopEverythingStoppingCriteria())
        #print("generate_params:\t",generate_params)
        t0 = time.time()
        try:
            # Generate the entire reply at once.
            if self.isstream:
                with torch.no_grad():
                    output = self.model.generate(**generate_params)[0]
                    if self.device:
                        output = output.cuda(self.device)
                        new_tokens = len(output) - len(input_ids[0])
                yield  self.tokenizer.decode(output[-new_tokens:], skip_special_tokens = True)

            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator.
            else:

                def generate_with_callback(callback=None, *args, **kwargs):
                    kwargs['stopping_criteria'].append(Stream(callback_func=callback))
                    clear_torch_cache()
                    with torch.no_grad():
                        self.model.generate(**kwargs)

                def generate_with_streaming(**kwargs):
                    return Iteratorize(generate_with_callback, [], kwargs, callback=None)

                with generate_with_streaming(**generate_params) as generator:
                    for output in generator:
                        new_tokens = len(output) - len(input_ids[0])
                        yield self.tokenizer.decode(output[-new_tokens:], skip_special_tokens = True)
                        if output[-1] in eos_token_ids:
                            break

        except Exception:
            traceback.print_exc()
        finally:
            t1 = time.time()
            original_tokens = len(original_input_ids[0])
            new_tokens = len(output) 
            print(f'Output generated in {(t1-t0):.2f} seconds ({new_tokens/(t1-t0):.2f} tokens/s, {new_tokens} tokens, context {original_tokens}, seed {seed})')
            return

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()



class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).

    Adapted from: https://stackoverflow.com/a/9969000
    """

    def __init__(self, func, args=None, kwargs=None, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.args = args or []
        self.kwargs = kwargs or {}
        self.stop_now = False
        def _callback(val):
            if self.stop_now or config.STOPEVERYTHING:
                raise ValueError
            self.q.put(val)
        def gentask():
            try:
                ret = self.mfunc(callback=_callback, *args, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            clear_torch_cache()
            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()
    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        clear_torch_cache()

    
    
    
