from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, GenerationConfig, ModelOutput
from transformers.generation.logits_process import LogitsProcessor

from langchain import LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import torch
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
    
    
class LLMModel(LLM):
    max_token: int = 4096
    temperature: float = 0.8
    top_p = 0.9
    tokenizer: object = None
    model: object = None
    history_len: int = 1024
    device: str = 'cuda:5'
    do_sample: bool =True
    
    def __init__(self):
        super().__init__()
        
    @property
    def _llm_type(self) -> str:
        return "AutoGPTQ"
    def load_model(self, model_name_or_path,device):
        use_triton = False
        # print(self.max_token)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,use_safetensors=True,
                                                        trust_remote_code=False,device=self.device,
                                                        use_triton=use_triton,quantize_config=None)

    
    def process_response(self, response):
        response = response.strip()
        response = response.replace("[[训练时间]]", "2023年")
        return response

    def chats(self,query, **kwargs):
        gen_kwargs = {"max_length": self.max_token, "do_sample": self.do_sample, "max_new_tokens":512 ,"top_p": self.top_p,"temperature": self.temperature, **kwargs}
        input_ids = self.tokenizer([query], return_tensors="pt").input_ids.cuda(self.device) 
        outputs = self.model.generate(inputs=input_ids,**gen_kwargs) 
        output = self.tokenizer.decode(outputs.tolist()[0][len(input_ids[0]):])
        response = self.process_response(output)
        # history = history + [(query, response)]
        return response
    
    def _call(self,query:str,stop=None):
        print('_call query',query)
        response = self.chats(query)
        print('responce',response,'-------')
        return response