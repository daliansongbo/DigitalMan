import torch.cuda
import torch.backends
import os
import logging
import uuid

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)


USER_NAME = "You"
CHAR_NAME = "Assistant"
SYSTEM_PROMPT = "system_prompt"

#LLM模型
LLM_DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
LLM_MODEL = 'chatglm-6b'

#EMBEDDING模型
EMBEDDING_MODEL = "text2vec"
# Embedding running device
EMBEDDING_DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#语言类型
LANGUAGE_TYPE = "EN"

#人设默认路径
CHARACTER_DIR = "./characters/"
DEFAULT_CHARACTER_DIR = "./characters/default_charaters/"
#是否人设和用户作为独立主键，目的：不同用户创建相同角色
IS_USER_CHARACTER = True

#在线搜索参数
useSearch = False
SERPER_API_KEY = None
SERPAPI_API_KEY = None
GOOGLE_API_KEY = None 
GOOGLE_CSE_ID = None

#是否短期记忆
IS_SHORT_MEMORY =True
#短期记忆是否在本地保存
SHORT_MEMORY_IS_LOCAL = True
SHORT_MEMORY_LOCAL_DIR = "./data/short_history/"
SHORT_MEMORY_TOP_K = 30
#短期记忆SQL
SQL_LINK = "********"
SQL_USER = 'user_name'
PORT = '3306'
SQL_PASSWD = '123456'
SQL_DB = 'chatbot'
DB_URI = 'mysql+pymysql://{}:{}@{}:{}/{}'.format(SQL_USER, SQL_PASSWD, SQL_LINK, PORT, SQL_DB)



#是否使用长期记忆
IS_LONG_MEMORY = True
#长期记忆本地路径
##可以在本地采用文件夹的方式来进行分区，采用用户名称+人设名称的方式；
LONG_MEMORY_CHROMA_DIR = "./data/long_memory/"
#最终选择条数
LONG_SEACH_MEMORY_LEN = 5
# 检索时返回的匹配内容条数
VECTOR_SEARCH_TOP_K = 30 
#对话保存是否提取摘要
LONG_MEMORY_SUMMARY = False
# 知识检索内容Score, 数值范围约为0-1100，如果为0，则不生效，建议设置为500左右，经测试设置为小于500时，匹配结果更精准
SCORE_THRESHOLD = 700

#向量检索类型
LONG_MEMORY_TYPE = "faiss" #milvus、faiss、chroma
#milvus向量配置
milvus_host = "127.0.0.1"
milvus_port = "19530"
milvus_user = "user"
milvus_password= "123456"
milvus_db_name="chatbot_DB"
milvus_collection_name = "chatbot_vec"


#提取摘要的prompt
ENGLISH_SUMMARY_PROMPT = '''
    <s>[INST] <<SYS>>          
    <</SYS>>
    '''
CHINESE_SUMMARY_PROMPT = '''
    <s>[INST] <<SYS>>          
    <</SYS>>
    '''


#重要性评分
ENGLISH_IMPORTANT_SCORE_PROMPT = '''
'''
CHINESE_IMPORTANT_SCORE_PROMPT = '''
'''


# LLM 模型集合
LLM_MODEL_DICT = {
    "MythoMax-L2-13B-GPTQ": {
        "name": "MythoMax-L2-13B-GPTQ",
        "pretrained_model_name": "TheBloke/MythoMax-L2-13B-GPTQ",
        "local_model_path": "./checkpoint",
        "provides": "gptq",
        "mdoel_param" :None
    },
    "chatglm-6b": {
        "name": "chatglm-6b",
        "pretrained_model_name": "THUDM/chatglm-6b",
        "local_model_path": "./checkpoint",
        "provides": "AutoModel",
        "mdoel_param" :None
    },
    "chatglm2-6b": {
        "name": "chatglm2-6b",
        "pretrained_model_name": "THUDM/chatglm2-6b",
        "local_model_path": "./checkpoint",
        "provides": "AutoModel",
        "mdoel_param" :None
    },
    "chatyuan": {
        "name": "chatyuan",
        "pretrained_model_name": "ClueAI/ChatYuan-large-v2",
        "local_model_path": "./checkpoint",
        "provides": "AutoModel",
        "mdoel_param" :None
    },
    "vicuna-13b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": "./checkpoint",
        "provides": "LLama",
        "mdoel_param" :None
    },
    "vicuna-7b-hf": {
        "name": "vicuna-13b-hf",
        "pretrained_model_name": "vicuna-13b-hf",
        "local_model_path": "./checkpoint",
        "provides": "LLama",
        "mdoel_param" :None
    },
    # 如果报出：raise NewConnectionError(
    # urllib3.exceptions.NewConnectionError: <urllib3.connection.HTTPSConnection object at 0x000001FE4BDB85E0>:
    # Failed to establish a new connection: [WinError 10060]
    # 则是因为内地和香港的IP都被OPENAI封了，需要切换为日本、新加坡等地
    "openai-chatgpt-3.5": {
        "name": "gpt-3.5-turbo",
        "pretrained_model_name": "gpt-3.5-turbo",
        "provides": "FastChatOpenAILLMChain",
        "local_model_path": "./checkpoint",
        "api_base_url": "https://api.openai.com/v1",
        "api_key": "",
        "mdoel_param" :None
    },

}

LLM_MODEL_HALF = None
LLM_MODEL_QUANTIZE = None 
LLM_LLAMACPP = False
LLM_CHATGLMCPP = False
LORA_DIR = "./checkpoint/lora_modle/"
ISLORA = False
LORALIST = []
#模型加载相关参数
LLM_MODEL_PARAMS = {
    'model_basename': 'model', 'device': LLM_DEVICE, 'use_triton': False, 'inject_fused_attention': True, 
    'inject_fused_mlp': True, 'use_safetensors': True, 'trust_remote_code': False, "torch_dtype":torch.float16,
    'max_memory': None, 'quantize_config': None, 'use_cuda_fp16': True, 'disable_exllama': False
}
#文本生成相关参数
CACHE = False
DEEPSPEED = False
ISSTREAM = True
LLM_MODEL_GENERATION_PARAMS = {'max_new_tokens': 200, 'do_sample': True, 'temperature': 1.3, 'top_p': 0.14, 
    'typical_p': 1, 'repetition_penalty': 1.5, 'encoder_repetition_penalty': 1, 'top_k': 49, 
    'min_length': 0, 'no_repeat_ngram_size': 0, 'num_beams': 1, 'penalty_alpha': 0, 'length_penalty': 1,
    'early_stopping': False
}
STOPEVERYTHING = False
#Embedding 模型集合
EMBEDDING_MODEL_DICT = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec-base-multilingual": "shibing624/text2vec-base-multilingual",
    "text2vec-base-chinese-sentence": "shibing624/text2vec-base-chinese-sentence",
    "text2vec-base-chinese-paraphrase": "shibing624/text2vec-base-chinese-paraphrase",
    "m3e-small": "moka-ai/m3e-small",
    "m3e-base": "moka-ai/m3e-base",
}

