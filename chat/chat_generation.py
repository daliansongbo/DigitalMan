import configs.config  as config 
from model.loader import ChatLLM
from modules.embedding import Embedding
from modules.character import Character,CharacterStoge
from modules.character_template_english import EnglishCharacterTemplate
from memory.memory_storage import MemoryStorageDriver

class Chatbot():
    def __init__(self):
        super().__init__()
        #初始化模型
        self.llm = ChatLLM()
        self.model,self.tokenizer = self.llm.load_model()
        config.LLM_ChatBot_Model = self.model
        config.LLM_ChatBot_Tokenizer = self.tokenizer
        #加载embedding模型
        self.embeddings = Embedding()
        config.Embedding = self.embeddings
        #人设模块初始化
        #需要设计一个人设模块的字典/数据库，加载所有人设信息，每次直接检索
        self.Character_Dict = CharacterStoge()
        self.default_charater = {}
        for k in self.Character_Dict.user_charater_dict.keys() :
            if "default_charater" in k:
                keys = str(k).replace("default_charaters_","")
                self.default_charater[keys] = self.Character_Dict.user_charater_dict[k]

        # 初始化记忆模块
        self.memory =  MemoryStorageDriver()
        # 系统prompt
        self.prompt =  EnglishCharacterTemplate()
        
    def call(self,query,user_name,charater_name):
        stopping_strings = []
        user_character_name = user_name+'_'+charater_name
        print(self.default_charater.keys())
        if user_character_name in self.Character_Dict.user_charater_dict.keys():
            Character_info = self.Character_Dict.user_charater_dict[user_character_name]
        elif charater_name in self.default_charater.keys():
            Character_info = self.default_charater[charater_name]

        prompt_character = self.prompt.format(Character_info)
            
        # 检索关联的短期记忆和长期记忆
        short_history = self.memory.search_short_memory(query_text=query, user_name=user_name, charater_name=charater_name)
        long_history = self.memory.search_long_memory(query_text=query, user_name=user_name, charater_name=charater_name)
        config.USER_NAME = user_name
        config.CHAR_NAME = charater_name
        stopping_strings += [f"\n{config.USER_NAME}:",f"\n{config.CHAR_NAME}:",f"\n{config.SYSTEM_PROMPT}:"]

        # 调用大语言模型流式生成对话       
        prompt = prompt_character.format(user =config.USER_NAME,char = config.CHAR_NAME, input=query, short_history=short_history, long_history=long_history)
        response = self.llm.chatbot_generate(question=prompt,stopping_strings = stopping_strings)
        print("response",response)
        self.memory.save(you_name = user_name,query_text = query,role_name = charater_name,answer_text = response)
        # except Exception as e:
        return response
query = "hi"
user_name = "username"
charater_name = "Aubrey Drake Graham"
bot =  Chatbot()
bot.call(query,user_name,charater_name)
