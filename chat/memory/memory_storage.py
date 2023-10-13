import json
from typing import Tuple

import configs.config as config
from typing import List
from memory.chroma_memory.chroma_memory import ChromaStorage
from memory.faiss_memory.faiss_memory import FaissStorage
from memory.local_memory.local_memory import LocalStorage
from memory.milvus_memory.milvus_storage_impl import MilvusStorage
from memory.sql_memory.sql_memory  import  SQLStorage
from utils.snowflake_utils import SnowFlake
from memory.memory_side_info import MemoryImportance,MemorySummary
from typing import Any, Dict, List


class MemoryStorageDriver():
    def __init__(self) -> None:
        self.snow_flake = SnowFlake(data_center_id=5, worker_id=5)
        if config.IS_SHORT_MEMORY:
            if config.SHORT_MEMORY_IS_LOCAL:
                self.short_memory_storage = LocalStorage()
            else:
                self.short_memory_storage = SQLStorage()
        
        if config.IS_LONG_MEMORY :
            if config.LONG_MEMORY_TYPE == "milvus": #milvus、faiss、chroma
                self.long_memory_storage = MilvusStorage()
                
            elif  config.LONG_MEMORY_TYPE == "faiss":
                self.long_memory_storage = FaissStorage()
                
            elif config.LONG_MEMORY_TYPE == "chroma":
                self.long_memory_storage = ChromaStorage()
                
    def search_short_memory(self, query_text: str, user_name: str, charater_name: str) -> List[Dict[str, str]]:
        dict_list = []
        if config.IS_SHORT_MEMORY:
            local_memory = self.short_memory_storage.search(charater_name,  user_name)
            for json_string in local_memory:
                json_dict = json.loads(json_string)
                # "a","b"
                dict_list.append(json_dict)
        return dict_list

    def search_long_memory(self, query_text: str, user_name: str, charater_name: str) -> str:
        long_history = ""
        summary_historys = []
        if config.IS_LONG_MEMORY:
            # 获取长期记忆，按照角色划分
            #faiss: search(self,query: str,user_name: str,character_name:str):
            #milvus: search(self, query_text: str, user_name: str,character_name:str)
            #chroma: search(self,query: str,user_name: str,character_name:str)
            long_memory = self.long_memory_storage.search(query_text, user_name, charater_name)
            if len(long_memory) > 0:
                # 将json字符串转换为字典
                for i in range(len(long_memory)):
                    summary_historys.append(long_memory[i])
                long_history = ";".join(summary_historys)
            return long_history
        else:
            return ""

    def save(self,  you_name: str, query_text: str, role_name: str, answer_text: str) -> None:
        # 存储短期记忆
        pk = self.get_current_entity_id()
        local_history = {
            f'{role_name}': self.format_role_history(role_name=role_name, answer_text=answer_text),
            f'{you_name}': self.format_you_history(you_name=you_name, query_text=query_text)
        }
        
        #sql: def save(self,query_text,user_name,charater_name,query,responce) -> None:
        #local:def save(self,query_text,user_name,character_name,query,responce)
        self.short_memory_storage.save(json.dumps(local_history),you_name,role_name,query_text,answer_text)
        
        # 是否开启长期记忆
        if config.IS_LONG_MEMORY:
            # 将当前对话语句生成摘要
            history = self.format_history(you_name=you_name, query_text=query_text, role_name=role_name, answer_text=answer_text)
            importance_score = 3
            if config.LONG_MEMORY_SUMMARY:
                memory_summary = MemorySummary()
                summary = memory_summary.summary(input=history)
                history = summary["summary"]
                # 计算记忆的重要程度
                memory_importance = MemoryImportance()
                importance_score = memory_importance.importance(input=history)
                # save(self, pk: int, owner :str , character_name : str,  query_text: str,responce :str , importance_score: int)
                # def save(self,pk,user_name,character_name,query,responce,importance_score)
                # def save(self,pk,user_name,character_name,query,responce,importance_score):
                # def save(self, pk: int,  query_text: str, owner: str, importance_score: int)
            self.long_memory_storage.save(pk,you_name, role_name, history,answer_text, importance_score)

    def format_history(self, you_name: str, query_text: str, role_name: str, answer_text: str):
        
        you_history = self.format_you_history(you_name=you_name, query_text=query_text)
        
        role_history = self.format_role_history(role_name=role_name, answer_text=answer_text)
        
        chat_history = you_history + '\001' + role_history
        return chat_history
    
    def get_current_entity_id(self) -> int:
        '''生成唯一标识'''
        return self.snow_flake.task()
    
    def format_you_history(self, you_name: str, query_text: str):
        you_history = f"{you_name}:{query_text}"
        return you_history

    def format_role_history(self, role_name: str, answer_text: str):
        role_history = f"{role_name}:{answer_text}"
        return role_history

    # def clear(self, owner: str) -> None:
    #     self.long_memory_storage.clear(owner)
    #     self.short_memory_storage.clear(owner)

