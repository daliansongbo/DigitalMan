# 导入所需模块
import os
from sentence_transformers import SentenceTransformer
import time
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import json
from typing import List,Tuple
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from langchain.document_loaders import TextLoader
from pathlib import Path
from langchain.vectorstores import Chroma
from llama_index import SimpleDirectoryReader
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import configs.config as config


import re
remove_chars = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

def torch_gc():
    if torch.cuda.is_available():
        # with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif torch.backends.mps.is_available():
        try:
            from torch.mps import empty_cache
            empty_cache()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
            
    
def seperate_list(ls: List[int]) -> List[List[int]]:
    lists = []
    ls1 = [ls[0]]
    for i in range(1, len(ls)):
        if ls[i - 1] + 1 == ls[i]:
            ls1.append(ls[i])
        else:
            lists.append(ls1)
            ls1 = [ls[i]]
    lists.append(ls1)
    return lists



def load_txt(hsitory_dir,user_character_name):
    data = []
    directory_path = os.path.join(hsitory_dir,user_character_name)
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = f'{directory_path}/{filename}'
            with open(file_path) as f:
                text = f.read()
        text = text.replace("</s>","")    
        metadata = {"source": file_path,"times":filename.replace('.txt',''),"user_character_name":user_character_name}
        data +=  [Document(page_content=text, metadata=metadata)]
    return data


def load_txt_update(hsitory_dir,user_character_name,file_list):
    data = []
    directory_path = os.path.join(hsitory_dir,user_character_name)
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = f'{directory_path}/{filename}'
            with open(file_path) as f:
                text = f.read()
        text = text.replace("</s>","")    
        metadata = {"source": file_path,"times":filename.replace('.txt',''),"user_character_name":user_character_name}
        data +=  [Document(page_content=text, metadata=metadata)]
    return data


class ChromaStorage():
    def __init__(self):
        self.history_dir = config.CHROMA_HISTORY_DIR
        if not os.path.exists(self.history_dir):
            os.mkdir(self.history_dir)

        self.embeddings = config.Embedding
        self.user_realtime_record_dict = {}
        self.user_character_vs_dict = {}
        self.schroma_memory = Chroma(collection_name="chatbot",embedding_function=self.embeddings)
        chroma_user_dir_list = os.listdir(self.history_dir)
        if len(chroma_user_dir_list)>0:
            for user_character_name in chroma_user_dir_list:
                docs = load_txt(self.history_dir,user_character_name)
                print("docs",docs)
                if len(docs)>0:
                    self.schroma_memory.add_documents(docs)  ##docs 为Document列表
                    self.user_character_vs_dict[user_character_name] = self.schroma_memory
                elif len(docs)==0:
                    os.system("rm -rf " + os.path.join(self.history_dir,user_character_name))
    
    def update_vec(self,user_name,character_name):
        user_character_name = user_name+"_"+character_name
        ids_list = self.schroma_memory.get(where={"user_character_name":user_character_name})['ids']
        self.schroma_memory._collection.delete(ids=ids_list)
        docs = load_txt(self.history_dir,user_character_name)
        self.schroma_memory.add_documents(docs)  ##docs 为Document列表
        
        

    def get_docs_with_score(self,docs_with_score):
        memories = []
        current_time = int(time.time())
        print(docs_with_score)
        for doc, score in docs_with_score:
            if int(score)<config.SCORE_THRESHOLD:
                temp_memory ={}
                temp_rext = doc.page_content 
                temp_metadata =  doc.metadata 
                time_diff = current_time - int(temp_metadata["times"])
                #时间衰减
                # temp_memory["recent"] = 0.99 ** (time_diff / 3600)
                temp_memory["recent"] = 0.99**pow(time_diff / 3600, 0.5) 
                temp_memory["text"] = temp_rext
                #相似度转换为正比例
                temp_memory["relevance"] = 1 - score/1000
                temp_memory["total_score"] = temp_memory["relevance"] + temp_memory["recent"]
                memories.append(temp_memory)
        # 排序获得最高分的记忆
        memories = sorted(memories, key=lambda m: m["total_score"], reverse=True)
        if len(memories) > 0:
            memories_text = [item['text'] for item in memories]
            memories_size = config.LONG_SEACH_MEMORY_LEN
            memories_text = memories_text[:memories_size] if len(memories_text) >= memories_size else memories_text
            memories_text = [[item.split('\001')[0],item.split('\001')[1]]  for item in memories_text]
            return memories_text
        else:
            return []
            

    def search(self,query: str,user_name: str,character_name:str):
        context =[]
        user_character_name = user_name+"_"+character_name
        if user_character_name in self.user_character_vs_dict.keys():
            # user_character_vec = self.user_character_vs_dict[user_character_name]
            print("get vecstore ....")
            related_docs_with_score = self.schroma_memory.similarity_search_with_score(query, 
                                                                                      k=config.VECTOR_SEARCH_TOP_K,
                                                                                      filter = {"user_character_name":user_character_name})
            print("related_docs_with_score",len(related_docs_with_score),related_docs_with_score)
            context = self.get_docs_with_score(related_docs_with_score)
            print("context",len(context))
        return context
            
    def save(self,pk,user_name,character_name,query,responce,importance_score):
        user_character_name = user_name+"_"+character_name
        user_character_path = os.path.join(self.history_dir,user_character_name)
        if not os.path.exists(user_character_path):
            os.mkdir(user_character_path) 
        prompt = "{}:{}\001{}:{}".format(user_name,query,character_name, responce)
        file_name = str(int(time.time()))+'.txt'
        file_path = user_character_path +'/'+ file_name
        fl = open(file_path,'w+')
        fl.write(prompt)
        fl.close()
        



   