# 导入所需模块
import os
import time
import configs.config as config
from typing import List


class LocalStorage():
    def __init__(self):
        self.history_dir = config.SHORT_MEMORY_LOCAL_DIR
        self.embeddings = config.Embedding
        self.user_character_local_history_dict = {}
        
        if os.path.exists(self.history_dir):
            user_dir_list = os.listdir(self.history_dir)
            if len(user_dir_list)>0:
                for user_character_name in user_dir_list:
                    temp_history_dict = {}
                    directory_path = os.path.join(self.history_dir , user_character_name)
                    if len(os.listdir(directory_path))>0:
                        for filename in os.listdir(directory_path):
                            if filename.endswith(".txt"):
                                file_path = f'{directory_path}/{filename}'
                                with open(file_path) as f:
                                    text = f.read()
                                text = text.replace("</s>","")
                                temp_time = filename.replace(".txt","")
                                if  int(time.time() - int(int(temp_time)/1000))/3600/60>10:
                                    continue
                                temp_history_dict[int(temp_time)] = text
                                # temp_history_list.append([text,filename.replace(".txt","")])
                    self.user_character_local_history_dict[user_character_name] = temp_history_dict
        else:
            os.mkdir(self.history_dir)
    #提取最近的对话记录
    def search(self, limit: int, user_charater_name:str) -> List[str]:
        results = []
        if user_charater_name in self.user_character_local_history_dict.keys():
            temp_history_dict = self.user_character_local_history_dict[user_charater_name] 
            temp_keys = list(temp_history_dict.keys())
            temp_keys.sort(reverse = True)
            #提取最近30条对话记录
            key_topk =  temp_keys[:config.SHORT_MEMORY_TOP_K]
            results =[temp_history_dict[k] for k in key_topk if int(time.time() - int(int(k)/1000))/3600/60<10]
        return list(results)
          
        
    def save(self,query_text,user_name,character_name,query,responce):
        temp_history_dict = {}
        user_character_name = user_name+"_"+character_name
        user_character_path = os.path.join(self.history_dir,user_character_name)
        if not os.path.exists(user_character_path):
            os.mkdir(user_character_path) 
        prompt = "{}:{}\001{}:{}".format(user_name,query,character_name, responce)
        temp_time = int(time.time()*1000)
        temp_history_dict[temp_time] = prompt
        self.user_character_local_history_dict[user_character_name] = temp_history_dict
        file_name = str(temp_time)+'.txt'
        file_path =  f'{user_character_path}/{file_name}' 
        fl = open(file_path,'w+')
        fl.write(prompt)
        fl.close()
        



   