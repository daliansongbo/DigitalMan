
import configs.config as config
import json


class MemorySummary():
    def __init__(self) -> None:
        if config.LANGUAGE_TYPE.lower() == "en":
            self.prompt = config.ENGLISH_SUMMARY_PROMPT
        else:   
            self.prompt = config.CHINESE_SUMMARY_PROMPT

    def summary(self, input: str) -> str:
        prompt = self.prompt + f"input:{input} [/INST]"
        
        result = config.chat_model.chat(prompt=prompt)
        print("=> summary:", result)
        # 寻找 JSON 子串的开始和结束位置
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        if start_idx != -1 and end_idx != -1:
            json_str = result[start_idx:end_idx+1]
            json_data = json.loads(json_str)
        else:
            json_data = {}
            print("未找到匹配的JSON字符串")
        return json_data

class MemoryImportance():
    def __init__(self) -> None:
        if config.LANGUAGE_TYPE.lower() == "en":
            self.prompt = config.ENGLISH_IMPORTANT_SCORE_PROMPT
        else:   
            self.prompt = config.CHINESE_IMPORTANT_SCORE_PROMPT
    def importance(self, input: str) -> int:
        if config.LANGUAGE_TYPE.lower() == "en":
            user_input = f"memory:{input} [/INST]"
        else:   
            user_input = f"记忆:{input} [/INST]"

        prompt = self.prompt + user_input
        
        result = config.chat_model.chat(prompt=prompt)
        
        print("=> importance:", result)
        # 寻找 JSON 子串的开始和结束位置
        start_idx = result.find('{')
        end_idx = result.rfind('}')
        score = 3
        if start_idx != -1 and end_idx != -1:
            json_str = result[start_idx:end_idx+1]
            json_data = json.loads(json_str)
            score = int(json_data["score"])
        else:
            print("未找到匹配的JSON字符串")
        return score
