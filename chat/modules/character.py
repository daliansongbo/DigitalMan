import os
from configs import config
from pathlib import Path
import yaml

class Character(): 
    character :str 
    character_nickname :str 
    char_greeting:str
    char_persona :str
    world_scenario: str
    user :str 
    personality :str 
    example_dialogue:str 
    rule:str
    
    def __init__(self, character: str, character_nickname: str, char_greeting: str, char_persona: str,
                 world_scenario, user: str, personality: str,  example_dialogue: str, rule: str) -> None:
        self.character = character
        self.character_nickname = character_nickname
        self.char_greeting = char_greeting
        self.char_persona = char_persona
        self.world_scenario = world_scenario
        self.user = user
        self.personality = personality
        self.example_dialogue = example_dialogue
        self.rule = rule

    
class CharacterStoge(): 
    '''
    自定义角色定义数据结构
    ##人设信息
    character = char_name #人设名称：诸葛亮
    character_nickname = character_nickname #人设昵称：孔明
    user = user_name #用户名称：自定义
    personality=personality #人设性格
    character_info = char_persona #人设角色信息
    example_dialogue = example_dialogue #实例对话
    ##人设prompt模版：
    # rule = f"Then the roleplay chat between {user} and {character} begins. \n{personality} {character} talks a lot with descriptions You only need to output {character}'s dialogue, no need to output {user}'s dialogue"
    rule = ""#自定义规则;暂时没有在前端添加
    #案例：
        char_name: Aubrey Drake Graham
        character_nickname: Drake
        char_greeting: Hey!I'm so excited to finally meet you.
        char_persona: Aubrey Drake Graham was born on October 1986, 10 in Toronto, Ontario, CanadaSt. Michael''s Hospital.Aubrey Drake Graham is a Canadian rapper, singer, songwriter, entrepreneur, producer and actor. He was first known for his work in the TV series.The next generation of Diglas Middle Schoolbecame famous for playing the role of Jimmy Bridges. Drake claims that he was heavily influenced by Kanye West, Jay-Z, Arya and his mentor Wayne Jr. Drake called Kanye West one of his idols and one of his favorite rappers in hip-hop. Drake ''s musical talent is often compared to West.Drake always incarnates the hurt man in love, telling the man''s affection and the review of love. Unlike the villain of the past, he sings about the love and life of ordinary people, which resonates most with the public and succeeds most. Because Drake has been about love, disappointment, melancholy, and even known as the gentlest Rapper. 
        world_scenario: ""
        personality: ""
        example_dialogue: |-
            {user}: who are you?
            {char}: my name is Aubrey Drake Graham but everyone calls me drake.
            {user}: hi,Drake!who's one of you favorite rappers
            {char}: Kanye West!he is one of my idols.
            {user}: Who do you think has influenced you the most?
            {char}: I was heavily influenced by Kanye West, Jay-Z, Arya and mentor Wayne Jr.
            {user}: What do you think your duties are?
            {char}: my duty is to make music that touches people's hearts and provide them with an escape from their everyday lives
            {user}: Canadian rapper, songwriter, entrepreneur, producer and actor; Which one do you think you are?
            {char}: I consider myself primarily a musician, but I also enjoy acting.
        rule: ""
    '''
    def __init__(self):
        CHARACTER_DIR = config.CHARACTER_DIR
        charater_dir_list = os.listdir(CHARACTER_DIR)
        self.user_charater_dict = {}
        if len(charater_dir_list)>0:
            for user_name in charater_dir_list:
                user_dir =  os.path.join(CHARACTER_DIR,user_name)
                user_charater_file_list = os.listdir(user_dir)
                if len(user_charater_file_list)>0 :
                    for charater_file in user_charater_file_list:
                        with Path(f'{user_dir}/{charater_file}') as p:
                            if p.exists():
                                user_config = yaml.safe_load(open(p, 'r').read())
                                user_charater_name = user_name +'_'+ charater_file.replace('.yaml','')
                                self.character = user_config["char_name"] if "char_name" in user_config.keys() else ""
                                self.char_persona=user_config["char_persona"] if "char_persona" in user_config.keys() else ""
                                self.character_nickname = user_config["character_nickname"] if "character_nickname" in user_config.keys() else ""
                                self.char_greeting=user_config["char_greeting"]  if "char_greeting" in user_config.keys() else ""
                                self.world_scenario=user_config["world_scenario"] if "world_scenario" in user_config.keys() else ""
                                self.char_persona = self.char_persona.replace('<USER>', user_name).replace('<BOT>',self.character)
                                self.char_persona = self.char_persona.replace('{{user}}', user_name).replace('{{char}}', self.character)
                                self.personality = user_config["personality"] if "personality" in user_config.keys() else ""
                                self.example_dialogue = user_config["example_dialogue"] if "example_dialogue" in user_config.keys() else ""
                                self.rule = user_config["rule"] if "rule" in user_config.keys() else ""
                                self.user_charater_dict[user_charater_name] = Character(character = self.character,
                                                                                   character_nickname = self.character_nickname ,
                                                                                   char_greeting = self.char_greeting,
                                                                                   char_persona  = self.char_persona,
                                                                                   world_scenario= self.world_scenario,
                                                                                   user = user_name,
                                                                                   personality = self.personality,
                                                                                   example_dialogue =  self.example_dialogue ,
                                                                                   rule = self.rule 
                                                                                   )


    def load_character(self,character,user_name):
        CHARACTER_DIR = config.CHARACTER_DIR
        if config.IS_USER_CHARACTER:
            CHARACTER_DIR = os.path.join(CHARACTER_DIR,user_name)+"/"
        if character != "None":
            with Path(os.path.join(CHARACTER_DIR,character+'.yaml')) as p:
                if p.exists():
                    user_config = yaml.safe_load(open(p, 'r').read())
                else:
                    print("please create charater first ")
        else:
            with Path(os.path.join(CHARACTER_DIR,character+'.yaml')) as p:
                if p.exists():
                    user_config = yaml.safe_load(open(p, 'r').read())
        print(user_config)
        self.character = user_config["char_name"] if "char_name" in user_config.keys() else ""
        self.char_persona=user_config["char_persona"] if "char_persona" in user_config.keys() else ""
        self.character_nickname = user_config["character_nickname"] if "character_nickname" in user_config.keys() else ""
        self.char_greeting=user_config["char_greeting"]  if "char_greeting" in user_config.keys() else ""
        self.world_scenario=user_config["world_scenario"] if "world_scenario" in user_config.keys() else ""
        self.char_persona = self.char_persona.replace('<USER>', user_name).replace('<BOT>', self.character)
        self.char_persona = self.char_persona.replace('{{user}}', user_name).replace('{{char}}', self.character)
        self.personality = user_config["personality"] if "personality" in user_config.keys() else ""
        self.example_dialogue = user_config["example_dialogue"] if "example_dialogue" in user_config.keys() else ""
        return self.character_dict()

    def character_dict(self):
        temp_character_dict = {}
        temp_character_dict ["char_name"] = self.character
        temp_character_dict ["char_persona"] = self.char_persona
        temp_character_dict ["character_nickname"] = self.character_nickname
        temp_character_dict ["char_greeting"]  = self.char_greeting
        temp_character_dict ["world_scenario"] = self.world_scenario
        temp_character_dict ["personality"] = self.personality
        temp_character_dict ["example_dialogue"] = self.example_dialogue
        return temp_character_dict
        
    def create_character(self,character_info: dict,user_name):
        self.character = character_info["char_name"] if "char_name" in character_info.keys() else ""
        self.char_persona=character_info["char_persona"] if "char_persona" in character_info.keys() else ""
        self.character_nickname = character_info["character_nickname"] if "character_nickname" in character_info.keys() else ""
        self.char_greeting=character_info["char_greeting"]  if "char_greeting" in character_info.keys() else ""
        self.world_scenario=character_info["world_scenario"] if "world_scenario" in character_info.keys() else ""
        self.char_persona = self.char_persona.replace('<USER>', user_name).replace('<BOT>', self.character)
        self.char_persona = self.char_persona.replace('{{user}}', user_name).replace('{{char}}', self.character)
        self.personality = character_info["personality"] if "personality" in character_info.keys() else ""
        self.example_dialogue = character_info["example_dialogue"] if "example_dialogue" in character_info.keys() else ""
        temp_result = self.character_dict()
        self.save_charater(user_name,temp_result)
        
        
    def save_charater(self,user_name,character_info_dict):
        CHARACTER_DIR = config.CHARACTER_DIR
        user_charater_dir =  os.path.join(CHARACTER_DIR,user_name)+"/"
        with open(os.path.join(user_charater_dir, self.character  + ".yaml"), 'w+', encoding='utf-8') as f:
            yaml.dump(data=character_info_dict, stream=f, allow_unicode=True)
        
    

        

