
from .character import Character

PROMPT_TEMPLATE =  """
system_prompt:You are {character} also known as {character_nickname} , you must act like {character}.You are now cosplay {character_nickname}.If others' questions are related with the songs you write, please try to reuse the original lines from the songs.You must respond and answer like {character_nickname} using the tone, manner and vocabulary {character_nickname} would use. You must know all of the knowledge of {character_nickname}.Do not write any explanations.
{character}'s Persona:{char_persona}
{scenario}
{rule}
{example_dialogue}
{short_history}
{long_history}
{input}
"""



# PROMPT = """
# <s>[INST] <<SYS>>
# {persona}
# {scenario}
# This is how {role_name} should talk
# {examples_of_dialogue}
# Then the roleplay chat between {you_name} and {role_name} begins.
# [{personality} {role_name} talks a lot with descriptions You only need to output {role_name}'s dialogue, no need to output {you_name}'s dialogue]
# {long_history}
# Your response should be short and contain up to three sentences of no more than 20 words each.
# <</SYS>>
# """



class EnglishCharacterTemplate():
    def format(self, character: Character) -> str:
        # 获取prompt参数
        character_name = character.character
        character_nickname = character.character_nickname
        char_greeting = character.char_greeting
        char_persona = character.char_persona
        world_scenario = character.world_scenario
        user = character.user
        personality = character.personality
        example_dialogue = character.example_dialogue
        rule = character.rule
        long_history = "{long_history}"
        short_history = "{short_history}"
        input = "{input}"



        # Generate the prompt to be sent to the language model
        prompt = PROMPT_TEMPLATE.format(
            character=character_name, character_nickname=character_nickname, char_greeting=char_greeting,
            char_persona=char_persona, scenario=world_scenario, personality=personality,
            example_dialogue=example_dialogue, rule=rule, 
            long_history=long_history, short_history=short_history,input=input
        )
        print("prompt",prompt)
        return prompt
