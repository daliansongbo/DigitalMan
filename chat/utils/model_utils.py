import gc
import torch
import ..configs.config as config

def clear_torch_cache():
    gc.collect()
    torch.cuda.empty_cache()

def unload_model():
    config.LLM_ChatBot_Model = config.LLM_ChatBot_Tokenizer  = None
    config.LORALIST = []
    clear_torch_cache()


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)

    return text

def get_stopping_strings(state):
    stopping_strings = []
    # if config['mode'] in ['instruct', 'chat-instruct']:
    #     stopping_strings += [
    #         state['turn_template'].split('<|user-message|>')[1].split('<|bot|>')[0] + '<|bot|>',
    #         state['turn_template'].split('<|bot-message|>')[1] + '<|user|>'
    #     ]

    #     replacements = {
    #         '<|user|>': state['name1_instruct'],
    #         '<|bot|>': state['name2_instruct']
    #     }

    #     for i in range(len(stopping_strings)):
    #         stopping_strings[i] = replace_all(stopping_strings[i], replacements).rstrip(' ').replace(r'\n', '\n')

    if state['mode'] in ['chat', 'chat-instruct']:
        stopping_strings += [
            f"\n{state['name1']}:",
            f"\n{state['name2']}:",
            f"\n{state['name3']}:"
        ]

    if state['stop_at_newline']:
        stopping_strings.append("\n")

    return stopping_strings