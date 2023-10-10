import time
import random
import google.generativeai as palm
import re
import json
import os

'''
Ask LLM randomly generate ~100 states in the format of set
'''

def extract_states(input_string):
    # Regular expression to find all occurrences of sets
    set_regex = re.compile(r'\{(.*?)\}')
    # Find all matches
    matches = set_regex.findall(input_string)
    # Process matches into sets and store them in a list
    all_sets = []
    for match in matches:
        # Split elements by comma and strip whitespaces, then create a set
        elements = [element.strip() for element in match.split(',')]
        all_sets.append(elements)
    return all_sets

palm.configure(api_key='AIzaSyAET1f2IPytWawCqnaDBWL-ueGrrW0O3oA')

prompt = """Please generate a random description of a person's initial \
state and ensure that it is presented in a set format. Each set should \
include more than three description of the external state (such as the environment they are in)\
, more than one internal state (such as emotion or mental state), \
and more than one behavior or posture. Please provide a rich collection of \
states, including as many elements as possible. For example: \
{table, windows, coffee shop, sunny, anxiety, sad, ordering, taking out, wallet, standing}."""
states = []
while len(states) < 1000:
    time.sleep(5+5*random.random())
    response = palm.chat(messages=prompt, candidate_count=4)
    for cand in response.candidates:
        new_states = extract_states(cand['content'])
        states += new_states

with open(os.getcwd()+'/states_3.json', 'w', encoding='utf-8') as f:
    json.dump(states, f, ensure_ascii=False, indent=4)
