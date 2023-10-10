import json
import os
import google.generativeai as palm
import random
import time
import re
from tqdm import tqdm

def extract_state(response):
    # Extract the data using a regular expression to find content within square brackets.
    match = re.search(r'\[([^\]]+)\]', response)
    if not match:
        raise ValueError("Could not extract list from response")

    # Extract the matched content and split it into strings, then strip any excess whitespace.
    data = [item.strip().strip("'") for item in match.group(1).split(',')]

    # print('extracted', sorted(data))
    # Sort the list and return it
    return sorted(data)

with open(os.path.join(os.getcwd(), 'sequences_1.json'), 'r') as file:
    sequences = json.load(file)
new_sequences = []
total_sequences = len(sequences)
sequence_cnt = 1
miss_cnt = 0
for sequence in (bar := tqdm(sequences, desc='PROG', position=0)):
    if sequence is None:
        continue
    new_sequence = []
    total_states = len(sequence)
    state_cnt = 1
    # print(f'S[{sequence_cnt}]')
    for state in sequence:
        if state is None:
            continue
        try_cnt = 1
        for i in range(5):
            bar.set_postfix_str(f"S{state_cnt}/{total_states}, T{try_cnt}/5, M{miss_cnt}")
            bar.update
            try:
                palm.configure(api_key='AIzaSyAET1f2IPytWawCqnaDBWL-ueGrrW0O3oA')
                time.sleep(0.5 + 2 * random.random())
                prompt = f'''Input:
        A state represented as a list containing multiple strings, where each string signifies an "idea unit" (a single concept or perception of an individual’s external and internal world).
        
        Task:
        Analyze each string within the state, ensuring it is condensed into the smallest possible "idea unit".
        If a string can be divided, split it into individual words or phrases, ensuring they do not construct sentences.
        Ensure that any adjectives remains unchanged (e.g., "cloudy" becomes "cloudy", "interesting" becomes "interesting").
        Ensure that any verb is presented in its base form (e.g., "took" becomes "take", "ordering" becomes "order").
        Ensure that any plural noun is converted to its singular form (e.g., "windows" becomes "window").
        Replace the original strings with the split words or phrases.
        Keep all other aspects unchanged and return the modified data in the same list format.
        
        For instance, if the input is:
        ['taking out wallet', 'sad', 'windows', 'the weather is sunny', 'table', 'ordering', 'standing in a coffee shop', 'I feel anxiety']
        The modified state should be in the following format:
        ['take out', 'coffee shop', 'wallet', 'sad', 'window', 'sunny', 'table', 'order', 'stand', 'anxiety']
        
        Now consider the state:
        {str(state)}
        Provide me the modified state
        '''
                # print(f'  |--[{state_cnt}] PRE ADJ:', state)
                response = palm.chat(messages=prompt)
                resp_state = response.last
                extracted_state = extract_state(resp_state)
                # print(f'  |--[{state_cnt}] POS ADJ:', extracted_state)
                new_sequence.append(extracted_state)
                break
            except Exception as e:
                # print(f"      |--An error occurred: {str(e)}")
                # print(f"      |--Retrying {retry_cnt}")
                time.sleep(5)  # Wait for 5 seconds before retrying
                try_cnt += 1
        if try_cnt > 5:
            miss_cnt += 1
            new_sequence.append(state)
        state_cnt += 1
    new_sequences.append(new_sequence)
    # sequence_cnt += 1
    with open(os.getcwd()+'/sequences_1_recoded.json', 'w', encoding='utf-8') as new_file:
        json.dump(new_sequences, new_file, ensure_ascii=False, indent=4)