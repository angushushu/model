import time
import random
import google.generativeai as palm
import re
import json
import os

'''
Based on the initial states, ask LLM generate a 10 state long sequence
'''


def extract_sequence(input_string):
    # Regular expression to find all occurrences of 'State <number>: {<set>}'
    set_regex = re.compile(r'State \d+: \{(.*?)\}')
    # Find all matches
    matches = set_regex.findall(input_string)
    # Process matches into sets and store them in a list
    sequence = []
    for state_str in matches:
        # Ignore empty strings
        if state_str:
            # Split elements by comma and strip whitespaces, then create a set
            state = [element.strip() for element in state_str.split(',')]
            sequence.append(state)

    if len(sequence) > 0:
        return sequence
    else:
        return None

with open(os.getcwd()+'/states_3.json', 'r') as file:
    data = json.load(file)
    init_states = [state for state in data]
    sequences = []

    for init_state in init_states:
        palm.configure(api_key='AIzaSyAET1f2IPytWawCqnaDBWL-ueGrrW0O3oA')

        time.sleep(1 + 2 * random.random())
#             prompt = f'''Start with the initial state: {str(set(init_state)).replace("'", "")}, \
# please generate a sequence of subsequent states, ensuring each state represents a plausible and \
# non-contradictory situation for a person at a single point in time. Each state should be in a set \
# format and include more than three external states, more than one internal state, and more than one \
# behavior or posture, ensuring that no conflicting actions (such as 'sitting' and 'walking') are present in the same state. \
# Present the sequence in a chain format, like so: {{state1}} -> {{state2}} -> {{state3}}. \
# Each state should independently describe a moment without necessarily including all elements from the previous state, \
# and should not simply be an expansion of the previous state. Your response should strictly contain only \
# the sets in the described format, without any additional explanations or text.'''
        prompt = f'''Begin with the initial state: {str(set(init_state)).replace("'", "")}. Your task is to generate a sequence of subsequent states for a person transitioning through different scenarios over time. Adhere to the steps and format outlined below to ensure accurate output:

Step 1: Envision the next scenario.
- Consider how the person’s situation changes after the initial state, including changes in the environment, interactions, and the person’s emotions.

Step 2: Formulate the next state based on the envisioned scenario.
- External states (at least three): [list here]
- Internal states (at least one): [list here]
- Actions or Postures (at least one): [list here]

Step 3: Number the next state and present it in set format.
- State 2: {{external states, internal states, actions or postures}}

Step 4: Repeat Steps 1 to 3 to generate a sequence of states, ensuring a logical and coherent transition from one state to the next. 

Step 5: Compile the numbered states in a list.
- State 1: {{initial state}}
- State 2: {{next state}}
- State 3: {{following state}}
... and so on.

Example Transition:
- Initial State: {{ordering, anxiety, windows, sad, taking out, standing, sunny, coffee shop, wallet, table}}
- Envisioned Scenario: The person completes their order, feels relieved, exits the coffee shop, and walks down the street on a cloudy day.
- Next State: State 2: {{paying, relief, door, happy, leaving, walking, cloudy, street, wallet, no table}}

Now, proceed to generate a list of numbered states following the steps and format provided above. Ensure each state is unique, plausible, and presented in the specified set format without any additional explanations or text.'''

        print(prompt)
        response = palm.chat(messages=prompt, candidate_count=4)
        for cand in response.candidates:
            resp_seq = cand['content']
            print(resp_seq)
            sequences.append(extract_sequence(resp_seq))

    with open(os.getcwd()+'/sequences_3.json', 'w', encoding='utf-8') as new_file:
        json.dump(sequences, new_file, ensure_ascii=False, indent=4)