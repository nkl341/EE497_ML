# program		data_preprocess.py
# purpose	    Proprocess and standardize for training data
# usage         script
# notes         (1) 
# date			2/7/2024
# programmer    Colton Vandenburg
import json
import string


def write_dialogue_to_file(filename):
    with open(filename, 'w') as file:
        while True:
            speaker1 = input("Enter dialogue for Speaker 1 (or type 'exit' to quit): ")
            if speaker1.lower() == 'exit':
                break
            file.write(f"Speaker 1: {speaker1}\n")
            
            speaker2 = input("Enter dialogue for Speaker 2 (or type 'exit' to quit): ")
            if speaker2.lower() == 'exit':
                break
            file.write(f"Speaker 2: {speaker2}\n")

write_dialogue_to_file('conversation.txt')
def read_text_file(filename):
    with open(filename, 'r') as file:
        content = file.readlines()
    return content

# Usage
#filename = 'conversation.txt'
#content = read_text_file(filename)