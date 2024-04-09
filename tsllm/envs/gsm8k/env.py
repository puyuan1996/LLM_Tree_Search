import copy
import re
from typing import List, Optional
import numpy as np
from tsllm.envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR, SEP

ANS_RE = re.compile(r"The answer is (\-?[0-9\.\,]+)")
STOP_STR = "The answer is "


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
    else:
        return INVALID_ANS
    return match_str

# 从github上下载gsm8k 
# def extract_groundtruth(groundtruth_str: str):
#     import pdb; pdb.set_trace()
#     x = groundtruth_str.split("#### ")[1].strip().replace(",", "")
#     print('x', x)
#     try:
#         float(x)
#     except:
#         raise ValueError(
#             "Warning: Error should raise since the extracted groundtruth string {}\
#              cannot be converted to float".format(
#                 x
#             )
#         )
#     return x

# 使用处理好的/mnt/afs/niuyazhe/code/LLM_Tree_Search/tsllm/envs/gsm8k/train_data/sft_init.jsonl
def extract_groundtruth(groundtruth_list: list):
    # Initialize an empty list to hold all the extracted numbers
    extracted_numbers = []
    
    # Iterate over each element in the list
    for groundtruth in groundtruth_list:
        # Ensure that each element is a dictionary and has the 'text' key
        if not isinstance(groundtruth, dict) or 'text' not in groundtruth:
            raise ValueError("Each element in the list must be a dictionary with a 'text' key.")
        
        # Extract the 'text' value
        text = groundtruth['text']
        
        # Look for the pattern "The answer is " and extract the number following it
        answer_prefix = "The answer is "
        start_index = text.find(answer_prefix)
        if start_index == -1:
            raise ValueError("The pattern 'The answer is ' was not found in the input string.")
        
        # Move the start index to the beginning of the number
        start_index += len(answer_prefix)
        
        # Find the end of the number by looking for a space or end of the string
        end_index = start_index
        while end_index < len(text) and text[end_index].isdigit():
            end_index += 1
        
        # Extract the number as a string
        number_str = text[start_index:end_index]
        
        # Convert the extracted number to an integer
        try:
            number = int(number_str)
        except ValueError:
            raise ValueError(f"The extracted substring '{number_str}' could not be converted to an integer.")
        
        # Add the extracted number to the list
        extracted_numbers.append(number)
    
    return extracted_numbers[0]

# # Example usage:
# groundtruth_str = [
#     {'text': 'Natalia sold 48/2 = 24 clips in May.\nNatalia sold 48+24 = 72 clips altogether in April and May.\nThe answer is 72', 'correct': True}
# ]
# numbers = extract_groundtruth(groundtruth_str)
# print("Extracted numbers:", numbers)

def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str):
    float_groundtruth = float(extracted_groundtruth)
    try:
        return abs(float(answer) - float_groundtruth) < 1e-5
    except Exception:
        return False


class Gsm8kEnv(CoTEnv):
    sep = SEP

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fn,
        tokenizer,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fn,
            tokenizer,
            task_desc_str,
            cot_example_str,
            problem_format_str,
            reset,
        )

    @property
    def stop_str(self):
        return STOP_STR

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        # print("Compare: {} -- {}".format(extrated_answer,
        #  self.math_problem['answer']))
        # return extrated_answer == self.math_problem['answer']
        return judge_correct(
            self.math_problem["question"], self.math_problem["answer"], extracted_answer
        )

    def init_action_history(self):
        # add the first prompted questions
        return ([self.task_prefix] if self.task_prefix is not None else []) + [
            f"Question: {self.math_problem['question']}\nAnswer: Let's think step by step"
        ]

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
