from openai import OpenAI
import pandas as pd
import re
import pickle
import json
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
from utils import extract_logic_components
from Z3solve import *
import copy
import random

client = OpenAI(base_url="your_url", api_key="your_api_key")
#client = OpenAI(base_url="your_url", api_key="your_api_key")

record_path = 'outputs/ours/output0133.txt'

def print_and_write(s, prt = True, path = record_path):
    if prt:
        print(str(s))
    with open(path, 'a', encoding='utf-8') as file:
        file.write(str(s) + '\n')

def ask(prompt, input, prt = False, write = True):
    if write:
        print_and_write('input:', prt=prt)
        print_and_write(prompt, prt=prt)
        print_and_write(input, prt=prt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input}
        ],
        stream=False
    )
    output = response.choices[0].message
    if output.content != None:
        response = output.content
    else:
        response = output.reasoning_content
    if write:
        print_and_write('response:', prt=prt)
        print_and_write(response, prt=prt)
    return response

class LogicProgrammer:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.save_path = args.save_path
        self.max_subgoal_trial = args.max_subgoal_trial
        self.max_outline_trial = args.max_outline_trial
        self.max_depth = args.max_depth

        self.prompt_creator = {
                               'AR-LSAT': {'outline': self.outline_prompt_folio, 
                                         'subgoal': self.subgoal_prompt_folio, 
                                         'answer_feedback': self.answer_feedback_folio, 
                                         'outline_feedback': self.outline_feedback_folio, 
                                         'translate': self.translate_prompt_folio
                                         }
                               }
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_dir = f'./models/prompts/{self.dataset_name}'
        
        if not os.path.exists(f'{prompt_dir}_outline_ours.txt'):
            print_and_write(f"Warning: Prompt files for '{self.dataset_name}' not found. Defaulting to 'FOLIO' prompts.")
            prompt_dir = './models/prompts/FOLIO'

        outline_prompt_file = f'{prompt_dir}_outline_ours.txt'
        with open(outline_prompt_file, 'r', encoding='utf-8') as f:
            self.outline_prompt_template = f.read()

        subgoal_prompt_file = f'{prompt_dir}_subgoal_ours.txt'
        with open(subgoal_prompt_file, 'r', encoding='utf-8') as f:
            self.subgoal_prompt_template = f.read()

        answer_feedback_file = f'{prompt_dir}_answer_feedback_ours.txt'
        with open(answer_feedback_file, 'r', encoding='utf-8') as f:
            self.answer_feedback_template = f.read()

        outline_feedback_file = f'{prompt_dir}_outline_feedback_ours.txt'
        with open(outline_feedback_file, 'r', encoding='utf-8') as f:
            self.outline_feedback_template = f.read()

        translate_prompt_file = f'{prompt_dir}_translate_ours.txt'
        with open(translate_prompt_file, 'r', encoding='utf-8') as f:
            self.translate_prompt_template = f.read()

    def outline_prompt_folio(self, data, hints, feedback):
        premises = data['context'].split('. ')
        premise_text = '\n'.join([f"[PREMISE {i+1}] {premise.strip('.')}." for i, premise in enumerate(premises)])

        if hints == []:
            hints_text = ''
        else:
            hints_text = 'You can use the following hints as additional premises. They are proposed in your former trials and are proved to be true through rigid verification:\n' 
            hints_text += '\n'.join([f"[PREMISE {i+len(premises)+1}] {hint.strip('.')}." for i, hint in enumerate(hints)]) if hints != [] else ''

        question_text = f"[QUESTION] {data['question'].strip()}"
        full_prompt = self.outline_prompt_template.replace('[[PREMISES]]', premise_text).replace('[[HINTS]]', f'\n{hints_text}\n').replace('[[QUESTION]]', question_text).replace('[[FEEDBACK]]', feedback)
        return full_prompt

    def subgoal_prompt_folio(self, data, hints, outline_text, feedback):
        premises = data['context'].split('. ')
        premise_text = '\n'.join([f"[PREMISE {i+1}] {premise.strip('.')}." for i, premise in enumerate(premises)])
        
        if hints == []:
            hints_text = ''
        else:
            hints_text = 'You can use the following hints as additional premises:\n' 
            hints_text += '\n'.join([f"[PREMISE {i+len(premises)+1}] {hint.strip('.')}." for i, hint in enumerate(hints)]) if hints != [] else ''

        question_text = f"[QUESTION] {data['question'].strip()}"
        outline_text = f"[DEDUCTION]\n {outline_text}"
        full_prompt = self.subgoal_prompt_template.replace('[[PREMISES]]', premise_text).replace('[[HINTS]]', f'\n{hints_text}\n').replace('[[QUESTION]]', question_text).replace('[[DEDUCTION]]', outline_text).replace('[[FEEDBACK]]', feedback)
        return full_prompt

    def answer_feedback_folio(self, trial_id, failure_text = '', summary_text=''):
        feedback_prompt = self.answer_feedback_template.replace('[[SUMMARY]]', summary_text).replace('[[TRIALID]]', str(trial_id)).replace('[[FAILURE]]', failure_text)
        return feedback_prompt

    def outline_feedback_folio(self, trial_id, outline, correct_subgoals, incorrect_subgoals):
        subconclusion_result_text = ''
        if correct_subgoals != []:
            subconclusion_result_text += '\n[Your correct subconclusions]:\n'
            for correct_subgoal in correct_subgoals:
                subconclusion_result_text += f"{correct_subgoal.strip().strip('.')}.\n"
        if incorrect_subgoals != []:
            subconclusion_result_text += '\n[Your incorrect subconclusions]:\n'
            for incorrect_subgoal in incorrect_subgoals:
                subconclusion_result_text += f"{incorrect_subgoal.strip().strip('.')}.\n"
        feedback_prompt = self.outline_feedback_template.replace('[[DEDUCTION]]', outline).replace('[[SUBCONCLUSIONRESULTS]]', subconclusion_result_text).replace('[[TRIALID]]', str(trial_id))
        return feedback_prompt

    def translate_prompt_folio(self, data):
        problem = data['context'].replace('. ', '.\n').strip()
        question = data['question'].replace('. ', '.\n').strip()
        full_prompt = self.translate_prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, self.dataset_name, f'{self.split}.json'), 'r', encoding='utf-8') as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def generate_outline(self, example, hints = [], feedback=''):
        full_prompt = self.prompt_creator[self.dataset_name]['outline'](example, hints, feedback)

        output = ask('You are a helpful assistant.', full_prompt)

        return output

    def clean_subgoals(self, subgoals, example, hints):
        premises = example['context'].split('. ')
        for target in list(subgoals.keys()):
            
            for subconclusion_id in subgoals[target]['proofs']['subconclusions']:
                if subconclusion_id == target or subconclusion_id not in list(subgoals.keys()):
                    subgoals[target]['proofs']['subconclusions'].remove(subconclusion_id)

            for premise_id in subgoals[target]['proofs']['premises']:
                if int(premise_id) not in range(1, len(premises+hints)+1):
                    subgoals[target]['proofs']['premises'].remove(premise_id)

            if target == 'answer':
                continue
            for premise_id, premise in enumerate(premises+hints+[example['question']]):
                if subgoals[target]['content'].strip().strip('.').lower() == premise.strip().strip('.').lower():
                    print_and_write(f'statement {target} is identical to premise {premise_id+1}')
                    subgoals.pop(target)
                    if premise_id == len(premises+hints):
                        continue
                    for target_to_adjust in list(subgoals.keys()):
                        if target in subgoals[target_to_adjust]['proofs']['subconclusions']:
                            subgoals[target_to_adjust]['proofs']['subconclusions'].remove(target)
                            if str(premise_id+1) not in subgoals[target_to_adjust]['proofs']['premises']:
                                subgoals[target_to_adjust]['proofs']['premises'].append(str(premise_id+1))
                    
                    break
        return subgoals

    def generate_subgoals(self, example, hints = [], feedback=''):
        outline = self.generate_outline(example, hints, feedback)
        
        answer_feedback = ''
        for trial_id in range(self.max_subgoal_trial):
            subgoal_prompt = self.prompt_creator[self.dataset_name]['subgoal'](example, hints, outline, answer_feedback)
            subgoal_response = ask('You are a helpful assistant.', subgoal_prompt)
            subgoals = extract_logic_components(subgoal_response)
            subgoals = self.clean_subgoals(subgoals, example, hints)
            
            valid = False if any(subgoals[id]['proofs']['subconclusions'] == [] and subgoals[id]['proofs']['premises'] == [] for id in list(subgoals.keys())) else True
                
            if not valid or 'answer' not in subgoals:
                if valid and 'answer' not in subgoals:
                    failure_text = "but you didn't denote the answer (should start with '[ANSWER]') and list its proofs (should be like '[PROOFS] premise i, premise j,..., subconclusion k, subconclusion l') in the correct form above"
                elif not valid and 'answer' in subgoals:
                    failure_text = "but you didn't list subconclusions (should start with '[SUBCONCLUSION i]') and their proofs (should be like '[PROOFS] premise i, premise j,..., subconclusion k, subconclusion l') in the correct form above"
                elif not valid and 'answer' not in subgoals:
                    failure_text = "but you didn't denote the answer (should start with '[ANSWER]') and list subconclusions (should start with '[SUBCONCLUSION i]'), answer and their proofs (should be like '[PROOFS] premise i, premise j,..., subconclusion k, subconclusion l') in the correct form above"
                
                answer_feedback += self.prompt_creator[self.dataset_name]['answer_feedback'](trial_id+1, failure_text, subgoal_response)
            else:
                break
        
        print_and_write(f"unsupplemented subgoals:\n{subgoals}")
        
        if 'answer' not in subgoals:
            subconclusions = [id for id in list(subgoals.keys())]
            subgoals['answer'] = {'content': None,
                            'proofs': {'subconclusions': subconclusions,
                                       'premises': []
                                       },
                            'verified': False
            }
        for target in list(subgoals.keys()):
            if subgoals[target]['proofs']['subconclusions'] == [] and subgoals[target]['proofs']['premises'] == []:
                subgoals['answer']['proofs']['premises'] = [str(i+1) for i in range(len(example['context'].split(". ")+hints))]
                if target == 'answer':
                    subconclusions = [id for id in list(subgoals.keys())]
                    subconclusions.remove('answer')
                    subgoals['answer']['proofs']['subconclusions'] = subconclusions
                                            
        return subgoals, outline

    def translate(self, example):
        full_prompt = self.prompt_creator[self.dataset_name]['translate'](example)
        output = ask('You are a helpful assistant.', full_prompt)
        return output

    def verify_step(self, target, example, hints, subgoals, require_verified_premises=True):
        if subgoals[target]['verified']:
            return subgoals[target]['content'].upper() if target == 'answer' else 'TRUE' 
        
        required_premises = []
        required_subconclusions = []
        premises = example['context'].split(". ")
        for subconclusion_id in subgoals[target]['proofs']['subconclusions']:
            if subgoals[subconclusion_id]['verified'] or not require_verified_premises:
                required_subconclusions.append(subgoals[subconclusion_id]['content'])
        for premise_id in subgoals[target]['proofs']['premises']:
            required_premises.append((premises+hints)[int(premise_id)-1])
        
        if (required_premises + required_subconclusions) == []:
            return 'UNCERTAIN'
        
        premise_text = '\n'.join(required_premises + required_subconclusions)

        if target == 'answer':
            verification_question = example['question']
        else:
            verification_question = subgoals[target]['content']
            verification_question = 'Based on the above information, is the following statement true, false, or uncertain? '+verification_question
        
        translation = self.translate({'context': premise_text, 'question': verification_question})
        answer = verify_logic_problem(translation)

        return answer

    def solve_logic_problem(self, example, hints = [], depth = 0, id='0'):
        all_feedback = ''
        
        if id =='0' or depth == 0:
            print_and_write(f'hints: {hints}')
        
        for trial_id in range(self.max_outline_trial):
            print_and_write(f'id: {id}, outline trial: {trial_id+1}/{self.max_outline_trial}')
            if self.max_depth == 0 or depth >= self.max_depth:
                translation = self.translate(example)
                answer = verify_logic_problem(translation)
                return answer, [{'answer':{'content': None, 'proofs': {'sunconclusions':[], 'premises':[]}, 'verified': False}}], ''
                
            subgoals, outline = self.generate_subgoals(example, hints, all_feedback)
            
            print_and_write(f'subgoals:')
            for item in list(subgoals.items()):
                print_and_write(f"{item[0]} proofs: {item[1]['proofs']}")
            
            def verify_conclusion(target):
                further_subgoals, further_outline = [], ''
                
                required_subconclusions = subgoals[target]['proofs']['subconclusions']
                
                for required_subconclusion in required_subconclusions:
                    verify_conclusion(required_subconclusion)
                
                if target == 'answer':
                    if subgoals[target]['content']:
                        print_and_write(f'verifying step {id}.{target}')
                        result = self.verify_step(target, example, hints, subgoals)
                        subgoals[target]['verified'] = True if result == subgoals[target]['content'] else False
                    
                    else:
                        print_and_write(f'Answer not concluded after verification completed, continue deduction')
                        verified_hints = []
                        for i in subgoals:
                            if subgoals[i]['verified'] == True:
                                verified_hints.append(subgoals[i]['content'])
        
                        example_to_further_solve = example.copy()
                        example_to_further_solve['question'] = example['question']

                        result, further_subgoals, further_outline = self.solve_logic_problem(example_to_further_solve, hints + verified_hints, depth, id+f'.{target}')
                        subgoals[target]['verified'] = further_subgoals[-1]['answer']['verified']
                        subgoals[target]['content'] = result

                else:
                    print_and_write(f'verifying step {id}.{target}')
                    result = self.verify_step(target, example, hints, subgoals)
                    if result == 'TRUE':
                        subgoals[target]['verified'] = True
                    elif result == 'FALSE':
                        subgoals[target]['verified'] = False
                    elif result == 'UNCERTAIN':
                        verified_hints = []
                        for i in subgoals:
                            if subgoals[i]['verified'] == True:
                                verified_hints.append(subgoals[i]['content'])
        
                        example_with_subgoal_to_solve = example.copy()
                        example_with_subgoal_to_solve['question'] = 'Based on the above information, is the following statement true, false, or uncertain? ' + subgoals[target]['content']

                        result, subsubgoals, suboutline = self.solve_logic_problem(example_with_subgoal_to_solve, hints + verified_hints, depth+1, id+f'.{target}')
                        subgoals[target]['verified'] = (result == 'TRUE')
                
                return further_subgoals, further_outline


            further_subgoals, further_outline = verify_conclusion('answer')
            
            if not subgoals['answer']['verified'] and trial_id < self.max_outline_trial-1:    
                correct_subgoals = []
                incorrect_subgoals = []
                for target in subgoals:
                    if target == 'answer':
                        continue
                    if subgoals[target]['verified']:
                        correct_subgoals.append(subgoals[target]['content'])
                    else:
                        incorrect_subgoals.append(subgoals[target]['content'])
                
                for further_subgoal in further_subgoals:
                    for target in further_subgoal:
                        if target == 'answer':
                            continue
                        if further_subgoal[target]['verified']:
                            correct_subgoals.append(further_subgoal[target]['content'])
                        else:
                            incorrect_subgoals.append(further_subgoal[target]['content'])
                
                hints += correct_subgoals
                feedback = self.prompt_creator[self.dataset_name]['outline_feedback'](trial_id+1, outline + further_outline, correct_subgoals, incorrect_subgoals)
                all_feedback += feedback
                print_and_write(f'verification failed, now is trial {trial_id+1}, continue to have anothe trial!')
                continue
            else:
                break

        return subgoals['answer']['content'], [subgoals]+further_subgoals, outline + further_outline

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--max_subgoal_trial', type=int, default=2)
    parser.add_argument('--max_outline_trial', type=int, default=2)
    parser.add_argument('--max_depth', type=int, default=2)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    logic_programmer = LogicProgrammer(args)
    raw_dataset = logic_programmer.load_raw_dataset()
    results_path = 'outputs/ours/processing_results37.jsonl'
    
    processed_ids = set()
    all_results = []
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip(): 
                    result = json.loads(line)
                    processed_ids.add(result['id'])
                    all_results.append(result)
        print(f"Resuming from previous run. Found {len(processed_ids)} already processed items.")
    except FileNotFoundError:
        print("Starting a new run. No previous results file found.")
    
    correct_predictions_this_run = 0
    processed_this_run = 0

    for i, example in enumerate(tqdm(raw_dataset, desc="Processing dataset")):
        if example['id'] in processed_ids:
            continue
        print_and_write("\n" + "="*80)
        print_and_write(f"Processing item {i+1}/{len(raw_dataset)} | ID: {example['id']}")
        print_and_write(f"Ground truth answer: {example['answer']}")
        print_and_write("="*80 + "\n")

        predicted_answer_letter = "N/A"
        ground_truth_letter = example['answer']
        is_correct = False
        all_option_subgoals = {}
        uncertain_options = []
        true_options = []

        print_and_write(f"Problem is multi-choice. Evaluating each option individually.")
        
        for option in example['options']:
            try:
                option_letter = option.split(')')[0].strip()
                option_text = option.split(') ', 1)[1].strip()
            except IndexError:
                print_and_write(f"Could not parse option: {option}. Skipping.")
                continue

            print_and_write("\n" + "-"*30 + f" Evaluating Option {option_letter} " + "-"*30)
            
            sub_problem = {
                'context': example['context'],
                'question': f"Based on the provided rules, is the following arrangement valid? '{option_text}'"
            }
            
            option_result, subgoals, outline = logic_programmer.solve_logic_problem(sub_problem, hints = [], depth = 0, id=f'0_{option_letter}')
            
            all_option_subgoals[option_letter] = subgoals
            
            print_and_write(f"--- Verdict for Option {option_letter}: {option_result} ---")

            option_result_str = str(option_result).strip().lower()
            if option_result_str == 'true':
                true_options.append(option_letter)
            elif option_result_str == 'uncertain':
                uncertain_options.append(option_letter)
        
        if true_options:
            predicted_answer_letter = random.choice(true_options)
            print_and_write(f"Found {len(true_options)} TRUE options ({true_options}). Randomly selecting: {predicted_answer_letter}")
        elif uncertain_options:
            predicted_answer_letter = random.choice(uncertain_options)
            print_and_write(f"No TRUE option found. Randomly selecting from UNCERTAIN options: {predicted_answer_letter}")

        is_correct = (predicted_answer_letter == ground_truth_letter)

        print_and_write("\n--- Final Verification ---")
        print_and_write(f"Predicted Answer Letter: {predicted_answer_letter}")
        print_and_write(f"Ground Truth Answer Letter: {ground_truth_letter}")
        print_and_write(f"Detailed subgoals for all evaluated options: {json.dumps(all_option_subgoals, indent=2)}")

        if is_correct:
            correct_predictions_this_run += 1
            print_and_write("Result: CORRECT")
        else:
            print_and_write("Result: INCORRECT")
        
        processed_this_run += 1

        new_result = {
            'id': example['id'],
            'predicted_answer': predicted_answer_letter,
            'ground_truth_answer': ground_truth_letter,
            'is_correct': is_correct,
            'reasoning_trace': all_option_subgoals
        }
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(new_result) + '\n')
        all_results.append(new_result)

        print_and_write("\n" + "="*80)
        print_and_write("                Current Cumulative Results:")
        print_and_write("="*80 + "\n")

        total_processed = len(all_results)
        if total_processed > 0:
            total_correct = sum(1 for res in all_results if res['is_correct'])
            accuracy = (total_correct / total_processed) * 100
            print_and_write(f"Total items in dataset: {len(raw_dataset)}")
            print_and_write(f"Total items processed so far: {total_processed}")
            print_and_write(f"Total correct predictions: {total_correct}")
            print_and_write(f"Cumulative Accuracy: {accuracy:.2f}%")
        else:
            print_and_write("No items were processed successfully. Accuracy could not be calculated.")

    print_and_write("\n" + "="*80)
    print_and_write("                 Run finished. Final Cumulative Results:")
    print_and_write("="*80 + "\n")
    
    total_processed = len(all_results)
    if total_processed > 0:
        total_correct = sum(1 for res in all_results if res['is_correct'])
        accuracy = (total_correct / total_processed) * 100
        print_and_write(f"Total items in dataset: {len(raw_dataset)}")
        print_and_write(f"Total items processed so far: {total_processed}")
        print_and_write(f"Total correct predictions: {total_correct}")
        print_and_write(f"Cumulative Accuracy: {accuracy:.2f}%")
    else:
        print_and_write("No items were processed successfully. Accuracy could not be calculated.")

def one_shot_test(data_id):
    args = parse_args()
    logic_programmer = LogicProgrammer(args)
    raw_dataset = logic_programmer.load_raw_dataset()
    print_and_write(f'=====solving problem {data_id}=====')
    print_and_write(f"ground truth answer: {raw_dataset[data_id]['answer']}")
    answer, subgoals, outline = logic_programmer.solve_logic_problem(raw_dataset[data_id])
    
    print_and_write(f"ground truth answer: {raw_dataset[data_id]['answer']}")
    print_and_write(f"predicted answer: {answer.lower()}")
    print_and_write(f"result: {raw_dataset[data_id]['answer']==answer.lower()}")

if __name__ == '__main__':
    main()