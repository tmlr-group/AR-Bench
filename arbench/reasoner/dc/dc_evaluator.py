import json
import os
import random
import re
import sys
import string
from collections import deque
from typing import List, Dict, Tuple, Optional, Any, Union
from tqdm import tqdm

from arbench.reasoner.dc.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_dc import (
    ANSWER_CHOICES,
    CHOICE_TO_INDEX,
    convert_initial_info_to_string,
    extract_answer_choice,
    format_choices,
    calculate_accuracy,
    is_valid_choice,
    choice_to_index,
    index_to_choice,
    extract_reasoning,
    CrimeDetectionSearchTreeNode
)
from fire import Fire
from dotenv import load_dotenv

load_dotenv()

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")

ANSWER_CHOICES = ["A", "B", "C", "D", "E"]
CHOICE_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "": ""}

METHOD_DICT = {
        "zero_shot": propose_template,
        "few_shot": propose_template_with_1_shot,
        "few_shot_inst": propose_template_with_1_shot_inst,
        "proactive_cot": propose_template,  # placeholder, proactive flow uses dedicated prompts
    }


def extract_answer_choice(input_string: str) -> str:
    match = re.search(r"Answer:\s*(.*)", input_string)
    if not match:
        return ""
    
    try:
        answer = match.group(1)
        choices = re.findall(r"A|B|C|D|E", answer)
        return choices[-1] if choices else ""
    except:
        print(f"==> input_string: {input_string}, ==> answer: {answer}. ==> choices: {choices}")
        raise ValueError("Failed to extract answer choice")


# Custom parse_keypoints for CD game (different from utils version)
def parse_keypoints(input_string: str) -> List[int]:
    input_string = input_string.lower()
    matches = re.findall(r"hit point: ([\d, ]+)", input_string)
    matches = [item for item in matches if item.strip()]
    
    numbers = []
    try:
        if matches:
            numbers = [int(num.strip()) for num in matches[0].split(",")]
    except:
        print(input_string, matches)
        raise ValueError("Failed to parse keypoints")
        
    return numbers


# Aliases for consistency
format_keypoints = lambda keypoints: "\n".join(f"{i+1}. {key}" for i, key in enumerate(keypoints)) + "\n"


def remove_commas(string: str) -> str:
    return string.replace(",", "") if "," in string else string


def remove_punctuation_at_ends(input_str: str) -> str:
    punctuations = string.punctuation
    if input_str and input_str[0] in punctuations:
        input_str = input_str[1:]
    if input_str and input_str[-1] in punctuations:
        input_str = input_str[:-1]
    return input_str


def convert_tree_to_json(node: 'SearchTreeNode') -> List[Dict]:
    if not node:
        return []

    result = []
    queue = deque([(node, 0)])

    while queue:
        current_node, level = queue.popleft()

        result.append({
            "name": current_node.suspect,
            "question": current_node.question,
            "feedback": current_node.answer,
            "level": level,
            "value": current_node.value,
            "record": current_node.step_record,
            "chosen": current_node.chosen,
        })

        for child in current_node.children:
            queue.append((child, level + 1))

    return result


class SearchTreeNode:
    
    def __init__(self, parent: Optional['SearchTreeNode'] = None):
        # Basic attributes
        self.suspect = ""  # Name of suspect being questioned
        self.question = ""  # Question asked to suspect
        self.answer = ""  # Answer received from suspect
        self.value = -1  # Node evaluation score
        self.children: List['SearchTreeNode'] = []
        self.parent = parent
        self.depth = 0
        self.total_value = 0  # Not used in tot search
        self.visits = 0  # Not used in tot search
        self.chosen = False

        # Output attributes for logging
        self.input_token = 0
        self.output_token = 0
        self.question_hit_cnt = 0
        self.right_question_list: List[str] = []
        self.matched_keyquestion_set: set = set()

        # Attributes for step sampling
        self.choice_suspect_prompt = ""
        self.ask_prompt = ""
        self.step_record: List[Dict] = []

    def add_child(self, child_node: 'SearchTreeNode') -> None:
        child_node.depth = self.depth + 1
        child_node.question_hit_cnt = self.question_hit_cnt
        child_node.right_question_list = self.right_question_list
        child_node.matched_keyquestion_set = self.matched_keyquestion_set
        self.children.append(child_node)

    def get_conversation_record(self) -> List[Dict]:
        record = []
        current_node = self
        
        # Traverse back to root, collecting conversation records
        while current_node.parent and current_node.parent.depth > 0:
            parent_info = {
                "suspect": current_node.parent.suspect,
                "question": current_node.parent.question,
                "feedback": current_node.parent.answer,
            }
            record.insert(0, parent_info)
            current_node = current_node.parent
            
        return record

    def display_node_info(self) -> None:
        print("Suspect:", self.suspect)
        print("Question:", self.question)
        print("Answer:", self.answer)
        print("Value:", self.value)
        print("Input Token:", self.input_token)
        print("Output Token:", self.output_token)
        print("Visits:", self.visits)
        print("Depth:", self.depth)
        print("Total Value:", self.total_value)
        print("Question Hit Count:", self.question_hit_cnt)
        print("Right Question List:", self.right_question_list)
        print("Matched Keyquestion Set:", self.matched_keyquestion_set)

    def calculate_token_sums(self) -> Tuple[int, int]:
        input_token_sum = 0
        output_token_sum = 0
        current_node = self

        while current_node:
            input_token_sum += current_node.input_token
            output_token_sum += current_node.output_token
            current_node = current_node.parent

        return input_token_sum, output_token_sum


def expand_node(
    node: SearchTreeNode,
    branch: int,
    current_round: int,
    max_round: int,
    init_info: str,
    suspect_name_str: str,
    suspects: List[Dict],
    key_question_dict: Dict[str, List[str]],
    model: str,
    response_model: str,
    policy_temperature: float,
    policy_top_p: float,
    response_temperature: float,
    response_top_p: float
) -> None:
    # Initialize evaluation agents for suspects with key questions
    keypoint_eval_agents = {
        item["name"]: [{
            "role": "system",
            "content": keypoint_hits_prompt.format(
                question=init_info,
                name=item["name"],
                answer=item["story"],
                keypoints=format_keypoints(item["key_question"]),
            ),
        }]
        for item in suspects
        if "key_question" in item
    }
    
    # Initialize response agents for role-playing
    response_agents = {
        item["name"]: [{
            "role": "system",
            "content": respond_template.format(
                name=item["name"], task=item["task"], story=item["story"]
            ),
        }]
        for item in suspects
    }

    idx = 0
    while idx < branch:
        # Select suspect
        choice_suspect_prompt = select_suspect_template.format(
            turn=current_round, suspect_names=suspect_name_str
        )
        choice_suspect_agent = [{"role": "user", "content": choice_suspect_prompt}]
        
        try:
            response = inference(choice_suspect_agent, model=model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            selected_suspect = response.choices[0].message.content
            selected_suspect = remove_punctuation_at_ends(selected_suspect)

            assert selected_suspect in response_agents.keys(), \
                f"{selected_suspect} is not in {response_agents.keys()}"
                
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except:
            choice_suspect_agent.append({"role": "assistant", "content": selected_suspect})
            choice_suspect_agent.append({"role": "user", "content": refine_select_suspect_prompt})
            continue

        # Only increment when valid suspect is selected
        idx += 1

        # Create new child node
        current_node = SearchTreeNode(node)
        node.add_child(current_node)

        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.suspect = selected_suspect
        
        # Get conversation record
        record = current_node.get_conversation_record()
        current_node.record = record

        # Format record for prompt
        record_str = ""
        for entity in record:
            record_str += (
                f"Question for {entity['suspect']}: {entity['question']} "
                f"Feedback: {entity['feedback']}\n"
            )

        # Generate question for suspect
        ask_agent = [{
            "role": "system",
            "content": propose_template.format(turn=max_round, background=init_info),
        }]
        
        ask_agent.append({
            "role": "user",
            "content": question_propose_prompt_searching.format(
                record=record_str, suspect=selected_suspect
            ),
        })
        
        response = inference(ask_agent, model=model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        ask_prompt = (
            f"system: {propose_template.format(turn=max_round, background=init_info)}\n"
            f"user: {question_propose_prompt.format(turn=current_round, record=record, suspect=selected_suspect)}"
        )
        question = response.choices[0].message.content

        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.question = question

        # Evaluate question quality if suspect has key questions
        if selected_suspect in keypoint_eval_agents:
            keypoint_eval_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(keypoint_eval_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            check_results = response.choices[0].message.content
            current_node.input_token += response.usage.prompt_tokens
            current_node.output_token += response.usage.completion_tokens

            numbers = parse_keypoints(check_results)
            current_node.value = len(numbers)

            if numbers:
                try:
                    current_node.question_hit_cnt += 1
                    current_node.right_question_list.append(question)
                    for num in numbers:
                        current_node.matched_keyquestion_set.add(
                            key_question_dict[selected_suspect][num - 1]
                        )
                except IndexError:
                    pass

        current_node.ask_prompt = ask_prompt
        current_node.choice_suspect_prompt = choice_suspect_prompt
        current_node.step_record = record

        # Get suspect response
        response_agents[selected_suspect].append({"role": "user", "content": question})
        response = inference(response_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        suspect_response = response.choices[0].message.content
        current_node.input_token += response.usage.prompt_tokens
        current_node.output_token += response.usage.completion_tokens
        current_node.answer = suspect_response

        response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})


def select_best_child_node(node: SearchTreeNode) -> Optional[SearchTreeNode]:
    if not node.children:
        return None

    max_value = max(child.value for child in node.children)
    max_value_children = [child for child in node.children if child.value == max_value]
    best_node = random.choice(max_value_children)
    best_node.chosen = True
    
    return best_node


def _run_tot_evaluation(
    dataset: List[Dict],
    logs: List[Dict], 
    output_path: str,
    policy_model: str,
    max_turn: int,
    branch: int,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float
) -> None:
    """Run evaluation using tot search method."""
    tree_logs = []
    tree_path = output_path.replace(".json", "") + "_tree.json"
    
    if os.path.exists(tree_path):
        with open(tree_path, "r", encoding="utf-8") as file:
            tree_logs = json.load(file)

    for i in tqdm(range(len(logs), len(dataset))):
        # Prepare case information
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]
        
        choice_str = ", ".join([
            f"{index}. {item['name']}"
            for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
        ])
        
        key_question_dict = {
            item["name"]: item["key_question"]
            for item in dataset[i]["suspects"]
            if "key_question" in item
        }
        
        suspect_name_str = ", ".join([
            item["name"] for item in dataset[i]["initial_information"]["suspect"]
        ])

        # Initialize search tree and perform tot search
        root = SearchTreeNode(None)
        current_node = root
        
        for j in range(max_turn):
            expand_node(
                node=current_node,
                branch=branch,
                current_round=j + 1,
                max_round=max_turn,
                init_info=init_info,
                suspect_name_str=suspect_name_str,
                suspects=dataset[i]["suspects"],
                key_question_dict=key_question_dict,
                model=policy_model,
                response_model=response_model,
                policy_temperature=policy_temperature,
                policy_top_p=policy_top_p,
                response_temperature=response_temperature,
                response_top_p=response_top_p
            )

            new_child_node = select_best_child_node(current_node)
            current_node = new_child_node

        # Get final prediction
        final_node = current_node
        record = final_node.get_conversation_record()
        record.append({
            "suspect": final_node.suspect,
            "question": final_node.question,
            "feedback": final_node.answer,
        })
        
        record_str = ""
        for entity in record:
            record_str += (
                f"Question for {entity['suspect']}: {entity['question']} "
                f"Feedback: {entity['feedback']}\n"
            )

        select_murderer_prompt = (
            propose_template.format(turn=max_turn, background=init_info) +
            select_murderer_template_searching.format(record=record_str, choice=choice_str)
        )
        
        select_murderer_agent = [{"role": "user", "content": select_murderer_prompt}]
        response = inference(select_murderer_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content

        get_result_input_token = response.usage.prompt_tokens
        get_result_output_token = response.usage.completion_tokens

        pred = CHOICE_TO_INDEX[extract_answer_choice(raw_pred).strip()]
        
        input_token, output_token = final_node.calculate_token_sums()
        
        logs.append({
            "idx": i,
            "raw_pred": raw_pred,
            "pred": pred,
            "label": label,
            "round": max_turn,
            "input_token_sum": input_token + get_result_input_token,
            "output_token_sum": output_token + get_result_output_token,
            "correctness": pred == label,
            "record": record,
        })
        
        # Save logs
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=4)

        # Save tree
        tree = convert_tree_to_json(root)
        tree_logs.append(tree)
        with open(tree_path, "w", encoding="utf-8") as file:
            json.dump(tree_logs, file, indent=4)



def _run_traditional_evaluation(
    method: str,
    dataset: List[Dict],
    logs: List[Dict],
    output_path: str,
    policy_model: str,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float,
    max_turn: int,
) -> None:
    """Run evaluation using traditional prompting methods."""

    for i in tqdm(range(len(logs), len(dataset))):
        # Prepare case information
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]

        total_input_token, total_output_token = 0, 0
        choice_str = ", ".join([
            f"{index}. {item['name']}"
            for index, item in zip(ANSWER_CHOICES, dataset[i]["initial_information"]["suspect"])
        ])

        # Initialize conversation
        propose_agent = [{
            "role": "system",
            "content": METHOD_DICT[method].format(background=init_info, turn=max_turn),
        }]

        # Initialize response agents for suspects
        response_agents = {
            item["name"]: [{
                "role": "system",
                "content": respond_template.format(
                    name=item["name"], task=item["task"], story=item["story"]
                ),
            }]
            for item in dataset[i]["suspects"]
        }
        
        suspect_name_str = ", ".join([
            item["name"] for item in dataset[i]["initial_information"]["suspect"]
        ])

        # Question-answer loop
        for turn in range(max_turn):
            # Select suspect
            try:
                propose_agent.append({
                    "role": "user",
                    "content": select_suspect_template.format(
                        turn=turn + 1, suspect_names=suspect_name_str
                    ),
                })
                
                response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                selected_suspect = response.choices[0].message.content
                total_input_token += response.usage.prompt_tokens
                total_output_token += response.usage.completion_tokens
                selected_suspect = remove_punctuation_at_ends(selected_suspect)

                assert selected_suspect in response_agents.keys(), \
                    f"{selected_suspect} is not in {response_agents.keys()}"
                    
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except:
                propose_agent.append({"role": "assistant", "content": selected_suspect})
                propose_agent.append({"role": "user", "content": refine_select_suspect_prompt})
                continue

            # Ask question to suspect
            propose_agent.append({"role": "assistant", "content": selected_suspect})
            propose_agent.append({
                "role": "user",
                "content": question_propose_prompt.format(turn=turn + 1),
            })
            
            response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            question = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            # Get suspect response
            response_agents[selected_suspect].append({"role": "user", "content": question})
            response = inference(response_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            suspect_response = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            response_agents[selected_suspect].append({"role": "assistant", "content": suspect_response})

            propose_agent.append({"role": "assistant", "content": question})
            propose_agent.append({"role": "user", "content": suspect_response})

        # Get final prediction
        propose_agent.append({
            "role": "user",
            "content": select_murderer_template.format(choice=choice_str),
        })
        
        response = inference(propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        raw_pred = response.choices[0].message.content
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens

        pred = CHOICE_TO_INDEX[extract_answer_choice(raw_pred).strip()]
        propose_agent.append({"role": "assistant", "content": raw_pred})

        logs.append({
            "idx": i,
            "record": propose_agent,
            "respond_conversation": [
                {"name": key, "conversation": value}
                for key, value in response_agents.items()
            ],
            "pred": pred,
            "label": label,
            "round": max_turn,
            "correctness": pred == label,
        })
        
        with open(output_path, "w", encoding="utf-8") as file:
            json.dump(logs, file, indent=4)

def _format_initial_info(init_info_dict: Dict[str, Any]) -> str:
    time_v = init_info_dict["time"]
    location = init_info_dict["location"]
    victim_name = init_info_dict["victim"]["name"]
    victim_intro = init_info_dict["victim"]["introduction"]
    cause_of_death = init_info_dict["victim"]["cause_of_death"]
    murder_weapon = init_info_dict["victim"]["murder_weapon"]
    suspect_list = [
        f"- suspect {i+1}:\n  - name: {item['name']}\n  - introduction: {item['introduction']}\n"
        for i, item in enumerate(init_info_dict["suspect"])
    ]
    suspect_info = "".join(suspect_list)
    init_info_str = (
        f"\ntime: {time_v}\nlocation: {location}\nvictim:\n- name: {victim_name}\n- introduction: {victim_intro}\n- cause of death: {cause_of_death}\n- murder weapon: {murder_weapon}\nThe investigation focuses on five suspects, one of whom is the true murderer:\n{suspect_info}"
    )
    return init_info_str


def _place_question_records(question_record: List[Dict[str, str]]) -> str:
    record_str = ""
    for idx, record in enumerate(question_record):
        record_str += (
            f"Turn {idx+1}: Question for {record['suspect']}: {record['question']} "
            f"Feedback: {record['feedback']}\n"
        )
    return record_str


def _extract_selected_suspect(text: str) -> str:
    match = re.search(r"Selected suspect:\s*(.+?)(?:$|\n)", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_choice_letter(text: str) -> str:
    match = re.search(r"Answer:\s*(.*)", text)
    if not match:
        return ""
    ans = match.group(1)
    letters = re.findall(r"A|B|C|D|E", ans)
    return letters[-1] if letters else ""


def _run_proactive_cot_evaluation(
    dataset: List[Dict],
    logs: List[Dict],
    output_path: str,
    policy_model: str,
    response_model: str,
    max_turn: int,
    policy_temperature: float,
    policy_top_p: float,
    response_temperature: float,
    response_top_p: float,
) -> None:
    for i in tqdm(range(len(logs), len(dataset))):
        init_info_dict = dataset[i]["initial_information"]
        init_info = _format_initial_info(init_info_dict)
        label = dataset[i]["label"]

        # build response agents for suspects
        response_agents = {
            item["name"]: [{
                "role": "system",
                "content": respond_template.format(
                    name=item["name"], task=item["task"], story=item["story"]
                ),
            }]
            for item in dataset[i]["suspects"]
        }

        question_record: List[Dict[str, str]] = []
        suspect_names = [s["name"] for s in init_info_dict["suspect"]]
        suspect_list_str = ", ".join(suspect_names)
        total_input_token, total_output_token = 0, 0

        for turn in range(1, max_turn + 1):
            history_str = _place_question_records(question_record)

            # select suspect
            select_prompt = proactive_cot_select_suspect_template.format(
                init_info=init_info,
                history=history_str,
                turn=turn,
                max_turn=max_turn,
                suspect_list=suspect_list_str,
            )
            select_messages = [
                {"role": "system", "content": proactive_cot_system_prompt},
                {"role": "user", "content": select_prompt},
            ]
            sel_resp = inference(select_messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            sel_content = sel_resp.choices[0].message.content
            total_input_token += sel_resp.usage.prompt_tokens
            total_output_token += sel_resp.usage.completion_tokens
            selected_suspect = _extract_selected_suspect(sel_content)
            if not selected_suspect or selected_suspect not in response_agents:
                # fallback to closest match by substring, else random
                matched = [name for name in response_agents.keys() if selected_suspect and (selected_suspect.lower() in name.lower() or name.lower() in selected_suspect.lower())]
                selected_suspect = matched[0] if matched else random.choice(list(response_agents.keys()))

            # ask question for selected suspect
            history_str = _place_question_records(question_record)
            ask_prompt = proactive_cot_question_template.format(
                init_info=init_info,
                history=history_str,
                turn=turn,
                max_turn=max_turn,
                selected_suspect=selected_suspect,
            )
            ask_messages = [
                {"role": "system", "content": proactive_cot_system_prompt},
                {"role": "user", "content": ask_prompt},
            ]
            ask_resp = inference(ask_messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            ask_content = ask_resp.choices[0].message.content

            total_input_token += ask_resp.usage.prompt_tokens
            total_output_token += ask_resp.usage.completion_tokens

            q_match = re.search(r"Question:\s*(.+?)(?:$|\n)", ask_content, re.DOTALL)
            if q_match:
                question = q_match.group(1).strip()
            else:
                parts = ask_content.strip().split("\n\n")
                question = parts[-1].strip() if parts else ask_content.strip()

            # get suspect response
            response_agents[selected_suspect].append({"role": "user", "content": question})
            resp = inference(response_agents[selected_suspect], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            feedback = resp.choices[0].message.content
            response_agents[selected_suspect].append({"role": "assistant", "content": feedback})

            total_input_token += resp.usage.prompt_tokens
            total_output_token += resp.usage.completion_tokens


            question_record.append({
                "suspect": selected_suspect,
                "question": question,
                "feedback": feedback,
            })

        # final decision
        index_list = ["A", "B", "C", "D", "E"]
        choice_str = ", ".join([f"{idx}. {item['name']}" for idx, item in zip(index_list, init_info_dict["suspect"])])

        final_prompt = (
            f"Initial case information:\n{init_info}\n\n"
            f"Interrogation record:\n{_place_question_records(question_record)}\n\n"
            f"Based on your investigation, who do you believe is the true murderer from these suspects?\n{choice_str}\n\n"
            f"Provide your reasoning and then your final answer in the format:\n"
            f"Reason: [Your detailed analysis]\nAnswer: [A, B, C, D, or E]"
        )
        final_messages = [
            {"role": "system", "content": proactive_cot_system_prompt_conclusion},
            {"role": "user", "content": final_prompt},
        ]
        final_resp = inference(final_messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        reasoning = final_resp.choices[0].message.content

        total_input_token += final_resp.usage.prompt_tokens
        total_output_token += final_resp.usage.completion_tokens

        ans = _extract_choice_letter(reasoning).strip()

        pred = ""
        success = False
        attempts = 3
        while not success and attempts > 0 and ans:
            attempts -= 1
            pred = CHOICE_TO_INDEX.get(ans, "")
            if pred in [0, 1, 2, 3, 4]:
                success = True
                break

            follow_up = "Based on your analysis above, who is the murderer? Please answer only with the letter A, B, C, D, or E."
            final_messages.append({"role": "assistant", "content": reasoning})
            final_messages.append({"role": "user", "content": follow_up})
            retry_resp = inference(final_messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            retry_text = retry_resp.choices[0].message.content

            total_input_token += retry_resp.usage.prompt_tokens
            total_output_token += retry_resp.usage.completion_tokens

            ans = _extract_choice_letter(retry_text).strip()
            if not ans:
                m = re.search(r"\b[A-E]\b", retry_text)
                if m:
                    ans = m.group(0)

        if not success:
            pred = random.randint(0, 4)

        logs.append({
            "idx": i,
            "pred": pred,
            "label": label,
            "record": question_record,
            "reasoning": reasoning,
            "round": len(question_record),
            "correctness": pred == label,
            "input_token": total_input_token,
            "output_token": total_output_token,
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)


def _uot_extract_question(question: str) -> Tuple[Optional[str], Optional[str]]:
    pattern = r"Question for ([^:]+): (.+)"
    match = re.match(pattern, question)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None


def _uot_suspect_mapper(selected_suspect: str, respond_agents: Dict[str, List[Dict]]) -> Optional[str]:
    for name in respond_agents.keys():
        if selected_suspect and (selected_suspect.lower() in name.lower() or name.lower() in selected_suspect.lower()):
            return name
    return None


def _uot_generate_candidate_questions(
    policy_model: str,
    init_info: str,
    qa_record: List[Dict[str, str]],
    policy_temperature: float,
    policy_top_p: float,
) -> Tuple[List[str], int, int]:
    qa_record_str = "\n".join([
        f"Turn {i}: Question for {r['suspect']}: {r['question']} Feedback: {r['feedback']}" for i, r in enumerate(qa_record)
    ])
    prompt_text = uot_generate_questions_template.format(
        init_info=init_info,
        qa_record=qa_record_str,
        current_suspects="",
    )
    messages = [{"role": "user", "content": prompt_text}]
    resp = inference(messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
    content = resp.choices[0].message.content
    input_tok = resp.usage.prompt_tokens
    output_tok = resp.usage.completion_tokens
    questions: List[str] = []
    for m in re.finditer(r"Q\d+:\s*(.*?)(?=Q\d+:|$)", content, re.DOTALL):
        q = m.group(1).strip()
        if q:
            questions.append(q)
    return questions, input_tok, output_tok


def _uot_update_answer_set(
    policy_model: str,
    init_info: str,
    question: str,
    feedback: str,
    answer_set: List[str],
    policy_temperature: float,
    policy_top_p: float,
) -> Tuple[List[str], int, int]:
    answer_set_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answer_set)])
    prompt_text = uot_update_answer_set_template.format(
        init_info=init_info,
        answer_set=answer_set_str,
        question=question,
        feedback=feedback,
    )
    messages = [{"role": "user", "content": prompt_text}]
    resp = inference(messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
    content = resp.choices[0].message.content
    input_tok = resp.usage.prompt_tokens
    output_tok = resp.usage.completion_tokens
    numbers = re.findall(r"\d+", content)
    valid_idx = [int(n) - 1 for n in numbers if int(n) <= len(answer_set)]
    reduced = [answer_set[i] for i in valid_idx]
    return reduced, input_tok, output_tok


def _uot_estimate_answer_set_reduction(
    policy_model: str,
    response_model: str,
    init_info: str,
    question: str,
    answer_set: List[str],
    respond_agents: Dict[str, List[Dict]],
    policy_temperature: float,
    policy_top_p: float,
    response_temperature: float,
    response_top_p: float,
    simulate_depth: int,
) -> Tuple[Dict[str, int], int, int]:
    total_in, total_out = 0, 0
    if simulate_depth <= 0:
        sel_name, q_text = _uot_extract_question(question)
        sel_name = _uot_suspect_mapper(sel_name, respond_agents)
        if not sel_name:
            return {}, 0, 0
        respond_agents[sel_name].append({"role": "user", "content": q_text})
        resp = inference(respond_agents[sel_name], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        feedback = resp.choices[0].message.content.strip()
        total_in += resp.usage.prompt_tokens
        total_out += resp.usage.completion_tokens
        respond_agents[sel_name].append({"role": "assistant", "content": feedback})
        reduced, ui, uo = _uot_update_answer_set(policy_model, init_info, question, feedback, answer_set, policy_temperature, policy_top_p)
        total_in += ui
        total_out += uo
        return {feedback: len(answer_set) - len(reduced)}, total_in, total_out

    # simulate deeper by sampling next-step questions and averaging
    sel_name, q_text = _uot_extract_question(question)
    sel_name = _uot_suspect_mapper(sel_name, respond_agents)
    if not sel_name:
        return {}, 0, 0
    respond_agents[sel_name].append({"role": "user", "content": q_text})
    resp = inference(respond_agents[sel_name], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
    feedback = resp.choices[0].message.content.strip()
    total_in += resp.usage.prompt_tokens
    total_out += resp.usage.completion_tokens
    respond_agents[sel_name].append({"role": "assistant", "content": feedback})
    reduced, ui, uo = _uot_update_answer_set(policy_model, init_info, question, feedback, answer_set, policy_temperature, policy_top_p)
    total_in += ui
    total_out += uo
    if len(reduced) <= 1 or simulate_depth == 1:
        return {feedback: len(answer_set) - len(reduced)}, total_in, total_out
    # generate next candidate questions on reduced set (we reuse template without explicit list)
    cand, gi, go = _uot_generate_candidate_questions(policy_model, init_info, [], policy_temperature, policy_top_p)
    total_in += gi
    total_out += go
    best_next = None
    best_score = -1.0
    for q in cand:
        red_map, ei, eo = _uot_estimate_answer_set_reduction(
            policy_model, response_model, init_info, q, reduced, respond_agents,
            policy_temperature, policy_top_p, response_temperature, response_top_p, 0
        )
        total_in += ei
        total_out += eo
        if red_map:
            exp = sum(red_map.values()) / len(red_map)
            if exp > best_score:
                best_score = exp
                best_next = q
    if best_next is None:
        return {feedback: len(answer_set) - len(reduced)}, total_in, total_out
    # one more recursive rollout
    deeper_map, di, do = _uot_estimate_answer_set_reduction(
        policy_model, response_model, init_info, best_next, reduced, respond_agents,
        policy_temperature, policy_top_p, response_temperature, response_top_p, simulate_depth - 1
    )
    total_in += di
    total_out += do
    if deeper_map:
        exp_next = sum(deeper_map.values()) / len(deeper_map)
        total = (len(answer_set) - len(reduced)) + exp_next
        return {feedback: int(total)}, total_in, total_out
    return {feedback: len(answer_set) - len(reduced)}, total_in, total_out


def _run_uot_evaluation(
    dataset: List[Dict],
    logs: List[Dict],
    output_path: str,
    policy_model: str,
    response_model: str,
    max_turn: int,
    policy_temperature: float,
    policy_top_p: float,
    response_temperature: float,
    response_top_p: float,
    simulate_depth: int,
) -> None:
    for i in tqdm(range(len(logs), len(dataset))):
        init_info = convert_initial_info_to_string(dataset[i]["initial_information"])
        label = dataset[i]["label"]
        index_list = ["A", "B", "C", "D", "E"]
        choice_str = ", ".join([f"{idx}. {item['name']}" for idx, item in zip(index_list, dataset[i]["initial_information"]["suspect"])])

        respond_agents = {
            item["name"]: [{
                "role": "system",
                "content": respond_template.format(name=item["name"], task=item["task"], story=item["story"]),
            }]
            for item in dataset[i]["suspects"]
        }

        answer_set = [item["name"] for item in dataset[i]["initial_information"]["suspect"]]
        qa_record: List[Dict[str, str]] = []
        failed_turns = 0
        total_input_token, total_output_token = 0, 0

        turn_used = 0
        for turn in range(max_turn):
            print(f"Turn {turn + 1}/{max_turn}")
            turn_used = turn + 1
            # generate candidate questions (retry up to 3 times to get 3 valid)
            valid_questions: List[str] = []
            retries = 0
            while retries < 3 and len(valid_questions) < 3:
                cand, gi, go = _uot_generate_candidate_questions(policy_model, init_info, qa_record, policy_temperature, policy_top_p)
                total_input_token += gi
                total_output_token += go
                for q in cand:
                    sel, q_text = _uot_extract_question(q)
                    if not (sel and q_text):
                        continue
                    mapped = _uot_suspect_mapper(sel, respond_agents)
                    if mapped:
                        valid_questions.append(q)
                retries += 1

            if not valid_questions:
                failed_turns += 1
                qa_record.append({"suspect": "system", "question": "failed", "feedback": "Failed to generate valid questions"})
                continue

            # score questions by expected reduction
            scored = []
            for q in valid_questions:
                red_map, ei, eo = _uot_estimate_answer_set_reduction(
                    policy_model, response_model, init_info, q, answer_set, respond_agents,
                    policy_temperature, policy_top_p, response_temperature, response_top_p, simulate_depth
                )
                total_input_token += ei
                total_output_token += eo
                if red_map:
                    exp = sum(red_map.values()) / len(red_map)
                    scored.append((q, exp, red_map))

            if not scored:
                failed_turns += 1
                qa_record.append({"suspect": "system", "question": "failed", "feedback": "Failed to evaluate any questions"})
                continue

            scored.sort(key=lambda x: x[1], reverse=True)
            best_q = scored[0][0].split("\n")[0]
            sel_name, q_text = _uot_extract_question(best_q)
            sel_name = _uot_suspect_mapper(sel_name, respond_agents)

            # get suspect feedback
            respond_agents[sel_name].append({"role": "user", "content": q_text})
            resp = inference(respond_agents[sel_name], model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            feedback = resp.choices[0].message.content.strip()
            total_input_token += resp.usage.prompt_tokens
            total_output_token += resp.usage.completion_tokens
            respond_agents[sel_name].append({"role": "assistant", "content": feedback})

            qa_record.append({"suspect": sel_name, "question": q_text, "feedback": feedback})
            answer_set, ui, uo = _uot_update_answer_set(policy_model, init_info, best_q, feedback, answer_set, policy_temperature, policy_top_p)
            total_input_token += ui
            total_output_token += uo
            print(f"Current answer set size: {len(answer_set)}")
            if len(answer_set) <= 1:
                break

        # final decision
        if len(answer_set) == 1:
            final_name = answer_set[0]
            pred = ""
            for idx, item in enumerate(dataset[i]["initial_information"]["suspect"]):
                if item["name"] == final_name:
                    pred = CHOICE_TO_INDEX[index_list[idx]]
                    break
        else:
            qa_record_str = "\n".join([
                f"Turn {k}: Question for {r['suspect']}: {r['question']} Feedback: {r['feedback']}"
                for k, r in enumerate(qa_record)
            ])
            final_prompt = uot_conclusion_template.format(
                init_info=init_info,
                qa_record=qa_record_str,
                choice=choice_str,
            )
            messages = [{"role": "user", "content": final_prompt}]
            successful = False
            attempts = 20
            pred = ""
            while not successful and attempts > 0:
                attempts -= 1
                r = inference(messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                total_input_token += r.usage.prompt_tokens
                total_output_token += r.usage.completion_tokens
                reasoning = r.choices[0].message.content
                ans = extract_answer_choice(reasoning).strip()
                pred = CHOICE_TO_INDEX.get(ans, "")
                if pred in [0, 1, 2, 3, 4]:
                    successful = True
            if not successful:
                qa_record.append({"suspect": "system", "question": "failed", "feedback": "Failed to generate final answer"})

        logs.append({
            "idx": i,
            "pred": pred,
            "label": label,
            "record": qa_record,
            "round": turn_used,
            "failed_turns": failed_turns,
            "final_answer_set_size": len(answer_set),
            "correctness": pred == label,
            "input_token": total_input_token,
            "output_token": total_output_token,
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=4)

def main(
    method: str, 
    data_path: str, 
    output_path: str, 
    policy_model: str,
    response_model: str,
    max_turn: int = 25, 
    branch: int = 3,
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7,
    simulate_depth: int = 3,
) -> None:
    with open(data_path, "r") as file:
        dataset = json.load(file)

    # Load existing logs if available
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Policy Model: {policy_model}, Response Model: {response_model}")
    print(f"Policy Temperature: {policy_temperature}, Policy Top_p: {policy_top_p}")
    print(f"Response Temperature: {response_temperature}, Response Top_p: {response_top_p}")
    print(f"Max turn: {max_turn}, Method: {method}, Simulate Depth: {simulate_depth}")

    if method == "tot":
        _run_tot_evaluation(dataset, logs, output_path, policy_model, max_turn, branch, 
                              policy_temperature, policy_top_p, response_model,
                              response_temperature, response_top_p)
    elif method == "proactive_cot":
        _run_proactive_cot_evaluation(
            dataset=dataset,
            logs=logs,
            output_path=output_path,
            policy_model=policy_model,
            response_model=response_model,
            max_turn=max_turn,
            policy_temperature=policy_temperature,
            policy_top_p=policy_top_p,
            response_temperature=response_temperature,
            response_top_p=response_top_p,
        )
    elif method == "uot":
        _run_uot_evaluation(
            dataset=dataset,
            logs=logs,
            output_path=output_path,
            policy_model=policy_model,
            response_model=response_model,
            max_turn=max_turn,
            policy_temperature=policy_temperature,
            policy_top_p=policy_top_p,
            response_temperature=response_temperature,
            response_top_p=response_top_p,
            simulate_depth=simulate_depth,
        )
    else:
        _run_traditional_evaluation(method, dataset, logs, output_path, policy_model,
                                   policy_temperature, policy_top_p, response_model,
                                   response_temperature, response_top_p, max_turn)

if __name__ == "__main__":
    Fire(main)
