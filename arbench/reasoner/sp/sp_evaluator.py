import json
import os
import random
import sys
import re
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional, Any, Union

from fire import Fire
from tqdm import tqdm

from arbench.reasoner.sp.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_sp import (
    f1_score,
    place_keypoints,
    parse_match_info,
    extract_questioner_respond
)
from dotenv import load_dotenv

load_dotenv()

# Constants
RESPONSE_TEMPLATE = system_prompt_with_2shots
VALID_ANSWERS = {"Yes", "No", "Unknown"}


# Method configuration
METHOD_DICT = {
    "zero_shot": propose_template,
    "few_shot": propose_template_with_1_shot,
    "few_shot_inst": propose_template_with_1_shot_inst,
    "proactive_cot": propose_template,  # placeholder; proactive flow uses dedicated prompts
    "uot": propose_template,  # placeholder; UoT flow uses dedicated prompts
}

POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")
RESPONSE_API_KEY = os.getenv("RESPONSE_API_KEY")
RESPONSE_BASE_URL = os.getenv("RESPONSE_BASE_URL")


# Aliases for consistency with existing code
calculate_f1_score = f1_score
format_keypoints = place_keypoints


class SearchTreeNode:
    def __init__(self, parent: Optional['SearchTreeNode'] = None):
        self.question = ""
        self.answer = ""
        self.value = -1
        self.record: List[Dict] = [] if not parent else parent.record.copy()
        self.children: List['SearchTreeNode'] = []
        self.input_token = 0
        self.output_token = 0
        self.visits = 0  # Not used in current search strategy
        self.parent = parent
        self.total_value = 0  # Not used in current search strategy
        self.chosen = False
        self.log_info: Dict = {}

    def add_child(self, child_node: 'SearchTreeNode') -> None:
        self.children.append(child_node)

    def get_record(self) -> str:
        record = ""
        current_node = self
        
        while current_node.parent:
            record = (
                current_node.parent.question + " A: " + current_node.parent.answer + "\n" + record
            )
            current_node = current_node.parent
            
        return record.strip()

    def display_all_values(self) -> None:
        print("Question:", self.question)
        print("Answer:", self.answer)
        print("Value:", self.value)
        print("Input Token:", self.input_token)
        print("Output Token:", self.output_token)
        print("Visits:", self.visits)
        print("Total Value:", self.total_value)


def expand_node(
    node: SearchTreeNode,
    depth: int = 0,
    max_depth: int = 25,
    respond_template: str = RESPONSE_TEMPLATE,
    surface: str = "",
    bottom: str = "",
    log_info: Dict = {},
    branch: int = 3,
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7,
) -> List[SearchTreeNode]:
    child_nodes = []
    policy_model = log_info["policy_model"]
    response_model = log_info["response_model"]

    for _ in range(branch):
        # Initialize child node
        new_node = SearchTreeNode(node)
        
        # Initialize agents to prevent interference
        respond_agent = [{
            "role": "system",
            "content": respond_template.format(question=surface, answer=bottom),
        }]
        
        input_token = 0
        output_token = 0

        # Get conversation record from ancestor nodes
        record = node.get_record() if node else ""
        
        # Generate question
        question_agent = [{
            "role": "user",
            "content": propose_template_Node.format(
                max_depth=str(max_depth),
                remain=str(max_depth - depth),
                question=surface,
                record=record,
            ),
        }]

        # Ask model to generate question
        response = inference(messages=question_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        question = response.choices[0].message.content

        new_node.question = question
        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        # Validate question format
        if "Q" not in question:
            print("Question format error")

        # Get answer using evaluation model
        respond_agent.append({"role": "user", "content": question})
        response = inference(messages=respond_agent, model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
        answer = response.choices[0].message.content.strip().strip(".")
        
        
        # Validate answer format
        if answer not in VALID_ANSWERS:
            print(f"Answer format error, it should be one of {VALID_ANSWERS}")
            


        new_node.answer = answer
        input_token += response.usage.prompt_tokens
        output_token += response.usage.completion_tokens

        # Update node record
        new_node.record.append({"question": question, "answer": answer})
        new_node.input_token = input_token
        new_node.output_token = output_token

        child_nodes.append(new_node)
        node.children.append(new_node)
        
        log_info["depth"] = depth
        new_node.log_info = log_info

    return child_nodes


def select_best_child_node(parent_node: SearchTreeNode) -> Optional[SearchTreeNode]:
    children = parent_node.children
    if not children:
        return None

    # Select nodes with maximum value
    max_value = max(child.value for child in children)
    candidates = [child for child in children if child.value == max_value]

    if len(candidates) == 1:
        return candidates[0]

    # Apply answer priority if multiple candidates have same max value
    answer_priority = {"Yes": 2, "No": 1, "Unknown": 0}
    best_candidate = None
    best_priority = -1

    for candidate in candidates:
        priority = answer_priority.get(candidate.answer, 0)
        if priority > best_priority:
            best_candidate = candidate
            best_priority = priority
        elif priority == best_priority:
            best_candidate = random.choice([best_candidate, candidate])

    return best_candidate


def get_final_prediction(node: SearchTreeNode, policy_model: str, temperature: float = 0.7, top_p: float = 0.7) -> Tuple[str, int, int, str]:
    pred_input_token = 0
    pred_output_token = 0
    
    
    record = node.get_record() if node else ""
    response = inference(
        [{"role": "user", "content": get_answer_prompt_Node.format(record=record)}],
        model=policy_model,
        temperature=temperature,
        top_p=top_p,
        api_key=POLICY_API_KEY,
        base_url=POLICY_BASE_URL
    )
    
    prediction = response.choices[0].message.content
    pred_input_token += response.usage.prompt_tokens
    pred_output_token += response.usage.completion_tokens

    return prediction, pred_input_token, pred_output_token, record


def evaluate_prediction(
    prediction: str, 
    surface: str, 
    bottom: str, 
    keypoints: List[str],
    response_model: str,
    temperature: float = 0.7,
    top_p: float = 0.7
) -> Tuple[str, int, int, Optional[str], int]:
    
    evaluate_input_token = 0
    evaluate_output_token = 0
    
    prompt = guess_eval_prompt.format(
        question=surface, 
        answer=bottom, 
        keypoints=format_keypoints(keypoints), 
        pred=prediction
    )
    
    response = inference([{"role": "user", "content": prompt}], model=response_model, temperature=temperature, top_p=top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
    eval_result = response.choices[0].message.content
    evaluate_input_token += response.usage.prompt_tokens
    evaluate_output_token += response.usage.completion_tokens
    
    match_point, accuracy_count = parse_match_info(eval_result)
    return eval_result, evaluate_input_token, evaluate_output_token, match_point, accuracy_count


def convert_tree_to_json(node: SearchTreeNode) -> List[Dict]:
    if not node:
        return []

    result = []
    queue = deque([(node, 0)])

    while queue:
        current_node, level = queue.popleft()

        result.append({
            "question": current_node.question,
            "answer": current_node.answer,
            "level": len(current_node.record),
            "chosen": current_node.chosen,
        })

        for child in current_node.children:
            queue.append((child, level + 1))

    return result


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
    tree_logs = []
    tree_logs_path = output_path.replace(".json", "_tree.json")
    
    if os.path.exists(tree_logs_path):
        with open(tree_logs_path, "r") as file:
            tree_logs = json.load(file)

    assert len(logs) == len(tree_logs), "Logs and tree_logs should have the same length for tot method"

    for i in tqdm(range(len(logs), len(dataset))):
        log_info = {
            "policy_model": policy_model,
            "response_model": response_model
        }

        surface, bottom = dataset[i]["surface"], dataset[i]["bottom"]
        root = SearchTreeNode(None)
        root.question = surface
        current_node = root
        
        # Perform tot search
        for j in range(max_turn):
            current_node.chosen = True
            child_list = expand_node(
                current_node,
                depth=j,
                max_depth=max_turn,
                respond_template=RESPONSE_TEMPLATE,
                surface=surface,
                bottom=bottom,
                log_info=log_info,
                branch=branch,
                policy_temperature=policy_temperature,
                policy_top_p=policy_top_p,
                response_temperature=response_temperature,
                response_top_p=response_top_p,
            )
            

            best_node = select_best_child_node(current_node)
            current_node = best_node


        final_node = current_node
        prediction, pred_input_token, pred_output_token, record = get_final_prediction(
            final_node, log_info["policy_model"], policy_temperature, policy_top_p
        )

        
        logs.append({
            "idx": i,
            "question": surface,
            "answer": bottom,
            "pred": prediction,
            "f1_score_char": calculate_f1_score(prediction, bottom),
            "f1_score_word": calculate_f1_score(prediction.split(), bottom.split()),
            "round": max_turn,
            "record": final_node.record,
        })

        tree_logs.append(convert_tree_to_json(root))

        # Save logs
        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)

        with open(tree_logs_path, "w") as f:
            json.dump(tree_logs, f, indent=4)


def _run_traditional_evaluation(
    method: str,
    dataset: List[Dict], 
    logs: List[Dict], 
    output_path: str, 
    policy_model: str, 
    max_turn: int,
    policy_temperature: float,
    policy_top_p: float,
    response_model: str,
    response_temperature: float,
    response_top_p: float
) -> None:

    for i in tqdm(range(len(logs), len(dataset))):
        surface, bottom, keypoints = (
            dataset[i]["surface"],
            dataset[i]["bottom"],
            dataset[i]["key_question"],
        )
        total_input_token, total_output_token = 0, 0

        # Initialize conversation
        propose_agent = [{
            "role": "system",
            "content": METHOD_DICT[method].format(turn=max_turn, question=surface),
        }]

        # Question-answer loop
        for turn in range(max_turn):
            respond_agent = [{
                "role": "system",
                "content": RESPONSE_TEMPLATE.format(question=surface, answer=bottom),
            }]
            
            # Generate question
            propose_agent.append({
                "role": "user",
                "content": f"Turn {turn+1}: please propose your next question.",
            })

            

            response = inference(messages=propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            question = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
           # Get answer
            respond_agent.append({"role": "user", "content": question})
            response = inference(messages=respond_agent, model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)

            answer = response.choices[0].message.content.strip().strip(".")
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
          
            # Update conversation
            propose_agent.append({"role": "assistant", "content": question})
            propose_agent.append({"role": "user", "content": answer})

        # Get final prediction
        propose_agent.append({"role": "user", "content": get_answer_prompt})
        response = inference(messages=propose_agent, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        prediction = response.choices[0].message.content
        
        propose_agent.append({"role": "assistant", "content": prediction})
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens

        logs.append({
            "idx": i,
            "question": surface,
            "answer": bottom,
            "pred": prediction,
            "f1_score_char": calculate_f1_score(prediction, bottom),
            "f1_score_word": calculate_f1_score(prediction.split(), bottom.split()),
            "round": max_turn,
            "input_token": total_input_token,
            "output_token": total_output_token,
            "record": propose_agent,
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)




def _format_history(question_record: List[Dict]) -> str:
    history = ""
    for record in question_record:
        q = record["question"].replace("Q:", "").replace("A:", "").strip()
        a = record["answer"].replace("Q:", "").replace("A:", "").strip()
        history += f"Q: {q}\nA: {a}\n"
    return history


def _extract_question_from_text(text: str) -> Optional[str]:
    m = re.search(r"Q:\s*(.*)", text)
    return m.group(1).strip() if m else None


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
        surface = dataset[i]["surface"]
        bottom = dataset[i]["bottom"]

        question_record: List[Dict] = []
        total_input_token, total_output_token = 0, 0

        for turn in range(1, max_turn + 1):
            history_str = _format_history(question_record)
            prompt = proactive_cot_question_prompt_template.format(
                puzzle=surface,
                history=history_str,
                turn=turn,
                max_turn=max_turn,
            )

            messages = [
                {"role": "system", "content": proactive_cot_system_prompt},
                {"role": "user", "content": prompt},
            ]
            print("turn: ", turn, "prompt: ", prompt)

            response = inference(messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            content = response.choices[0].message.content

            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            question = _extract_question_from_text(content)

            if not question:
                fallback_prompt = "Based on the above analysis, ask a specific yes-or-no question that would help solve the puzzle:"
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": fallback_prompt})
                fallback_resp = inference(messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                fallback_text = fallback_resp.choices[0].message.content

                total_input_token += fallback_resp.usage.prompt_tokens
                total_output_token += fallback_resp.usage.completion_tokens

                question = _extract_question_from_text(fallback_text)
                if not question:
                    continue

            referee_messages = [
                {"role": "system", "content": system_prompt_with_2shots.format(question=surface, answer=bottom)},
                {"role": "user", "content": question},
            ]
            answer_resp = inference(referee_messages, model=response_model, temperature=response_temperature, top_p=response_top_p, api_key=RESPONSE_API_KEY, base_url=RESPONSE_BASE_URL)
            answer = answer_resp.choices[0].message.content

            total_input_token += answer_resp.usage.prompt_tokens
            total_output_token += answer_resp.usage.completion_tokens


            question_record.append({"question": question, "answer": answer})

        final_prompt = (
            f"Puzzle: {surface}\n\n"
            f"Previous Questions and Answers:\n{_format_history(question_record)}\n\n"
            f"Based on all the questions and answers above, what do you think is the solution to this puzzle? Please provide a detailed explanation.\n\n"
            f"Your answer:"
        )
        final_messages = [
            {"role": "system", "content": proactive_cot_system_prompt_final_guess},
            {"role": "user", "content": final_prompt},
        ]
        final_resp = inference(final_messages, model=policy_model, temperature=policy_temperature, top_p=policy_top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        pred = final_resp.choices[0].message.content

        total_input_token += final_resp.usage.prompt_tokens
        total_output_token += final_resp.usage.completion_tokens


        f1_char = calculate_f1_score(pred, bottom)
        f1_word = calculate_f1_score(pred.split(), bottom.split())

        logs.append({
            "idx": i,
            "question": surface,
            "answer": bottom,
            "pred": pred,
            "f1_score_char": f1_char,
            "f1_score_word": f1_word,
            "round": max_turn,
            "record": question_record,
            "input_token": total_input_token,
            "output_token": total_output_token,
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)


def _extract_questions_from_text(text: str) -> List[str]:
    """Extract questions from the model's response."""
    questions = []
    # Match Q1:, Q2:, Q3: format
    q_matches = re.finditer(r'Q\d+:\s*(.*?)(?=Q\d+:|$)', text, re.DOTALL)
    for match in q_matches:
        question = match.group(1).strip()
        if question:
            questions.append(question)
    
    # If no Q1/Q2/Q3 format found, try matching Q: format
    if not questions:
        matches = re.finditer(r'Q:\s*(.*?)(?=Q:|$)', text, re.DOTALL)
        for match in matches:
            question = match.group(1).strip()
            if question:
                questions.append(question)
    
    return questions


def _extract_explanations_from_text(text: str) -> List[str]:
    """Extract explanations from the model's response."""
    explanations = []
    matches = re.finditer(r'\d+\.\s*(.*?)(?=\d+\.|$)', text, re.DOTALL)
    for match in matches:
        explanation = match.group(1).strip()
        if explanation:
            explanations.append(explanation)
    return explanations


def _extract_numbers_from_text(text: str) -> List[int]:
    """Extract numbers from the model's response."""
    numbers = re.findall(r'\d+', text)
    return [int(n) for n in numbers]


def _generate_initial_answer_set(
    model: str,
    puzzle: str,
    temperature: float = 0.7,
    top_p: float = 0.7,
    api_key: str = None,
    base_url: str = None
) -> Tuple[List[str], int, int]:
    """Generate initial set of possible answers for the puzzle."""
    prompt = uot_initial_answer_set_prompt.format(puzzle=puzzle)
    messages = [{"role": "user", "content": prompt}]
    
    response = inference(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url
    )
    
    explanations = _extract_explanations_from_text(response.choices[0].message.content)
    return explanations, response.usage.prompt_tokens, response.usage.completion_tokens


def _generate_candidate_questions(
    model: str,
    puzzle: str,
    answer_set: List[str],
    qa_record: List[Dict],
    temperature: float = 0.7,
    top_p: float = 0.7,
    api_key: str = None,
    base_url: str = None
) -> Tuple[List[str], int, int]:
    """Generate 3 candidate questions that would help narrow down the answer set."""
    qa_record_str = "\n".join([f"Q: {record['question']}\nA: {record['answer']}" for record in qa_record])
    answer_set_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answer_set)])
    
    prompt = uot_candidate_questions_prompt.format(
        puzzle=puzzle,
        qa_record=qa_record_str,
        answer_set=answer_set_str
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = inference(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url
    )
    
    questions = _extract_questions_from_text(response.choices[0].message.content)
    return questions, response.usage.prompt_tokens, response.usage.completion_tokens


def _simulate_rounds(
    model: str,
    puzzle: str,
    question: str,
    answer_set: List[str],
    depth: int = 3,
    temperature: float = 0.1,
    top_p: float = 0.9,
    api_key: str = None,
    base_url: str = None
) -> Tuple[float, Dict[str, int], int, int]:
    """Simulate future rounds to calculate total uncertainty reduction."""
    total_input_token = 0
    total_output_token = 0
    
    if depth == 0 or len(answer_set) <= 1:
        return 0, {}, total_input_token, total_output_token
    
    # Get initial reduction estimates
    reductions, input_token, output_token = _estimate_answer_set_reduction(
        model, puzzle, question, answer_set, temperature, top_p, api_key, base_url, simulate_depth=0
    )
    total_input_token += input_token
    total_output_token += output_token
    
    # Initialize total uncertainty reduction
    total_uncertainty = 0
    path_reductions = {}
    
    # For each possible answer (Yes/No-Unknown)
    for answer, remaining_count in reductions.items():
        if remaining_count == 0:
            continue
        
        # Calculate probability of this answer
        prob = 1 / 2 if answer == 'Yes' else 1 / 2  # Equal probability for Yes vs No-Unknown
        
        # Get the reduced answer set for this answer
        reduced_set, input_token, output_token = _update_answer_set(
            model, puzzle, question, answer, answer_set, temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
        
        if len(reduced_set) <= 1:
            path_reductions[answer] = len(answer_set) - len(reduced_set)
            total_uncertainty += prob * (len(answer_set) - len(reduced_set))
            continue
        
        # For depth 1, just calculate the uncertainty reduction directly
        if depth == 1:
            path_reductions[answer] = len(answer_set) - len(reduced_set)
            total_uncertainty += prob * (len(answer_set) - len(reduced_set))
            continue
        
        # For deeper simulations, find the best next question
        best_next_question = None
        best_next_reduction = -1
        
        # Generate candidate questions for the next round
        candidate_questions, input_token, output_token = _generate_candidate_questions(
            model, puzzle, reduced_set, [], temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
        
        for next_question in candidate_questions:
            next_reductions, input_token, output_token = _estimate_answer_set_reduction(
                model, puzzle, next_question, reduced_set, temperature, top_p, api_key, base_url, simulate_depth=0
            )
            total_input_token += input_token
            total_output_token += output_token
            
            expected_reduction = sum(next_reductions.values()) / 2  # Changed from 3 to 2
            if expected_reduction > best_next_reduction:
                best_next_reduction = expected_reduction
                best_next_question = next_question
        
        # Recursive simulation with reduced depth
        further_uncertainty, _, input_token, output_token = _simulate_rounds(
            model, puzzle, best_next_question, reduced_set, depth-1, temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
        
        # Calculate total uncertainty reduction for this path
        path_reduction = len(answer_set) - len(reduced_set) + further_uncertainty
        path_reductions[answer] = path_reduction
        
        # Weight by probability of this answer
        total_uncertainty += prob * path_reduction
    
    return total_uncertainty, path_reductions, total_input_token, total_output_token


def _estimate_answer_set_reduction(
    model: str,
    puzzle: str,
    question: str,
    answer_set: List[str],
    temperature: float = 0.1,
    top_p: float = 0.9,
    api_key: str = None,
    base_url: str = None,
    simulate_depth: int = 0
) -> Tuple[Dict[str, int], int, int]:
    """Estimate how much the answer set would be reduced for each possible answer (yes/no-unknown)."""
    if simulate_depth > 0:
        # Use simulation for deeper analysis
        total_reduction, path_reductions, input_token, output_token = _simulate_rounds(
            model, puzzle, question, answer_set, simulate_depth, temperature, top_p, api_key, base_url
        )
        return path_reductions, input_token, output_token
    
    # Original single-round estimation
    answer_set_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answer_set)])
    
    prompt = uot_answer_set_reduction_prompt.format(
        puzzle=puzzle,
        answer_set=answer_set_str,
        question=question
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = inference(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url
    )
    
    # Extract numbers from response
    yes_match = re.search(r'Yes:\s*(\d+)', response.choices[0].message.content)
    no_unknown_match = re.search(r'No-Unknown:\s*(\d+)', response.choices[0].message.content)
    
    reductions = {
        'Yes': len(answer_set) - int(yes_match.group(1)) if yes_match else len(answer_set) // 2,
        'No-Unknown': len(answer_set) - int(no_unknown_match.group(1)) if no_unknown_match else len(answer_set) // 2
    }
    
    return reductions, response.usage.prompt_tokens, response.usage.completion_tokens


def _update_answer_set_with_specific_answer(
    model: str,
    puzzle: str,
    question: str,
    answer: str,
    answer_set: List[str],
    temperature: float = 0.1,
    top_p: float = 0.9,
    api_key: str = None,
    base_url: str = None
) -> Tuple[List[str], int, int]:
    """Update the answer set based on a specific answer (Yes, No, or Unknown)."""
    answer_set_str = "\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answer_set)])
    
    prompt = uot_update_answer_set_prompt.format(
        puzzle=puzzle,
        answer_set=answer_set_str,
        question=question,
        answer=answer
    )
    
    messages = [{"role": "user", "content": prompt}]
    response = inference(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        api_key=api_key,
        base_url=base_url
    )
    
    # Extract numbers from response
    numbers = _extract_numbers_from_text(response.choices[0].message.content)
    valid_indices = [n - 1 for n in numbers if n <= len(answer_set)]
    
    updated_set = [answer_set[i] for i in valid_indices]
    return updated_set, response.usage.prompt_tokens, response.usage.completion_tokens


def _update_answer_set(
    model: str,
    puzzle: str,
    question: str,
    answer: str,
    answer_set: List[str],
    temperature: float = 0.1,
    top_p: float = 0.9,
    api_key: str = None,
    base_url: str = None
) -> Tuple[List[str], int, int]:
    """Update the answer set based on the question and answer."""
    total_input_token = 0
    total_output_token = 0
    
    # If answer is No-Unknown, we'll check both No and Unknown
    if answer == 'No-Unknown':
        # First try with No
        no_set, input_token, output_token = _update_answer_set_with_specific_answer(
            model, puzzle, question, 'No', answer_set, temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
        
        # Then try with Unknown
        unknown_set, input_token, output_token = _update_answer_set_with_specific_answer(
            model, puzzle, question, 'Unknown', answer_set, temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
        
        # Combine the results
        updated_set = list(set(no_set + unknown_set))
    else:
        updated_set, input_token, output_token = _update_answer_set_with_specific_answer(
            model, puzzle, question, answer, answer_set, temperature, top_p, api_key, base_url
        )
        total_input_token += input_token
        total_output_token += output_token
    
    return updated_set, total_input_token, total_output_token


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
    simulate_depth: int = 1,
) -> None:
    """Run the UoT (Uncertainty of Thought) evaluation."""

    print(f"Start uot evaluation, simulate_depth={simulate_depth}")
    for i in tqdm(range(len(logs), len(dataset))):
        puzzle = dataset[i]['surface']
        true_answer = dataset[i]['bottom']
        
        total_input_token = 0
        total_output_token = 0
        
        # Step 1: Generate initial answer set
        answer_set, input_token, output_token = _generate_initial_answer_set(
            policy_model, puzzle, policy_temperature, policy_top_p, POLICY_API_KEY, POLICY_BASE_URL
        )
        total_input_token += input_token
        total_output_token += output_token
        
        qa_record = []
        
        for turn in range(max_turn):
            print(f"Turn {turn + 1}/{max_turn}")
            # Step 2: Generate candidate questions
            candidate_questions, input_token, output_token = _generate_candidate_questions(
                policy_model, puzzle, answer_set, qa_record, policy_temperature, policy_top_p,
                POLICY_API_KEY, POLICY_BASE_URL
            )
            total_input_token += input_token
            total_output_token += output_token
            
            # Step 3: Evaluate each question's potential reduction
            question_scores = []
            for question in candidate_questions:
                reductions, input_token, output_token = _estimate_answer_set_reduction(
                    policy_model, puzzle, question, answer_set, 0.1, 0.9,
                    POLICY_API_KEY, POLICY_BASE_URL, simulate_depth
                )
                total_input_token += input_token
                total_output_token += output_token
                
                # Calculate expected reduction (assuming equal probability for each answer)
                expected_reduction = sum(reductions.values()) / 2
                question_scores.append((question, expected_reduction, reductions))
            
            # Sort by expected reduction and select best question
            question_scores.sort(key=lambda x: x[1], reverse=True)
            best_question = question_scores[0][0].split("\n")[0] if question_scores else "Unknown"
            
            # Step 4: Get answer from evaluator
            respond_agent = [
                {"role": "system", "content": RESPONSE_TEMPLATE.format(question=puzzle, answer=true_answer)},
                {"role": "user", "content": best_question}
            ]
            
            response = inference(
                messages=respond_agent,
                model=response_model,
                temperature=response_temperature,
                top_p=response_top_p,
                api_key=RESPONSE_API_KEY,
                base_url=RESPONSE_BASE_URL
            )
            answer = response.choices[0].message.content
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
            
            # Step 5: Update answer set
            qa_record.append({"question": best_question, "answer": answer})
            answer_set, input_token, output_token = _update_answer_set(
                policy_model, puzzle, best_question, answer, answer_set, 0.1, 0.9,
                POLICY_API_KEY, POLICY_BASE_URL
            )
            print(f"Turn {turn + 1}/{max_turn}, Current answer set size: {len(answer_set)}")
            total_input_token += input_token
            total_output_token += output_token
            
            # Stop if we have only one answer left
            if len(answer_set) <= 1:
                break
        
        # Step 6: Generate final answer
        # If we have exactly one answer in the set, use it
        if len(answer_set) == 1:
            final_answer = answer_set[0]
        else:
            # Otherwise, generate a final answer based on all collected information
            qa_record_str = "\n".join([
                f"Q: {record['question']}\nA: {record['answer']}"
                for record in qa_record
            ])
            
            prompt = uot_final_answer_prompt.format(puzzle=puzzle, qa_record=qa_record_str)
            messages = [{"role": "user", "content": prompt}]
            response = inference(
                messages=messages,
                model=policy_model,
                temperature=0.1,
                top_p=0.9,
                api_key=POLICY_API_KEY,
                base_url=POLICY_BASE_URL
            )
            final_answer = response.choices[0].message.content.strip()
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens
        
        # Calculate scores for the final answer
        f1_char = calculate_f1_score(final_answer, true_answer)
        f1_word = calculate_f1_score(final_answer.split(), true_answer.split())
        
        # Log the result for this puzzle
        logs.append({
            "idx": i,
            "question": puzzle,
            "answer": true_answer,
            "pred": final_answer,
            "f1_score_char": f1_char,
            "f1_score_word": f1_word,
            "round": turn + 1,
            "record": qa_record,
            "input_token": total_input_token,
            "output_token": total_output_token,
        })
        
        # Save progress after each puzzle
        with open(output_path, 'w') as f:
            json.dump(logs, f, indent=4)

def main(
    method: str, 
    data_path: str, 
    output_path: str, 
    policy_model: str,
    response_model: str,
    branch: int = 3,
    max_turn: int = 25, 
    policy_temperature: float = 0.7,
    policy_top_p: float = 0.7,
    response_temperature: float = 0.7,
    response_top_p: float = 0.7,
    simulate_depth: int = 1
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
        _run_traditional_evaluation(method, dataset, logs, output_path, policy_model, max_turn,
                                   policy_temperature, policy_top_p, response_model,
                                   response_temperature, response_top_p)


if __name__ == "__main__":
    Fire(main)
