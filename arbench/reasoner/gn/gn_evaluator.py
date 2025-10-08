import itertools
import json
import os
import random
import re
import sys
import time
import math
import itertools
from datetime import timedelta
from typing import Any, Dict, Generator, List, Union, Optional, Tuple
from dotenv import load_dotenv
import openai
from openai import OpenAI
from tqdm import tqdm
from fire import Fire
from arbench.reasoner.gn.prompt import *
from arbench.utils.inference import inference
from arbench.utils.utils_gn import (
    NotNumberError,
    generate_unique_four_digit_number,
    extract_and_convert_guess,
    compare_guess
)

load_dotenv()
# Constants
MAX_RETRIES = 5
CORRECT_POSITION_SCORE = 2
DIFFERENT_POSITION_SCORE = 1
TARGET_FEEDBACK = [4, 0]
POLICY_API_KEY = os.getenv("POLICY_API_KEY")
POLICY_BASE_URL = os.getenv("POLICY_BASE_URL")

# Method configuration
METHOD_DICT = {
    "zero_shot": propose_template,
    "few_shot": propose_template_with_1_shot,
    "few_shot_inst": propose_template_with_1_shot_inst,
    "proactive_cot": propose_template,  # placeholder; proactive flow uses dedicated prompts
}


def extract_four_digit_numbers(input_string: str) -> List[str]:
    # Remove all types of brackets from input string
    input_string = input_string.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace("{", "").replace("}", "").replace("*", "")
    guess_pattern = r"guess:\s*([0-9,\s]+)"
    guess_matches = re.findall(guess_pattern, input_string.lower())
    guess_matches = [s for s in guess_matches if len(re.findall(r"\d", s)) == 4]

    if guess_matches:
        last_guess = guess_matches[-1].strip()
        return last_guess
    else:
        four_digit_matches = re.findall(r"\d{4}", input_string)
        if four_digit_matches:
            return four_digit_matches[-1].strip()
        else:
            return []


class SearchNode:
    
    def __init__(self, guess_number: str, parent: Optional['SearchNode'] = None):
        self.guess_number = guess_number
        self.parent = parent
        self.children: List['SearchNode'] = []
        self.value = -1.0  # Node evaluation score
        self.feedback = [-1, -1]  # [correct_position, different_position]

    def add_child(self, child: 'SearchNode') -> None:
        self.children.append(child)

    def get_ancestor_guess_record(self, target: List[int]) -> Tuple[str, List[List]]:
        ancestor_guesses = []
        current_node = self
        
        # Collect ancestor guesses from current to root
        while current_node.parent:
            ancestor_guesses.append(current_node.guess_number)
            current_node = current_node.parent
        
        ancestor_guesses.reverse()
        
        # Generate feedback for each guess
        feedback_prompt = ""
        guess_history = []
        
        for i, guess in enumerate(ancestor_guesses):
            correct_pos, different_pos, _ = compare_guess(target, guess)
            
            feedback_str = (
                f"The guess number {i + 1} is {guess}, and the feedback is: "
                f"{correct_pos} digits are present in the answer and in the correct positions, "
                f"{different_pos} digits are present in the answer but in the different positions."
            )
            guess_history.append([guess, correct_pos, different_pos])
            feedback_prompt += feedback_str + "\n"
            
        return feedback_prompt, guess_history

    def get_value(self, target: List[int]) -> None:
        correct_pos, different_pos, self.value = compare_guess(target, self.guess_number)
        self.feedback = [correct_pos, different_pos]


class ToTSearch:
    
    def __init__(self, target: List[int], model_to_use: str, temperature: float = 0.7, top_p: float = 0.7):
        self.root = SearchNode("root")
        self.target = target
        self.model_to_use = model_to_use
        self.temperature = temperature
        self.top_p = top_p

    def select(self, max_depth: int) -> Tuple[SearchNode, List[List]]:
        current_node = self.root
        
        for depth in range(max_depth):
            # Check if target is reached
            if current_node.feedback == TARGET_FEEDBACK:
                ancestor_record, history = current_node.get_ancestor_guess_record(self.target)
                return current_node, history

            # Get guess candidates from model
            ancestor_record, _ = current_node.get_ancestor_guess_record(self.target)
            guess_prompt = [{
                "role": "user",
                "content": Game_rule + Guess_number_prompt.format(guess_record=ancestor_record)
            }]
            
            # Retry logic for getting valid guesses
            retry_count = 0
            valid_numbers = []
            
            while not valid_numbers and retry_count < MAX_RETRIES:
                response = inference(guess_prompt, model=self.model_to_use, json_format=False, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL,
                temperature=self.temperature, top_p=self.top_p)
                guess_response = response.choices[0].message.content
                valid_numbers = extract_four_digit_numbers(guess_response)
                
                if not valid_numbers:
                    retry_count += 1
                    if retry_count == MAX_RETRIES:
                        warning = "Check your output must be in this format:Guess: [num1], [num2], [num3],and each num should be 4-digit"
                        guess_prompt[0]["content"] += "\n" + warning

            if not valid_numbers:
                raise ValueError("Failed to get valid four-digit numbers after multiple retries")
            
            # Select best candidate based on value
            best_value = -float("inf")
            best_child = None
            
            for number in valid_numbers:
                candidate_node = SearchNode(number, parent=current_node)
                candidate_node.get_value(self.target)
                if candidate_node.value > best_value:
                    best_value = candidate_node.value
                    best_child = candidate_node
            # Fallback to random selection if no best child found
            if best_child is None:
                candidates = [SearchNode(num, parent=current_node) for num in valid_numbers]
                best_child = random.choice(candidates)
                best_child.get_value(self.target)
            current_node = best_child

        # Get final result after reaching max depth
        ancestor_record, history = current_node.get_ancestor_guess_record(self.target)
        final_prompt = [{
            "role": "user",
            "content": Game_rule + Final_prompt.format(guess_record=ancestor_record)
        }]
        
        response = inference(final_prompt, model=self.model_to_use, json_format=False, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL,
                           temperature=self.temperature, top_p=self.top_p)
        final_number = extract_and_convert_guess(response.choices[0].message.content)

        return current_node, history





def _run_tot_evaluation(dataset: List, logs: List, output_path: str, model: str, max_turn: int, 
                          temperature: float, top_p: float) -> None:
    """Run evaluation using tot search method."""
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = [int(digit) for digit in dataset[i]]
        
        tot_search = ToTSearch(true_number, model, temperature, top_p)
        root, guess_list = tot_search.select(max_turn)
        
        logs.append({
            "idx": i,
            "true_number": dataset[i],
            "guess_list": guess_list,
            "guess_round": max_turn,
            "correctness": guess_list[-1][1] == 4,
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)



def _run_traditional_evaluation(dataset: List, logs: List, output_path: str, model: str, method: str, max_turn: int, 
                               temperature: float, top_p: float) -> None:
    """Run evaluation using traditional prompting methods."""
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = [int(digit) for digit in dataset[i]]
        
        guess_list = []
        correctness = False
        output_fail_cnt = 0
        total_input_token, total_output_token = 0, 0

        propose_agent = [
            {"role": "system", "content": METHOD_DICT[method].format(turn=max_turn)}
        ]

        # Game turns
        for turn in range(max_turn):
            try:
                propose_agent.append({
                    "role": "user",
                    "content": guess_prompt.format(turn=turn + 1)
                })
                
                response = inference(propose_agent, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                guess = response.choices[0].message.content
                total_input_token += response.usage.prompt_tokens
                total_output_token += response.usage.completion_tokens

                propose_agent.append({"role": "assistant", "content": guess})

                guess_number = extract_and_convert_guess(guess)
                same_pos, diff_pos, _ = compare_guess(true_number, guess_number)
                
                guess_list.append([
                    "".join([str(digit) for digit in guess_number]),
                    same_pos,
                    diff_pos,
                ])
                
            except NotNumberError:
                propose_agent.append({"role": "user", "content": refine_prompt})
                guess_list.append(["", 0, 0])
                output_fail_cnt += 1
                continue

            propose_agent.append({
                "role": "user",
                "content": eval_prompt.format(same_pos=same_pos, diff_pos=diff_pos)
            })

        # Final guess
        propose_agent.append({"role": "user", "content": final_guess_prompt})
        response = inference(propose_agent, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
        
        final_guess = response.choices[0].message.content
        total_input_token += response.usage.prompt_tokens
        total_output_token += response.usage.completion_tokens
        
        try:
            guess_number = extract_and_convert_guess(final_guess)
            same_pos, diff_pos, _ = compare_guess(true_number, guess_number)
            guess_list.append([
                "".join([str(digit) for digit in guess_number]),
                same_pos,
                diff_pos,
            ])
        except NotNumberError:
            guess_list.append(["", 0, 0])
            output_fail_cnt += 1

        logs.append({
            "idx": i,
            "true_number": "".join([str(digit) for digit in true_number]),
            "conversation": propose_agent,
            "guess_list": guess_list,
            "prediction": guess_list[-1],
            "guess_round": len(guess_list),
            "output_fail_cnt": output_fail_cnt,
            "input_token": total_input_token,
            "output_token": total_output_token,
            "correctness": guess_list[-1][1] == 4,
        })

        with open(output_path, "w") as file:
            json.dump(logs, file, indent=4)

def _gn_extract_multiple_guesses(s: str) -> List[List[int]]:
    m = re.findall(r"\d{4}", s)
    result: List[List[int]] = []
    for token in m:
        result.append([int(ch) for ch in token])
    return result[:3]

def _gn_generate_all_answers() -> List[List[int]]:
    answers: List[List[int]] = []
    digits = list(range(10))
    
    for combo in itertools.permutations(digits, 4):
        answers.append(list(combo))
    return answers


def _gn_calculate_remaining(answer_set: List[List[int]], guess: List[int], same_pos: int, diff_pos: int) -> List[List[int]]:
    remaining: List[List[int]] = []
    for cand in answer_set:
        _, _, score = compare_guess(cand, guess)
        # utils compare_guess returns (same_pos, diff_pos, score)
        s, d, _ = compare_guess(cand, guess)
        if s == same_pos and d == diff_pos:
            remaining.append(cand)
    return remaining


def _gn_calculate_remaining_with_model(model: str, answer_set: List[List[int]], guess: List[int], same_pos: int, diff_pos: int,
                                       temperature: float, top_p: float) -> Tuple[List[List[int]], int, int]:
    guess_str = "".join([str(d) for d in guess])
    feedback_str = f"{same_pos} digits are present in the answer and in the correct positions, {diff_pos} digits are present in the answer but in the different positions"
    sample_size = min(50, len(answer_set))
    sample_answers = random.sample(answer_set, sample_size) if answer_set else []
    answer_set_str = "\n".join(["".join(map(str, ans)) for ans in sample_answers])
    if len(answer_set) > sample_size:
        answer_set_str += f"\n... (and {len(answer_set) - sample_size} more possible answers)"

    prompt = f"""I am playing a number guessing game. The secret is a 4-digit number where all digits are different (0-9).
When I guess: {guess_str}
I receive feedback: {feedback_str}

Here is my current set of possible answers (showing {sample_size} out of {len(answer_set)} total):
{answer_set_str}

Based on this feedback and my current set of possible answers, please determine which numbers from my answer set are still valid.
A number is valid if it would give exactly the same feedback ({same_pos} correct digits in correct positions, {diff_pos} correct digits in wrong positions) when compared to my guess {guess_str}.

Format your answer as a list of 4-digit numbers, one per line. Only include numbers that satisfy the feedback criteria."""

    messages = [{"role": "user", "content": prompt}]
    resp = inference(messages, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
    content = resp.choices[0].message.content
    in_tok = resp.usage.prompt_tokens
    out_tok = resp.usage.completion_tokens
    extracted = re.findall(r"\d{4}", content)
    verified: List[List[int]] = []
    for number_str in extracted:
        candidate = [int(d) for d in number_str]
        s, d, _ = compare_guess(candidate, guess)
        if s == same_pos and d == diff_pos:
            verified.append(candidate)
    return verified, in_tok, out_tok


def _gn_estimate_uncertainty_reduction(answer_set: List[List[int]], guess: List[int]) -> Tuple[float, Dict[str, int]]:
    feedback_partition: Dict[str, int] = {}
    for candidate in answer_set:
        s, d, _ = compare_guess(candidate, guess)
        key = f"{s}_{d}"
        feedback_partition[key] = feedback_partition.get(key, 0) + 1
    total = len(answer_set)
    entropy_reduction = 0.0
    for count in feedback_partition.values():
        probability = count / total if total else 0
        if probability > 0:
            
            entropy_reduction -= probability * math.log2(probability)
    return entropy_reduction, feedback_partition


def _gn_simulate_rounds(answer_set: List[List[int]], guess: List[int], depth: int = 3,
                        model: Optional[str] = None,
                        temperature: float = 0.1, top_p: float = 0.9) -> Tuple[float, Dict[str, int], int, int]:
    if depth == 0 or len(answer_set) <= 1:
        return 0.0, {}, 0, 0
    print(f"[Sim] Enter depth={depth}  answer_set={len(answer_set)}  guess={''.join(map(str, guess))}")
    _, feedback_partition = _gn_estimate_uncertainty_reduction(answer_set, guess)
    total_in, total_out = 0, 0
    total_uncertainty = 0.0
    path_reductions: Dict[str, int] = {}

    for key, count in feedback_partition.items():
        same_pos, diff_pos = map(int, key.split('_'))
        if count == 0:
            continue
        prob = count / len(answer_set)
        print(f"[Sim]  Branch key={key}  count={count}  prob={prob:.3f}")
        new_answer_set, ui, uo = _gn_calculate_remaining_with_model(model, answer_set, guess, same_pos, diff_pos, temperature, top_p)
        total_in += ui
        total_out += uo

        if len(new_answer_set) <= 1:
            path_reductions[key] = len(answer_set) - len(new_answer_set)
            total_uncertainty += prob * (len(answer_set) - len(new_answer_set))
            continue

        if depth == 1:
            path_reductions[key] = len(answer_set) - len(new_answer_set)
            total_uncertainty += prob * (len(answer_set) - len(new_answer_set))
            continue

        candidates = random.sample(new_answer_set, min(3, len(new_answer_set)))
        best_next_guess = None
        best_next_reduction = -1.0
        for cand in candidates:
            uncertainty, _ = _gn_estimate_uncertainty_reduction(new_answer_set, cand)
            if uncertainty > best_next_reduction:
                best_next_reduction = uncertainty
                best_next_guess = cand

        if depth == 2 or len(new_answer_set) <= 5:
            further_reduction = len(new_answer_set) // 2
            path_reduction = len(answer_set) - len(new_answer_set) + further_reduction
            path_reductions[key] = path_reduction
            total_uncertainty += prob * path_reduction
            continue

        deeper_uncertainty, _, di, do = _gn_simulate_rounds(new_answer_set, best_next_guess, depth-1, model, temperature, top_p)
        total_in += di
        total_out += do
        path_reduction = len(answer_set) - len(new_answer_set) + int(deeper_uncertainty)
        path_reductions[key] = path_reduction
        total_uncertainty += prob * path_reduction
        print(f"Deeper -> deeper_uncertainty={deeper_uncertainty:.3f}  path_reduction={path_reduction}")

    print(f"Exit depth={depth}  total_uncertainty={total_uncertainty:.3f}  tokens(in,out)=({total_in},{total_out})")
    return total_uncertainty, path_reductions, total_in, total_out


def _gn_analyze_candidates_with_simulation(answer_set: List[List[int]], candidates: List[List[int]], depth: int = 3,
                                        model: Optional[str] = None,
                                           temperature: float = 0.1, top_p: float = 0.9) -> Tuple[List[Tuple[List[int], float, Dict[str, int]]], int, int]:
    candidate_scores: List[Tuple[List[int], float, Dict[str, int]]] = []
    total_in, total_out = 0, 0
    print(f"candidates={len(candidates)}  answer_set={len(answer_set)}  depth={depth}")
    for cand in candidates:
        uncertainty_reduction, path_reductions, ui, uo = _gn_simulate_rounds(answer_set, cand, depth, model, temperature, top_p)
        total_in += ui
        total_out += uo
        candidate_scores.append((cand, uncertainty_reduction, path_reductions))
        print(f"cand={''.join(map(str, cand))}  uncertainty={uncertainty_reduction:.3f}  +in={ui} +out={uo}")
    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    if candidate_scores:
        head = ", ".join([f"{''.join(map(str, c[0]))}:{c[1]:.3f}" for c in candidate_scores[:3]])
        print(f"Top scores -> {head}")
    return candidate_scores, total_in, total_out


def _run_uot_evaluation(dataset: List, logs: List, output_path: str, model: str, max_turn: int, temperature: float, top_p: float,
                        simulate_depth: int) -> None:

    print(f"Start uot evaluation, simulate_depth={simulate_depth}")
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = [int(d) for d in str(dataset[i]).zfill(4)]

        guess_list: List[List[Union[str, int]]] = []
        current_answer_set = _gn_generate_all_answers()
        guess_record = ""
        total_input_token, total_output_token = 0, 0

        for turn in range(max_turn):
            print(f"Turn {turn + 1}/{max_turn}")
            if turn == 0:
                prompt = SYSTEMPROMPT.format(turn=max_turn) + guess_record + MULTI_GUESS_PROMPT
            else:
                answer_set_str = "\n".join(["".join(map(str, ans)) for ans in current_answer_set])
                prompt = SYSTEMPROMPT.format(turn=max_turn) + guess_record + MULTI_GUESS_PROMPT_WITH_ANSWERS.format(answer_set=answer_set_str)

            propose_agent = [{"role": "user", "content": prompt}]
            response = inference(model=model, messages=propose_agent, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens

            candidate_guesses = _gn_extract_multiple_guesses(response.choices[0].message.content)
            def _fmt_candidates(cands: List[List[int]], limit: int = 5) -> str:
                as_str = ["".join(map(str, c)) for c in cands]
                if len(as_str) > limit:
                    return ", ".join(as_str[:limit]) + f" ... (+{len(as_str)-limit} more)"
                return ", ".join(as_str)

            print(f"Candidate guesses size: {len(candidate_guesses)}  -> [{_fmt_candidates(candidate_guesses)}]")
            while len(candidate_guesses) < 3:
                if len(current_answer_set) > 3:
                    candidate_guesses.append(random.choice(current_answer_set))
                else:
                    random_guess = random.sample(range(10), 4)
                    if random_guess not in candidate_guesses:
                        candidate_guesses.append(random_guess)
            print(f"Finalized candidates (>=3): [{_fmt_candidates(candidate_guesses)}]")

            # Analyze with simulation (optionally model-assisted)
            candidate_scores, ui, uo = _gn_analyze_candidates_with_simulation(
                current_answer_set, candidate_guesses, depth=simulate_depth,
                model=model, temperature=0.1, top_p=0.9
            )
            total_input_token += ui
            total_output_token += uo
            if candidate_scores:
                preview = ", ".join([f"{''.join(map(str, cs[0]))}:{cs[1]:.3f}" for cs in candidate_scores[:3]])
                print(f"Top candidate scores: {preview}")
            best_guess = candidate_scores[0][0]
            best_score = candidate_scores[0][1]
            print(f"Chosen best guess: {''.join(map(str, best_guess))}  score: {best_score:.3f}")

            guess_number_str = "".join([str(d) for d in best_guess])
            same_pos, diff_pos, _ = compare_guess(true_number, best_guess)
            guess_list.append([guess_number_str, same_pos, diff_pos])
            guess_record += FEEDBACKPROMPT.format(turn=turn, guess=guess_number_str, in_pos=same_pos, out_pos=diff_pos) + "\n"

            current_answer_set = _gn_calculate_remaining(current_answer_set, best_guess, same_pos, diff_pos)
            print(f"Current answer set size: {len(current_answer_set)}")
            if len(current_answer_set) > 0:
                print(f"Sample remain heads: [{_fmt_candidates(current_answer_set[:5])}]")
            if same_pos == 4 or len(current_answer_set) <= 1:
                if same_pos == 4:
                    print("Stopping early: correct guess achieved.")
                else:
                    print("Stopping early: answer set collapsed to <= 1.")
                break

        if not any(g[1] == 4 for g in guess_list) and len(current_answer_set) == 1:
            final_answer = current_answer_set[0]
            guess_number_str = "".join([str(d) for d in final_answer])
            same_pos, diff_pos, _ = compare_guess(true_number, final_answer)
            guess_list.append([guess_number_str, same_pos, diff_pos])
            print(f"Finalizing with unique remaining answer: {guess_number_str} (same_pos={same_pos}, diff_pos={diff_pos})")

        logs.append({
            "idx": i,
            "true_number": true_number,
            "guess_list": guess_list,
            "prediction": guess_list[-1] if guess_list else ["", 0, 0],
            "guess_round": len(guess_list),
            "input_token": total_input_token,
            "output_token": total_output_token,
            "correctness": any(g[1] == 4 for g in guess_list),
            "method": "uot",
        })

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)
        print(f"Saved logs to: {output_path}")


        
def _history_to_text(history: List[Dict]) -> str:
    history_str = ""
    for i, record in enumerate(history):
        guess = record["guess"] if isinstance(record["guess"], list) else record["guess"]
        if isinstance(guess, list):
            guess_str = "".join(str(d) for d in guess)
        else:
            guess_str = str(guess)
        result = record["result"]
        history_str += (
            f"Turn {i+1}: Guess: {guess_str}. Feedback: {result[0]} digits are present in the answer and in the correct positions, {result[1]} digits are present in the answer but in the different positions.\n"
        )
    return history_str


# def _extract_guess_from_text(text: str) -> List[int]:
#     pattern = r"\b\d{4}\b"
#     match = re.search(pattern, text)
#     if match:
#         guess_str = match.group(0)
#         return [int(d) for d in guess_str]
#     return []


def _run_proactive_cot_evaluation(
    dataset: List,
    logs: List,
    output_path: str,
    model: str,
    max_turn: int,
    temperature: float,
    top_p: float,
) -> None:
    for i in tqdm(range(len(logs), len(dataset))):
        true_number = dataset[i]
        secret_digits = [int(d) for d in str(true_number).zfill(4)]

        guess_history: List[Dict[str, Any]] = []
        correct = False
        total_input_token, total_output_token = 0, 0

        for turn in range(1, max_turn + 1):
            history_str = _history_to_text(guess_history)
            prompt = proactive_cot_prompt_template.format(history=history_str)

            messages = [
                {"role": "system", "content": proactive_cot_system_prompt},
                {"role": "user", "content": prompt},
            ]
            response = inference(messages, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)

            total_input_token += response.usage.prompt_tokens
            total_output_token += response.usage.completion_tokens


            reasoning = response.choices[0].message.content
            guess = extract_four_digit_numbers(reasoning)

            if not guess or len(guess) != 4:
                fallback_prompt = "Based on the analysis, your next guess for the 4-digit number (with no repeated digits) is: "
                messages.append({"role": "assistant", "content": reasoning})
                messages.append({"role": "user", "content": fallback_prompt})
                fallback_response = inference(messages, model=model, temperature=temperature, top_p=top_p, api_key=POLICY_API_KEY, base_url=POLICY_BASE_URL)
                fallback_text = fallback_response.choices[0].message.content

                total_input_token += fallback_response.usage.prompt_tokens
                total_output_token += fallback_response.usage.completion_tokens

                guess = extract_four_digit_numbers(fallback_text)

                if not guess or len(guess) != 4:
                    used_digits = set()
                    for rec in guess_history:
                        used_digits.update(rec["guess"])
                    remaining = [d for d in range(10) if d not in used_digits]
                    if len(remaining) < 4:
                        remaining = list(range(10))
                    guess = random.sample(remaining, 4)

            same_pos, diff_pos, _ = compare_guess(secret_digits, guess)

            guess_history.append({
                "guess": guess,
                "result": [same_pos, diff_pos],
                "reasoning": reasoning,
            })

            if same_pos == 4:
                correct = True
                break

        log_entry = {
            "idx": i,
            "true_number": secret_digits,
            "guess_list": [
                ["".join(str(d) for d in rec["guess"]), rec["result"][0], rec["result"][1]]
                for rec in guess_history
            ],
            "prediction": [
                "".join(str(d) for d in guess_history[-1]["guess"]),
                guess_history[-1]["result"][0],
                guess_history[-1]["result"][1],
            ] if guess_history else ["", 0, 0],
            "guess_round": len(guess_history),
            "correctness": correct,
            "input_token": total_input_token,
            "output_token": total_output_token,
        }

        logs.append(log_entry)

        with open(output_path, "w") as f:
            json.dump(logs, f, indent=4)




def main(model: str, method: str, data_path: str, output_path: str, max_turn: int = 25, 
         temperature: float = 0.7, top_p: float = 0.7, simulate_depth: int = 3) -> None:
    
    # Load dataset
    with open(data_path, "r") as file:
        dataset = json.load(file)

    # Load existing logs if available
    logs = []
    if os.path.exists(output_path):
        with open(output_path, "r") as file:
            logs = json.load(file)

    print(f"Model: {model}, Max turn: {max_turn}, Method: {method}")
    print(f"Temperature: {temperature}, Top_p: {top_p}")

    if method == "tot":
        _run_tot_evaluation(dataset, logs, output_path, model, max_turn, temperature, top_p)
    elif method == "proactive_cot":
        _run_proactive_cot_evaluation(dataset, logs, output_path, model, max_turn, temperature, top_p)
    elif method == "uot":
        _run_uot_evaluation(dataset, logs, output_path, model, max_turn, temperature, top_p, simulate_depth)
    else:
        _run_traditional_evaluation(dataset, logs, output_path, model, method, max_turn, temperature, top_p)
    

if __name__ == "__main__":
    Fire(main)
