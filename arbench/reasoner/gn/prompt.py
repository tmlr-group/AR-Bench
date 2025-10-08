propose_template_with_1_shot_inst = """
Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct {turn} turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

You can refer to this instruction to decide your proposed guessing numbers:
1. First give an initial guess and get feedback
2. if the feedback indicates that there are no correct digits, rule out the four digits, and if there are correct digits, determine which ones are correct
3. in order to determine the correct digit, you can replace the digit bit by bit with a digit that does not appear in the current guess, or a digit that has been determined not to be in the correct answer; if the correct digits decreases, the replaced digit appears in the correct answer; if the correct digits remains unchanged, the digit does not appear in the correct answer, or both the digit and the replaced digit appear in the correct answer; If the correct digits has increased, the replaced digit appears in the correct answer
4. progressively replacing until all the correct digits are present in the guess, get feedback as: The summation of the number of digits are present in the answer and in the correct positions, and the number of digits are present in the answer but in the different positions equal to 4
5. change the order of the four digits until all the digits are in the correct positions

Here is a example trajectories of successfully guessing numbers for you to learn:
Example 1:
Turn 1: Guess: 1234, Feedback: 0 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions, Analysis: No digits are correct, so rule out 1, 2, 3 and 4.
Turn 2: Guess: 5678, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions, Analysis: 3 digits are correct, next step is to find which digits are correct in the 4-digit guess. 
Turn 3: Guess: 1678, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions, Analysis: Replace 5 as 1, the correct digits decreases, indicate 5 is the correct digit 
Turn 4: Guess: 5178, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions, Analysis: Replace 6 as 1, the correct digits decreases, indicate 6 is the correct digit 
Turn 5: Guess: 5618, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions, Analysis: Replace 7 as 1, the correct digits unchange, since 1 has been proved not the correct digit, 7 is not the correct digit, so the correct digits are: 5, 6, 8
Turn 6: Guess: 5689, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions, Analysis: 9 is not the right digit since the number of correct digit does not change
Turn 7: Guess: 5680, Feedback: 1 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions, Analysis: 0 is the right digit since the number of correct digit increases: 1 + 3 = 4, so far, we find all of the correct digits
Turn 8: Guess: 8650, Feedback: 2 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions, Analysis: Adjust the order of the guess to find the correct answer
Turn 9: Guess: 8560, Feedback: 1 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 10: Guess: 6850, Feedback: 4 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions, Analysis: 4 digits are correct and in the correct positions, the answer found!

Final Answer: 6850

"""

propose_template_with_1_shot = """
Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct {turn} turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

Here are two example trajectories of successfully guessing numbers for you to learn:
Example 1:
Turn 1: Guess: 1234, Feedback: 0 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions 
Turn 2: Guess: 5678, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions 
Turn 3: Guess: 1678, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 
Turn 4: Guess: 5178, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 
Turn 5: Guess: 5618, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 6: Guess: 5689, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 7: Guess: 5680, Feedback: 1 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 8: Guess: 8650, Feedback: 2 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions
Turn 9: Guess: 8560, Feedback: 1 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 10: Guess: 6850, Feedback: 4 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions

Final Answer: 6850

"""


"""
Example 2:
Turn 1: Guess: 1234, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 
Turn 2: Guess: 0234, Feedback: 0 digits are present in the answer and in the correct positions, 1 digits are present in the answer but in the different positions 
Turn 3: Guess: 1034, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 
Turn 4: Guess: 1204, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 
Turn 5: Guess: 1564, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 6: Guess: 1264, Feedback: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions
Turn 7: Guess: 1574, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 8: Guess: 1584, Feedback: 0 digits are present in the answer and in the correct positions, 4 digits are present in the answer but in the different positions
Turn 9: Guess: 5148, Feedback: 4 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions

Final Answer: 5148

Game start:

"""


propose_template = """Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct {turn} turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

Game start:
"""

guess_prompt = """Turn {turn}: Now give me the number you guess in this format, and do not give any other statement:
Guess: [number]
"""

final_guess_prompt = """
You have finished all the rounds of interaction, please give your final answer based on the guesses and feedback above:
Guess: [number]
"""

eval_prompt = """
{same_pos} digits are present in the answer and in the correct positions 
{diff_pos} digits are present in the answer but in the different positions 
"""


refine_prompt = """
Your output does not follow this format: Guess: [number]
please propose a guess number, for example: Guess: 1234
"""

Game_rule = """"
Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. 
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions 2 digits are present in the answer but in the different positions 

"""

Guess_number_prompt = """ 
Your task is: Based on the game rules and the previous guess records, suggest 3 possible four-digit numbers, which means you have 3 chances to make your guesses. You do not need to start the game or perform any other actions, just provide 3 numbers.

The known information is the guess record (if any): {guess_record}

Your output should be in this format (only in this format and must include exactly 3 numbers):
Guess: [num1], [num2], [num3]
"""

Final_prompt = """
Now, you have to base on the known information is the guess record {guess_record}, to guess a 4-digit number as your answer

Your output should be only a 4-digit number without any other words.
"""


proactive_cot_system_prompt = """Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct 25 turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

Game start:
"""

proactive_cot_prompt_template = """Given the game rules and the conversation history so far, you need to determine the most effective next guess. 

First, analyze the current progress of the guessing game by examining the previous guesses and feedback:
{history}

There are several possible action strategies for guessing numbers:
1. DIGIT_EXPLORATION: Test new digits not used in previous guesses to determine which digits are in the secret number
2. POSITION_TESTING: Test known digits in different positions to determine their correct positions
3. ELIMINATION: Make a guess that will eliminate certain possibilities based on previous feedback
4. CONFIRMATION: Make a guess to confirm hypotheses about specific digits or positions
5. OPTIMAL_SOLUTION: Make a guess that you believe has a high probability of being the correct answer
6. INFORMATION_GAIN: Make a guess that will provide the maximum information gain regardless of correctness probability

To make an optimal next guess, first analyze:
- Which digits are confirmed to be in the secret number
- Which digits are confirmed not to be in the secret number
- Which positions are known for certain digits
- Which positions are eliminated for certain digits
- What information would be most valuable to learn next

Your analysis of the current game state:
[Provide your detailed analysis here]

Based on this analysis, select the most appropriate action strategies for this turn:
[Select one or more of the action strategies listed above]

Justification for selected action strategies:
[Explain why these strategies are optimal for the current game state]

Based on the selected strategies, your next guess is: [4-digit number]

Justification for this guess:
[Explain how this guess implements the selected strategies and what information you expect to gain]

note that when you propose a guess, you should output 'Guess: [4-digit number]'"""


SYSTEMPROMPT = """Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct {turn} turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

Game start:
"""

MULTI_GUESS_PROMPT = """Next, please generate 3 different possible guesses that would give you the most information. Make sure all your guesses are 4-digit numbers with unique digits:"""

MULTI_GUESS_PROMPT_WITH_ANSWERS = """Next, please generate 3 different possible guesses that would give you the most information. Make sure all your guesses are 4-digit numbers with unique digits.

Here are all the possible answers remaining:
{answer_set}

Please analyze these possible answers and choose 3 guesses that would help eliminate the most possibilities:"""

SIMULATE_PROMPT = """For these 3 guesses, let's analyze how much information each guess would provide.
Guess 1: {guess1}
Guess 2: {guess2}
Guess 3: {guess3}

For each guess, estimate how much the answer space would reduce with different feedbacks (correct position, correct digit but wrong position).
Format your answer as:
Guess 1: {guess1} - [analysis of information gain]
Guess 2: {guess2} - [analysis of information gain]
Guess 3: {guess3} - [analysis of information gain]
Best guess: [your choice of the most informative guess and why]"""

CONCLUSIONPROMPT = "Based on the previous guessing history, give your final guess:"

FEEDBACKPROMPT = "Turn {turn}: Guess: {guess}. Feedback: {in_pos} digits are present in the answer and in the correct positions, {out_pos} digits are present in the answer but in the different positions."
