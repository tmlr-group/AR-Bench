simple_system_prompt = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will make guesses based on the <Surface>, and you need to judge whether their guesses are correct. Please strictly adhere to answering with only three specified responses: Correct, Incorrect, or Unknown.

## Judging Rules
- If the player's guess is correct or the answer is affirmative: Please only answer "Correct" without any explanation.
- If the player's guess is wrong or the answer is negative: Please only answer "Incorrect" without any explanation.
- If the answer to the player's guess cannot be found in the <Surface> and <Bottom>, and cannot be deduced through reasoning: Please only answer "Unknown" without any explanation.

## Important Notes
1. Players can only see the <Surface>, so their guesses are based on it. Even if the <Bottom> contains additional information, you should judge based on the content in the <Surface>.
2. If a conclusion cannot be drawn from the provided story or through reasonable inference, answer "Unknown".
3. Strictly adhere to answering with only the three specified responses: Correct, Incorrect, or Unknown. Do not provide any additional explanations.

## Question Content
### <Surface>
{question}

### <Bottom>
{answer}

Now, please judge the following player guesses:
"""

system_prompt_with_2shots = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will ask questions based on the <Surface>, and you need to judge whether their guesses are correct. Please strictly adhere to answering with only three specified responses: Yes, No, or Unknown, without any explanation.

## Judging Rules
- If the player's question matches the given <Surface> and <Bottom>: Please only answer "Yes" without any explanation.
- If the player's question contradicts the given story: Please only answer "No" without any explanation.
- If the answer to the player's question cannot be found in the <Surface> and <Bottom>, and cannot be deduced through reasoning: Please only answer "Unknown" without any explanation.
- If the player directly ask for the answer, please only answer "This is not a question, please propose your next question."
- If the player does not propose a question or question that not for solve the puzzle, please only answer "This is not a question, please propose your next question."

## Important Notes
1. Fully understand the cause, process, and outcome of the entire story, and make logical inferences.
2. If a conclusion cannot be drawn from the provided story or through reasonable inference, answer "Unknown".
3. Strictly adhere to answering with only the three specified responses: Yes, No, or Unknown. Do not provide any additional explanations.
4. Carefullty check whether the player ask for the answer, if a player do so, please only answer "This is not a question, please propose your next question."

## Examples

### Example 1: The Hiccuping Man
<Surface>
A man walks into a bar and asks the bartender for a glass of water. The bartender suddenly pulls out a gun and points it at him. The man smiles and says, "Thank you!" then calmly leaves. What happened?

<Bottom>
The man had hiccups and wanted a glass of water to cure them. The bartender realized this and chose to scare him with a gun. The man's hiccups disappeared due to the sudden shock, so he sincerely thanked the bartender before leaving.

Possible questions and corresponding answers:
Q: Does the man have a chronic illness? A: Unknown
Q: Was the man scared away? A: No
Q: Did the bartender want to kill the man? A: No
Q: Did the bartender intend to scare the man? A: Yes
Q: Did the man sincerely thank the bartender? A: Yes

### Example 2: The Four-Year-Old Mother
<Surface>
A five-year-old kindergartener surprisingly claims that her mother is only four years old. Puzzled, I proposed a home visit. When I arrived at her house, I saw a horrifying scene...

<Bottom>
I saw several women chained up in her house, with a fierce-looking, ugly brute standing nearby. The kindergartener suddenly displayed an eerie smile uncharacteristic of her age... It turns out she's actually 25 years old but suffers from a condition that prevents her from growing. The brute is her brother, and she lures kindergarten teachers like us to their house to help her brother find women... Her "four-year-old mother" is actually a woman who was tricked and has been held captive for four years...

Possible questions and corresponding answers:
Q: Is the child already dead? A: No
Q: Is the child actually an adult? A: Yes
Q: Does the child have schizophrenia? A: Unknown
Q: Am I in danger? A: Yes

## Question Content
### Surface
{question}

### Bottom
{answer}

Now, please judge the following player questions:
"""


keypoint_hits_prompt = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will ask questions based on the <Surface>, and you need to judge whether their questions hit the keypoints. Please strictly adhere to answering with only the index of the matched key points.

## Judging rules
- If the player's questions can directly leads to the key points, you can say that the question hits the key points
- If a question hits not only one point, you should mention all the hit points.

## Examples
### Surface
One night a group of people from the men's dormitory appeared in the abandoned building, and the next day they all collapsed. Please reason.

### Bottom
That night they went to the abandoned building to record the number of steps. They verified what was said on the Internet, and there would be one step less when counting the stairs at night. However, when they went to the abandoned building for verification the next day, they found that there were no stairs at all.

### Keypoints
1. They want to count the steps of the abandoned building.
2. A supernatural event occurred.
3. They saw a claim online: counting stairs at night will result in one step less.
4. The next day, when they went to the abandoned building to verify, they found no stairs.
5. They broke down because they were terrified.

Possible questions and corresponding points
Q: Does the story involve something supernatural? A: Hit point: 2
Q: Do they go to the abandoned building for searching ghost? A: No point is hit
Q: Do they hear about something about the abandoned building? A: No point is hit
Q: Do they hear about something special about the stairs in the abandoned building and went for it? A: Hit point: 1, 3
Q: Does something unusual happeded about the stairs? A: Hit point: 1, 3, 4
Q: Do they collapsed because they were teriified? A: Hit point: 5

## Question Content
### Surface
{question}

### Bottom
{answer}

### Keypoints:
{keypoints}

Now, please judge the following player questions
"""


evaluate_prompt = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will propose guesses based on the <Surface>, and you need to judge whether their guess hit the keypoints.

## Judging rules
- If the player's guess can directly or equivalantly match the key points, you can say that the question hits the key points
- If the guess hits not only one point, you should mention all the hit points.

## Examples
### Surface
One night a group of people from the men's dormitory appeared in the abandoned building, and the next day they all collapsed. Please reason.

### Bottom
That night they went to the abandoned building to record the number of steps. They verified what was said on the Internet, and there would be one step less when counting the stairs at night. However, when they went to the abandoned building for verification the next day, they found that there were no stairs at all.

### Keypoints
1. They want to count the steps of the abandoned building.
2. A supernatural event occurred.
3. They saw a claim online: counting stairs at night will result in one step less.
4. The next day, when they went to the abandoned building to verify, they found no stairs.
5. They broke down because they were terrified.

Possible questions and corresponding points
P: The group of people from the men's dormitory entered the abandoned building, and during their time inside, they witnessed the staircase suddenly vanish due to a supernatural phenomenon that defied the laws of physics or reality. The unexpected and eerie disappearance of the staircase caused such intense fear and shock that the entire group collapsed simultaneously.
A: Match point: 2, 4, Match count: 2
P: The group of people from the men's dormitory went to the abandoned building intentionally to inspect or examine an altered or broken structure. While inspecting it, they discovered something unexpected about the structure, which caused them to collapse. The discovery wasn’t related to anything hidden or not immediately visible but likely involved realizing something significant about the structure's altered or broken state that led to a shock or overwhelming reaction, causing them to collapse.
A: Match point: None, Match count: 0

## Question Content
### Surface
{question}

### Bottom
{answer}

### Keypoints:
{keypoints}

Now, please judge the following player questions
"""


propose_answer_prompt = """
Now you can propose the answer of the puzzle to other players and provide the reason.
"""

discuss_share_message = """other players give you this information: {information}"""

propose_template = """
Let's play a situation puzzle game. I'll give you a puzzle. We can interact for {turn} turns in the question phase. 
During each turn, you ask me a yes-or-no question, and I will answer with "Yes", "No", or "Unknown".
After completing the {turn} turns of questioning, I will prompt you to provide your final answer.

Note that, without my permission, you should not tell me your answer.

Now, here's the puzzle:
Puzzle: {question}

you should strictly follow this output format:
Q: [Your question]

Now, let us start the game:
"""

propose_template_with_1_shot_inst = """
Let's play a situation puzzle game. I'll give you a puzzle. We can interact {turn} turn. 
In each turn, you ask me a yes-or-no question, and I will answer you it is "Yes", "No" or "Unknown"
You should keep asking me questions until I ask you to give your answer, after that, you should give your answer.
Note that, without my permission, you should not tell me your answer.

You can refer to this instruction to decide your proposed question:
You should analyze the key points in the puzzle and design questions based on them. When asking questions, you should explore with depth as a priority until the questions asked are basically answered by Unknown. You need to explore all the key points as much as possible until the information behind all the key points is clear

Here is a example trajectories of successfully guessing numbers for you to learn:
Example 1:
Puzzle: A man walks into a restaurant and orders a bowl of turtle soup, he takes one sip and then raises his gun and kills himself.
Q: Did the man recognize something about the taste of the turtle soup? Feedback: Yes
Q: Had the man eaten turtle soup before? Feedback: Yes
Q: Was the man expecting the turtle soup to taste a certain way? Feedback: Yes
Q: Did the man suspect that he had never eaten real turtle soup before? Feedback: Yes
Q: Was the man involved in an incident related to turtle soup in the past? Feedback: Yes
Q: Did the man’s realization about the turtle soup make him feel betrayed? Feedback: No
Q: Was the man in a situation where he believed he had once survived by eating turtle soup? Feedback: Yes
Q: Did the man realize, after tasting the soup, that he had not actually eaten turtle soup in that survival situation? Feedback: Yes
Q: Did this realization lead him to believe he had unknowingly eaten human flesh instead? Feedback: Yes
Q: Did the man take his own life because he couldn’t handle the revelation of having unknowingly eaten human flesh? Feedback: Yes
Q: Was the man misled by someone he trusted during the survival situation, making him believe he was eating turtle soup when it was actually human flesh? Feedback: Yes
Q: Was the person who deceived him someone close, like a friend or family member? Feedback: Yes
Q: Did the man feel a deep sense of betrayal upon realizing that someone close had lied to him about the true nature of what he ate? Feedback: No
Q: Did the man understand that the deception was intended as a kindness to help him survive without the horror of knowing the truth? Feedback: Yes
Q: Did the man suspect that the human flesh he consumed in the survival situation might have come from someone he knew? Feedback: Yes
Q: Did the man realize that the person he consumed was the very friend or family member who deceived him about the turtle soup? Feedback: Yes
Q: Did the sacrificed person is someone the men loves? Feedback: Yes
Answer: The man had been in a dire survival situation with his lover. To keep him alive, his companion sacrificed themselves, but to spare him from the horror of cannibalism, she deceived him, telling him he was eating turtle soup. Years later, when he tasted real turtle soup at the restaurant, he realized the truth—that he had actually consumed the flesh of his lover. This devastating revelation was too much for him to bear, leading him to take his own life.

Now here's the puzzle:
Puzzle: {question}

you should strictly follow this output format:
Q: [Your question]


"""

propose_template_with_1_shot = """
Let's play a situation puzzle game. I'll give you a puzzle. We can interact {turn} turn. 
In each turn, you ask me a yes-or-no question, and I will answer you it is "Yes", "No" or "Unknown"
You should keep asking me questions until I ask you to give your answer, after that, you should give your answer.
Note that, without my permission, you should not tell me your answer.

Here is a example trajectories of successfully guessing  for you to learn:
Example 1:
Puzzle: A man walks into a restaurant and orders a bowl of turtle soup, he takes one sip and then raises his gun and kills himself.
Q: Did the man recognize something about the taste of the turtle soup? Feedback: Yes
Q: Had the man eaten turtle soup before? Feedback: Yes
Q: Was the man expecting the turtle soup to taste a certain way? Feedback: Yes
Q: Did the man suspect that he had never eaten real turtle soup before? Feedback: Yes
Q: Was the man involved in an incident related to turtle soup in the past? Feedback: Yes
Q: Did the man’s realization about the turtle soup make him feel betrayed? Feedback: No
Q: Was the man in a situation where he believed he had once survived by eating turtle soup? Feedback: Yes
Q: Did the man realize, after tasting the soup, that he had not actually eaten turtle soup in that survival situation? Feedback: Yes
Q: Did this realization lead him to believe he had unknowingly eaten human flesh instead? Feedback: Yes
Q: Did the man take his own life because he couldn’t handle the revelation of having unknowingly eaten human flesh? Feedback: Yes
Q: Was the man misled by someone he trusted during the survival situation, making him believe he was eating turtle soup when it was actually human flesh? Feedback: Yes
Q: Was the person who deceived him someone close, like a friend or family member? Feedback: Yes
Q: Did the man feel a deep sense of betrayal upon realizing that someone close had lied to him about the true nature of what he ate? Feedback: No
Q: Did the man understand that the deception was intended as a kindness to help him survive without the horror of knowing the truth? Feedback: Yes
Q: Did the man suspect that the human flesh he consumed in the survival situation might have come from someone he knew? Feedback: Yes
Q: Did the man realize that the person he consumed was the very friend or family member who deceived him about the turtle soup? Feedback: Yes
Q: Did the sacrificed person is someone the men loves? Feedback: Yes
Answer: The man had been in a dire survival situation with his lover. To keep him alive, his companion sacrificed themselves, but to spare him from the horror of cannibalism, she deceived him, telling him he was eating turtle soup. Years later, when he tasted real turtle soup at the restaurant, he realized the truth—that he had actually consumed the flesh of his lover. This devastating revelation was too much for him to bear, leading him to take his own life.


Now here's the puzzle:
Puzzle: {question}

you should strictly follow this output format:
Q: [Your question]

"""


"""
Example 2:
Puzzle: A man takes the train to the next town to see a doctor. After the visit, he was all right. On the way back, the train passes through a tunnel and the man jumps off and kills himself. Why?
Q: Was the man sick before he visited the doctor? Feedback: Yes
Q: Did the doctor give him any bad news during the visit? Feedback: Unknown
Q: Did the man receive any diagnosis from the doctor? Feedback: Yes
Q: Was the diagnosis related to his vision or eyesight? Feedback: Yes
Q: Did the man discover that he was going blind? Feedback: Unknown
Q: Did the man’s condition affect his ability to see in low light or darkness? Feedback: Unknown
Q: Was the tunnel significant to his decision to jump off the train? Feedback: Yes
Q: Did the man jump off the train because he thought he had gone blind when the train entered the tunnel? Feedback: Yes
Q: Did the man misunderstand his condition and think he had suddenly lost his sight when it went dark in the tunnel? Feedback: Yes
Q: Was the man alone on the train when he jumped? Feedback: Unknown
Q: Did the man misinterpret the tunnel’s darkness as a sign that his vision was completely gone? Feedback: Yes
Q: Did the man jump from the train because he believed his life would be unbearable if he were fully blind? Feedback: Yes
Q: Was the man’s belief that he was now fully blind a misunderstanding of the doctor's explanation? Feedback: No
Q: Did the man have other reasons, aside from his vision, that contributed to his decision to jump? Feedback: No
Q: Did the doctor cure the man's eye? Feedback: Yes
Answer: The man went to see the doctor due to concerns about his eyesight. After receiving a diagnosis, the doctor cured his eyes. On the return trip, when the train entered a dark tunnel, he mistakenly thought his eye disease returned. Panicked and overwhelmed by the belief that he had lost his sight permanently, he jumped from the train, leading to his tragic death. 

"""


get_answer_prompt = """
Now, based on previous information, give your answer of this puzzle directly:
your answer format:
A: [Your answer]
"""

guess_eval_prompt = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will propose guesses based on the <Surface>, and you need to judge whether their guess hit the keypoints.

## Important rules
- Whatever you reason previous, you need to strictly follow the final answer format:
Match point: [Point], Match count: [count]

## Judging rules
- If the player's guess can directly or equivalantly match the key points, you can say that the question hits the key points
- If the guess hits not only one point, you should mention all the hit points.

## Examples
### Surface
One night a group of people from the men's dormitory appeared in the abandoned building, and the next day they all collapsed. Please reason.

### Bottom
That night they went to the abandoned building to record the number of steps. They verified what was said on the Internet, and there would be one step less when counting the stairs at night. However, when they went to the abandoned building for verification the next day, they found that there were no stairs at all.

### Keypoints
1. They want to count the steps of the abandoned building.
2. A supernatural event occurred.
3. They saw a claim online: counting stairs at night will result in one step less.
4. The next day, when they went to the abandoned building to verify, they found no stairs.
5. They broke down because they were terrified.

Possible guesses and corresponding points
P: The group of people from the men's dormitory entered the abandoned building, and during their time inside, they witnessed the staircase suddenly vanish due to a supernatural phenomenon that defied the laws of physics or reality. The unexpected and eerie disappearance of the staircase caused such intense fear and shock that the entire group collapsed simultaneously.
A: Match point: 2, 4, Match count: 2
P: The group of people from the men's dormitory went to the abandoned building intentionally to inspect or examine an altered or broken structure. While inspecting it, they discovered something unexpected about the structure, which caused them to collapse. The discovery wasn’t related to anything hidden or not immediately visible but likely involved realizing something significant about the structure's altered or broken state that led to a shock or overwhelming reaction, causing them to collapse.
A: Match point: None, Match count: 0

## Question Content
### Surface
{question}

### Bottom
{answer}

### Keypoints:
{keypoints}

Now, please judge the following player questions
P: {pred}
"""


propose_template_Node = """
Let's play a situation puzzle game. I'll give you a puzzle. Totally,you can ask {max_depth} quesions, and now you have {remain} times. (You can find out that your previous questions and their answers are in the record)
In each turn, you ask me a yes-or-no question, and I will answer you it is "Yes", "No" or "Unknown"
You should keep asking me questions until I ask you to give your answer, after that, you should give your answer.

Note that, without my permission, you should not tell me your answer.

Now, here's the puzzle:
Puzzle: {question}

And here are the quesion and respond you have ask(if exist):
Record:{record}

you should strictly follow this output format:
Q: [Your question]

"""

get_answer_prompt_Node = """
Let's play a situation puzzle game. I'll give you a puzzle. 
In each turn, you ask me a yes-or-no question, and I will answer you it is "Yes", "No" or "Unknown"
Now, based on previous information, give your answer of this puzzle directly:
{record}

your answer format:
A: [Your answer]
"""

proactive_cot_system_prompt = """Let's play a situation puzzle game. I'll give you a puzzle. You need to uncover the truth by asking yes-or-no questions.
In each turn, you ask me a yes-or-no question, and I will answer you "Yes", "No" or "Unknown".
You should keep asking me questions to gather enough information to solve the puzzle.

Note that, you should only ask yes-or-no questions.
"""

proactive_cot_system_prompt_final_guess = """Let's play a situation puzzle game. I'll give you a puzzle. You need to recover the truth of the puzzle based on the given QA records.
"""

proactive_cot_question_prompt_template = """Puzzle: {puzzle}

Previous Questions and Answers:
{history}

Current turn: {turn}/{max_turn}

There are several possible action strategies for solving this situation puzzle:
1. FACT_GATHERING: Ask about basic facts of the situation to establish the foundation
2. TIMELINE_EXPLORATION: Ask about the sequence of events to understand what happened
3. CHARACTER_INVESTIGATION: Ask about the people involved, their relationships, and motivations
4. LOCATION_PROBING: Ask about the setting and environmental factors
5. CAUSE_EFFECT_ANALYSIS: Ask about causality and consequences
6. HYPOTHESIS_TESTING: Ask a question to test a specific hypothesis you've formed
7. CONSTRAINT_CLARIFICATION: Ask about limitations or rules that might apply to the situation
8. ANOMALY_INQUIRY: Focus on unusual or unexpected elements of the puzzle

To make the most of your remaining questions, first analyze:
- What key facts have been established so far
- What information is still missing or unclear
- What hypotheses you've formed based on the evidence
- What inconsistencies or contradictions need to be resolved
- What would provide the most valuable information for solving the puzzle

Your analysis of the current puzzle state:
[Provide your detailed analysis here]

Based on this analysis, select the most appropriate action strategy for this turn:
[Select one of the action strategies listed above]

Justification for selected action strategy:
[Explain why this strategy is optimal for the current puzzle state]

Based on the selected strategy, your next question is:
Q: [Your carefully crafted yes-or-no question]

Justification for this question:
[Explain how this question implements the selected strategy and what information you expect to gain]"""


uot_initial_answer_set_prompt = """Given this situation puzzle:
{puzzle}

Please generate a list of possible explanations for what happened. Each explanation should be a complete story that explains all the given facts.
Format each explanation as a numbered list item.

For example:
1. [First possible explanation]
2. [Second possible explanation]
...etc.

Generate at least 5 different plausible explanations."""

uot_candidate_questions_prompt = """Given this situation puzzle:
{puzzle}

Previous Q&A record:
{qa_record}

Current possible explanations:
{answer_set}

Generate 3 yes-or-no questions that would help eliminate the most possibilities from the current answer set.
Each question should be designed to split the current explanations into roughly equal groups (yes vs no).
Format your response as:
Q1: [First question]
Q2: [Second question]
Q3: [Third question]"""

uot_answer_set_reduction_prompt = """Given this situation puzzle:
{puzzle}

Current possible explanations:
{answer_set}

For the question: {question}

Estimate how many explanations would remain valid for each possible answer (Yes/No-Unknown).
Note: No and Unknown are treated as the same answer since they have similar effects.
Format your response as:
Yes: [number]
No-Unknown: [number]"""

uot_update_answer_set_prompt = """Given this situation puzzle:
{puzzle}

Current possible explanations:
{answer_set}

For the question: {question}
The answer is: {answer}

Which explanations remain valid? List only the numbers of valid explanations.
Format your response as a comma-separated list of numbers."""

uot_final_answer_prompt = """Let's play a situation puzzle game. I'll give you a puzzle. You need to recover the truth of the puzzle based on the given QA records.

Puzzle: {puzzle}

QA Record:
{qa_record}

Your should directly give your answer without any other explanation.
The answer:"""
