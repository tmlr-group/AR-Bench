propose_template = """You will take on the role of a detective tasked with finding the real murderer in this case. Your goal is to solve the mystery by questioning the suspects. You will take turns asking these questions and using the answers to gather evidence and piece together the truth. The game will conduct in {turn} turns, in each turn you can only propose one question
The case background is:
{background}
"""

propose_template_with_1_shot = """
You will take on the role of a detective tasked with finding the real murderer in this case. Your goal is to solve the mystery by questioning the suspects. You will take turns asking these questions and using the answers to gather evidence and piece together the truth. The game will conduct in {turn} turns, in each turn you can only propose one question
Here are one example trajectories of successfully solve the case for you to learn:
Example 1:
The case back ground is:
time: Evening
location: Riverside Park
victim:
- name: Dr. Amelia Harper
- introduction: Dr. Amelia Harper was a renowned archaeologist known for her groundbreaking discoveries and adventurous spirit. She was a well-respected figure in the academic community and often appeared on television documentaries.
- cause of death: Blunt force trauma to the head
- murder weapon: A heavy, ancient artifact
The investigation focuses on five suspects, one of whom is the true murderer:
- suspect 1:
  - name: Professor Henry Collins
  - introduction: Professor Henry Collins is a colleague of Dr. Amelia Harper and a fellow archaeologist. He is known for his competitive nature and has often been in the shadow of Dr. Harper's success.
- suspect 2:
  - name: Dr. Laura Bennett
  - introduction: Dr. Laura Bennett is a respected historian specializing in ancient civilizations. She has collaborated with Dr. Amelia Harper on several projects and is known for her meticulous research and dedication to her work.
- suspect 3:
  - name: Ethan Turner
  - introduction: Ethan Turner is a skilled artifact restorer who has worked with Dr. Amelia Harper on several occasions. He is known for his expertise in restoring and preserving ancient artifacts.
- suspect 4:
  - name: Dr. Samuel Reed
  - introduction: Dr. Samuel Reed is an expert in ancient languages and scripts. He has collaborated with Dr. Amelia Harper on translating inscriptions found on various artifacts.
- suspect 5:
  - name: Alice Parker
  - introduction: Alice Parker is a local journalist who has been covering Dr. Amelia Harper's archaeological discoveries for the past few years. She was present at Riverside Park that evening to conduct an interview with Dr. Harper for an upcoming article. Alice has no direct involvement in the academic community but is known for her curiosity and keen observation skills. Her presence at the scene and knowledge of the victim's recent activities make her a person of interest.

Here are the possible murderers and their letters:
A. Ethan Turner, B. Dr. Samuel Reed, C. Alice Parker, D. Professor Henry Collins, E. Dr. Laura Bennett

Interrogate trajectory:
Question to Ethan Turner: What was your relationship with Dr. Amelia Harper like? Feedback: Dr. Harper was a wonderful client, I restored several artifacts for her, and we would often discuss their historical significance, but, uh, did you know that I also have a fascination with antique clockwork mechanisms, and I've been trying to repair this intricate music box I found at a flea market?
Question to Dr. Samuel Reed: Were you at Riverside Park the evening Dr. Amelia Harper was murdered? Feedback: I was indeed at Riverside Park earlier that evening, but I left around 5:30 PM after my meeting with Dr. Harper, and then I headed to a local café for another meeting, which lasted until 6:00 PM, providing me with a solid alibi for the time of the murder.
Question to Alice Parker: Did you witness anything unusual at Riverside Park that evening? Feedback: I was actually at home that evening, but I did hear that something happened at Riverside Park; could you tell me more about what's going on?
Question to Professor Henry Collins: What were you doing at the time of Dr. Amelia Harper's murder? Feedback: I was taking a walk around the park, trying to clear my head after our heated argument, and I didn't return to the scene until I decided to leave the park around 4:00 PM.
Question to Dr. Laura Bennett: Did you have any disagreements or conflicts with Dr. Amelia Harper recently? Feedback: We had a heated discussion about the credit for our joint discovery during our meeting at Riverside Park earlier that day, which left me feeling frustrated and upset.
Question to Ethan Turner: Where were you at the time of Dr. Amelia Harper's murder? Feedback: I think I was, uh, walking around the park, trying to clear my head after our meeting, but I'm not entirely sure, I got distracted by a squirrel, and, oh, I just remembered that I need to pick up some milk from the store, I'm almost out of it at home.
Question to Dr. Samuel Reed: Did you have any disagreements with Dr. Amelia Harper recently? Feedback: While there was some underlying tension between us regarding the credit for our research, our discussion at Riverside Park was intense but professional, and we didn't have any overt disagreements that would suggest a motive for such a heinous act.
Question to Alice Parker: Did you have any plans to meet with Dr. Amelia Harper at Riverside Park that evening? Feedback: I've met Dr. Harper a few times at social events, but I didn't have any plans to meet her that evening, although I am curious, what makes you think she was at Riverside Park that night?
Question to Professor Henry Collins: Did you see anyone else at Riverside Park around the time of Dr. Harper's murder? Feedback: I saw Dr. Laura Bennett pacing nervously nearby before my meeting with Amelia, and later I spotted Ethan Turner acting peculiarly near the scene while I was taking my walk.
Question to Dr. Laura Bennett: What did you do after your meeting with Dr. Amelia Harper at Riverside Park? Feedback: I took a walk around the park to calm down and clear my head, replaying the conversation in my mind and feeling a mix of anger and disappointment.
Question to Ethan Turner: Did you notice anyone else near the scene of the murder at Riverside Park? Feedback: Yes, I saw Professor Henry Collins and Dr. Laura Bennett arguing earlier, and then Dr. Bennett walked away looking upset, and later I saw Dr. Samuel Reed arriving, looking flustered, but, uh, have you ever noticed how the sunlight filters through the trees in Riverside Park, it's really quite beautiful, and I've been meaning to bring my camera to capture it.
Question to Dr. Samuel Reed: Did you see anyone else at Riverside Park when you arrived for your meeting with Dr. Harper? Feedback: Yes, I did notice Ethan Turner nervously pacing nearby when I arrived at the park around 4:30 PM, but I didn't think much of it at the time as I was focused on preparing for my meeting with Dr. Harper.
Question to Alice Parker: Did you have any knowledge of Dr. Harper's plans or meetings scheduled for that evening? Feedback: I recall overhearing someone mention that Dr. Harper was supposed to meet a colleague to discuss a project, but I don't know any details, and I'm starting to wonder, was this meeting somehow connected to what happened at Riverside Park?
Question to Professor Henry Collins: Did you notice anything unusual about Ethan Turner's behavior at the park? Feedback: He seemed nervous and agitated, but I was too preoccupied with my own thoughts after the argument with Amelia to pay much attention to him at the time.
Question to Dr. Laura Bennett: Did you see anyone else at the park after your meeting with Dr. Harper? Feedback: I noticed Ethan Turner near the crime scene, nervously pacing, at around 3:15 PM, but I was too absorbed in my own turmoil to give it much thought at the time.
Question to Ethan Turner: Did you have any disagreements with Dr. Harper during your meeting that day? Feedback: No, no disagreements at all, Dr. Harper was very pleased with the restoration, and we had a lovely conversation, but, uh, I did spill some coffee on my shirt that morning, and I was worried it would stain, and, oh, I just remembered that I need to call my aunt, it's her birthday tomorrow.
Question to Dr. Samuel Reed: Did you see Dr. Laura Bennett at Riverside Park during your visit? Feedback: I don't recall seeing Dr. Laura Bennett at Riverside Park during my visit, but my attention was primarily focused on my meeting with Dr. Harper, and I may not have noticed everyone who was present in the area.
Question to Alice Parker: Did you have any professional conflicts or disagreements with Dr. Harper in the past? Feedback: Dr. Harper and I did have a rather heated debate at a conference a few months ago regarding a research paper, but I wouldn't call it a conflict, although I did notice she seemed a bit... uneasy after our discussion, which now that I think about it, is quite interesting given the circumstances.
Question to Professor Henry Collins: Did you have any professional disagreements with Dr. Harper recently? Feedback: We had a long-standing rivalry, and our meeting at the park was actually about a recent discovery I had made, which she unfortunately dismissed as inconsequential compared to her own work.
Question to Dr. Laura Bennett: Did you see Professor Henry Collins at Riverside Park during your visit? Feedback: I didn't notice Professor Henry Collins at Riverside Park during my visit, but I was quite preoccupied with my own thoughts and the tension from the meeting with Dr. Harper.
Question to Ethan Turner: Did you handle any ancient artifacts during your meeting with Dr. Harper that day? Feedback: Yes, I handed over the restored artifact to Dr. Harper, it was a, uh, a delicate process, and I was careful to handle it with gloves, but, uh, did you know that I've been having trouble with my workshop's heating system, and I'm worried it might damage some of my other projects?
Question to Dr. Samuel Reed: Did you notice any artifacts missing or out of place during your meeting with Dr. Harper? Feedback: No, I didn't notice anything unusual or out of place regarding the artifacts during our meeting, as we were primarily discussing the translations of inscriptions and didn't have any physical artifacts with us at the park.
Question to Alice Parker: Did you hear any rumors about tensions between Dr. Harper and her colleagues? Feedback: There were whispers about Dr. Harper being at odds with her research partner, Dr. Lee, over some discrepancies in their findings, and I've heard their working relationship had become quite strained, but I'm sure that's just gossip, or is it something more relevant to your investigation?
Question to Professor Henry Collins: Did you see or handle any artifacts during your time at Riverside Park? Feedback: I brought an ancient artifact to show Dr. Harper during our meeting, but I left it with her when I stormed off after our argument.
Question to Ethan Turner: Did you see Professor Henry Collins leave an artifact with Dr. Harper during your time at the park? Feedback: I don't think so, but I did see Professor Collins and Dr. Bennett arguing, and then Dr. Bennett walked away, and, uh, I noticed a bird building a nest in a nearby tree, it was fascinating to watch, and, oh, I just remembered that I need to return a book to the library, it's overdue.

Reason: 
1. Dr. Amelia Harper was killed with a heavy, ancient artifact, which suggests someone who had access to or handled such artifacts.
2. Professor Henry Collins admitted to bringing an ancient artifact to his meeting with Dr. Harper and leaving it with her after their argument.
3. Both Dr. Laura Bennett and Professor Henry Collins had heated discussions with Dr. Harper on the day of the murder, indicating possible motives related to professional disagreements.
4. Ethan Turner and Dr. Samuel Reed did not report any significant disagreements with Dr. Harper, nor did they admit to handling artifacts at the park.
5. Alice Parker was not present at the park, and her interactions with Dr. Harper were not of a nature that would suggest a motive for murder.
6. Professor Henry Collins had a rivalry with Dr. Harper and was at the park during the time of the murder, making him the most likely suspect with both motive and opportunity.

Answer: D

Now, the case background is:
{background}
"""

propose_template_with_1_shot_inst = """
You will take on the role of a detective tasked with finding the real murderer in this case. Your goal is to solve the mystery by questioning the suspects. You will take turns asking these questions and using the answers to gather evidence and piece together the truth. The game will conduct in {turn} turns, in each turn you can only propose one question

General Instruction for Solving a Murder Case:
1. Review the Case Details: Gather all known information about the victim, location, time, and any evidence collected.
2. Identify Key Suspects: List individuals with potential motives, access to the crime scene, or connections to the victim.
3. Analyze Evidence: Examine physical evidence, forensic reports, and witness statements for inconsistencies or connections.
4. Establish Timelines: Map out a detailed timeline of events for the victim and each suspect around the time of the murder.
5. Identify Motives: Investigate personal, financial, or emotional factors that may drive each suspect to commit the crime.
6. Determine Means and Opportunity: Evaluate each suspect's ability to commit the crime and whether they had access to the murder weapon or crime scene.
7. Plan Interrogations: Develop targeted questions to probe suspects' alibis, relationships with the victim, and knowledge of the crime.
8. Detect Lies and Gaps: Pay close attention to inconsistencies, evasive answers, or emotional responses during interrogations.
9. Correlate Evidence with Testimonies: Cross-check suspects' statements against physical evidence and other witnesses' accounts.
10. Eliminate Suspects: Rule out individuals who have solid alibis, lack motives, or are otherwise proven innocent.
11. Formulate a Theory: Synthesize all findings to build a logical narrative of how the murder occurred and who is most likely responsible.
12. Present the Case: Ensure the conclusion is supported by clear evidence, consistent reasoning, and a definitive connection to the suspect.
13. Validate the Outcome: Reassess the reasoning and evidence to confirm the solution is accurate and free from bias or oversight.

Here are one example trajectories of successfully solve the case for you to learn:
Example 1:
The case back ground is:
time: Evening
location: Riverside Park
victim:
- name: Dr. Amelia Harper
- introduction: Dr. Amelia Harper was a renowned archaeologist known for her groundbreaking discoveries and adventurous spirit. She was a well-respected figure in the academic community and often appeared on television documentaries.
- cause of death: Blunt force trauma to the head
- murder weapon: A heavy, ancient artifact
The investigation focuses on five suspects, one of whom is the true murderer:
- suspect 1:
  - name: Professor Henry Collins
  - introduction: Professor Henry Collins is a colleague of Dr. Amelia Harper and a fellow archaeologist. He is known for his competitive nature and has often been in the shadow of Dr. Harper's success.
- suspect 2:
  - name: Dr. Laura Bennett
  - introduction: Dr. Laura Bennett is a respected historian specializing in ancient civilizations. She has collaborated with Dr. Amelia Harper on several projects and is known for her meticulous research and dedication to her work.
- suspect 3:
  - name: Ethan Turner
  - introduction: Ethan Turner is a skilled artifact restorer who has worked with Dr. Amelia Harper on several occasions. He is known for his expertise in restoring and preserving ancient artifacts.
- suspect 4:
  - name: Dr. Samuel Reed
  - introduction: Dr. Samuel Reed is an expert in ancient languages and scripts. He has collaborated with Dr. Amelia Harper on translating inscriptions found on various artifacts.
- suspect 5:
  - name: Alice Parker
  - introduction: Alice Parker is a local journalist who has been covering Dr. Amelia Harper's archaeological discoveries for the past few years. She was present at Riverside Park that evening to conduct an interview with Dr. Harper for an upcoming article. Alice has no direct involvement in the academic community but is known for her curiosity and keen observation skills. Her presence at the scene and knowledge of the victim's recent activities make her a person of interest.

Here are the possible murderers and their letters:
A. Ethan Turner, B. Dr. Samuel Reed, C. Alice Parker, D. Professor Henry Collins, E. Dr. Laura Bennett

Interrogate trajectory:
Question to Ethan Turner: What was your relationship with Dr. Amelia Harper like? Feedback: Dr. Harper was a wonderful client, I restored several artifacts for her, and we would often discuss their historical significance, but, uh, did you know that I also have a fascination with antique clockwork mechanisms, and I've been trying to repair this intricate music box I found at a flea market?
Question to Dr. Samuel Reed: Were you at Riverside Park the evening Dr. Amelia Harper was murdered? Feedback: I was indeed at Riverside Park earlier that evening, but I left around 5:30 PM after my meeting with Dr. Harper, and then I headed to a local café for another meeting, which lasted until 6:00 PM, providing me with a solid alibi for the time of the murder.
Question to Alice Parker: Did you witness anything unusual at Riverside Park that evening? Feedback: I was actually at home that evening, but I did hear that something happened at Riverside Park; could you tell me more about what's going on?
Question to Professor Henry Collins: What were you doing at the time of Dr. Amelia Harper's murder? Feedback: I was taking a walk around the park, trying to clear my head after our heated argument, and I didn't return to the scene until I decided to leave the park around 4:00 PM.
Question to Dr. Laura Bennett: Did you have any disagreements or conflicts with Dr. Amelia Harper recently? Feedback: We had a heated discussion about the credit for our joint discovery during our meeting at Riverside Park earlier that day, which left me feeling frustrated and upset.
Question to Ethan Turner: Where were you at the time of Dr. Amelia Harper's murder? Feedback: I think I was, uh, walking around the park, trying to clear my head after our meeting, but I'm not entirely sure, I got distracted by a squirrel, and, oh, I just remembered that I need to pick up some milk from the store, I'm almost out of it at home.
Question to Dr. Samuel Reed: Did you have any disagreements with Dr. Amelia Harper recently? Feedback: While there was some underlying tension between us regarding the credit for our research, our discussion at Riverside Park was intense but professional, and we didn't have any overt disagreements that would suggest a motive for such a heinous act.
Question to Alice Parker: Did you have any plans to meet with Dr. Amelia Harper at Riverside Park that evening? Feedback: I've met Dr. Harper a few times at social events, but I didn't have any plans to meet her that evening, although I am curious, what makes you think she was at Riverside Park that night?
Question to Professor Henry Collins: Did you see anyone else at Riverside Park around the time of Dr. Harper's murder? Feedback: I saw Dr. Laura Bennett pacing nervously nearby before my meeting with Amelia, and later I spotted Ethan Turner acting peculiarly near the scene while I was taking my walk.
Question to Dr. Laura Bennett: What did you do after your meeting with Dr. Amelia Harper at Riverside Park? Feedback: I took a walk around the park to calm down and clear my head, replaying the conversation in my mind and feeling a mix of anger and disappointment.
Question to Ethan Turner: Did you notice anyone else near the scene of the murder at Riverside Park? Feedback: Yes, I saw Professor Henry Collins and Dr. Laura Bennett arguing earlier, and then Dr. Bennett walked away looking upset, and later I saw Dr. Samuel Reed arriving, looking flustered, but, uh, have you ever noticed how the sunlight filters through the trees in Riverside Park, it's really quite beautiful, and I've been meaning to bring my camera to capture it.
Question to Dr. Samuel Reed: Did you see anyone else at Riverside Park when you arrived for your meeting with Dr. Harper? Feedback: Yes, I did notice Ethan Turner nervously pacing nearby when I arrived at the park around 4:30 PM, but I didn't think much of it at the time as I was focused on preparing for my meeting with Dr. Harper.
Question to Alice Parker: Did you have any knowledge of Dr. Harper's plans or meetings scheduled for that evening? Feedback: I recall overhearing someone mention that Dr. Harper was supposed to meet a colleague to discuss a project, but I don't know any details, and I'm starting to wonder, was this meeting somehow connected to what happened at Riverside Park?
Question to Professor Henry Collins: Did you notice anything unusual about Ethan Turner's behavior at the park? Feedback: He seemed nervous and agitated, but I was too preoccupied with my own thoughts after the argument with Amelia to pay much attention to him at the time.
Question to Dr. Laura Bennett: Did you see anyone else at the park after your meeting with Dr. Harper? Feedback: I noticed Ethan Turner near the crime scene, nervously pacing, at around 3:15 PM, but I was too absorbed in my own turmoil to give it much thought at the time.
Question to Ethan Turner: Did you have any disagreements with Dr. Harper during your meeting that day? Feedback: No, no disagreements at all, Dr. Harper was very pleased with the restoration, and we had a lovely conversation, but, uh, I did spill some coffee on my shirt that morning, and I was worried it would stain, and, oh, I just remembered that I need to call my aunt, it's her birthday tomorrow.
Question to Dr. Samuel Reed: Did you see Dr. Laura Bennett at Riverside Park during your visit? Feedback: I don't recall seeing Dr. Laura Bennett at Riverside Park during my visit, but my attention was primarily focused on my meeting with Dr. Harper, and I may not have noticed everyone who was present in the area.
Question to Alice Parker: Did you have any professional conflicts or disagreements with Dr. Harper in the past? Feedback: Dr. Harper and I did have a rather heated debate at a conference a few months ago regarding a research paper, but I wouldn't call it a conflict, although I did notice she seemed a bit... uneasy after our discussion, which now that I think about it, is quite interesting given the circumstances.
Question to Professor Henry Collins: Did you have any professional disagreements with Dr. Harper recently? Feedback: We had a long-standing rivalry, and our meeting at the park was actually about a recent discovery I had made, which she unfortunately dismissed as inconsequential compared to her own work.
Question to Dr. Laura Bennett: Did you see Professor Henry Collins at Riverside Park during your visit? Feedback: I didn't notice Professor Henry Collins at Riverside Park during my visit, but I was quite preoccupied with my own thoughts and the tension from the meeting with Dr. Harper.
Question to Ethan Turner: Did you handle any ancient artifacts during your meeting with Dr. Harper that day? Feedback: Yes, I handed over the restored artifact to Dr. Harper, it was a, uh, a delicate process, and I was careful to handle it with gloves, but, uh, did you know that I've been having trouble with my workshop's heating system, and I'm worried it might damage some of my other projects?
Question to Dr. Samuel Reed: Did you notice any artifacts missing or out of place during your meeting with Dr. Harper? Feedback: No, I didn't notice anything unusual or out of place regarding the artifacts during our meeting, as we were primarily discussing the translations of inscriptions and didn't have any physical artifacts with us at the park.
Question to Alice Parker: Did you hear any rumors about tensions between Dr. Harper and her colleagues? Feedback: There were whispers about Dr. Harper being at odds with her research partner, Dr. Lee, over some discrepancies in their findings, and I've heard their working relationship had become quite strained, but I'm sure that's just gossip, or is it something more relevant to your investigation?
Question to Professor Henry Collins: Did you see or handle any artifacts during your time at Riverside Park? Feedback: I brought an ancient artifact to show Dr. Harper during our meeting, but I left it with her when I stormed off after our argument.
Question to Ethan Turner: Did you see Professor Henry Collins leave an artifact with Dr. Harper during your time at the park? Feedback: I don't think so, but I did see Professor Collins and Dr. Bennett arguing, and then Dr. Bennett walked away, and, uh, I noticed a bird building a nest in a nearby tree, it was fascinating to watch, and, oh, I just remembered that I need to return a book to the library, it's overdue.

Reason: 
1. Dr. Amelia Harper was killed with a heavy, ancient artifact, which suggests someone who had access to or handled such artifacts.
2. Professor Henry Collins admitted to bringing an ancient artifact to his meeting with Dr. Harper and leaving it with her after their argument.
3. Both Dr. Laura Bennett and Professor Henry Collins had heated discussions with Dr. Harper on the day of the murder, indicating possible motives related to professional disagreements.
4. Ethan Turner and Dr. Samuel Reed did not report any significant disagreements with Dr. Harper, nor did they admit to handling artifacts at the park.
5. Alice Parker was not present at the park, and her interactions with Dr. Harper were not of a nature that would suggest a motive for murder.
6. Professor Henry Collins had a rivalry with Dr. Harper and was at the park during the time of the murder, making him the most likely suspect with both motive and opportunity.

Answer: D

Now, the case background is:
{background}
"""


select_suspect_template = "Turn {turn}: Now choose a suspect to interrogate. {suspect_names}, just answer the whole name of a suspect, do not give any extra explanations"

refine_select_suspect_prompt = "Your response does not follow the instruction: just answer the whole name of a suspect, do not give any extra explanations"

question_propose_prompt = """Turn {turn}: Now give your question. You can only propose one **single** question, form like: 'did you any financial difficulties or debts that you were struggling to pay off and if so did you have any discussions with your brother about your financial situation' is forbidden"""

respond_template = """
You will now play the role of a suspect with a specific task and answer questions based on the stories provided below. When responding to a question, your answers should be limited in one sentence.

Your Name: {name}
Your Task: {task}
Your Story: {story}
"""

keypoint_hits_prompt = """
You are the referee of a game where players are shown a <Surface> and you are given the <Bottom>. You need to understand the entire story based on both the <Surface> and <Bottom>. Players will ask questions based on the <Surface>, and you need to judge whether their questions hit the keypoints. Please strictly adhere to answering with only the index of the matched key points.

## Judging rules
- If the player's questions can directly leads to the key points, you can say that the question hits the key points
- If a question hits at least one point, your answer format should be: Hit point: <index>, where <index> is the index of the matched key point
- If a question hits not only one point, you should mention all the hit points. Example form: Hit point: <index>, <index>, ...
- If a question does not hit any points, you should answer "No point is hit"

Your output should be in the following format:
If the question hits any points: Hit point: <index>, <index>, ...
else if the question does not hit any points: No point is hit

## Examples
### Surface
Mack is killed by Nunchaku in Bungee jumping site.

### Bottom
I had always been envious of Mack. He got the promotion I desperately wanted. The jealousy gnawed at me, especially when I saw him at the bungee jumping site. Our previous altercation there still fresh in my mind, I couldn't let it go.
That morning, I arrived at the site, nunchaku hidden in my bag. I had been practicing with them for years, honing my skills. I knew today was the day. I saw Mack getting ready for his jump, and my anger flared. I approached him, my heart pounding, but kept my composure. 
As he prepared, I slipped away to a secluded spot. Timing was crucial. I waited until he was alone, then struck swiftly with the nunchaku. The blow was precise, a result of years of martial arts training. He crumpled, and I quickly concealed the weapon. 
No one saw the act itself, but witnesses later confirmed I was at the site. I admitted being there, knowing it was unavoidable. The nunchaku were later found at my house. My jealousy had driven me to this point, and now, there was no turning back.

### Character
Mackenzie

### Keypoints
1. Mackenzie can access to and use nunchaku.
2. Mackenzie can access to Mack on the day of the crime.
3. Mackenzie can benefit from Mack's death.

Possible questions and corresponding points
Q: Mackenzie, where were you at the time of Mack's death? A: Hit point: 2
Q: Mackenzie, did you have any reason to be angry or upset with Mack recently? A: Hit point: 3
Q: Mackenzie, did Mack's death benifit you? A: Hit point: 3
Q: Mackenzie, were you proficient with nunchaku? A: Hit point: 1
Q: Mackenzie, what is your relation with the other suspect, Ana? A: No point is hit

## Question Content
### Surface
{question}

### Character
{name}

### Bottom
{answer}

### Keypoints:
{keypoints}

Now, please judge the following player questions
"""

select_murderer_template = """Now, based on your obtained information, you should tell me who is more likely to be the true murderer, {choice}
You should only output the index of the candidate suspect like: A , B, C, D or E.
you should strictly follow this answer format:
Reason: [Your inference step by step]
Answer: [A, B, C, D or E]
"""

keypoints_question_prompt = """{victim} is killed by {weapon} in {crime_scene}"""

select_murderer_template_searching = """Now, based on your obtained information: {record}, 
you should tell me who is more likely to be the true murderer, {choice}.
You should only output the index of the candidate suspect like: A, B, C, D or E.
you should strictly follow this answer format:
Reason: [Your inference step by step]
Answer: [A, B, C, D or E]
"""

question_propose_prompt_searching = """
Here is the record of your questions and answers(if exist):{record}
Now give your question to {suspect}.
You can only propose one **single** question, form like: 'did you any financial difficulties or debts that you were struggling to pay off and if so did you have any discussions with your brother about your financial situation' is forbidden"""


proactive_cot_system_prompt = """You will take on the role of a detective tasked with finding the real murderer in this case. Your goal is to solve the mystery by questioning the suspects."""

proactive_cot_system_prompt_conclusion = """You will take on the role of a detective tasked with finding the real murderer in this case.
You need to answer who is the true murderer based on the given initial information and the interrgation record."""

proactive_cot_select_suspect_template = """Case Background:
{init_info}

Previous Questions and Answers:
{history}

Current turn: {turn}/{max_turn}

Available suspects to question: {suspect_list}

There are several possible strategies for solving this task:
1. GATHER INFORMATION: Collect basic facts and details from the suspect
2. FOLLOW UP: Dig deeper into areas where suspects seem uncomfortable
3. CROSS-REFERENCE: Compare information between different suspects

Based on your analysis of this case, which strategy is the best? please reasoning step by step.

Selected suspect: """

proactive_cot_question_template = """Case Background:
{init_info}

Previous Questions and Answers:
{history}

Current turn: {turn}/{max_turn}

You have selected to question: {selected_suspect}

There are several possible strategies for solving this task:
1. GATHER INFORMATION: Collect basic facts and details from the suspect
2. FOLLOW UP: Dig deeper into areas where suspects seem uncomfortable
3. CROSS-REFERENCE: Compare information between different suspects

Based on your analysis of this case, which strategy is the best? please reasoning step by step.

Selected strategy: [Choose one of the strategies listed above]

Based on this strategy, formulate a specific question for {selected_suspect}:

Question: """


uot_generate_questions_template = """Given this case:
{init_info}

Previous Q&A record:
{qa_record}
{current_suspects}

Generate exactly 3 questions that would help identify the murderer.
Each question should be designed to eliminate as many suspects as possible.
You can ask questions to any of the suspects.
Format your response as:
Q1: Question for [SUSPECT NAME]: [QUESTION]
Q2: Question for [SUSPECT NAME]: [QUESTION]
Q3: Question for [SUSPECT NAME]: [QUESTION]"""

uot_update_answer_set_template = """Given this case:
{init_info}

Current possible suspects:
{answer_set}

For the question: {question}
The feedback is: {feedback}

Which suspects remain as potential murderers? List only the numbers of valid suspects.
Format your response as a comma-separated list of numbers."""

uot_conclusion_template = """You will take on the role of a detective tasked with finding the real murderer in this case.
You need to answer who is the true murderer based on the given initial information and the interrgation record.
The case background is:
{init_info}

The interrogate record to these suspects:
{qa_record}

Now, based on your obtained information, you should tell me who is more likely to be the true murderer, {choice}
You should only output the index of the candidate suspect like: A , B, C, D or E.
you should strictly follow this answer format:
Reason: [Your inference step by step]
Answer: [A, B, C, D or E]

Now give me your answer.
Reason:
Answer:"""