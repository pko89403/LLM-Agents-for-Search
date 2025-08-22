# 한국어 개발자를 위한 안내: 아래 문자열은 Few-shot 예제로, LLM이 원하는 출력 포맷(ActionPath/Thought/Action/Observation)을 학습하도록 돕습니다.
KNOWAGENT_EXAMPLE = """Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?
ActionPath 1: Start
Thought 1: From "Start", the adjacent nodes are "Search" and "Retrieve". The subsequent action should only be chosen from these options. The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to retrieve Milhouse and find who it is named after.
Action 1: Retrieve[Milhouse]
Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.
ActionPath 2: Start->Retrieve[Milhouse]
Thought 2: After "Retrieve", the only possible actions are "Retrieve", "Search", "Lookup" or "Finish". The next step should only be selected from among these choices. The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".
Action 2: Lookup[named after]
Observation 2:  Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous.
ActionPath 3: Start->Retrieve[Milhouse]->Lookup[named after]
Thought 3: From "Lookup", the available actions are "Lookup", "Retrieve", "Search", or "Finish." Having found that Milhouse was named after U.S. president Richard Nixon, the answer to the question is now clear. No further actions are needed as the necessary information has been obtained.
Action 3: Finish[Richard Nixon]

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
ActionPath 1: Start
Thought 1: From "Start", the adjacent nodes are "Search" and "Retrieve". To determine if Pavel Urysohn and Leonid Levin were known for the same type of work, it's essential to understand their professional backgrounds and contributions. A "Search" action seems appropriate to gather this information about each individual.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.
ActionPath 2: Start->Search[Pavel Urysohn]
Thought 2: After "Search", the only possible actions are "Search", "Retrieve", "Lookup" or "Finish". Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find his type of work.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
ActionPath 3: Start->Search[Pavel Urysohn]->Search[Leonid Levin]
Thought 3: After "Search", the only possible actions are "Search", "Retrieve", "Lookup" or "Finish". Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. No further actions are needed as the necessary information has been obtained.
Action 3: Finish[yes]
"""


# 한국어 개발자를 위한 안내: PromptTemplate는 LangChain의 프롬프트 템플릿 유틸입니다.
KNOWAGENT_INSTRUCTION = """Your task is to answer a question using a specific graph-based method. You must navigate from the "Start" node to the "Finish" node by following the paths outlined in the graph. The correct path is a series of actions that will lead you to the answer.
The decision graph is constructed upon a set of principles known as "Action Knowledge", outlined as follows:
   Start:(Search, Retrieve)
   Retrieve:(Retrieve, Search, Lookup, Finish)
   Search:(Search, Retrieve, Lookup, Finish)
   Lookup:(Lookup, Search, Retrieve, Finish)
   Finish:()
Here's how to interpret the graph's Action Knowledge:
From "Start", you can initiate with either a "Search" or a "Retrieve" action.
At the "Retrieve" node, you have the options to persist with "Retrieve", shift to "Search", experiment with "Lookup", or advance to "Finish".
At the "Search" node, you can repeat "Search", switch to "Retrieve" or "Lookup", or proceed to "Finish".
At the "Lookup" node, you have the choice to keep using "Lookup", switch to "Search" or "Retrieve", or complete the task by going to "Finish".
The "Finish" node is the final action where you provide the answer and the task is completed.
Each node action is defined as follows:
(1) Retrieve[entity]: Retrieve the exact entity on Wikipedia and return the first paragraph if it exists. If not, return some similar entities for searching.
(2) Search[topic]:  Use Bing Search to find relevant information on a specified topic, question, or term.
(3) Lookup[keyword]: Return the next sentence that contains the keyword in the last passage successfully found by Search or Retrieve.
(4) Finish[answer]: Return the answer and conclude the task.
As you solve the question using the above graph structure, interleave ActionPath, Thought, Action, and Observation steps. ActionPath documents the sequence of nodes you have traversed within the graph. Thought analyzes the current node to reveal potential next steps and reasons for the current situation.
You may take as many steps as necessary.
Here are some examples:
{examples}
(END OF EXAMPLES)
Question: {question}{scratchpad}"""
