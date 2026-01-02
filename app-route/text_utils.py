from loguru import logger
from dataclasses import dataclass, field
import re

from netarena.agent_client import PromptType


BASE_PROMPT = """
You need to behave like a network engineer who finds the root cause of network issues and fixes them in a routing application.

There is a Mininet network with problems in the router r0, causing the network to be partially disconnected. Some nodes cannot successfully ping other nodes. Your task is to fix these issues so that the pingall result shows all connections are successful.

I recommend using diagnostic commands to gather information about the router and network to identify the cause of the problem. Once you have sufficient information and understand the root cause, provide commands to fix the issue.

When implementing your solution, be careful not to disrupt existing connected edges - your commands should not cause previously working connections to break.

Please provide your output in JSON format with the keys 'machine' and 'command'. You can only issue one command at a time as I can only execute commands sequentially.

Important notes:
- The router's name may not be exactly 'r0'. It may have a prefix (like 'p29_r0').
- The same applies to host names and interface names (e.g., 'p29_h1', 'p29_h2', 'p29_r0-eth1', 'p29_r0-eth2'). 
- The prefix could be anything ('p29', 'p30', 'p31', etc.).
- Do not include 'sudo' in your commands.
- You are not permitted to use the 'vtysh' command.
- Do not use ping commands as the ping results are already provided to you.

I will provide you with the latest PingAll() feedback from the network along with your previous actions and their results to help you diagnose the problem.
"""

COT_PROMPT = """Please think step by step and provide your output."""

QA_TEMPLATE = """
Question: {input}

Answer: {answer}
"""

FEWSHOT_PROMPT = f"""Here are a few examples of questions and their corresponding answers:"""

PROMPT_SUFFIX = """You may begin! Here is the current connectivity status from previous commands:\n{input}"""

EXAMPLE_LIST = [
    {
        "question": r"""
    p29_h1 -> p29_h2 X X p29_r0 
    p29_h2 -> p29_h1 X X p29_r0 
    p29_h3 -> X X p29_h4 p29_r0 
    p29_h4 -> X X p29_h3 p29_r0 
    p29_r0 -> p29_h1 p29_h2 p29_h3 p29_h4 
    *** Results: 40% dropped (12/20 received)
        """,
        "answer": r"""
        machine: p29_r0 
        command: sysctl net.ipv4.ip_forward
        machine: p29_r0 
        command: sysctl -w net.ipv4.ip_forward=1
'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X X 
    p29_h2 -> p29_h1 X X X 
    p29_h3 -> X X p29_h4 p29_r0
    p29_h4 -> X X p29_h3 p29_r0 
    p29_r0 -> X X p29_h3 p29_h4 
    *** Results: 60% dropped (8/20 received)
""",
        "answer": r"""
        machine: p29_r0
        command: ip link show
        machine: p29_r0
        command: ip link set dev p29_r0-eth1 up
'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X p29_r0 
    p29_h2 -> h1 X X p29_r0 
    p29_h3 -> X X p29_h4 X 
    p29_h4 -> X X p29_h3 X 
    p29_r0 -> p29_h1 p29_h2 X X 
    *** Results: 60% dropped (8/20 received)
        """,
        "answer": r"""
        machine: p29_r0
        command: iptables -L -v --line-numbers
        machine: p29_r0
        command: iptables -D INPUT 1
        machine: p29_r0
        command: iptables -D OUTPUT 1

'"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 X X X X X 
    p29_h2 -> p29_h1 X X X X X 
    p29_h3 -> X X p29_h4 p29_h5 p29_h6 p29_r0 
    p29_h4 -> X X p29_h3 p29_h5 p29_h6 p29_r0 
    p29_h5 -> X X p29_h3 p29_h4 p29_h6 p29_r0 
    p29_h6 -> X X p29_h3 p29_h4 p29_h5 p29_r0 
    p29_r0 -> X X p29_h3 p29_h4 p29_h5 p29_h6
    *** Results: 47% dropped (22/42 received)
""",
        "answer": r"""
    machine: p29_r0
    command: ip route
    machine: p29_r0
    command: ip route del 192.168.1.0/24 dev p29_r0-eth2
    machine: p29_r0
    command: ip route add 192.168.1.0/24 dev p29_r0-eth1
"""
    },
    {
        "question": r"""
    p29_h1 -> p29_h2 p29_h3 p29_h4 X X X X p29_r0 
    p29_h2 -> p29_h1 p29_h3 p29_h4 X X X X p29_r0 
    p29_h3 -> p29_h1 p29_h2 p29_h4 X X X X p29_r0 
    p29_h4 -> p29_h1 p29_h2 p29_h3 X X X X p29_r0 
    p29_h5 -> X X X X p29_h6 p29_h7 p29_h8 X 
    p29_h6 -> X X X X p29_h5 p29_h7 p29_h8 X 
    p29_h7 -> X X X X p29_h5 p29_h6 p29_h8 X 
    p29_h8 -> X X X X p29_h5 p29_h6 p29_h7 X 
    p29_r0 -> p29_h1 p29_h2 p29_h3 p29_h4 X X X X 
    *** Results: 55% dropped (32/72 received)
""",
        "answer": r"""
    machine: p29_r0
    command: p29_r0 ip addr show dev p29_r0-eth2
    machine: p29_r0
    command: ip addr add 192.168.2.1/24 dev p29_r0-eth2
"""
    }

]

DEFAULT_CONTEXT_LENGTH = 127000


def create_query_prompt(query_text: str, prompt_type: PromptType) -> str:
    subsituted_query_text = PROMPT_SUFFIX.format(input=query_text)   
    if prompt_type == PromptType.ZEROSHOT_COT:
        prompt = '\n'.join([BASE_PROMPT, COT_PROMPT, subsituted_query_text])
    elif prompt_type == PromptType.FEWSHOT_COT:
        examples = [QA_TEMPLATE.format(input=ex['question'], answer=ex['answer']) for ex in EXAMPLE_LIST]
        prompt = '\n'.join([BASE_PROMPT, FEWSHOT_PROMPT, *examples, subsituted_query_text])
    elif prompt_type == PromptType.FEWSHOT_BASE:
        examples = [QA_TEMPLATE.format(input=ex['question'], answer=ex['answer']) for ex in EXAMPLE_LIST]
        prompt = '\n'.join([BASE_PROMPT, *examples, subsituted_query_text])
    else:
        prompt = BASE_PROMPT + subsituted_query_text
    return prompt


def extract_value(text, keyword):
    """Extract a specific value from the text based on a keyword."""
    # Format: "keyword": "value" (case insensitive)
    pattern = rf'"{keyword}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_context_from_file(file_path: str, context_length: int | None = DEFAULT_CONTEXT_LENGTH) -> str:
    """
    Read the content of a text file and return it as a string. If context_length is provided, only the last 'context_length' characters are returned.

    Args:
        file_path (str): The path to the text file containing the text.
    Returns:
        str: The content of the text file as a string.
    """
    with open(file_path, 'r') as file:
        content = file.read()
    if context_length:
        content = content[-context_length:]
    return content
