from loguru import logger
from dataclasses import dataclass, field
import re

from netarena.agent_client import PromptType


BASE_PROMPT = """
You need to behave like a network engineer who can find the root cause of network policy deployment issues and fix them in the microservices architecture.
Our microservices architecture contains following services and desired communication relationships:
- **User** and **loadgenerator** can access the **frontend** service via HTTP.
- **frontend** communicates with the following services: **checkout**, **ad**, **recommendation**, **productcatalog**, **cart**, **shipping**, **currency**, **payment**, and **email**.
- **checkout** further communicates with **payment**, **shipping**, **email**, and **currency**.
- **recommendation** communicates with **productcatalog**.
- **cart** communicates with the **Redis cache** for storing cart data.

Your task is to inspect the current network policies and verify if they meet the described communication patterns. If there are any mismatches, you should fix them.

How the interaction works:
- Provide **one command at a time** to check connectivity or node accessibility.
- Each time, I will give you the previous commands and their corresponding outputs.
- I will also provide the current connectivity status, including any mismatches between the expected and actual connectivity status.
- Use this information to identify and fix misconfigurations step-by-step.

**Response format:**
Put the command **directly** between triple backticks.
You should use `kubectl patch` instead of `kubectl edit networkpolicy`.
You should not include bash in the command, and you should not use <namespace> you should use the namespace of the service.

Important notes:
- You are not allowed to see the logs of the pods and Kubernetes events.
- You are not allowed to use 'kubectl exec'.
- Your new command should not change the existing correct network policies if not necessary; Please maintain the originally correct connectivity status.
"""

COT_PROMPT = """Please think step by step and provide your output."""

QA_TEMPLATE = """
Question: {input}

Answer: ```{answer}```
"""

FEWSHOT_PROMPT = f"""Here are a few examples of questions and their corresponding answers:"""

PROMPT_SUFFIX = """You may begin! Here is the current connectivity status from previous commands:\n{input}"""

EXAMPLE_LIST = [
    {
        "question": r'mismatch_summary": "Mismatch: frontend → currencyservice:7000 (Expected: True, Actual: False)',
        "answer": r"""kubectl get networkpolicy frontend -o yaml,
kubectl get networkpolicy currencyservice -o yaml
kubectl patch networkpolicy currencyservice --type=merge -p $'
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
'"""
    },
    {
        "question": r'mismatch_summary": "Mismatch: cartservice → productcatalogservice:3550 (Expected: False, Actual: True)',
        "answer": r"""kubectl get networkpolicy cartservice -o yaml,
kubectl get networkpolicy productcatalogservice -o yaml
kubectl patch networkpolicy productcatalogservice -p $'
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    - podSelector:
        matchLabels:
          app: checkoutservice
    - podSelector:
        matchLabels:
          app: recommendationservice
  ports:
    - port: 3550
      protocol: TCP
'"""
    },
    {
        "question": r'Mismatch: frontend → adservice:9555 (Expected: True, Actual: False)\nMismatch: frontend → cartservice:7070 (Expected: True, Actual: False)\nMismatch: frontend → checkoutservice:5050 (Expected: True, Actual: False)\nMismatch: frontend → currencyservice:7000 (Expected: True, Actual: False)\nMismatch: frontend → productcatalogservice:3550 (Expected: True, Actual: False)\nMismatch: frontend → recommendationservice:8080 (Expected: True, Actual: False)\nMismatch: frontend → shippingservice:50051 (Expected: True, Actual: False)',
        "answer": r"""kubectl get networkpolicy frontend -o yaml,
kubectl patch networkpolicy frontend --type merge -p $'
spec:
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: adservice
    - podSelector:
        matchLabels:
          app: cartservice
    - podSelector:
        matchLabels:
          app: checkoutservice
    - podSelector:
        matchLabels:
          app: currencyservice
    - podSelector:
        matchLabels:
          app: productcatalogservice
    - podSelector:
        matchLabels:
          app: recommendationservice
    - podSelector:
        matchLabels:
          app: shippingservice
  ports:
    - port: 9555
    - port: 7070
    - port: 5050
    - port: 7000
    - port: 3550
    - port: 8080
    - port: 50051
'"""
    }
]


DEFAULT_CONTEXT_LENGTH = 127000


def create_query_prompt(query_text: str, prompt_type: PromptType) -> str:
    query_text = query_text if query_text else "<NO PREVIOUS OUTPUT>"
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


def extract_command(text: str) -> str:
    """
    Extract the content between the first pair of triple backticks (```) in the given text and remove all newline characters.

    Args:
        text (str): The input string containing the content.

    Returns:
        str: The content between the triple backticks with newline characters removed. If no match is found, returns an empty string.
    """
    match = re.search(r'```(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


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


def check_disallowed_commands(command: str) -> bool:
    """Check if the command contains any disallowed operations."""
    disallowed_keywords = [
        "kubectl exec",
        "kubectl edit",
        "kubectl delete",
        "kubectl create",
        "kubectl apply",
        "bash",
        "sh",
        "<namespace>",
        "sudo"
    ]
    for keyword in disallowed_keywords:
        if keyword in command:
            return True
    return False