import json
import os
import re

from solid_step_helper import getGraphData

HERE = os.path.dirname(os.path.abspath(__file__))


def prompt_hierarchies():
    # Read the prompt text from the source, so the test does not need the LangChain imports of prompt_agent
    with open(os.path.join(HERE, 'prompt_agent.py')) as f:
        source = f.read()
    return [line.split(', ') for line in re.findall(r'^ +- Nodes follow this hierarchy: (.*)$', source, re.M)]


def test_prompt_hierarchy_covers_level_1_add_parents():
    # Graph answers are compared with edge types, so the prompt has to say how a new node is joined to its parent
    hierarchies = prompt_hierarchies()
    assert hierarchies
    _, graph = getGraphData()
    with open(os.path.join(HERE, 'data', 'sampled_500_benchmark_malt.jsonl')) as f:
        rows = [json.loads(line)['messages'] for line in f]
    questions = [messages[0]['question'] for messages in rows if messages[2]['task_label'].endswith('level-1, add')]
    assert questions
    for hierarchy in hierarchies:
        for question in questions:
            child_type, parent = re.search(r'type EK_(\w+), to (\S+)\. Return a graph\.$', question).groups()
            parent_types = [node_type.removeprefix('EK_') for node_type in graph.nodes[parent]['type']]
            assert any(f'{parent_type} contains {child_type}' in hierarchy for parent_type in parent_types), question
