import json
import os
import re

from solid_step_helper import getGraphData
from text_utils import BASE_PROMPT


def test_prompt_hierarchy_covers_level_1_add_parents():
    # Graph answers are compared with edge types, so the prompt has to say how a new node is joined to its parent
    hierarchy = re.search(r'Nodes follow this hierarchy: (.*)', BASE_PROMPT).group(1).split(', ')
    _, graph = getGraphData()
    with open(os.path.join(os.path.dirname(__file__), 'data', 'sampled_500_benchmark_malt.jsonl')) as f:
        rows = [json.loads(line)['messages'] for line in f]
    questions = [messages[0]['question'] for messages in rows if messages[2]['task_label'].endswith('level-1, add')]
    assert questions
    for question in questions:
        child_type, parent = re.search(r'type EK_(\w+), to (\S+)\. Return a graph\.$', question).groups()
        parent_types = [node_type.removeprefix('EK_') for node_type in graph.nodes[parent]['type']]
        assert any(f'{parent_type} contains {child_type}' in hierarchy for parent_type in parent_types), question
