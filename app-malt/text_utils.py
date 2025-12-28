from loguru import logger
from dataclasses import dataclass, field
import re

from netarena.agent_client import PromptType


BASE_PROMPT = """
You need to behave like a network engineer who processes graph data to answer user queries about capacity planning.

Your task is to generate the Python code needed to process the network graph to answer the user question or request. The code should take the form of a function named process_graph that accepts a single input argument graph_data and returns a single object return_object.

Graph Structure:
- The input graph_data is a networkx graph object with nodes and edges
- The graph is directed and each node has a 'name' attribute to represent itself
- Each node has a 'type' attribute in the format of EK_TYPE. 'type' must be a list, which can include ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN']
- Each node can have other attributes depending on its type
- Each directed edge also has a 'type' attribute, including RK_CONTAINS, RK_CONTROL

Important Guidelines:
- Check relationships based on edge, check name based on node attribute
- Nodes follow this hierarchy: CHASSIS contains PACKET_SWITCH, JUPITER contains SUPERBLOCK, SUPERBLOCK contains AGG_BLOCK, AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT
- Each PORT node has an attribute 'physical_capacity_bps'. For example, a PORT node name is ju1.a1.m1.s2c1.p3
- When calculating capacity of a node, sum the physical_capacity_bps on the PORT of each hierarchy contained in this node
- When updating a graph, always create a graph copy, do not modify the input graph
- To find a node based on type, check the name and type list. For example, [node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']]
- node.startswith will not work for the node name. You have to check the node name with the node['name']


Output Format:
- Do not use multi-layer functions. The output format should only return one object
- The return_object must be a JSON object with three keys: 'type', 'data', and 'updated_graph'
- The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table', or 'graph'
- The 'data' key should contain the data needed to render the output:
    * If output type is 'text': 'data' should contain a string
    * If output type is 'list': 'data' should contain a list of items
    * If output type is 'table': 'data' should contain a list of lists where each list represents a row in the table
    * If output type is 'graph': 'data' should contain a graph JSON
- The 'updated_graph' key should always contain the updated graph as "graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)"
    
Response Format:
- Your reply should always start with string "\\nAnswer:\\n"
- You should generate a function called "def process_graph"
- All of your output should only contain the defined function without example usages, no additional text, and displayed in a Python code block
- Do not include any package imports in your answer
"""

COT_PROMPT = """Please think step by step and provide your output."""

QA_TEMPLATE = """
Question: {input}

Answer:
```python
${{Code that will answer the user question or request}}
```
"""

FEWSHOT_PROMPT = f"""Here are a few examples of questions and their corresponding answers:"""

PROMPT_SUFFIX = f"""Begin! Remember to ensure that you generate valid Python code in the following format: {QA_TEMPLATE}"""

EXAMPLE_LIST = [
            {
                "question": "Update the physical capacity value of ju1.a3.m2.s2c4.p10 to 72. Return a graph.",
                "answer": r'''def process_graph(graph_data):    
                                graph_copy = copy.deepcopy(graph_data)    
                                for node in graph_copy.nodes(data=True):        
                                    if node[1]['name'] == 'ju1.a3.m2.s2c4.p10' and 'EK_PORT' in node[1]['type']:            
                                        node[1]['physical_capacity_bps'] = 72           
                                    break    
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)    
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'.
                                return return_object''',
            },
            {
                "question": "Add new node with name new_EK_PORT_82 type EK_PORT, to ju1.a2.m4.s3c6. Return a graph.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = copy.deepcopy(graph_data)
                                graph_copy.add_node('new_EK_PORT_82', type=['EK_PORT'], physical_capacity_bps=1000)
                                graph_copy.add_edge('ju1.a2.m4.s3c6', 'new_EK_PORT_82', type=['RK_CONTAINS'])
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)  
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
            {
                "question": "Count the EK_PACKET_SWITCH in the ju1.a2.dom. Return only the count number.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = graph_data.copy()
                                count = 0
                                for node in graph_copy.nodes(data=True):
                                    if 'EK_PACKET_SWITCH' in node[1]['type'] and node[0].startswith('ju1.a2.'):
                                        count += 1
                                # the return_object should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
            {
                "question": "Remove ju1.a1.m4.s3c6.p1 from the graph. Return a graph.",
                "answer": r'''def process_graph(graph_data):
                                graph_copy = graph_data.copy()
                                node_to_remove = None
                                for node in graph_copy.nodes(data=True):
                                    if node[0] == 'ju1.a1.m4.s3c6.p1':
                                        node_to_remove = node[0]
                                        break
                                if node_to_remove:
                                    graph_copy.remove_node(node_to_remove)
                                graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)
                                # in the return_object, it should be a json object with three keys, 'type', 'data' and 'updated_graph'. 
                                return return_object''',
            },
        ]


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


def extract_code_output(answer: str):
    '''
    Extract only the def process_graph() funtion from the output of LLM
    :param answer: output of LLM
    :return: cleaned function
    '''
    # If the function has "process_graph" in it, assume that this is the answer code.
    regex = re.compile(r'def\s+([a-zA-Z_0-9]*process_graph[a-zA-Z_0-9]*)')
    answer = regex.sub('def process_graph', answer)
    start = answer.find("def process_graph")
    if start == -1:
        return ""  # Return empty string if process_graph function not found
        
    # Find the code block ending
    code_block_end = answer.find("```", answer.find("```", start))
    
    # If we found proper code block markers
    if code_block_end != -1:
        clean_code = answer[start:code_block_end].strip()
    else:
        # Fallback to extract until the end of the string
        clean_code = answer[start:].strip()
    
    # Remove the lines that have "import package" in the code
    clean_code = '\n'.join([line for line in clean_code.split('\n') if not line.strip().startswith("import")])

    return clean_code
