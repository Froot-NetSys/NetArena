import json
import traceback
from dotenv import load_dotenv
import openai
import pandas as pd
from collections import Counter
from prototxt_parser.prototxt import parse
import os
from solid_step_helper import clean_up_llm_output_func
import networkx as nx
import jsonlines
import json
import re
import time
import sys
import numpy as np
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.chains import LLMChain 
import warnings
from langchain._api import LangChainDeprecationWarning
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import AzureOpenAIEmbeddings
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)

prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """

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

class BasePromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        # old_prompt = """
        # Generate the Python code needed to process the network graph to answer the user question or request. The network graph data is stored as a networkx graph object, the Python code you generate should be in the form of a function named process_graph that takes a single input argument graph_data and returns a single object return_object. The input argument graph_data will be a networkx graph object with nodes and edges.
        # The graph is directed and each node has a 'name' attribute to represent itself.
        # Each node has a 'type' attribute, in the format of EK_TYPE. 'type' must be a list, can include ['EK_SUPERBLOCK', 'EK_CHASSIS', 'EK_RACK', 'EK_AGG_BLOCK', 'EK_JUPITER', 'EK_PORT', 'EK_SPINEBLOCK', 'EK_PACKET_SWITCH', 'EK_CONTROL_POINT', 'EK_CONTROL_DOMAIN'].
        # Each node can have other attributes depending on its type.
        # Each directed edge also has a 'type' attribute, include RK_CONTAINS, RK_CONTROL.
        # You should check relationship based on edge, check name based on node attribute. 
        # Nodes has hierarchy: CHASSIS contains PACKET_SWITCH, JUPITER contains SUPERBLOCK, SUPERBLOCK contains AGG_BLOCK, AGG_BLOCK contains PACKET_SWITCH, PACKET_SWITCH contains PORT.
        # Each PORT node has an attribute 'physical_capacity_bps'. For example, a PORT node name is ju1.a1.m1.s2c1.p3. 
        # When calculating capacity of a node, you need to sum the physical_capacity_bps on the PORT of each hierarchy contains in this node.
        # When update a graph, always create a graph copy, do not modify the input graph. 
        # To find node based on type, check the name and type list. For example, [node[0] == 'ju1.a1.m1.s2c1' and 'EK_PACKET_SWITCH' in node[1]['type']].

        # Do not use multi-layer function. The output format should only return one object. The return_object will be a JSON object with two keys, 'type' and 'data' and "updated_graph". The 'type' key should indicate the output format depending on the user query or request. It should be one of 'text', 'list', 'table' or 'graph'.
        # The 'data' key should contain the data needed to render the output. If the output type is 'text' then the 'data' key should contain a string. If the output type is 'list' then the 'data' key should contain a list of items.
        # The 'updated_graph' key should contain the updated graph, no matter what the output type is. It should be a graph json "graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)".
        # If the output type is 'table' then the 'data' key should contain a list of lists where each list represents a row in the table.If the output type is 'graph' then the 'data' key should be a graph json "graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)".
        # node.startswith will not work for the node name. you have to check the node name with the node['name'].

        # Context: When the user requests to make changes to the graph, it is generally appropriate to return the graph. 
        # In the Python code you generate, you should process the networkx graph object to produce the needed output.

        # Remember, your reply should always start with string "\nAnswer:\n", and you should generate a function called "def process_graph".
        # All of your output should only contain the defined function without example usages, no additional text, and display in a Python code block.
        # Do not include any package import in your answer.
        # """

        prompt = """
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

        return prompt    


class ZeroShot_CoT_PromptAgent:
    def __init__(self):
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        cot_prompt_prefix = """
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

        Please think step by step and provide your output.
        """
        return cot_prompt_prefix


class FewShot_Basic_PromptAgent(ZeroShot_CoT_PromptAgent):
    def __init__(self):
        super().__init__()
        self.examples = EXAMPLE_LIST
        self.cot_prompt_prefix = super().generate_prompt()
    
    def get_few_shot_prompt(self):
        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nAnswer: {answer}"
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=self.examples,
            example_prompt=example_prompt,
            prefix=self.cot_prompt_prefix + "Here are some example question-answer pairs:\n",
            suffix=prompt_suffix,
            input_variables=["input"]
        )
        return few_shot_prompt


class FewShot_Semantic_PromptAgent(ZeroShot_CoT_PromptAgent):
    def __init__(self):
        self.examples = EXAMPLE_LIST
        self.cot_prompt_prefix = super().generate_prompt()

    def get_few_shot_prompt(self, query):
        embeddings = AzureOpenAIEmbeddings(
            model="text-embedding-3-large"
        )
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            self.examples,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            Chroma,
            # This is the number of examples to produce.
            k=1)

        example_prompt = PromptTemplate(
            input_variables=["question", "answer"],
            template="Question: {question}\nAnswer: {answer}"
        )
        
        few_shot_prompt = FewShotPromptTemplate(
            examples=example_selector.select_examples({"question": query}),
            example_prompt=example_prompt,
            prefix=self.cot_prompt_prefix + "Here are some example question-answer pairs:\n",
            suffix=prompt_suffix,
            input_variables=["input"]
        )
        return few_shot_prompt

class ReAct_PromptAgent(BasePromptAgent):
    def __init__(self):
        self.base_prompt_prefix = BasePromptAgent.generate_prompt(self)
        # Now set prompt_prefix manually instead of through super().__init__()
        self.prompt_prefix = self.generate_prompt()

    def generate_prompt(self):
        react_prompt_prefix = """ 
                                Answer the following question as best you can. Please use a tool if you need to.
                                
                                """
        react_prompt = react_prompt_prefix + self.base_prompt_prefix

        return react_prompt
