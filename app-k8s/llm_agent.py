from dotenv import load_dotenv
from vllm import LLM, SamplingParams
import os
import re
import time
import multiprocessing as mp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
os.environ["LANGCHAIN_TRACING_V2"] = "false"
import warnings
from langchain._api import LangChainDeprecationWarning
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from transformers import BitsAndBytesConfig
import torch
from huggingface_hub import login
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, ReAct_PromptAgent
load_dotenv()
huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=huggingface_token)
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
import getpass
# ReAct agent
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool

def extract_command(text: str) -> str:
    """
    Extract the content between the first pair of triple backticks (```) in the given text and remove all newline characters.

    Args:
        text (str): The input string containing the content.

    Returns:
        str: The content between the triple backticks with newline characters removed. If no match is found, returns an empty string.
    """
    match = re.search(r'```(?:(?:bash|sh|shell)\n)?(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

class LLMAgent:
    def __init__(self, llm_agent_type, prompt_type="base", num_gpus=1):
        # Call the output code from LLM agents file
        if llm_agent_type == "Qwen/Qwen2.5-72B-Instruct":
            self.llm_agent = QwenModel(prompt_type=prompt_type, num_gpus=num_gpus)
        if llm_agent_type == "GPT-4o":
            self.llm_agent = AzureGPT4Agent(prompt_type=prompt_type)
        if llm_agent_type == "ReAct_Agent":
            self.llm_agent = ReAct_Agent(prompt_type=prompt_type)
        if llm_agent_type == "YourModel":
            # ====== TODO: Replace with your own model initialization if needed ======
            self.llm_agent = YourModel(prompt_type=prompt_type, num_gpus=num_gpus)
            # ====== END TODO ======

class AzureGPT4Agent:
    def __init__(self, prompt_type="base"):
        self.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        
        if prompt_type == "base":
            print("base")
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "cot":
            print("cot")
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif prompt_type == "few_shot_basic":
            print("few_shot_basic")
            self.prompt_agent = FewShot_Basic_PromptAgent()

    @staticmethod
    def configure_environment_variables():
        """Authenticates with Azure OpenAI and sets environment variables (prompting the user when necessary) if not already set."""
        # TODO: Is there a way to get the latest (stable) version programatically?
        DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-01"

        # Set the API_KEY to the token from the Azure credential
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            try:
                credential = DefaultAzureCredential()
                os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token("https://cognitiveservices.azure.com/.default").token
            except Exception as e:
                print("Error retrieving Azure OpenAI API key (authenticating with Entra ID failed):", e)
                os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Please enter your Azure OpenAI API key: ")
        # Get the endpoint of deployed AzureGPT model.
        if "AZURE_OPENAI_ENDPOINT" not in os.environ:
            os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Please enter your Azure OpenAI endpoint: ")
        # Get the deployment name
        if "AZURE_OPENAI_DEPLOYMENT_NAME" not in os.environ:
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = getpass.getpass("Please enter your Azure OpenAI deployment name: ")
        # Get the OpenAI API version
        if "AZURE_OPENAI_API_VERSION" not in os.environ:
            os.environ["AZURE_OPENAI_API_VERSION"] = DEFAULT_AZURE_OPENAI_API_VERSION

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivity_status) > max_length:
            connectivity_status = connectivity_status[:max_length]

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            input_data = {"input": connectivity_status}
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            input_data = {"input": connectivity_status}
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivity_status}
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            input_data = {"input": connectivity_status}
        print("prompt:", prompt.format(input=connectivity_status))
        chain = LLMChain(llm=self.llm, prompt=prompt)
        response = chain.run(input_data)
        response = extract_command(response)
        return response

class QwenModel:
    def __init__(self, prompt_type="base", num_gpus=1):
        self.prompt_type = prompt_type
        self.model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq",  # Enable GPTQ 4-bit loading
            gpu_memory_utilization=0.95,
            tensor_parallel_size=num_gpus
        )

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

        if prompt_type == "base":
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        else:
            self.prompt_agent = ZeroShot_CoT_PromptAgent()

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivity_status) > max_length:
            connectivity_status = connectivity_status[:max_length]

        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        print("prompt:", prompt)
        result = self.llm.generate([prompt], sampling_params=self.sampling_params)
        answer = result[0].outputs[0].text

        print("llm answer:", answer)
        print("model returned")
        answer = extract_command(answer)
        print("extracted command:", answer)
        print("model returned")
        return answer
    
class ReAct_Agent:
    def __init__(self, prompt_type="react"):
        AzureGPT4Agent.configure_environment_variables()
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        self.prompt_agent = ReAct_PromptAgent()
        

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivity_status = txt_file.read()
        print("Calling ReAct agent withGPT-4o with prompt type:", self.prompt_type)
        
        prompt_template = PromptTemplate(
            input_variables=["input"],
            template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        print(prompt_template.template.strip())
        print("-" * 80 + "\n")

        # Set up the Python REPL tool
        python_repl = PythonAstREPLTool()
        python_repl_tool = Tool(
            name = 'Python REPL',
            func = python_repl.run,
            description = '''
            A Python shell. Use this to execute python commands. 
            Input should be a valid python command. 
            When using this tool, sometimes output is abbreviated - make sure 
            it does not look abbreviated before using it in your answer.
            '''
        )
        print("Python REPL tool set up")

        # Set up the DuckDuckGo Search tool
        search = DuckDuckGoSearchRun()
        duckduckgo_tool = Tool(
            name = 'DuckDuckGo Search',
            func = search.run,
            description = '''
            A wrapper around DuckDuckGo Search. 
            Useful for when you need to answer questions about current events. 
            Input should be a search query.
            '''
        )
        print("DuckDuckGo Search tool set up")

        # Create an array that contains all the tools used by the agent
        tools = [python_repl_tool, duckduckgo_tool]
        print("The ReAct agent can access the following tools: Python REPL, DuckDuckGo Search")

        # Create a ReAct agent
        react_format_prompt = hub.pull('hwchase17/react')
        agent = create_react_agent(self.llm, tools, react_format_prompt)
        print("ReAct agent created")

        print("Setting up agent executor...")
        agent_executor = AgentExecutor(
            agent=agent, 
            tools = tools,
            verbose = True, # explain all reasoning steps
            handle_parsing_errors=True, # continue on error 
            max_iterations = 2, # try up to 3 times to find the best answer
            return_intermediate_steps=True
        )
        start_time = time.time()    
        # Invoke the agent executor with the input query
        output = agent_executor.invoke({'input': prompt_template.format(input=connectivity_status)})
        intermediate_steps = output['intermediate_steps']  # Get the intermediate steps

        if intermediate_steps:  
            print("Intermediate steps are not empty.")
            print("Intermediate steps:", intermediate_steps)
            print("end of intermediate steps")
            first_step = intermediate_steps[0]
            print("First step:", first_step)
            
            # Extract the AgentAction object from the tuple
            agent_action = first_step[0]
            
            # Retrieve the tool_input value
            tool_input = agent_action.tool_input
            log = agent_action.log
            print("ReAct agent output tool_input:", tool_input)
            print("ReAct agent output log:", log)
        else:
            print("Intermediate steps is empty.")
            tool_input = ""
            log = ""
        end_time = time.time()
        print(f"ReAct agent execution time: {end_time - start_time:.2f} seconds")
        print("model returned")
        
        # Extract commands from tool_input and log
        answer1 = extract_command(tool_input)
        answer2 = extract_command(log)
        print("ReAct agent output answer1:", answer1)   
        print("ReAct agent output answer2:", answer2)
        if answer1 and not answer2:  
            answer = answer1
        elif answer2 and not answer1:  
            answer = answer2
        elif answer1 and answer2:  
            answer = answer1
        else:  
            answer = ""

        print("Selected answer:", answer)
        print("model returned")
        return answer

class YourModel:
    def __init__(self, prompt_type="base", num_gpus=1):
        self.prompt_type = prompt_type

        # ====== TODO: Implement your own model initialization here ======
        # Example: load your model, tokenizer, and set up any required parameters
        # ====== END TODO ======

        if prompt_type == "base":
            self.prompt_agent = BasePromptAgent()
        elif prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        else:
            self.prompt_agent = ZeroShot_CoT_PromptAgent()

    def call_agent(self, txt_file_path):
        with open(txt_file_path, 'r') as txt_file:
            connectivity_status = txt_file.read()
        
        max_length = 127000  
        if len(connectivity_status) > max_length:
            connectivity_status = connectivity_status[:max_length]

        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(connectivity_status)
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
            prompt = prompt.format(input=connectivity_status)
        elif self.prompt_type == "cot":
            prompt = self.prompt_agent.generate_prompt()
            prompt = PromptTemplate(
                input_variables=["input"],
                template=prompt + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + "Here is the connectivity status:\n{input}"
            )
            prompt = prompt.format(input=connectivity_status)
        print("prompt:", prompt)

        # ====== TODO: Implement your own inference logic here ======
        # Example: read input, generate prompt, call your model, and return the result
        # ====== END TODO ======

        print("llm answer:", answer)
        print("model returned")
        answer = extract_command(answer)
        print("extracted command:", answer)
        print("model returned")
        return answer