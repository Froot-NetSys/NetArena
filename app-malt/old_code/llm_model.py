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
warnings.simplefilter("ignore", category=LangChainDeprecationWarning)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from huggingface_hub import login
from vllm import LLM, SamplingParams
from prompt_agent import BasePromptAgent, ZeroShot_CoT_PromptAgent, FewShot_Basic_PromptAgent, FewShot_Semantic_PromptAgent, ReAct_PromptAgent
import torch.multiprocessing as mp
# For Azure OpenAI models
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import getpass

# TOML support (Python 3.11+ has tomllib built-in)
try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Load environ variables from .env, will not override existing environ variables
load_dotenv()

def login_huggingface():
    """Login to Huggingface Hub. Only needed for local LLMs."""
    huggingface_token = os.getenv("HUGGINGFACE_TOKEN")
    if huggingface_token:
        login(token=huggingface_token)
    else:
        print("Warning: HUGGINGFACE_TOKEN not found. You may need to set it for downloading gated models.")


mp.set_start_method('spawn', force=True)

prompt_suffix = """Begin! Remember to ensure that you generate valid Python code in the following format:

        Answer:
        ```python
        ${{Code that will answer the user question or request}}
        ```
        Question: {input}
        """

class GoogleGeminiAgent:
    def __init__(self, prompt_type="base"):
        # Only ask for API key when choosing Gemini.
        if "GOOGLE_API_KEY" not in os.environ:
            os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.prompt_type = prompt_type
        
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()
        
    def call_agent(self, query):
        print("Calling Google Gemini with prompt type:", self.prompt_type)

        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(query)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt.example_prompt)
            print("\nPrefix:", prompt.prefix)
            print("\nSuffix:", prompt.suffix)
        else:
            print(prompt.template.strip())
        print("-" * 80 + "\n")

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code


class AzureGPT4Agent:
    DEFAULT_TOML_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
    DEFAULT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "config.template.toml")
    DEFAULT_JSON_PATH = os.path.join(os.path.dirname(__file__), "config.json")
    
    def __init__(self, prompt_type="base", config_path=None):
        self.config = self.load_config(config_path)
        self.configure_environment_variables()
        
        # Get temperature setting from config (default to 0.0 if not supported)
        temperature = 0.0 if not self.config.get("supports_temperature", True) else 0.0
        
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=temperature,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    @classmethod
    def load_config(cls, config_path=None):
        """Load configuration from TOML (preferred) or JSON file."""
        # If explicit path provided, use it
        if config_path and os.path.exists(config_path):
            return cls._load_config_file(config_path)
        
        # Prefer TOML over JSON
        if os.path.exists(cls.DEFAULT_TOML_PATH):
            return cls._load_config_file(cls.DEFAULT_TOML_PATH)
        elif os.path.exists(cls.DEFAULT_JSON_PATH):
            return cls._load_config_file(cls.DEFAULT_JSON_PATH)
        else:
            # Check if template exists and suggest copying it
            if os.path.exists(cls.DEFAULT_TEMPLATE_PATH):
                print(f"No config.toml found. To get started:")
                print(f"  cp config.template.toml config.toml")
                print(f"  # Then edit config.toml with your credentials")
            print("Using environment variables/prompts for configuration")
            return {}
    
    @staticmethod
    def _load_config_file(config_path):
        """Load config from either TOML or JSON file."""
        print(f"Loading config from: {config_path}")
        
        if config_path.endswith('.toml'):
            with open(config_path, 'rb') as f:
                config = tomllib.load(f)
            # Extract Azure config from TOML structure
            azure = config.get("model", {}).get("azure", {})
            return {
                "model_endpoint": azure.get("endpoint", ""),
                "deployment_name": azure.get("deployment_name", ""),
                "api_version": azure.get("api_version", ""),
                "api_key": azure.get("api_key", ""),
                "scope": azure.get("scope", "https://cognitiveservices.azure.com/.default"),
                "supports_temperature": azure.get("supports_temperature", True),
                "use_azure_credential": not azure.get("api_key", ""),  # Use credential if no API key
            }
        else:
            # JSON config (legacy support)
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config.get("language_model", {})

    def configure_environment_variables(self):
        """Sets environment variables from config file, falling back to Azure credential or user prompt."""
        DEFAULT_AZURE_OPENAI_API_VERSION = "2024-10-01"

        # Set endpoint from config or prompt
        if "AZURE_OPENAI_ENDPOINT" not in os.environ:
            if "model_endpoint" in self.config:
                os.environ["AZURE_OPENAI_ENDPOINT"] = self.config["model_endpoint"]
            else:
                os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Please enter your Azure OpenAI endpoint: ")

        # Set deployment name from config or prompt
        if "AZURE_OPENAI_DEPLOYMENT_NAME" not in os.environ:
            if "deployment_name" in self.config:
                os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = self.config["deployment_name"]
            else:
                os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = getpass.getpass("Please enter your Azure OpenAI deployment name: ")

        # Set API version from config or default
        if "AZURE_OPENAI_API_VERSION" not in os.environ:
            os.environ["AZURE_OPENAI_API_VERSION"] = self.config.get("api_version", DEFAULT_AZURE_OPENAI_API_VERSION)

        # Set API key - check env var, then config file, then Azure credential, then prompt
        if "AZURE_OPENAI_API_KEY" not in os.environ:
            # First check if api_key is in config file
            if "api_key" in self.config and self.config["api_key"] and self.config["api_key"] != "YOUR_API_KEY_HERE":
                os.environ["AZURE_OPENAI_API_KEY"] = self.config["api_key"]
                print("Using API key from config file")
            # Check if user wants to use Azure credential (explicit flag or no api_key)
            elif self.config.get("use_azure_credential", False):
                scope = self.config.get("scope", "https://cognitiveservices.azure.com/.default")
                try:
                    credential = DefaultAzureCredential()
                    os.environ["AZURE_OPENAI_API_KEY"] = credential.get_token(scope).token
                    print("Successfully authenticated with Azure credential")
                except Exception as e:
                    print(f"Azure credential authentication failed: {e}")
                    os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Please enter your Azure OpenAI API key: ")
            else:
                # No api_key in config and not using Azure credential - prompt user
                os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Please enter your Azure OpenAI API key: ")

    def call_agent(self, query):
        print("Calling GPT-4o with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt = self.prompt_agent.get_few_shot_prompt(query)
        elif self.prompt_type in ["few_shot_basic"]:
            prompt = self.prompt_agent.get_few_shot_prompt()
        else:
            prompt = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )

        # Print prompt template
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt.example_prompt)
            print("\nPrefix:", prompt.prefix)
            print("\nSuffix:", prompt.suffix)
        else:
            print(prompt.template.strip())
        print("-" * 80 + "\n")

        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(query)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
   
class QwenModel:
    def __init__(self, prompt_type="base"):
        login_huggingface()  # Login to Huggingface for local LLM
        self.model_name = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm = LLM(
            model=self.model_name,
            device=self.device,
            quantization="gptq"
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512
        )

        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    def call_agent(self, query):
        print("Calling Qwen with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(query)
            # For few-shot semantic, we need to format with the specific query
            prompt_text = prompt_template.format(input=query)
        elif self.prompt_type == "few_shot_basic":
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            # For few-shot basic, format with the query
            prompt_text = prompt_template.format(input=query)
        else:
            # For base/cot prompts
            prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )
            prompt_text = self.prompt_agent.prompt_prefix + prompt_suffix
            prompt_text = prompt_text.format(input=query)

        # Print prompt template (keeping the same debugging output)
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt_template, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt_template.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt_template.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt_template.example_prompt)
            print("\nPrefix:", prompt_template.prefix)
            print("\nSuffix:", prompt_template.suffix)
        else:
            print(prompt_text.strip())
        print("-" * 80 + "\n")

        # Use vLLM's native interface for generation
        outputs = self.llm.generate(prompt_text, self.sampling_params)
        answer = outputs[0].outputs[0].text
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code

class QwenModel_finetuned:
    def __init__(self, prompt_type="base", model_path=None):
        login_huggingface()  # Login to Huggingface for local LLM
        self.model_name = "Fine-tuned-Qwen-7B"
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        )
        # Set padding token to be same as EOS token, but with explicit attention mask handling
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="auto", 
            trust_remote_code=True, 
            fp16=True
        ).eval()
        
        # Use the default generation config from the model
        self.llm.generation_config = self.llm.generation_config
        
        self.prompt_type = prompt_type
        # Store prompt agent for later use
        if self.prompt_type == "cot":
            self.prompt_agent = ZeroShot_CoT_PromptAgent()
        elif self.prompt_type == "few_shot_basic":
            self.prompt_agent = FewShot_Basic_PromptAgent()
        elif self.prompt_type == "few_shot_semantic":
            self.prompt_agent = FewShot_Semantic_PromptAgent()
        else:
            self.prompt_agent = BasePromptAgent()

    def call_agent(self, query):
        print("Calling Fine-tuned Qwen with prompt type:", self.prompt_type)
        
        # Create prompt based on type
        if self.prompt_type == "few_shot_semantic":
            prompt_template = self.prompt_agent.get_few_shot_prompt(query)
            # For few-shot semantic, we need to format with the specific query
            prompt_text = prompt_template.format(input=query)
        elif self.prompt_type == "few_shot_basic":
            prompt_template = self.prompt_agent.get_few_shot_prompt()
            # For few-shot basic, format with the query
            prompt_text = prompt_template.format(input=query)
        else:
            # For base/cot prompts
            prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
            )
            prompt_text = self.prompt_agent.prompt_prefix + prompt_suffix
            prompt_text = prompt_text.format(input=query)

        # Print prompt template (keeping the same debugging output)
        print("\nPrompt Template:")
        print("-" * 80)
        if isinstance(prompt_template, FewShotPromptTemplate):
            print("Few Shot Prompt Template Configuration:")
            print("\nInput Variables:", prompt_template.input_variables)
            print("\nExamples:")
            for i, example in enumerate(prompt_template.examples, 1):
                print(f"\nExample {i}:")
                print(f"Question: {example['question']}")
                print(f"Answer: {example['answer']}")
            print("\nExample Prompt Template:", prompt_template.example_prompt)
            print("\nPrefix:", prompt_template.prefix)
            print("\nSuffix:", prompt_template.suffix)
        else:
            print(prompt_text.strip())
        print("-" * 80 + "\n")

        # Explicitly create the attention mask to handle the warning
        # Use the modelscope chat method with attention mask handling
        try:
            # First try the existing method which might work despite the warning
            answer, _ = self.llm.chat(self.tokenizer, prompt_text, history=None)
        except Exception as e:
            print(f"Standard chat method failed with: {e}. Using fallback method with explicit attention mask.")
            # Fallback to manual generation with explicit attention mask
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.llm.device)
            # Generate with explicit attention mask
            with torch.no_grad():
                outputs = self.llm.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    max_new_tokens=512,
                    temperature=0.0
                )
            # Decode the output skipping the input tokens
            answer = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
 

# ReAct agent
from langchain import hub
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.tools.python.tool import PythonAstREPLTool
class ReAct_Agent:
    def __init__(self, prompt_type="react", config_path=None):
        # Use AzureGPT4Agent to load config and set environment variables
        azure_agent = AzureGPT4Agent.__new__(AzureGPT4Agent)
        azure_agent.config = AzureGPT4Agent.load_config(config_path)
        azure_agent.configure_environment_variables()
        
        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
            temperature=0.0,
            max_tokens=4000,
        )
        self.prompt_type = prompt_type
        self.prompt_agent = ReAct_PromptAgent()
        

    def call_agent(self, query):
        print("Calling ReAct agent withGPT-4o with prompt type:", self.prompt_type)
        
        prompt_template = PromptTemplate(
                input_variables=["input"],
                template=self.prompt_agent.prompt_prefix + prompt_suffix
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
            max_iterations = 2 # try up to 3 times to find the best answer
        )
        print("ReAct agent executor set up")
        # Format the input correctly based on the prompt type

        output = agent_executor.invoke({'input': prompt_template.format(input=query)})
        # Return the output in the same format as other agents
        answer = output['output']
        print("ReActanswer: ", answer)
        print("model returned")
        code = clean_up_llm_output_func(answer)
        return code
