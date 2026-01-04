import argparse
from dataclasses import dataclass
import uvicorn
import litellm
from loguru import logger
from litellm import acompletion
from litellm import ModelResponse, CustomStreamWrapper, ModelResponseStream, StreamingChoices, Choices

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message


class LitellmAgent:
    """Uses LiteLLM to make a text completion."""

    def __init__(
            self, 
            model_name: str, 
            api_key: str | None = None,
            api_base_url: str | None = None, 
            api_version: str | None = None
        ):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.api_version = api_version

    async def invoke(self, input_text: str) -> str:
        messages = [{"role": "user", "content": input_text}]
        response = await acompletion(
            self.model_name, messages, 
            stream=True, 
            base_url=self.api_base_url, 
            api_version=self.api_version,
            api_key=self.api_key
        )
        if isinstance(response, CustomStreamWrapper):
            chunks = [chunk async for chunk in response]
            response = litellm.stream_chunk_builder(chunks, messages=messages)
        # Extract text content from aggregated output.
        if isinstance(response, ModelResponseStream):
            return response.choices[0].delta.content or ""
        elif isinstance(response, ModelResponse):
            return response.choices[0].message.content or ""
    

class LitellmAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

    def __init__(self, agent: LitellmAgent):
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        input = context.get_user_input()
        result = await self.agent.invoke(input)
        logger.info(f'Result: {result}')
        
        await event_queue.enqueue_event(new_agent_text_message(result))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception('cancel not supported')
    

@dataclass
class Args:
    model_name: str
    api_key: str | None = None
    api_base_url: str | None = None
    api_version: str | None = None
    host: str = '127.0.0.1'
    port: int = 8000
    

# Define a configuration for the benchmark
def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Expose common LLMs as an A2A compatible server.")
    parser.add_argument('--model-name', type=str, help='Name of model to serve.')
    parser.add_argument('--api-key', type=str, default=None, help='Passes the API key supplied directly.')
    parser.add_argument('--api-base-url', type=str, default=None, help='The base URL endpoint of your LLM.')
    parser.add_argument('--api-version', type=str, default=None, help='Version of API to use (Azure).')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to expose the server on.')
    parser.add_argument('--port', type=int, default=8000, help='Port to expose the server on.')
    args = parser.parse_args()
    config = Args(**vars(args))
    return config


if __name__ == "__main__":
    args = parse_args()

    server_url = f'http://{args.host}:{args.port}/'

    skill = AgentSkill(
        id='litellm_agent',
        name='LiteLLM Agent',
        description='Calls an LLM agent using the LiteLLM library.',
        tags=['llm', 'chatbot', 'litellm', 'text']
    )

    public_agent_card = AgentCard(
        name='A2A LLM Agent',
        description='Exposes LLM agents as A2A servers with LiteLLM as a compatability layer.',
        url=server_url,
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
    )

    llm_agent = LitellmAgent(model_name=args.model_name, api_key=args.api_key, api_base_url=args.api_base_url, api_version=args.api_version)
    request_handler = DefaultRequestHandler(
        agent_executor=LitellmAgentExecutor(llm_agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host=args.host, port=args.port)