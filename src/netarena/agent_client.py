import httpx
from loguru import logger
from uuid import uuid4
from typing import Any
from enum import Enum
import json
from dataclasses import dataclass, field
import uuid
import cattrs
from cattrs.gen import make_dict_unstructure_fn, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_data_artifact, new_task
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig, BaseClient, Client, A2AClient
from a2a.client.middleware import ClientCallContext
from a2a.types import (
    MessageSendParams,
    MessageSendConfiguration,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    Message,
    Part,
    TextPart,
    DataPart,
    Role,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)


DEFAULT_HTTP_TIMEOUT_S = 60


class PromptType(Enum):
    """
    Represents a prompting strategy used for an LLM agent (e.g. zero-shot chain-of-thought).
    """
    ZEROSHOT_BASE = "zeroshot_base"
    FEWSHOT_BASE = "fewshot_base"
    ZEROSHOT_COT = "zeroshot_cot"
    FEWSHOT_COT = "fewshot_cot"


@dataclass
class AgentClientConfig:
    """
    Contains information and settings for an AgentClient (e.g. agent endpoint, HTTP options, etc).
    """
    base_url: str
    name: str
    prompt_type: PromptType = PromptType.FEWSHOT_COT
    http_kwargs: dict[str, Any] = field(default_factory=lambda: dict(timeout=DEFAULT_HTTP_TIMEOUT_S))

    def serialize_omit_secrets(self) -> dict[str, Any]:
        """
        Serialize to JSON while omitting sensitive information (e.g. HTTP arguments).
        """
        # Serialize info for each agent without any potentially sensitive info (e.g. HTTP headers).
        converter = cattrs.Converter()
        exclude_hook = make_dict_unstructure_fn(AgentClientConfig, converter, http_kwargs=override(omit=True))
        converter.register_unstructure_hook(AgentClientConfig, exclude_hook)
        return converter.unstructure(self)


class AgentClient:
    """
    An async client that handles making text completion requests to an (LLM-powered) A2A-compatible agent.
    """

    def __init__(self, config: AgentClientConfig):
        self.config = config

    async def start(self, http_client: httpx.AsyncClient | None = None):
        logger.info(f'Starting agent client for agent "{self.config.name}" connected to {self.config.base_url}')
        # Initialize the A2AClient with the provided configuration
        self.http_client = httpx.AsyncClient() if not http_client else http_client

        client_config = ClientConfig(
            httpx_client=self.http_client,
            streaming=True,
        )
        factory = ClientFactory(client_config)
        # Fetch and set the agent card.
        self.agent_card = await _fetch_agent_card(self.http_client, self.config.base_url)
        self.a2a_client: BaseClient = factory.create(self.agent_card) # type: ignore
        # Start the A2AClient
        logger.info("Agent Server started successfully.")
        return self

    async def handle_query(self, prompt: str) -> str | None:
        """
        Queries the underlying agent with the given prompt. Returns None if the client fails to communicate.
        """
        try:
            logger.debug(f"Processing query with prompt: {prompt}")
            response = await _call_a2a_agent(self.a2a_client, prompt, http_kwargs=self.config.http_kwargs)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            response = None
        return response


async def _fetch_agent_card(http_client: httpx.AsyncClient, base_url: str):
    resolver = A2ACardResolver(
        httpx_client=http_client,
        base_url=base_url,
        # agent_card_path uses default, extended_agent_card_path also uses default
    )
    logger.info(f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}')
    public_card = await resolver.get_agent_card()  # Fetches from default public path
    logger.info('Successfully fetched public agent card:')
    logger.info(public_card.model_dump_json(indent=2, exclude_none=True))
    final_agent_card_to_use = public_card
    return final_agent_card_to_use


async def _call_a2a_agent(a2a_client: BaseClient, query_text: str, http_kwargs: dict = {}) -> str | None:
    message = _create_message(role=Role.user, text=query_text)
    call_context = ClientCallContext(state=dict(http_kwargs=http_kwargs))
    
    # if streaming == False, only one event is generated
    llm_answer = None
    last_event = None
    response_json = None
    i = 0
    async for event in a2a_client.send_message(message, context=call_context):
        logger.trace(f'Event Stream (i={i}): \n{event}')
        last_event = event
    match last_event:
        case Message() as msg:
            llm_answer = _extract_text_from_message(msg)
            response_json = msg.model_dump_json(indent=2, exclude_none=True)

        case (task, update):
            llm_answer = _extract_text_from_task(task)
            response_json = task.model_dump_json(indent=2, exclude_none=True)
        case _:
            pass
    if response_json:
        logger.debug(f'Received response from agent: \n{response_json}')
    else:
        logger.warning(f'Agent endpoint did not respond with data.')
    return llm_answer



def _extract_text_from_message(message: Message, separator='\n') -> str | None:
    if isinstance(message, dict):
        message = Message.model_validate(message)
    text = _merge_parts(message.parts, separator=separator)
    return text


def _extract_text_from_task(task: Task | dict, separator='\n') -> str | None:
    """Extract a text reply from a Task model or mapping by inspecting its history and artifacts."""
    if isinstance(task, dict):
        task = Task.model_validate(task)

    # Try artifacts first (prefer last artifact with text)
    artifacts = task.artifacts if task.artifacts else []
    for art in reversed(artifacts):
        # Some artifacts may include parts or textual content. Return first one found.
        text_parts = [part.root.text for part in art.parts if isinstance(part.root, TextPart)]
        return separator.join(text_parts).strip()
            
    # Try history as fallback (prefer latest agent message with text).
    history = task.history if task.history else []
    for msg in reversed(history):
        if msg.role == Role.agent:  
            text = _merge_parts(msg.parts, separator=separator)
            if text:
                return text
    return None


def _merge_parts(parts: list[Part], separator='\n') -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return separator.join(chunks)


def _create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id
    )

