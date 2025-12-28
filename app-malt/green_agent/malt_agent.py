import uvicorn
from loguru import logger
import uuid
from typing import Any
import sys
import os
import argparse
# Include path to user libraries.
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.dirname(file_dir)))

from pydantic import BaseModel, HttpUrl, ValidationError
from cattrs import structure

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    Task,
    TaskState,
    InternalError,
    InvalidParamsError,
    UnsupportedOperationError,
    DataPart
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from main import evaluate_on_queries, MaltConfig
from netarena.agent_client import AgentClientConfig, PromptType


class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl] # role-endpoint mapping
    config: dict[str, Any]


class MaltEvalAgent:

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        config: MaltConfig = structure(request.config, MaltConfig)

        # Only one agent allowed to be evaluated at a time.
        role, url = next(iter(request.participants.items())) 
        agent_config = AgentClientConfig(base_url=str(url), name=role, prompt_type=config.prompt_type)
        config.agent_client_configs = [agent_config]

        # TODO: Support multiple agent evals at once?
        query_eval_results = evaluate_on_queries(config)

        # Each agent (name) mapped to a unique artifact containing all its evaluation results.
        artifact_id = None

        avg_correct = 0
        avg_latency = 0
        avg_safety = 0
        n = 0
        async for res in query_eval_results:
            append = True
            n += 1

            # TODO: Accessing hardcoded keys.
            agent_name = res['agent_info']['name']

            avg_correct += ((1 if res["Result-Correctness"] == "Pass" else 0) - avg_correct) / n
            avg_safety += ((1 if res["Result-Safety"] == "Pass" else 0) - avg_safety) / n
            avg_latency += (res["Result-Latency"] - avg_latency) / n

            # Can only append once the artifact is already created (set to False to create it first).
            if artifact_id is None:
                artifact_id = str(uuid.uuid4())
                append = False

            part = DataPart(data=res)
            logger.info(part)
            await updater.add_artifact(
                parts=[part],
                artifact_id=artifact_id,
                append=append
            )
            await updater.update_status(TaskState.working, new_agent_text_message(f'Query {n} processed by agent: "{agent_name}"'))

        # Final evaluation results.
        query_res = {
            'correctness': avg_correct,
            'safety': avg_safety,
            'latency_s': avg_latency
        }
        part = DataPart(data=query_res)
        logger.info(part)
        await updater.add_artifact(
            parts=[part],
            artifact_id=artifact_id,
            append=append
        )

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        ok = True
        msg = "Evaluation request is valid."
        if len(request.participants.keys()) != 1:
            ok = False
            msg = f'Must have exactly one participant.'
            return ok, msg
        try:
            _ = structure(request.config, MaltConfig)
        except Exception as e:
            ok = False
            msg = f'App config invalid: {e}'
        return ok, msg


class GreenExecutor(AgentExecutor):

    def __init__(self, green_agent: MaltEvalAgent):
        self.agent = green_agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        request_text = context.get_user_input()
        try:
            req: EvalRequest = EvalRequest.model_validate_json(request_text)
            logger.debug(f'Incoming request: {req}')
            ok, msg = self.agent.validate_request(req)
            if not ok:
                raise ServerError(error=InvalidParamsError(message=msg))
        except ValidationError as e:
            raise ServerError(error=InvalidParamsError(message=e.json()))

        msg = context.message
        if msg:
            task = new_task(msg)
            await event_queue.enqueue_event(task)
        else:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"Starting assessment.\n{req.model_dump_json()}", context_id=context.context_id)
        )

        try:
            await self.agent.run_eval(req, updater)
            await updater.complete()
        except Exception as e:
            print(f"Agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context.context_id))
            raise ServerError(error=InternalError(message=str(e)))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
    

def parse_args():
    parser = argparse.ArgumentParser(description="Expose the NetArena benchmark as an A2A server.")
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to expose the server on.')
    parser.add_argument('--port', type=int, default=9999, help='Port to expose the server on.')
    parser.add_argument('--card-url', type=str, default=None, help='Port to expose the server on.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    skill = AgentSkill(
        id='malt_eval',
        name='MALT Evaluation',
        description='Benchmark LLM agents on dynamically generated data center planning queries.',
        tags=['llm', 'chatbot', 'litellm', 'text']
    )

    agent_url = args.card_url or f"http://{args.host}:{args.port}/"
    public_agent_card = AgentCard(
        name='NetArena MALT Evaluation Agent',
        description='An LLM chatbot powered by Azure and LiteLLM.',
        url=agent_url,
        version='1.0.0',
        default_input_modes=['data'],
        default_output_modes=['data'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],  # Only the basic skill for the public card
    )

    request_handler = DefaultRequestHandler(
        agent_executor=GreenExecutor(MaltEvalAgent()),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host=args.host, port=args.port)