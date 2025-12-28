import sys
import json
import asyncio
from pathlib import Path

import tomllib

from malt_agent import EvalRequest
from a2a.types import (
    AgentCard,
    Message,
    TaskStatusUpdateEvent,
    TaskArtifactUpdateEvent,
    Artifact,
    TextPart,
    DataPart,
    Part,
    Role,
)
import asyncio
import json
import logging
from uuid import uuid4

import httpx
from a2a.client import (
    A2ACardResolver,
    ClientConfig,
    ClientFactory,
    Consumer,
)

DEFAULT_TIMEOUT = 300

def create_message(*, role: Role = Role.user, text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=role,
        parts=[Part(TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id
    )

def merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)

async def send_message(message: str, base_url: str, context_id: str | None = None, streaming=False, consumer: Consumer | None = None):
    """Returns dict with context_id, response and status (if exists)"""
    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(
            httpx_client=httpx_client,
            streaming=streaming,
        )
        factory = ClientFactory(config)
        client = factory.create(agent_card)
        if consumer:
            await client.add_event_consumer(consumer)

        outbound_msg = create_message(text=message, context_id=context_id)
        last_event = None
        outputs = {
            "response": "",
            "context_id": None
        }

        # if streaming == False, only one event is generated
        async for event in client.send_message(outbound_msg):
            last_event = event

        match last_event:
            case Message() as msg:
                outputs["context_id"] = msg.context_id
                outputs["response"] += merge_parts(msg.parts)

            case (task, update):
                outputs["context_id"] = task.context_id
                outputs["status"] = task.status.state.value
                msg = task.status.message
                if msg:
                    outputs["response"] += merge_parts(msg.parts)
                if task.artifacts:
                    for artifact in task.artifacts:
                        outputs["response"] += merge_parts(artifact.parts)

            case _:
                pass

        return outputs


def parse_toml(d: dict[str, object]) -> tuple[EvalRequest, str, dict[str, str]]:
    green = d.get("green_agent")
    if not isinstance(green, dict) or "endpoint" not in green:
        raise ValueError("green.endpoint is required in TOML")
    green_endpoint: str = green["endpoint"]

    parts: dict[str, str] = {}
    role_to_id: dict[str, str] = {}

    for p in d.get("participants", []):
        if isinstance(p, dict):
            role = p.get("role")
            endpoint = p.get("endpoint")
            agentbeats_id = p.get("agentbeats_id")
            if role and endpoint:
                parts[role] = endpoint
            if role and agentbeats_id:
                role_to_id[role] = agentbeats_id

    eval_req = EvalRequest(
        participants=parts,
        config=d.get("config", {}) or {}
    )
    return eval_req, green_endpoint, role_to_id

def parse_parts(parts) -> tuple[list, list]:
    text_parts = []
    data_parts = []

    for part in parts:
        if isinstance(part.root, TextPart):
            try:
                data_item = json.loads(part.root.text)
                data_parts.append(data_item)
            except Exception:
                text_parts.append(part.root.text.strip())
        elif isinstance(part.root, DataPart):
            data_parts.append(part.root.data)

    return text_parts, data_parts

def print_parts(parts, task_state: str | None = None):
    text_parts, data_parts = parse_parts(parts)

    output = []
    if task_state:
        output.append(f"[Status: {task_state}]")
    if text_parts:
        output.append("\n".join(text_parts))
    if data_parts:
        output.extend(json.dumps(item, indent=2) for item in data_parts)

    print("\n".join(output) + "\n")

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client_cli.py <scenario.toml> [output.json]")
        sys.exit(1)

    scenario_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not scenario_path.exists():
        print(f"File not found: {scenario_path}")
        sys.exit(1)

    toml_data = scenario_path.read_text()
    data = tomllib.loads(toml_data)

    req, green_url, role_to_id = parse_toml(data)

    artifacts: list[Artifact] = []

    async def event_consumer(event, card: AgentCard):
        nonlocal artifacts
        match event:
            case Message() as msg:
                print_parts(msg.parts)

            case (task, TaskStatusUpdateEvent() as status_event):
                status = status_event.status
                parts = status.message.parts if status.message else []
                print_parts(parts, status.state.value)
                if status.state.value == "completed":
                    print(task.artifacts)
                    artifacts = task.artifacts
                elif status.state.value not in ["submitted", "working"]:
                    print(f"Agent returned status {status.state.value}. Exiting.")
                    exit(1)

            case (task, TaskArtifactUpdateEvent() as artifact_event):
                print_parts(artifact_event.artifact.parts, "Artifact update")

            case task, None:
                status = task.status
                parts = status.message.parts if status.message else []
                print_parts(parts, task.status.state.value)
                if status.state.value == "completed":
                    print(task.artifacts)
                    artifacts = task.artifacts
                elif status.state.value not in ["submitted", "working"]:
                    print(f"Agent returned status {status.state.value}. Exiting.")
                    exit(1)

            case _:
                print("Unhandled event")

    msg = req.model_dump_json()
    await send_message(msg, green_url, streaming=True, consumer=event_consumer)

    if output_path:
        all_data_parts = []
        for artifact in artifacts:
            _, data_parts = parse_parts(artifact.parts)
            all_data_parts.extend(data_parts)

        output_data = {
            "participants": role_to_id,
            "results": all_data_parts
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
            print(f"Results written to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())