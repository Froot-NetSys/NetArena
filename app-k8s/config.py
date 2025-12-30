from dataclasses import dataclass, field
from netarena.agent_client import PromptType, AgentClientConfig


@dataclass
class K8sConfig:
    """
    Structured container for the app settings.
    """
    prompt_type: PromptType = PromptType.ZEROSHOT_BASE
    num_queries: int = 10
    output_dir: str = 'output'
    microservice_dir: str = 'microservices-demo'
    output_file: str = 'eval_results.jsonl'
    benchmark_path: str = 'error_config.jsonl'
    config_gen: bool = False
    max_iterations: int = 10
    agent_client_configs: list[AgentClientConfig] = field(default_factory=list)

    def __post_init__(self):
        # TODO: Limit to only one agent for now (maybe use K8s namespaces to allow parallel assessments on a single cluster).
        if len(self.agent_client_configs) > 1:
            raise ValueError(f'Must have exactly one agent client config, got {len(self.agent_client_configs)}')