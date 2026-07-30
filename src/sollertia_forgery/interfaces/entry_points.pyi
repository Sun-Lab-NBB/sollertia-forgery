from .plan import plan_cli as plan_cli
from .forge import forge_command as forge_command
from .manage import (
    manifest_cli as manifest_cli,
    clean_command as clean_command,
    reset_command as reset_command,
    checksum_command as checksum_command,
    dataset_state_command as dataset_state_command,
)
from .server import server_cli as server_cli
from .process import process_cli as process_cli

_CONTEXT_SETTINGS: dict[str, int]

def slf_cli() -> None: ...
def run_mcp_server_command() -> None: ...
def _register_subcommands() -> None: ...
