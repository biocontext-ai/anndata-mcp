import enum
import logging
import os
import sys

import click

from .tools import *  # noqa: F403 import all tools to register them


class EnvironmentType(enum.Enum):
    """Enum to define environment type."""

    PRODUCTION = enum.auto()
    DEVELOPMENT = enum.auto()


@click.command(name="run")
@click.option(
    "-t",
    "--transport",
    "transport",
    type=str,
    help="MCP transport option. Defaults to 'stdio'.",
    default="stdio",
    envvar="MCP_TRANSPORT",
)
@click.option(
    "-p",
    "--port",
    "port",
    type=int,
    help="Port of MCP server. Defaults to '8000'",
    default=8000,
    envvar="MCP_PORT",
    required=False,
)
@click.option(
    "-h",
    "--host",
    "hostname",
    type=str,
    help="Hostname of MCP server. Defaults to '0.0.0.0'",
    default="0.0.0.0",
    envvar="MCP_HOSTNAME",
    required=False,
)
@click.option("-v", "--version", "version", is_flag=True, help="Get version of package.")
@click.option(
    "-e",
    "--env",
    "environment",
    type=click.Choice(EnvironmentType, case_sensitive=False),
    default=EnvironmentType.DEVELOPMENT,
    envvar="MCP_ENVIRONMENT",
    help="MCP server environment. Defaults to 'development'.",
)
@click.option(
    "--max-output-len",
    "max_output_len",
    type=int,
    default=1000,
    envvar="MCP_MAX_OUTPUT_LEN",
    help="Maximum output length for truncation. Defaults to 1000.",
)
def run_app(
    transport: str = "stdio",
    port: int = 8000,
    hostname: str = "0.0.0.0",
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT,
    version: bool = False,
    max_output_len: int = 1000,
):
    """Run the MCP server "anndata-mcp".

    Allows to retrieve information about an AnnData object via MCP
    If the environment variable MCP_ENVIRONMENT is set to "PRODUCTION", it will run the Starlette app with streamable HTTP for the MCP server. Otherwise, it will run the MCP server via stdio.
    The port is set via "-p/--port" or the MCP_PORT environment variable, defaulting to "8000" if not set.
    The hostname is set via "-h/--host" or the MCP_HOSTNAME environment variable, defaulting to "0.0.0.0" if not set.
    To specify to transform method of the MCP server, set "-e/--env" or the MCP_TRANSPORT environment variable, which defaults to "stdio".
    The maximum output length for truncation is set via "--max-output-len" or the MCP_MAX_OUTPUT_LEN environment variable, defaulting to "1000" if not set.
    """
    if version is True:
        from anndata_mcp import __version__

        click.echo(__version__)
        sys.exit(0)

    logger = logging.getLogger(__name__)

    from anndata_mcp.mcp import mcp

    # Set the max output length environment variable from command line
    os.environ["MCP_MAX_OUTPUT_LEN"] = str(max_output_len)

    # click.echo("Registered MCP tools:")
    # tools = asyncio.run(mcp._list_tools())
    # click.echo(tools)

    if environment == EnvironmentType.DEVELOPMENT:
        logger.info("Starting MCP server (DEVELOPMENT mode)")
        if transport == "http":
            mcp.run(transport=transport, port=port, host=hostname)
        else:
            mcp.run(transport=transport)
    else:
        raise NotImplementedError()
        # logger.info("Starting Starlette app with Uvicorn in PRODUCTION mode.")
        # uvicorn.run(app, host=hostname, port=port)


if __name__ == "__main__":
    run_app()
