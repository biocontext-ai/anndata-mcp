from fastmcp import FastMCP

mcp: FastMCP = FastMCP(
    name="anndata-mcp",
    instructions="Allows to retrieve information about an AnnData object via MCP",
    on_duplicate_tools="error",
)
