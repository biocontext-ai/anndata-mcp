# anndata-mcp

[![BioContextAI - Registry](https://img.shields.io/badge/Registry-package?style=flat&label=BioContextAI&labelColor=%23fff&color=%233555a1&link=https%3A%2F%2Fbiocontext.ai%2Fregistry)](https://biocontext.ai/registry)
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/dschaub95/anndata-mcp/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/anndata-mcp

Allows to retrieve information about an AnnData object via MCP

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

You can also find the project on [BioContextAI](https://biocontext.ai), the community-hub for biomedical MCP servers: [anndata-mcp on BioContextAI](https://biocontext.ai/registry/dschaub95/anndata-mcp).

## Installation

You need to have Python 3.10 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install anndata-mcp:

1. Use `uvx` to run it immediately:

```bash
uvx anndata_mcp
```

2. Include it in one of various clients that supports the `mcp.json` standard, please use:

```json
{
  "mcpServers": {
    "server-name": {
      "command": "uvx",
      "args": ["anndata_mcp"],
    }
  }
}
```

3. Install it through `pip`:

```bash
pip install --user anndata_mcp
```

4. Install the latest development version:

```bash
pip install git+https://github.com/dschaub95/anndata-mcp.git@main
```

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/dschaub95/anndata-mcp/issues
[tests]: https://github.com/dschaub95/anndata-mcp/actions/workflows/test.yaml
[documentation]: https://anndata-mcp.readthedocs.io
[changelog]: https://anndata-mcp.readthedocs.io/en/latest/changelog.html
[api documentation]: https://anndata-mcp.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/anndata-mcp
