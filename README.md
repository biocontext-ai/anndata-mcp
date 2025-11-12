# AnnData MCP

[![BioContextAI - Registry](https://img.shields.io/badge/Registry-package?style=flat&label=BioContextAI&labelColor=%23fff&color=%233555a1&link=https%3A%2F%2Fbiocontext.ai%2Fregistry)](https://biocontext.ai/registry)
[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/biocontext-ai/anndata-mcp/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/anndata-mcp

Allows to retrieve information about an AnnData object via MCP using the `read_lazy` function from `anndata`.

## Getting started

Please refer to the [documentation][],
in particular, the [API documentation][].

You can also find the project on [BioContextAI](https://biocontext.ai), the community-hub for biomedical MCP servers: [anndata-mcp on BioContextAI](https://biocontext.ai/registry/biocontext-ai/anndata-mcp).

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install anndata-mcp:

1. Use `uvx` to run it immediately:

```bash
uvx anndata-mcp
```

2. Include it in one of various clients that supports the `mcp.json` standard, please use:

```json
{
  "mcpServers": {
    "anndata-mcp": {
      "command": "uvx",
      "args": ["anndata-mcp"]
    }
  }
}
```

3. Install it through `pip`:

```bash
pip install --user anndata-mcp
```

4. Install the latest development version:

```bash
pip install git+https://github.com/biocontext-ai/anndata-mcp.git@main
```

## Contact

If you found a bug, please use the [issue tracker][].

## Citation

If this MCP server is useful to your research, please cite the `BioContextAI` and the `anndata` publications:

```bibtex
@article{BioContext_AI_Kuehl_Schaub_2025,
  title={BioContextAI is a community hub for agentic biomedical systems},
  url={http://dx.doi.org/10.1038/s41587-025-02900-9},
  urldate = {2025-11-06},
  doi={10.1038/s41587-025-02900-9},
  year = {2025},
  month = nov,
  journal={Nature Biotechnology},
  publisher={Springer Science and Business Media LLC},
  author={Kuehl, Malte and Schaub, Darius P. and Carli, Francesco and Heumos, Lukas and Hellmig, Malte and Fernández-Zapata, Camila and Kaiser, Nico and Schaul, Jonathan and Kulaga, Anton and Usanov, Nikolay and Koutrouli, Mikaela and Ergen, Can and Palla, Giovanni and Krebs, Christian F. and Panzer, Ulf and Bonn, Stefan and Lobentanzer, Sebastian and Saez-Rodriguez, Julio and Puelles, Victor G.},
  year={2025},
  month=nov,
  language={en},
}
```

```bibtex
@article{Virshup2024,
  title = {anndata: Access and store annotated data
matrices},
  volume = {9},
  ISSN = {2475-9066},
  url = {http://dx.doi.org/10.21105/joss.04371},
  DOI = {10.21105/joss.04371},
  number = {101},
  journal = {Journal of Open Source Software},
  publisher = {The Open Journal},
  author = {Virshup,  Isaac and Rybakov,  Sergei and Theis,  Fabian J. and Angerer,  Philipp and Wolf,  F. Alexander},
  year = {2024},
  month = sep,
  pages = {4371}
}
```

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/biocontext-ai/anndata-mcp/issues
[tests]: https://github.com/biocontext-ai/anndata-mcp/actions/workflows/test.yaml
[documentation]: https://anndata-mcp.readthedocs.io
[changelog]: https://anndata-mcp.readthedocs.io/en/latest/changelog.html
[api documentation]: https://anndata-mcp.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/anndata-mcp
