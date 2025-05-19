# bio2parquet

[![ci](https://github.com/bio2parquet/bio2parquet/workflows/ci/badge.svg)](https://github.com/bio2parquet/bio2parquet/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://bio2parquet.github.io/bio2parquet/)
[![pypi version](https://img.shields.io/pypi/v/bio2parquet.svg)](https://pypi.org/project/bio2parquet/)

Convert your genomic data to high-performance Parquet for seamless integration and faster ML pipeline training.

## 🚀 Features

- **FASTA Support**: Convert FASTA files (.fasta, .fa, .fna) to Parquet format
- **Compression Support**: Handle both plain and gzipped FASTA files
- **Hugging Face Integration**: Direct upload to Hugging Face Hub for dataset sharing
- **Python Native**: Built with Python for easy integration into your workflow
- **Type Safety**: Full type hints support for better development experience

## 📦 Installation

```bash
pip install bio2parquet
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install bio2parquet
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Basic conversion
bio2parquet fasta input.fasta

# Specify output file
bio2parquet fasta input.fasta -o output.parquet

# Upload to Hugging Face Hub
bio2parquet fasta input.fasta --hf-repo-id username/dataset-name --hf-token your_token
```

### Python API

```python
from bio2parquet import create_dataset_from_fasta

# Convert FASTA to Parquet
dataset = create_dataset_from_fasta("input.fasta")
dataset.to_parquet("output.parquet")

# Upload to Hugging Face Hub
dataset.push_to_hub("username/dataset-name", token="your_token")
```

## 📚 Documentation

For detailed documentation, visit our [documentation site](https://bio2parquet.github.io/bio2parquet/).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
