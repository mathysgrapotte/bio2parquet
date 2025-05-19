# bio2parquet

[![ci](https://github.com/bio2parquet/bio2parquet/workflows/ci/badge.svg)](https://github.com/bio2parquet/bio2parquet/actions?query=workflow%3Aci)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://bio2parquet.github.io/bio2parquet/)
[![pypi version](https://img.shields.io/pypi/v/bio2parquet.svg)](https://pypi.org/project/bio2parquet/)
[![gitter](https://badges.gitter.im/join%20chat.svg)](https://app.gitter.im/#/room/#bio2parquet:gitter.im)

Convert your genomic data to high-performance Parquet for seamless integration and faster ML pipeline training.

## 🚀 Features

- **High Performance**: Convert bioinformatics files to Parquet format for faster data processing
- **Seamless Integration**: Easy integration with modern data science and ML pipelines
- **Multiple Formats**: Support for various bioinformatics file formats
- **Optimized Storage**: Efficient data compression and storage
- **Python Native**: Built with Python for easy integration into your workflow

## 📦 Installation

```bash
pip install bio2parquet
```

With [`uv`](https://docs.astral.sh/uv/):

```bash
uv tool install bio2parquet
```

## 🎯 Quick Start

```python
from bio2parquet import convert

# Convert your bioinformatics file to Parquet
convert("input.bam", "output.parquet")
```

## 📚 Documentation

For detailed documentation, visit our [documentation site](https://bio2parquet.github.io/bio2parquet/).

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 💬 Community

Join our community on [Gitter](https://app.gitter.im/#/room/#bio2parquet:gitter.im) to discuss features, ask questions, and connect with other users.
