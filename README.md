# repo-sage

A conversational Q&A agent that answers questions about a GitHub repository.

> This project is a work in progress and not yet complete.

## Overview

repo-sage aims to make it easy to ask natural language questions about the
documentation in any public GitHub repository. It downloads markdown files,
intelligently chunks them using an LLM, and stores embeddings in a local vector
database for semantic search.

The motivation is to enable quick, context-aware answers without manually
searching through large documentation sets.

## How It Works

1. **Fetch**: Downloads and processes `.md` and `.mdx` files from a GitHub repository.
2. **Chunk**: Uses Google Gemini to split documents into logical sections.
3. **Embed & Store**: Generates embeddings using a pretrained [Sentence Transformers](https://www.sbert.net/index.html) (a.k.a SBERT)
   model and stores them locally via [Qdrant](https://qdrant.tech/).
4. **Search**: Performs semantic search to retrieve relevant context for Q&A.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
