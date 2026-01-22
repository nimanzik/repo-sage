import io
import zipfile
from typing import Any

import frontmatter
import requests

REQUEST_TIMEOUT_SECONDS = 30


def read_repo_markdown_files(
    repo_owner: str, repo_name: str, branch: str = "main"
) -> list[dict[str, Any]]:
    """Read Markdown and React Markdown files from a GitHub repository."""
    zip_url = (
        f"https://codeload.github.com/{repo_owner}/{repo_name}/zip/refs/heads/{branch}"
    )
    response = requests.get(zip_url, timeout=REQUEST_TIMEOUT_SECONDS)

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to download repository: {e}") from e

    extracted_data = []
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        for file_info in zf.infolist():
            if not file_info.filename.lower().endswith((".md", ".mdx")):
                continue

            with zf.open(file_info) as file:
                content = file.read().decode(encoding="utf-8", errors="ignore")

                post = frontmatter.loads(content)
                data = post.to_dict()
                data["filename"] = file_info.filename
                extracted_data.append(data)

    return extracted_data
