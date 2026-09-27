from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True, slots=True)
class ModelStore:
    root: Path | None = None

    def __init__(self, root: str | Path | None = None) -> None:
        object.__setattr__(self, "root", Path(root).expanduser().resolve() if root else None)

    def get(
        self,
        repo_id: str,
        filename: str,
        *,
        repo_filename: str | None = None,
    ) -> Path:
        requested = repo_filename or filename
        kwargs = {
            "repo_id": repo_id,
            "filename": requested,
            "token": False,
        }
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(self.root)

        return Path(hf_hub_download(**kwargs))
