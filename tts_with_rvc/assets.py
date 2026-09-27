from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download


@dataclass(frozen=True, slots=True)
class ModelStore:
    root: Path | None = None

    def __init__(self, root: str | Path | None = None) -> None:
        object.__setattr__(self, "root", Path(root).expanduser().resolve() if root else None)

    def resolve(
        self,
        path: str | Path,
        *,
        repo_id: str,
        repo_filename: str | None = None,
    ) -> Path:
        candidate = Path(path).expanduser()
        if candidate.is_file():
            return candidate.resolve()

        if self.root is not None:
            rooted = self.root / candidate.name
            if rooted.is_file():
                return rooted.resolve()

        filename = repo_filename or candidate.name
        kwargs = {"repo_id": repo_id, "filename": filename, "token": False}
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            kwargs["local_dir"] = str(self.root)

        return Path(hf_hub_download(**kwargs)).resolve()
