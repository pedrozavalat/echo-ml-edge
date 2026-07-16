from dataclasses import dataclass
from pathlib import Path
from typing import Literal


DriveEntryType = Literal["file", "directory"]


@dataclass(frozen=True)
class DriveClientInitArgs:
    client_name: Literal["google_api"]
    credentials_file: str | Path
    root_folder_id: str
    shared_drive_id: str | None = None



@dataclass(frozen=True)
class DriveEntry:
    """Normalized file or directory metadata returned by the client."""

    id: str
    name: str
    path: str
    entry_type: DriveEntryType
    mime_type: str
    size_bytes: int | None = None
    modified_time: str | None = None
    web_view_link: str | None = None
