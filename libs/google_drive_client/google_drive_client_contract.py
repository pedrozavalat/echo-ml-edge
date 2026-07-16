from abc import ABC, abstractmethod
from pathlib import Path

from .types.google_drive_client_types import DriveEntry


class GoogleDriveClientContract(ABC):
    @abstractmethod
    def list_dir(self, path: str) -> list[DriveEntry]:
        """List direct children of a Google Drive directory."""
        pass

    @abstractmethod
    def list_files(self, path: str) -> list[DriveEntry]:
        """List files directly contained in a Google Drive directory."""
        pass

    @abstractmethod
    def upload_to(
        self, source_path: str | Path, destination_path: str = "/"
    ) -> list[DriveEntry]:
        """Upload a local file or directory using the same relative path in Drive."""
        pass

    def download(self, source_path: str | Path, destination_path: str = "/"):
        """Download a Drive file or directory into a local directory."""
        pass

    def download_entries(
        self,
        entries: list[DriveEntry],
        destination_path: str | Path,
        max_workers: int = 6,
    ):
        pass
