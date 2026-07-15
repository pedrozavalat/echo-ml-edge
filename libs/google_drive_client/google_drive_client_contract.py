from abc import ABC, abstractmethod
from pathlib import Path

from .types.google_drive_client_types import DriveEntry


class GoogleDriveClientContract(ABC):
    @abstractmethod
    def list_dir(self, path: str) -> list[DriveEntry]:
        """List direct children of a Google Drive directory.

        :param path: Drive path relative to the configured Drive root.
        :return: Direct children, including files and directories.
        """
        raise NotImplementedError

    @abstractmethod
    def list_files(self, path: str) -> list[DriveEntry]:
        """List files directly contained in a Google Drive directory.

        :param path: Drive path relative to the configured Drive root.
        :return: Direct child files only.
        """
        raise NotImplementedError

    @abstractmethod
    def upload_to(self, source_path:str | Path, destination_path: str  = "/") -> list[DriveEntry]:
        """Upload a local file or directory using the same relative path in Drive.
        """
        raise NotImplementedError
