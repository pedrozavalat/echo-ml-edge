from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from ..google_drive_client_contract import GoogleDriveClientContract
from ..types.google_drive_client_types import DriveClientInitArgs, DriveEntry


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleApiDriveClient(GoogleDriveClientContract):
    """Google Drive implementation backed by Google Drive API v3.
    """

    def __init__(self, args: DriveClientInitArgs) -> None:
        self.credentials_file = Path(args.credentials_file)
        self.root_folder_id = args.root_folder_id
        self.shared_drive_id = args.shared_drive_id

        credentials = self._load_credentials()
        self._thread_local = threading.local()

        self.service: Resource = build(
            "drive",
            "v3",
            credentials=credentials,
            cache_discovery=False,
        )

        root_metadata = (
            self.service.files()
            .get(
                fileId=self.root_folder_id,
                fields="id,name,mimeType",
                supportsAllDrives=True,
            )
            .execute()
        )

        if root_metadata["mimeType"] != FOLDER_MIME_TYPE:
            raise NotADirectoryError(
                f"root_folder_id does not reference a directory: {self.root_folder_id}"
            )

        self._root_id = root_metadata["id"]

        # "/" siempre representa root_folder_id
        self._path_cache: dict[str, str] = {
            "/": self._root_id,
        }

    def list_dir(self, path: str) -> list[DriveEntry]:
        normalized = self._normalize_relative_path(path)
        folder_id = self._resolve_relative_path(normalized)
        children = self._list_children(folder_id)

        return [
            self._to_entry(item, normalized)
            for item in sorted(
                children,
                key=lambda value: (
                    value["mimeType"] != FOLDER_MIME_TYPE,
                    value["name"].lower(),
                ),
            )
        ]

    def list_files(self, path: str) -> list[DriveEntry]:
        return [entry for entry in self.list_dir(path) if entry.entry_type == "file"]

    def upload_to(self, source_path: str | Path, destination_path) -> list[DriveEntry]:
        local_path = Path(source_path).expanduser().resolve()
        drive_destination = self._normalize_relative_path(destination_path)

        if not local_path.exists():
            raise FileNotFoundError(local_path)

        self._ensure_drive_directory(drive_destination)

        if local_path.is_file():
            remote_file_path = self._join_drive_path(
                drive_destination,
                local_path.name,
            )

            return [
                self._upload_file(
                    local_path,
                    remote_file_path,
                )
            ]

        if not local_path.is_dir():
            raise ValueError(
                f"The source path is neither a file nor a directory: {local_path}"
            )

        uploaded: list[DriveEntry] = []

        for child in sorted(local_path.rglob("*")):
            relative_child = child.relative_to(local_path)

            remote_path = self._join_drive_path(
                drive_destination,
                relative_child.as_posix(),
            )

            if child.is_dir():
                self._ensure_drive_directory(remote_path)

            elif child.is_file():
                uploaded.append(
                    self._upload_file(
                        child,
                        remote_path,
                    )
                )

        return uploaded

    def download(self, source_path: str, destination_path):
        normalized_source = self._normalize_relative_path(source_path)

        local_destination = Path(destination_path).expanduser().resolve()
        local_destination.mkdir(parents=True, exist_ok=True)

        source_item = self._resolve_item_path(normalized_source)

        if source_item["mimeType"] != FOLDER_MIME_TYPE:
            local_file = local_destination / source_item["name"]

            return [
                self._download_file(
                    source_item,
                    local_file,
                )
            ]

        if normalized_source == "/":
            local_folder = local_destination
        else:
            local_folder = local_destination / source_item["name"]
            local_folder.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []

        self._download_directory(
            folder_id=source_item["id"],
            local_directory=local_folder,
            downloaded=downloaded,
        )

        return downloaded

    def download_entries(
        self,
        entries: list[DriveEntry],
        destination_path: str | Path,
        max_workers: int = 6,
    ) -> dict[str, Path]:
        destination = Path(destination_path).expanduser().resolve()
        destination.mkdir(
            parents=True,
            exist_ok=True,
        )
        downloaded: dict[str, Path] = {}

        with ThreadPoolExecutor(
            max_workers=max_workers,
        ) as executor:
            futures = {
                executor.submit(
                    self._download_entry,
                    entry,
                    destination / entry.name,
                ): entry
                for entry in entries
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    name, local_path = future.result()
                    downloaded[name] = local_path
                except Exception as error:
                    print(f"Error downloading {entry.name}: {error}")
        return downloaded

    def _get_thread_service(self) -> Resource:
        """Create one Google Drive service per download thread."""
        service = getattr(
            self._thread_local,
            "drive_service",
            None,
        )

        if service is None:
            credentials = self._load_credentials()
            service = build(
                "drive",
                "v3",
                credentials=credentials,
                cache_discovery=False,
            )
            self._thread_local.drive_service = service
        return service

    def _download_entry(
        self,
        entry: DriveEntry,
        destination_path: Path,
    ) -> tuple[str, Path]:
        service = self._get_thread_service()
        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        request = service.files().get_media(
            fileId=entry.id,
            supportsAllDrives=True,
        )
        with destination_path.open("wb") as output_file:
            downloader = MediaIoBaseDownload(
                output_file,
                request,
            )
            done = False
            while not done:
                _, done = downloader.next_chunk()
        return entry.name, destination_path

    def _download_directory(
        self,
        folder_id: str,
        local_directory: Path,
        downloaded: list[Path],
    ) -> None:
        local_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        children = self._list_children(folder_id)
        for child in sorted(
            children,
            key=lambda item: item["name"].lower(),
        ):
            child_local_path = local_directory / child["name"]
            if child["mimeType"] == FOLDER_MIME_TYPE:
                self._download_directory(
                    folder_id=child["id"],
                    local_directory=child_local_path,
                    downloaded=downloaded,
                )
            else:
                downloaded_file = self._download_file(
                    item=child,
                    local_path=child_local_path,
                )
                downloaded.append(downloaded_file)

    def _resolve_item_path(self, path: str) -> dict[str, Any]:
        normalized = self._normalize_relative_path(path)
        if normalized == "/":
            return (
                self.service.files()
                .get(
                    fileId=self._root_id,
                    fields="id,name,mimeType,size,modifiedTime,webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )
        drive_path = PurePosixPath(normalized)
        parent_path = str(drive_path.parent)
        if parent_path == ".":
            parent_path = "/"

        parent_id = self._resolve_relative_path(parent_path)
        item = self._find_child(
            parent_id=parent_id,
            name=drive_path.name,
            mime_type=None,
        )
        if item is None:
            raise FileNotFoundError(f"Google Drive path not found: {normalized}")
        return item

    def _download_file(
        self,
        item: dict[str, Any],
        local_path: Path,
    ) -> Path:
        file_id = item["id"]
        mime_type = item["mimeType"]
        if mime_type.startswith("application/vnd.google-apps."):
            raise NotImplementedError(
                f"Google Workspace file export is not implemented: "
                f"{item['name']} ({mime_type})"
            )
        metadata = (
            self.service.files()
            .get(
                fileId=file_id,
                fields="id,name,mimeType,capabilities(canDownload)",
                supportsAllDrives=True,
            )
            .execute()
        )
        can_download = metadata.get("capabilities", {}).get("canDownload", False)
        if not can_download:
            raise PermissionError(f"The file cannot be downloaded: {item['name']}")
        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        request = self.service.files().get_media(
            fileId=file_id,
            supportsAllDrives=True,
        )
        with local_path.open("wb") as output_file:
            downloader = MediaIoBaseDownload(
                output_file,
                request,
            )
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    print(f"Downloading {item['name']}: {progress}%")
        return local_path

    def _load_credentials(self) -> Credentials:
        if not self.credentials_file.exists():
            raise FileNotFoundError(self.credentials_file)
        return service_account.Credentials.from_service_account_file(
            str(self.credentials_file),
            scopes=SCOPES,
        )

    def _resolve_relative_path(self, path: str) -> str:
        if path == "/":
            return self._root_id
        cached = self._path_cache.get(path)
        if cached:
            return cached
        current_id = self._root_id
        current_path = "/"
        for segment in PurePosixPath(path).parts:
            if segment == "/":
                continue
            current_path = self._join_drive_path(current_path, segment)
            cached_id = self._path_cache.get(current_path)
            if cached_id:
                current_id = cached_id
                continue
            item = self._find_child(
                parent_id=current_id,
                name=segment,
                mime_type=FOLDER_MIME_TYPE,
            )
            if item is None:
                raise FileNotFoundError(
                    f"Google Drive directory not found: {current_path}"
                )
            current_id = item["id"]
            self._path_cache[current_path] = current_id
        return current_id

    def _ensure_drive_directory(self, path: str) -> str:
        normalized = self._normalize_relative_path(path)
        if normalized == "/":
            return self._root_id
        current_id = self._root_id
        current_path = "/"
        for segment in PurePosixPath(normalized).parts:
            if segment == "/":
                continue
            current_path = self._join_drive_path(current_path, segment)
            cached = self._path_cache.get(current_path)
            if cached:
                current_id = cached
                continue
            item = self._find_child(
                parent_id=current_id,
                name=segment,
                mime_type=FOLDER_MIME_TYPE,
            )
            if item is None:
                metadata = {
                    "name": segment,
                    "mimeType": FOLDER_MIME_TYPE,
                    "parents": [current_id],
                }
                item = (
                    self.service.files()
                    .create(
                        body=metadata,
                        fields="id,name,mimeType",
                        supportsAllDrives=True,
                    )
                    .execute()
                )
            current_id = item["id"]
            self._path_cache[current_path] = current_id
        return current_id

    def _upload_file(self, local_file: Path, drive_path: str) -> DriveEntry:
        drive_path = self._normalize_relative_path(drive_path)
        parent_path = str(PurePosixPath(drive_path).parent)
        if parent_path == ".":
            parent_path = "/"
        parent_id = self._ensure_drive_directory(parent_path)
        name = PurePosixPath(drive_path).name
        existing = self._find_child(
            parent_id=parent_id,
            name=name,
            mime_type=None,
        )
        mime_type, _ = mimetypes.guess_type(local_file.name)
        media = MediaFileUpload(
            str(local_file),
            mimetype=mime_type or "application/octet-stream",
            resumable=True,
        )
        fields = "id,name,mimeType,size,modifiedTime,webViewLink"
        if existing and existing["mimeType"] != FOLDER_MIME_TYPE:
            item = (
                self.service.files()
                .update(
                    fileId=existing["id"],
                    media_body=media,
                    fields=fields,
                    supportsAllDrives=True,
                )
                .execute()
            )
        elif existing:
            raise IsADirectoryError(
                f"A Google Drive directory already exists at: {drive_path}"
            )
        else:
            metadata = {
                "name": name,
                "parents": [parent_id],
            }
            item = (
                self.service.files()
                .create(
                    body=metadata,
                    media_body=media,
                    fields=fields,
                    supportsAllDrives=True,
                )
                .execute()
            )

        return self._to_entry(item, parent_path)

    def _list_children(self, folder_id: str) -> list[dict[str, Any]]:
        query = f"'{folder_id}' in parents and trashed = false"
        items: list[dict[str, Any]] = []
        page_token: str | None = None

        while True:
            response = (
                self.service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields=(
                        "nextPageToken,"
                        "files(id,name,mimeType,size,modifiedTime,webViewLink)"
                    ),
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                    corpora="drive" if self.shared_drive_id else "user",
                    driveId=self.shared_drive_id,
                )
                .execute()
            )
            items.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                return items

    def _find_child(
        self,
        parent_id: str,
        name: str,
        mime_type: str | None,
    ) -> dict[str, Any] | None:
        escaped_name = name.replace("\\", "\\\\").replace("'", "\\'")
        query_parts = [
            f"'{parent_id}' in parents",
            f"name = '{escaped_name}'",
            "trashed = false",
        ]

        if mime_type is not None:
            query_parts.append(f"mimeType = '{mime_type}'")

        response = (
            self.service.files()
            .list(
                q=" and ".join(query_parts),
                spaces="drive",
                fields="files(id,name,mimeType,size,modifiedTime,webViewLink)",
                pageSize=10,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                corpora="drive" if self.shared_drive_id else "user",
                driveId=self.shared_drive_id,
            )
            .execute()
        )

        items = response.get("files", [])

        if len(items) > 1:
            raise RuntimeError(
                f"Ambiguous Google Drive path: multiple items named {name!r}"
            )

        return items[0] if items else None

    @staticmethod
    def _to_entry(item: dict[str, Any], parent_path: str) -> DriveEntry:
        path = GoogleApiDriveClient._join_drive_path(parent_path, item["name"])
        raw_size = item.get("size")

        return DriveEntry(
            id=item["id"],
            name=item["name"],
            path=path,
            entry_type=(
                "directory" if item["mimeType"] == FOLDER_MIME_TYPE else "file"
            ),
            mime_type=item["mimeType"],
            size_bytes=int(raw_size) if raw_size is not None else None,
            modified_time=item.get("modifiedTime"),
            web_view_link=item.get("webViewLink"),
        )

    @staticmethod
    def _normalize_drive_path(path: str) -> str:
        value = "/" + path.strip().strip("/")
        return "/" if value == "/" else value

    @staticmethod
    def _normalize_relative_path(path: str) -> str:
        normalized = GoogleApiDriveClient._normalize_drive_path(path)
        parts = PurePosixPath(normalized).parts

        if ".." in parts:
            raise ValueError("Parent traversal ('..') is not allowed")

        return normalized

    @staticmethod
    def _join_drive_path(parent: str, child: str) -> str:
        if parent == "/":
            return f"/{child}"
        return f"{parent.rstrip('/')}/{child}"
