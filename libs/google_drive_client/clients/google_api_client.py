from __future__ import annotations

import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaFileUpload

from ..google_drive_client_contract import GoogleDriveClientContract
from ..types.google_drive_client_types import DriveClientInitArgs, DriveEntry


FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
SCOPES = ["https://www.googleapis.com/auth/drive"]


class GoogleApiDriveClient(GoogleDriveClientContract):
    """Google Drive implementation backed by Google Drive API v3.

    Public methods use path strings rather than Drive IDs. Internally, paths are
    resolved segment by segment and cached.
    """

    def __init__(self, args: DriveClientInitArgs) -> None:
        self.credentials_file = Path(args.credentials_file)
        self.root_folder_id = args.root_folder_id
        self.shared_drive_id = args.shared_drive_id

        credentials = self._load_credentials()

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
                f"root_folder_id does not reference a directory: "
                f"{self.root_folder_id}"
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
