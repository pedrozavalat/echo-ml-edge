from .clients.google_api_client import GoogleApiDriveClient
from .google_drive_client_contract import GoogleDriveClientContract
from .types.google_drive_client_types import DriveClientInitArgs, DriveEntry


class GoogleDriveClient(GoogleDriveClientContract):
    CLIENTS = {"google_api"}

    def __init__(self, args: DriveClientInitArgs) -> None:
        if args.client_name not in self.CLIENTS:
            raise KeyError(f"Unsupported client: {args.client_name}")

        if args.client_name == "google_api":
            self.client_obj = GoogleApiDriveClient(args)
        else:
            raise KeyError(f"Unsupported client: {args.client_name}")

        self.client_name = args.client_name

    def list_dir(self, path: str) -> list[DriveEntry]:
        return self.client_obj.list_dir(path)

    def list_files(self, path: str) -> list[DriveEntry]:
        return self.client_obj.list_files(path)

    def upload_to(self, source_path: str, destination_path: str) -> list[DriveEntry]:
        return self.client_obj.upload_to(source_path, destination_path)
