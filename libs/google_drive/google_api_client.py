from googleapiclient.discovery import build
import os
import requests


class GoogleDriveClient:
    def __init__(self, api_key, output_path: str = "output"):
        self.service = build("drive", "v3", developerKey=api_key)
        self.output_path = output_path

    def download_file(self, file_id):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url, stream=True)
        with open(os.path.join(self.output_path, f"{file_id}"), "wb") as f:
            for chunk in r.iter_content(1024):
                if chunk:
                    f.write(chunk)

    def list_subfolders(self, folder_id):
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        res = self.service.files().list(q=query, fields="files(id, name)").execute()
        return res.get("files", [])

    def list_files(self, folder_id):
        query = f"'{folder_id}' in parents and trashed=false"
        res = (
            self.service.files()
            .list(q=query, fields="files(id, name, mimeType)")
            .execute()
        )
        return res.get("files", [])
