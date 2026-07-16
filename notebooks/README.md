# Notebooks

## Extractors
### 02 Extract EPII Dataset

This notebook uses the Google Drive Client library. 
```python
gdrive = GoogleDriveClient(
    DriveClientInitArgs(
        client_name="google_api",
        credentials_file="echo-saviia-credentials.json",
        root_folder_id="1LVUSQJjcuPEDt4taa_xgizsOih6vRTTb",
    )
)
```
> Notes: You need to include the credentials of the service account in the main directory and reference them at the `credentials_file` parameter. Furthermore, the `root_folder_id` declares the Folder ID where you want to stablish the connection at Google Drive. 
