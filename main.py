from libs.google_drive_client import GoogleDriveClient, DriveClientInitArgs


def main():

    client = GoogleDriveClient(
        DriveClientInitArgs(
            client_name="google_api",
            credentials_file=".credentials/echo-saviia-credentials.json",
            root_folder_id='1LVUSQJjcuPEDt4taa_xgizsOih6vRTTb'
        )
    )
    
    files = [(f.name, f.entry_type) for f in client.list_dir('/Data/')]
    print(files)
    


if __name__ == "__main__":
    main()
