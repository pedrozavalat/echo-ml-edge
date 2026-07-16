# ECHO-SAVIIA ML Edge


## Google Drive Client Library

The main reusable component lives in `libs/google_drive_client/`. It provides a thin, typed wrapper around the Google Drive API v3 for common Drive operations:

- `list_dir(path)` to list the contents of a Drive folder.
- `list_files(path)` to list only files in a folder.
- `upload_to(source_path, destination_path)` to upload a local file or directory into Drive.
- `download(source_path, destination_path)` to download a Drive file or folder locally.
- `download_entries(entries, destination_path, max_workers=6)` to download a batch of Drive entries concurrently.

The public entry point is `GoogleDriveClient`, together with `DriveClientInitArgs` and `DriveEntry`.

### Credentials Required

You need a service account credentials JSON file to use the library. The file must be available locally and passed through the `credentials_file` argument when creating the client.

The service account also needs the appropriate permissions on the target Drive folder or shared drive. The `root_folder_id` defines the folder that the client treats as its root, and `shared_drive_id` can be provided when the content lives in a shared drive.

### Installation

Install the Python dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from libs.google_drive_client import GoogleDriveClient, DriveClientInitArgs

gdrive = GoogleDriveClient(
	DriveClientInitArgs(
		client_name="google_api",
		credentials_file="path/to/service-account.json",
		root_folder_id="your-root-folder-id",
		shared_drive_id=None,
	)
)

entries = gdrive.list_dir("/")
files = gdrive.list_files("/")
gdrive.download("/datasets/example.csv", "./data")
gdrive.upload_to("./local-folder", "/uploads")
```

The root path `/` always points to the folder identified by `root_folder_id`.

## Project Distribution

```text
.
├── LICENSE                     
├── README.md                   # Project overview and setup instructions
├── requirements.txt            # Python dependencies
├── docs/
│   └── reports/                # Generated reports and documentation
├── libs/
│   └── google_drive_client/    # Google Drive client library
└── notebooks/
    ├── experimentation/        # Experimental analyses and prototypes
    └── extractors/             # Data extraction notebooks
```


The repository is organized by purpose:

- `libs/`: reusable Python code. This is where the Google Drive client library lives.
- `notebooks/`: exploratory and working notebooks used to extract data and run experiments.
- `notebooks/extractors/`: notebooks focused on data extraction and dataset preparation from Google Drive or other sources.
- `notebooks/experimentation/`: notebooks used for model experimentation, comparisons, and iterative training work.
- `docs/`: written documentation and reports.
- `docs/reports/`: evaluation reports and project results.

## Notebook Areas

The notebook folder is split into two main streams:

### Extractors

These notebooks prepare or extract datasets, usually pulling files from Google Drive and converting them into a format suitable for training or evaluation.

### Experimentation

These notebooks are for model exploration, classifier development, testing different configurations, and validating results.

