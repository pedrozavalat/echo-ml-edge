import sys
import os
from pathlib import Path
import json
from PytorchWildlife.models import detection as pw_detection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from libs.google_drive_client import GoogleDriveClient, DriveClientInitArgs

gdrive = GoogleDriveClient(
    DriveClientInitArgs(
        client_name="google_api",
        credentials_file=".credentials/echo-saviia-credentials.json",
        root_folder_id="1LVUSQJjcuPEDt4taa_xgizsOih6vRTTb",
    )
)


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as file:
        json.dump(data, file)


def load_json(path: str) -> dict:
    with open(path, "r") as file:
        json_file = json.load(file)
    return json_file


def check_routes(fpath: str, name: str) -> dict:
    path = str(Path(fpath) / name)
    entries = list(gdrive.list_dir(path))
    if any(f.entry_type == "file" for f in entries):
        return {name: [f.name for f in entries if f.entry_type == "file"]}

    dirs = [f.name for f in entries if f.entry_type == "directory"]
    children = {}
    for cname in dirs:
        children.update(check_routes(path, cname))
    return {name: children}


def extract_keys_path(name, map, keys):
    keys.append(name)
    if isinstance(map[name], list):
        return keys
    for key in map[name]:
        result = extract_keys_path(key, map[name], keys)
        if result:
            return result
    keys.pop()
    return None


def extract_last_value(map: dict, keys: list[str]):
    curr_map = map
    for key in keys:
        curr_map = curr_map[key]
    return [val for val in curr_map if not val.startswith("._")]


def main():
    DOWNLOAD_PATH_MAP = False
    PREPROCESS_PATH_MAP = False
    PATH_MAP = "scripts/data/"
    source = "Data"
    start_dir = "Sin clasificar"
    pathmap_name = "dirs.json"
    preprocessed_pathmap_name = "dirs_preprocessed.json"

    if DOWNLOAD_PATH_MAP:
        map = check_routes(source, start_dir)
        result = {}
        result[source] = map
        save_json(PATH_MAP + pathmap_name, result)
    else:
        map = load_json(PATH_MAP + pathmap_name)[source][start_dir]
        if PREPROCESS_PATH_MAP:
            preprocessed_map = {}
            for key in map.keys():
                keys = extract_keys_path(key, map, [])
                values = extract_last_value(map, keys)  # type: ignore
                dirpath = f"{source}/{start_dir}/" + "/".join(keys)  # type: ignore
                preprocessed_map[dirpath] = values
            save_json(PATH_MAP + preprocessed_pathmap_name, preprocessed_map)
        else:
            preprocessed_map = load_json(PATH_MAP + preprocessed_pathmap_name)

    # Create temporal directory
    if not os.path.exists(".tmp"):
        os.mkdir(".tmp")


if __name__ == "__main__":
    main()
