# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "beartype",
#     "pandas",
#     "requests",
#     "tqdm",
#     "tyro",
# ]
# ///

# Built on Michelle's download script: https://huggingface.co/datasets/imageomics/Comparison-Subset-Jiggins/blob/977a934e1eef18f6b6152da430ac83ba6f7bd30f/download_jiggins_subset.py with modification of David's redo loop: https://github.com/Imageomics/data-fwg/blob/anomaly-data-challenge/HDR-anomaly-data-challenge/notebooks/download_images.ipynb and expanded logging and file checks. Further added checksum calculation for all downloaded images at end.

# Script to download Jiggins images from any of the master CSV files. Generates Checksum file for all images downloaded (<master filename>_checksums.csv). Logs image downloads and failures in json files (<master filename>_log.json & <master filename>_error_log.json). Logs record numbers and response codes as strings, not int64.

import csv
import dataclasses
import hashlib
import json
import os
import shutil
import sys
import time

import beartype
import pandas as pd
import requests
import tqdm
import tyro

EXPECTED_COLS = [
    "CAMID",
    "X",
    "Image_name",
    "file_url",
    "Taxonomic_Name",
    "record_number",
    "Dataset",
]

REDO_CODE_LIST = [429, 500, 502, 503, 504]

# Reset to appropriate index if download gets interrupted.
STARTING_INDEX = 0


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    csv: str
    """Path to CSV file with urls."""
    output: str
    """Directory to write images to."""


@beartype.beartype
def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@beartype.beartype
def get_checksums(input_directory, output_filepath):
    with open(output_filepath, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filepath", "filename", "md5"])
        for root, dirs, files in os.walk(input_directory):
            n_files = len(files)
            for name in tqdm.tqdm(files, total=n_files, desc="MD5ing"):
                file_path = os.path.join(root, name)
                checksum = md5_checksum(file_path)
                writer.writerow([file_path, name, checksum])
        print(f"Checksums written to {output_filepath}")


@beartype.beartype
def log_response(
    log_data: dict,
    index: int,
    image,
    url,
    record_number,
    dataset,
    cam_id,
    response_code,
) -> dict[object, object]:
    # log status
    log_entry = {}
    log_entry["Image"] = image
    log_entry["file_url"] = url
    log_entry["record_number"] = str(record_number)  # int64 has problems sometimes
    log_entry["dataset"] = dataset
    log_entry["CAMID"] = cam_id
    log_entry["Response_status"] = str(response_code)
    log_data[index] = log_entry

    return log_data


@beartype.beartype
def update_log(log, index, filepath):
    # save logs
    with open(filepath, "a") as log_file:
        json.dump(log[index], log_file, indent=4)
        log_file.write("\n")


@beartype.beartype
def download_images(jiggins_data, image_folder, log_filepath, error_log_filepath):
    log_data = {}
    log_errors = {}

    for i in tqdm.tqdm(range(0, len(jiggins_data))):
        # species will really be <Genus> <species> ssp. <subspecies>, where subspecies indicated
        species = jiggins_data["Taxonomic_Name"][i]
        image_name = (
            jiggins_data["X"][i].astype(str) + "_" + jiggins_data["Image_name"][i]
        )
        record_number = jiggins_data["record_number"][i]

        # download the image from url if not already downloaded
        # Will attempt to download everything in CSV (image_name is unique: <X>_<Image_name>), unless download restarted
        if not os.path.exists(f"{image_folder}/{species}/{image_name}"):
            # get image from url
            url = jiggins_data["file_url"][i]
            dataset = jiggins_data["Dataset"][i]
            cam_id = jiggins_data["CAMID"][i]

            # download the image
            redo = True
            max_redos = 2
            while redo and max_redos > 0:
                try:
                    response = requests.get(url, stream=True)
                except Exception as e:
                    redo = True
                    max_redos -= 1
                    if max_redos <= 0:
                        log_errors = log_response(
                            log_errors,
                            index=i,
                            image=species + "/" + image_name,
                            url=url,
                            record_number=record_number,
                            dataset=dataset,
                            cam_id=cam_id,
                            response_code=str(e),
                        )
                        update_log(log=log_errors, index=i, filepath=error_log_filepath)

                if response.status_code == 200:
                    redo = False
                    # log status
                    log_data = log_response(
                        log_data,
                        index=i,
                        image=species + "/" + image_name,
                        url=url,
                        record_number=record_number,
                        dataset=dataset,
                        cam_id=cam_id,
                        response_code=response.status_code,
                    )
                    update_log(log=log_data, index=i, filepath=log_filepath)

                    # create the species appropriate folder if necessary
                    if not os.path.exists(f"{image_folder}/{species}"):
                        os.makedirs(f"{image_folder}/{species}", exist_ok=False)

                    # save image to appropriate folder
                    with open(
                        f"{image_folder}/{species}/{image_name}", "wb"
                    ) as out_file:
                        shutil.copyfileobj(response.raw, out_file)

                # check for too many requests
                elif response.status_code in REDO_CODE_LIST:
                    redo = True
                    max_redos -= 1
                    if max_redos <= 0:
                        log_errors = log_response(
                            log_errors,
                            index=i,
                            image=species + "/" + image_name,
                            url=url,
                            record_number=record_number,
                            dataset=dataset,
                            cam_id=cam_id,
                            response_code=response.status_code,
                        )
                        update_log(log=log_errors, index=i, filepath=error_log_filepath)

                    else:
                        time.sleep(1)
                else:  # other fail, eg. 404
                    redo = False
                    log_errors = log_response(
                        log_errors,
                        index=i,
                        image=species + "/" + image_name,
                        url=url,
                        record_number=record_number,
                        dataset=dataset,
                        cam_id=cam_id,
                        response_code=response.status_code,
                    )
                    update_log(log=log_errors, index=i, filepath=error_log_filepath)

            del response

        else:
            if i > STARTING_INDEX:
                # No need to print if download is restarted due to interruption (set STARTING_INDEX accordingly).
                print(
                    f"duplicate image: {jiggins_data['X']}, {jiggins_data['Image_name']}, from record {record_number}"
                )


@beartype.beartype
def main(cfg: Config):
    # log file location (folder of source CSV)
    log_filepath = cfg.csv.split(".")[0] + "_log.json"
    error_log_filepath = cfg.csv.split(".")[0] + "_error_log.json"

    # load csv
    jiggins_data = pd.read_csv(cfg.csv, low_memory=False)

    # Check for required columns
    missing_cols = []
    for col in EXPECTED_COLS:
        if col not in list(jiggins_data.columns):
            missing_cols.append(col)
    if len(missing_cols) > 0:
        sys.exit(f"The CSV is missing column(s): {missing_cols}")

    # dowload images from urls
    download_images(jiggins_data, cfg.output, log_filepath, error_log_filepath)

    # generate checksums and save CSV to same folder as CSV used for download
    checksum_path = cfg.csv.split(".")[0] + "_checksums.csv"
    get_checksums(cfg.output, checksum_path)

    print(f"Images downloaded from {cfg.csv} to {cfg.output}.")
    print(
        f"Checksums recorded in {checksum_path} and download logs are in {log_filepath} and {error_log_filepath}."
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
