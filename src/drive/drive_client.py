from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import Resource, build
from googleapiclient.http import MediaIoBaseDownload


DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/heic",
    "image/heif",
    "image/webp",
}


def create_drive_service(credentials_path: str | Path) -> Resource:
    """
    Authenticate with Google Drive using a service-account JSON file.
    """

    credentials_file = Path(credentials_path)

    if not credentials_file.exists():
        raise FileNotFoundError(
            f"Service-account credentials not found: {credentials_file}"
        )

    credentials = service_account.Credentials.from_service_account_file(
        str(credentials_file),
        scopes=[DRIVE_READONLY_SCOPE],
    )

    return build(
        "drive",
        "v3",
        credentials=credentials,
        cache_discovery=False,
    )


def list_images_in_folder(
    drive_service: Resource,
    folder_id: str,
) -> list[dict[str, Any]]:
    """
    Return all supported image files directly inside a Drive folder.
    """

    images: list[dict[str, Any]] = []
    page_token: str | None = None

    query = (
        f"'{folder_id}' in parents "
        "and trashed = false"
    )

    while True:
        response = (
            drive_service.files()
            .list(
                q=query,
                spaces="drive",
                fields=(
                    "nextPageToken, "
                    "files(id, name, mimeType, size, modifiedTime)"
                ),
                pageToken=page_token,
                pageSize=1000,
            )
            .execute()
        )

        files = response.get("files", [])

        for file in files:
            if file.get("mimeType") in SUPPORTED_IMAGE_TYPES:
                images.append(file)

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return images


def download_drive_file(
    drive_service: Resource,
    file_id: str,
    filename: str,
    output_directory: str | Path,
) -> Path:
    """
    Download one Google Drive file into the local output directory.
    """

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(filename).name
    destination = output_path / safe_filename

    request = drive_service.files().get_media(fileId=file_id)

    with destination.open("wb") as output_file:
        downloader = MediaIoBaseDownload(output_file, request)

        done = False

        while not done:
            status, done = downloader.next_chunk()

            if status:
                percentage = int(status.progress() * 100)
                print(f"Downloading {safe_filename}: {percentage}%")

    return destination