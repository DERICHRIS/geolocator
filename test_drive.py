
from pathlib import Path

from src.drive.drive_client import (
    create_drive_service,
    download_drive_file,
    list_images_in_folder,
)


CREDENTIALS_PATH = Path("credentials/service-account.json")
DOWNLOAD_DIRECTORY = Path("temp")

FOLDER_ID = "17YfJLMyImWspEqqzuyDSQd8HqRonEcWn"


def main() -> None:
    if not FOLDER_ID.strip():
        raise ValueError("Google Drive folder ID cannot be empty.")

    print("Connecting to Google Drive...")

    drive_service = create_drive_service(CREDENTIALS_PATH)

    print("Connected successfully.")
    print("Searching for images...")

    images = list_images_in_folder(
        drive_service=drive_service,
        folder_id=FOLDER_ID,
    )

    if not images:
        print("No supported images were found.")
        print(
            "Make sure the Google Drive folder is shared with the "
            "service-account email."
        )
        return

    print(f"Found {len(images)} image(s):")

    for index, image in enumerate(images, start=1):
        print(
            f"{index}. {image['name']} "
            f"({image['mimeType']})"
        )

    print("\nDownloading images...")

    for image in images:
        try:
            downloaded_path = download_drive_file(
                drive_service=drive_service,
                file_id=image["id"],
                filename=image["name"],
                output_directory=DOWNLOAD_DIRECTORY,
            )

            print(f"Saved: {downloaded_path}")

        except Exception as error:
            print(
                f"Failed to download {image['name']}: {error}"
            )

    print("\nImage download process completed.")


if __name__ == "__main__":
    main()
