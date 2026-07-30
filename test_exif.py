from pathlib import Path

from src.exif.exif_reader import extract_gps_from_directory


IMAGE_DIRECTORY = Path("temp")


def main() -> None:
    print("Reading images from temp folder...")
    print()

    results = extract_gps_from_directory(
        IMAGE_DIRECTORY
    )

    if not results:
        print("No supported images were found.")
        return

    print(f"Found {len(results)} image(s).")
    print("-" * 60)

    for result in results:
        print(f"Filename   : {result['filename']}")
        print(f"GPS Status : {result['gps_status']}")

        if result["gps_status"] == "Available":
            print(f"Latitude   : {result['latitude']}")
            print(f"Longitude  : {result['longitude']}")
        else:
            print("Latitude   : Not Available")
            print("Longitude  : Not Available")

        if result["error"]:
            print(f"Error      : {result['error']}")

        print("-" * 60)


if __name__ == "__main__":
    main()