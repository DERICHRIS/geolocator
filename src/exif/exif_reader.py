from __future__ import annotations

from pathlib import Path
from typing import Any

import exifread


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".webp",
}


def _ratio_to_float(value: Any) -> float:
    """
    Convert an EXIF ratio value into a normal float.

    Example:
        15/2 becomes 7.5
    """

    numerator = float(value.num)
    denominator = float(value.den)

    if denominator == 0:
        raise ValueError("EXIF ratio denominator cannot be zero.")

    return numerator / denominator


def _convert_dms_to_decimal(
    degrees: Any,
    minutes: Any,
    seconds: Any,
    direction: str,
) -> float:
    """
    Convert GPS coordinates from Degrees-Minutes-Seconds format
    into decimal format.

    Example:
        10 degrees, 47 minutes, 25 seconds North
        becomes approximately 10.790278
    """

    degrees_value = _ratio_to_float(degrees)
    minutes_value = _ratio_to_float(minutes)
    seconds_value = _ratio_to_float(seconds)

    decimal_coordinate = (
        degrees_value
        + minutes_value / 60
        + seconds_value / 3600
    )

    if direction.upper() in {"S", "W"}:
        decimal_coordinate *= -1

    return decimal_coordinate


def extract_gps_from_image(
    image_path: str | Path,
) -> dict[str, Any]:
    """
    Extract GPS latitude and longitude from an image's EXIF metadata.

    Returns a dictionary containing:
    - filename
    - latitude
    - longitude
    - gps_status
    - error
    """

    file_path = Path(image_path)

    result: dict[str, Any] = {
        "filename": file_path.name,
        "latitude": None,
        "longitude": None,
        "gps_status": "Not Available",
        "error": None,
    }

    if not file_path.exists():
        result["error"] = "Image file does not exist."
        return result

    if not file_path.is_file():
        result["error"] = "Path is not a file."
        return result

    if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        result["error"] = (
            f"Unsupported image extension: {file_path.suffix}"
        )
        return result

    try:
        with file_path.open("rb") as image_file:
            exif_tags = exifread.process_file(
                image_file,
                details=False,
            )

        latitude_tag = exif_tags.get("GPS GPSLatitude")
        latitude_reference_tag = exif_tags.get(
            "GPS GPSLatitudeRef"
        )

        longitude_tag = exif_tags.get("GPS GPSLongitude")
        longitude_reference_tag = exif_tags.get(
            "GPS GPSLongitudeRef"
        )

        if not all(
            [
                latitude_tag,
                latitude_reference_tag,
                longitude_tag,
                longitude_reference_tag,
            ]
        ):
            return result

        latitude_values = latitude_tag.values
        longitude_values = longitude_tag.values

        if len(latitude_values) != 3:
            result["error"] = "Invalid GPS latitude format."
            return result

        if len(longitude_values) != 3:
            result["error"] = "Invalid GPS longitude format."
            return result

        latitude = _convert_dms_to_decimal(
            degrees=latitude_values[0],
            minutes=latitude_values[1],
            seconds=latitude_values[2],
            direction=str(latitude_reference_tag),
        )

        longitude = _convert_dms_to_decimal(
            degrees=longitude_values[0],
            minutes=longitude_values[1],
            seconds=longitude_values[2],
            direction=str(longitude_reference_tag),
        )

        result["latitude"] = round(latitude, 7)
        result["longitude"] = round(longitude, 7)
        result["gps_status"] = "Available"

        return result

    except Exception as error:
        result["error"] = str(error)
        return result


def extract_gps_from_directory(
    directory_path: str | Path,
) -> list[dict[str, Any]]:
    """
    Extract GPS metadata from every supported image
    inside a directory.
    """

    folder_path = Path(directory_path)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {folder_path}"
        )

    if not folder_path.is_dir():
        raise NotADirectoryError(
            f"Path is not a directory: {folder_path}"
        )

    results: list[dict[str, Any]] = []

    image_files = sorted(
        file_path
        for file_path in folder_path.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower()
        in SUPPORTED_IMAGE_EXTENSIONS
    )

    for image_file in image_files:
        gps_result = extract_gps_from_image(image_file)
        results.append(gps_result)

    return results