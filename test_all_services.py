import os
from pathlib import Path

import pytesseract
import requests
from dotenv import load_dotenv
from google import genai
from google.cloud import vision
from PIL import Image


load_dotenv()

TEST_IMAGE = Path("temp/IMG_7248.JPG")
TEST_LATITUDE = 10.8072194
TEST_LONGITUDE = 78.6770306


def print_result(service: str, success: bool, message: str) -> None:
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {service}: {message}")


def test_environment() -> bool:
    required_values = {
        "GOOGLE_MAPS_API_KEY": os.getenv("GOOGLE_MAPS_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS"
        ),
        "TESSERACT_CMD": os.getenv("TESSERACT_CMD"),
    }

    all_valid = True

    for variable, value in required_values.items():
        if not value:
            print_result(variable, False, "Missing from .env")
            all_valid = False
        else:
            print_result(variable, True, "Configured")

    credentials = required_values["GOOGLE_APPLICATION_CREDENTIALS"]

    if credentials and not Path(credentials).exists():
        print_result(
            "Vision credentials file",
            False,
            f"File not found: {credentials}",
        )
        all_valid = False
    elif credentials:
        print_result("Vision credentials file", True, "File exists")

    if not TEST_IMAGE.exists():
        print_result("Test image", False, f"File not found: {TEST_IMAGE}")
        all_valid = False
    else:
        print_result("Test image", True, str(TEST_IMAGE))

    return all_valid


def test_google_maps() -> bool:
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params = {
        "latlng": f"{TEST_LATITUDE},{TEST_LONGITUDE}",
        "key": api_key,
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()
        status = data.get("status")

        if status == "OK":
            address = data["results"][0]["formatted_address"]
            print_result("Google Maps Geocoding", True, address)
            return True

        error_message = data.get("error_message", "No error message returned")
        print_result(
            "Google Maps Geocoding",
            False,
            f"Status={status}; {error_message}",
        )
        return False

    except Exception as exc:
        print_result("Google Maps Geocoding", False, str(exc))
        return False


def create_vision_image() -> vision.Image:
    with TEST_IMAGE.open("rb") as image_file:
        content = image_file.read()

    return vision.Image(content=content)


def test_cloud_vision_ocr() -> bool:
    try:
        client = vision.ImageAnnotatorClient()
        image = create_vision_image()

        response = client.text_detection(image=image)

        if response.error.message:
            print_result(
                "Cloud Vision OCR",
                False,
                response.error.message,
            )
            return False

        annotations = response.text_annotations

        if annotations:
            detected_text = annotations[0].description.replace("\n", " | ")
            print_result(
                "Cloud Vision OCR",
                True,
                f"Detected: {detected_text}",
            )
        else:
            print_result(
                "Cloud Vision OCR",
                True,
                "API responded successfully; no text detected",
            )

        return True

    except Exception as exc:
        print_result("Cloud Vision OCR", False, str(exc))
        return False


def test_cloud_vision_objects() -> bool:
    try:
        client = vision.ImageAnnotatorClient()
        image = create_vision_image()

        response = client.object_localization(image=image)

        if response.error.message:
            print_result(
                "Cloud Vision Object Localization",
                False,
                response.error.message,
            )
            return False

        objects = response.localized_object_annotations

        if objects:
            detected = ", ".join(
                f"{obj.name} ({obj.score:.2f})"
                for obj in objects
            )

            print_result(
                "Cloud Vision Object Localization",
                True,
                f"Detected: {detected}",
            )
        else:
            print_result(
                "Cloud Vision Object Localization",
                True,
                "API responded successfully; no supported objects detected",
            )

        return True

    except Exception as exc:
        print_result(
            "Cloud Vision Object Localization",
            False,
            str(exc),
        )
        return False


def test_gemini_text() -> bool:
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents="Reply with exactly: GEMINI_API_WORKING",
        )

        text = response.text.strip() if response.text else ""

        if "GEMINI_API_WORKING" in text:
            print_result("Gemini text API", True, text)
            return True

        print_result(
            "Gemini text API",
            False,
            f"Unexpected response: {text}",
        )
        return False

    except Exception as exc:
        print_result("Gemini text API", False, str(exc))
        return False


def test_gemini_vision() -> bool:
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        image = Image.open(TEST_IMAGE)

        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=[
                image,
                (
                    "Confirm that you can inspect this image. "
                    "Reply with one short sentence."
                ),
            ],
        )

        text = response.text.strip() if response.text else ""

        if text:
            print_result("Gemini Vision", True, text)
            return True

        print_result("Gemini Vision", False, "Empty response")
        return False

    except Exception as exc:
        print_result("Gemini Vision", False, str(exc))
        return False


from pathlib import Path

import pytesseract
from PIL import Image, ImageOps


def test_tesseract() -> bool:
    converted_path = Path("temp/tesseract_test_image.png")

    try:
        tesseract_path = os.getenv("TESSERACT_CMD")

        if not tesseract_path:
            print_result(
                "Tesseract OCR",
                False,
                "TESSERACT_CMD is missing from .env",
            )
            return False

        tesseract_path = tesseract_path.strip()

        if not Path(tesseract_path).is_file():
            print_result(
                "Tesseract OCR",
                False,
                f"Executable not found: {repr(tesseract_path)}",
            )
            return False

        pytesseract.pytesseract.tesseract_cmd = tesseract_path

        # Open the original image
        with Image.open(TEST_IMAGE) as original_image:
            print(
                "Original image details:",
                {
                    "format": original_image.format,
                    "mode": original_image.mode,
                    "size": original_image.size,
                },
            )

            # Correct iPhone EXIF orientation
            corrected_image = ImageOps.exif_transpose(original_image)

            # Convert into a standard 3-channel RGB image
            rgb_image = corrected_image.convert("RGB")

            # Save as a standard PNG that Tesseract can read
            rgb_image.save(
                converted_path,
                format="PNG",
            )

        version = pytesseract.get_tesseract_version()

        detected_text = pytesseract.image_to_string(
            str(converted_path),
            lang="eng",
            config="--psm 11",
        ).strip()

        preview = detected_text.replace("\n", " | ")[:150]

        if preview:
            message = (
                f"Version {version}; "
                f"detected: {preview}"
            )
        else:
            message = (
                f"Version {version}; "
                "Tesseract executed successfully; no readable text detected"
            )

        print_result("Tesseract OCR", True, message)
        return True

    except Exception as exc:
        print_result(
            "Tesseract OCR",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return False

def main() -> None:
    print("=" * 65)
    print("PROPERTY PHOTO PIPELINE - SERVICE HEALTH CHECK")
    print("=" * 65)

    environment_ok = test_environment()

    print("\n" + "-" * 65)

    if not environment_ok:
        print("Environment validation failed. Fix the missing values first.")
        return

    results = {
        "Google Maps Geocoding": test_google_maps(),
        "Cloud Vision OCR": test_cloud_vision_ocr(),
        "Cloud Vision Object Localization": test_cloud_vision_objects(),
        "Gemini Text": test_gemini_text(),
        "Gemini Vision": test_gemini_vision(),
        "Tesseract OCR": test_tesseract(),
    }

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    passed = 0

    for service, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{service:<40} {status}")

        if success:
            passed += 1

    print("-" * 65)
    print(f"Passed: {passed}/{len(results)}")

    if passed == len(results):
        print("All services are working.")
    else:
        print("One or more services need attention.")


if __name__ == "__main__":
    main()