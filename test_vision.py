import os
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import vision


load_dotenv()


def test_cloud_vision(image_path: str) -> None:
    """Run OCR and object localization on one local image."""

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not credentials_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS is missing from the .env file."
        )

    if not Path(credentials_path).exists():
        raise FileNotFoundError(
            f"Service-account JSON was not found: {credentials_path}"
        )

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    client = vision.ImageAnnotatorClient()

    with path.open("rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    print("\n========== OCR RESULTS ==========")

    text_response = client.text_detection(image=image)

    if text_response.error.message:
        raise RuntimeError(
            f"Cloud Vision OCR error: {text_response.error.message}"
        )

    texts = text_response.text_annotations

    if not texts:
        print("No text detected.")
    else:
        print("Complete detected text:")
        print(texts[0].description)

        print("\nIndividual text regions:")

        for text in texts[1:]:
            vertices = [
                (vertex.x, vertex.y)
                for vertex in text.bounding_poly.vertices
            ]

            print(
                {
                    "text": text.description,
                    "bounding_box": vertices,
                }
            )

    print("\n========== OBJECT RESULTS ==========")

    object_response = client.object_localization(image=image)

    if object_response.error.message:
        raise RuntimeError(
            f"Cloud Vision object detection error: "
            f"{object_response.error.message}"
        )

    objects = object_response.localized_object_annotations

    if not objects:
        print("No objects detected.")
    else:
        for detected_object in objects:
            normalized_box = [
                {
                    "x": vertex.x,
                    "y": vertex.y,
                }
                for vertex in detected_object.bounding_poly.normalized_vertices
            ]

            print(
                {
                    "object": detected_object.name,
                    "confidence": round(detected_object.score, 4),
                    "normalized_bounding_box": normalized_box,
                }
            )


if __name__ == "__main__":
    test_cloud_vision("temp/IMG_7259.JPG")