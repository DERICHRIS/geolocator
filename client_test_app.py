import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, ImageOps
from google import genai
from google.cloud import vision
from google.genai import types


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

APP_DIR = Path(__file__).resolve().parent
ENV_FILE = APP_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=True,
)

st.set_page_config(
    page_title="Property Address Extractor",
    page_icon="🏠",
    layout="wide",
)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash",
).strip()
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS"
)


# ============================================================
# PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .main-title {
            text-align: center;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .sub-title {
            text-align: center;
            color: #666;
            margin-bottom: 2rem;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #dddddd;
            border-radius: 10px;
            overflow: hidden;
        }

        .uploaded-image-card {
            width: 100%;
            border: 1px solid #d9d9d9;
            border-radius: 12px;
            padding: 8px;
            background: #ffffff;
            box-sizing: border-box;
        }

        .uploaded-image-card img {
            display: block;
            width: 100%;
            height: auto;
            border-radius: 8px;
        }

        .uploaded-image-caption {
            margin-top: 8px;
            text-align: center;
            color: #7a7a7a;
            font-size: 0.9rem;
            overflow-wrap: anywhere;
        }

        div[data-testid="stButton"] > button {
            border-radius: 10px;
            min-height: 46px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_environment() -> list[str]:
    missing = []

    required = {
        "GOOGLE_MAPS_API_KEY": GOOGLE_MAPS_API_KEY,
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "GEMINI_MODEL": GEMINI_MODEL,
        "GOOGLE_APPLICATION_CREDENTIALS": GOOGLE_APPLICATION_CREDENTIALS,
    }

    for key, value in required.items():
        if not value:
            missing.append(key)

    if GOOGLE_APPLICATION_CREDENTIALS:
        credential_path = Path(GOOGLE_APPLICATION_CREDENTIALS)
        if not credential_path.exists():
            missing.append(
                f"Google credentials file does not exist: "
                f"{GOOGLE_APPLICATION_CREDENTIALS}"
            )

    return missing


def normalize_uploaded_image(uploaded_file) -> tuple[Image.Image, bytes]:
    uploaded_file.seek(0)

    with Image.open(uploaded_file) as original:
        corrected = ImageOps.exif_transpose(original)
        rgb_image = corrected.convert("RGB").copy()

    buffer = io.BytesIO()
    rgb_image.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return rgb_image, buffer.getvalue()


def ratio_to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        numerator = getattr(value, "numerator", None)
        denominator = getattr(value, "denominator", None)

        if numerator is None or not denominator:
            raise ValueError(f"Unable to convert EXIF GPS value: {value}")

        return float(numerator) / float(denominator)


def dms_to_decimal(dms: Any, reference: str) -> float:
    degrees = ratio_to_float(dms[0])
    minutes = ratio_to_float(dms[1])
    seconds = ratio_to_float(dms[2])

    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)

    if reference.upper() in {"S", "W"}:
        decimal *= -1

    return decimal


def extract_gps(uploaded_file) -> dict[str, Any]:
    uploaded_file.seek(0)

    with Image.open(uploaded_file) as image:
        exif = image.getexif()

        if not exif:
            return {
                "latitude": None,
                "longitude": None,
                "gps_status": "Not Available",
            }

        try:
            gps_ifd = exif.get_ifd(0x8825)
        except Exception:
            gps_ifd = None

        if not gps_ifd:
            return {
                "latitude": None,
                "longitude": None,
                "gps_status": "Not Available",
            }

        latitude_dms = gps_ifd.get(2)
        latitude_ref = gps_ifd.get(1)
        longitude_dms = gps_ifd.get(4)
        longitude_ref = gps_ifd.get(3)

        if not all(
            [
                latitude_dms,
                latitude_ref,
                longitude_dms,
                longitude_ref,
            ]
        ):
            return {
                "latitude": None,
                "longitude": None,
                "gps_status": "Not Available",
            }

        latitude = dms_to_decimal(
            latitude_dms,
            str(latitude_ref),
        )

        longitude = dms_to_decimal(
            longitude_dms,
            str(longitude_ref),
        )

        return {
            "latitude": latitude,
            "longitude": longitude,
            "gps_status": "Available",
        }


def reverse_geocode(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    response = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json",
        params={
            "latlng": f"{latitude},{longitude}",
            "key": GOOGLE_MAPS_API_KEY,
        },
        timeout=30,
    )

    response.raise_for_status()
    payload = response.json()

    status = payload.get("status")

    if status != "OK" or not payload.get("results"):
        error_message = payload.get(
            "error_message",
            status or "Unknown Google Maps error",
        )
        raise RuntimeError(
            f"Google Maps Geocoding failed: {error_message}"
        )

    result = payload["results"][0]

    formatted_address = result.get(
        "formatted_address",
        "Not Available",
    )

    # Remove a leading Google Plus Code such as:
    # RM4G+WR3, SBI Officers Colony, ...
    if formatted_address and formatted_address != "Not Available":
        formatted_address = re.sub(
            r"^[A-Z0-9]{4,8}\+[A-Z0-9]{2,4},\s*",
            "",
            formatted_address,
            flags=re.IGNORECASE,
        ).strip()

    extracted = {
        "formatted_address": formatted_address,
        "map_house_number": None,
        "street": None,
        "neighborhood": None,
        "city": None,
        "district": None,
        "state": None,
        "postal_code": None,
        "country": None,
        "location_type": result.get(
            "geometry",
            {},
        ).get(
            "location_type",
            "Unknown",
        ),
    }

    component_mapping = {
        "street_number": "map_house_number",
        "route": "street",
        "sublocality": "neighborhood",
        "sublocality_level_1": "neighborhood",
        "locality": "city",
        "administrative_area_level_2": "district",
        "administrative_area_level_1": "state",
        "postal_code": "postal_code",
        "country": "country",
    }

    for component in result.get("address_components", []):
        for component_type in component.get("types", []):
            field_name = component_mapping.get(component_type)

            if field_name and not extracted[field_name]:
                extracted[field_name] = component.get(
                    "long_name",
                    None,
                )

    return extracted


def run_cloud_vision(image_bytes: bytes) -> dict[str, Any]:
    client = vision.ImageAnnotatorClient()

    vision_image = vision.Image(
        content=image_bytes,
    )

    response = client.annotate_image(
        {
            "image": vision_image,
            "features": [
                {
                    "type_": vision.Feature.Type.TEXT_DETECTION,
                },
                {
                    "type_": vision.Feature.Type.OBJECT_LOCALIZATION,
                },
            ],
        }
    )

    if response.error.message:
        raise RuntimeError(
            f"Cloud Vision failed: {response.error.message}"
        )

    annotations = response.text_annotations

    full_text = (
        annotations[0].description.strip()
        if annotations
        else ""
    )

    text_candidates = []

    for annotation in annotations[1:]:
        text = annotation.description.strip()

        bounding_box = [
            {
                "x": vertex.x or 0,
                "y": vertex.y or 0,
            }
            for vertex in annotation.bounding_poly.vertices
        ]

        text_candidates.append(
            {
                "text": text,
                "bounding_box": bounding_box,
            }
        )

    numeric_candidates = []
    seen = set()

    for candidate in text_candidates:
        text = candidate["text"]

        if re.search(r"\d", text) and text not in seen:
            numeric_candidates.append(candidate)
            seen.add(text)

    detected_objects = []

    for detected_object in response.localized_object_annotations:
        detected_objects.append(
            {
                "name": detected_object.name,
                "confidence": round(
                    float(detected_object.score),
                    4,
                ),
            }
        )

    return {
        "full_text": full_text,
        "text_candidates": text_candidates,
        "numeric_candidates": numeric_candidates,
        "detected_objects": detected_objects,
    }


def validate_house_number_with_gemini(
    image: Image.Image,
    vision_result: dict[str, Any],
    address_result: dict[str, Any] | None,
) -> dict[str, Any]:
    client = genai.Client(
        api_key=GEMINI_API_KEY,
    )

    evidence = {
        "cloud_vision_full_text": vision_result.get(
            "full_text",
            "",
        ),
        "numeric_ocr_candidates": vision_result.get(
            "numeric_candidates",
            [],
        ),
        "detected_objects": vision_result.get(
            "detected_objects",
            [],
        ),
        "gps_derived_address": address_result,
    }

    prompt = f"""
You are analyzing one exterior property photograph.

Your task is to identify the visible house number or building number
belonging specifically to the PRIMARY property shown in the photograph.

Available machine-generated evidence:

{json.dumps(evidence, indent=2, ensure_ascii=False)}

Accept a number only when it is clearly attached to or associated with:

- the main entrance gate
- the front door
- an address plaque
- a mailbox
- the front wall
- a building number sign

Reject numbers belonging to:

- cars or registration plates
- motorcycles
- electricity meters
- water meters
- gas meters
- utility boxes
- dates
- phone numbers
- advertisements
- posters
- construction markings
- permit numbers
- survey markings
- road signs
- neighbouring properties

Important rules:

1. Never guess.
2. GPS address is contextual information only.
3. Do not invent a house number from the GPS address.
4. If the visible evidence is insufficient, select manual_review.
5. If both a valid property number and unrelated numbers such as a vehicle
   registration are visible, ignore the unrelated numbers and return the
   valid property number.
6. Do not reject the whole image merely because a vehicle number is present.
7. Return only the best-supported answer.
"""

    response_schema = {
        "type": "OBJECT",
        "properties": {
            "decision": {
                "type": "STRING",
                "enum": [
                    "accept",
                    "reject",
                    "manual_review",
                ],
            },
            "house_number": {
                "type": "STRING",
                "nullable": True,
            },
            "confidence": {
                "type": "NUMBER",
            },
            "belongs_to_primary_property": {
                "type": "BOOLEAN",
            },
            "source_region": {
                "type": "STRING",
                "enum": [
                    "gate",
                    "door",
                    "address_plaque",
                    "mailbox",
                    "front_wall",
                    "building_sign",
                    "vehicle",
                    "utility_meter",
                    "road_sign",
                    "neighbouring_property",
                    "unknown",
                ],
            },
            "reason": {
                "type": "STRING",
            },
        },
        "required": [
            "decision",
            "house_number",
            "confidence",
            "belongs_to_primary_property",
            "source_region",
            "reason",
        ],
    }

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            image,
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    result = json.loads(response.text)

    confidence = float(
        result.get(
            "confidence",
            0.0,
        )
    )

    result["confidence"] = max(
        0.0,
        min(
            confidence,
            1.0,
        ),
    )

    return result


def apply_final_decision_rules(
    gemini_result: dict[str, Any],
    gps_result: dict[str, Any],
    address_result: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = gemini_result.get("decision")
    house_number = gemini_result.get("house_number")

    confidence = float(
        gemini_result.get(
            "confidence",
            0.0,
        )
    )

    belongs = bool(
        gemini_result.get(
            "belongs_to_primary_property",
            False,
        )
    )

    source_region = gemini_result.get(
        "source_region",
        "unknown",
    )

    ai_reason = gemini_result.get(
        "reason",
        "No reason provided",
    )

    gps_available = (
        gps_result.get("latitude") is not None
        and gps_result.get("longitude") is not None
    )

    address_available = bool(address_result)

    if not gps_available:
        final_status = "Manual Review"
        status_reason = "Geolocation metadata not available"

    elif not address_available:
        final_status = "Manual Review"
        status_reason = "Address could not be determined from geolocation"

    elif (
        decision == "accept"
        and house_number
        and belongs
        and confidence >= 0.85
    ):
        final_status = "Accepted"
        status_reason = "Geolocation and house number verified"

    elif decision == "reject":
        final_status = "Rejected"
        house_number = None

        rejection_reasons = {
            "vehicle": "Only a vehicle registration number was detected",
            "utility_meter": "Only a utility meter number was detected",
            "road_sign": "Detected number belongs to a road sign",
            "neighbouring_property": (
                "Detected number belongs to a neighbouring property"
            ),
        }

        status_reason = rejection_reasons.get(
            source_region,
            ai_reason or "Detected number does not belong to the property",
        )

    elif not house_number:
        final_status = "Manual Review"
        status_reason = "House number not visible or not readable"

    elif confidence < 0.85:
        final_status = "Manual Review"
        status_reason = "House number detected with low confidence"

    else:
        final_status = "Manual Review"
        status_reason = ai_reason or "Manual verification required"

    status_with_reason = f"{final_status} ({status_reason})"

    return {
        **gemini_result,
        "house_number": house_number,
        "final_status": final_status,
        "status_reason": status_reason,
        "status_with_reason": status_with_reason,
    }


def build_result_table(
    filename: str,
    gps_result: dict[str, Any],
    address_result: dict[str, Any] | None,
    final_result: dict[str, Any],
    vision_result: dict[str, Any],
) -> pd.DataFrame:
    address_result = address_result or {}

    detected_number = final_result.get(
        "house_number",
    )

    if not detected_number:
        detected_number = "Not Available"

    full_ocr_text = vision_result.get(
        "full_text",
        "",
    ).replace(
        "\n",
        " | ",
    )

    rows = [
        {
            "Field": "File Name",
            "Value": filename,
        },
        {
            "Field": "Final Status",
            "Value": final_result.get(
                "status_with_reason",
                "Unknown",
            ),
        },
        {
            "Field": "Status Reason",
            "Value": final_result.get(
                "status_reason",
                "No reason provided",
            ),
        },
        {
            "Field": "Detected House Number",
            "Value": detected_number,
        },
        {
            "Field": "Formatted Address",
            "Value": address_result.get(
                "formatted_address",
                "Not Available",
            ),
        },
        {
            "Field": "Street",
            "Value": address_result.get(
                "street",
            ) or "Not Available",
        },
        {
            "Field": "Neighbourhood",
            "Value": address_result.get(
                "neighborhood",
            ) or "Not Available",
        },
        {
            "Field": "City",
            "Value": address_result.get(
                "city",
            ) or "Not Available",
        },
        {
            "Field": "District",
            "Value": address_result.get(
                "district",
            ) or "Not Available",
        },
        {
            "Field": "State",
            "Value": address_result.get(
                "state",
            ) or "Not Available",
        },
        {
            "Field": "Postal Code",
            "Value": address_result.get(
                "postal_code",
            ) or "Not Available",
        },
        {
            "Field": "Country",
            "Value": address_result.get(
                "country",
            ) or "Not Available",
        },
        {
            "Field": "Latitude",
            "Value": (
                f"{gps_result['latitude']:.7f}"
                if gps_result.get("latitude") is not None
                else "Not Available"
            ),
        },
        {
            "Field": "Longitude",
            "Value": (
                f"{gps_result['longitude']:.7f}"
                if gps_result.get("longitude") is not None
                else "Not Available"
            ),
        },
        {
            "Field": "GPS Status",
            "Value": gps_result.get(
                "gps_status",
                "Not Available",
            ),
        },
        {
            "Field": "Google Location Type",
            "Value": address_result.get(
                "location_type",
            ) or "Not Available",
        },
        {
            "Field": "House Number Confidence",
            "Value": (
                f"{final_result.get('confidence', 0.0) * 100:.2f}%"
            ),
        },
        {
            "Field": "House Number Found On",
            "Value": final_result.get(
                "source_region",
                "unknown",
            ).replace(
                "_",
                " ",
            ).title(),
        },
        {
            "Field": "AI Reason",
            "Value": final_result.get(
                "reason",
                "No reason provided",
            ),
        },
        {
            "Field": "Cloud Vision OCR Text",
            "Value": full_ocr_text or "No text detected",
        },
    ]

    return pd.DataFrame(rows)


def build_csv_row(
    filename: str,
    gps_result: dict[str, Any],
    address_result: dict[str, Any] | None,
    final_result: dict[str, Any],
) -> pd.DataFrame:
    address_result = address_result or {}

    row = {
        "filename": filename,
        "house_number": final_result.get(
            "house_number",
        ) or "Not Available",
        "street": address_result.get(
            "street",
        ) or "Not Available",
        "neighborhood": address_result.get(
            "neighborhood",
        ) or "Not Available",
        "city": address_result.get(
            "city",
        ) or "Not Available",
        "district": address_result.get(
            "district",
        ) or "Not Available",
        "state": address_result.get(
            "state",
        ) or "Not Available",
        "postal_code": address_result.get(
            "postal_code",
        ) or "Not Available",
        "country": address_result.get(
            "country",
        ) or "Not Available",
        "formatted_address": address_result.get(
            "formatted_address",
        ) or "Not Available",
        "latitude": gps_result.get(
            "latitude",
        ),
        "longitude": gps_result.get(
            "longitude",
        ),
        "gps_status": gps_result.get(
            "gps_status",
        ),
        "confidence": round(
            float(
                final_result.get(
                    "confidence",
                    0.0,
                )
            ),
            4,
        ),
        "status": final_result.get(
            "final_status",
        ),
        "status_reason": final_result.get(
            "status_reason",
        ),
        "status_with_reason": final_result.get(
            "status_with_reason",
        ),
        "detected_on": final_result.get(
            "source_region",
        ),
        "ai_reason": final_result.get(
            "reason",
        ),
    }

    return pd.DataFrame([row])


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.markdown(
    '<div class="main-title">🏠 Property Address Extractor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Upload one original property image to extract GPS address details
        and identify the visible house number.
    </div>
    """,
    unsafe_allow_html=True,
)

missing_configuration = validate_environment()

if missing_configuration:
    st.error(
        "The application configuration is incomplete."
    )

    for item in missing_configuration:
        st.write(f"- {item}")

    st.info(
        "Add the required values to the .env file and restart Streamlit."
    )

    st.stop()


uploaded_file = st.file_uploader(
    "Upload property image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "mpo",
    ],
    help=(
        "Upload the original camera image whenever possible. "
        "WhatsApp and screenshot images often lose GPS metadata."
    ),
)

if uploaded_file is None:
    st.info(
        "Upload a property image to begin."
    )

else:
    try:
        normalized_image, normalized_bytes = normalize_uploaded_image(
            uploaded_file,
        )

        st.subheader("Uploaded Image")

        # Keep the preview and button inside the same compact center column
        # so both elements have exactly the same alignment and width.
        left, center, right = st.columns([2, 1.25, 2])

        with center:
            encoded_preview = base64.b64encode(
                normalized_bytes,
            ).decode("utf-8")

            safe_filename = (
                uploaded_file.name
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#x27;")
            )

            st.markdown(
                f"""
                <div class="uploaded-image-card">
                    <img
                        src="data:image/jpeg;base64,{encoded_preview}"
                        alt="Uploaded property image"
                    />
                    <div class="uploaded-image-caption">
                        {safe_filename}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            analyze_button = st.button(
                "🔍 Analyze Property",
                type="primary",
                use_container_width=True,
            )

        if analyze_button:
            with st.spinner(
                "Analyzing image and extracting property details..."
            ):
                uploaded_file.seek(0)

                gps_result = extract_gps(
                    uploaded_file,
                )

                address_result = None
                geocoding_warning = None

                latitude = gps_result.get(
                    "latitude",
                )

                longitude = gps_result.get(
                    "longitude",
                )

                if (
                    latitude is not None
                    and longitude is not None
                ):
                    try:
                        address_result = reverse_geocode(
                            latitude,
                            longitude,
                        )
                    except Exception as exc:
                        geocoding_warning = str(
                            exc,
                        )
                else:
                    geocoding_warning = (
                        "GPS metadata was not found. "
                        "The image may be compressed, edited, "
                        "downloaded from WhatsApp, or captured as a screenshot."
                    )

                vision_result = run_cloud_vision(
                    normalized_bytes,
                )

                gemini_result = validate_house_number_with_gemini(
                    image=normalized_image,
                    vision_result=vision_result,
                    address_result=address_result,
                )

                final_result = apply_final_decision_rules(
                    gemini_result=gemini_result,
                    gps_result=gps_result,
                    address_result=address_result,
                )

                details_table = build_result_table(
                    filename=uploaded_file.name,
                    gps_result=gps_result,
                    address_result=address_result,
                    final_result=final_result,
                    vision_result=vision_result,
                )

                csv_table = build_csv_row(
                    filename=uploaded_file.name,
                    gps_result=gps_result,
                    address_result=address_result,
                    final_result=final_result,
                )

            st.divider()

            status = final_result.get(
                "final_status",
            )

            if status == "Accepted":
                st.success(
                    "House number successfully identified and accepted."
                )

            elif status == "Rejected":
                st.error(
                    "No valid house number belonging to the property was identified."
                )

            else:
                st.warning(
                    "The image requires manual review."
                )

            metric_1, metric_2, metric_3 = st.columns(
                3,
            )

            with metric_1:
                st.metric(
                    "House Number",
                    final_result.get(
                        "house_number",
                    ) or "Not Available",
                )

            with metric_2:
                st.metric(
                    "Confidence",
                    f"{final_result.get('confidence', 0.0) * 100:.2f}%",
                )

            with metric_3:
                st.metric(
                    "Final Status",
                    final_result.get(
                        "status_with_reason",
                        "Unknown",
                    ),
                )

            st.subheader(
                "Property Details"
            )

            st.dataframe(
                details_table,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Field": st.column_config.TextColumn(
                        "Field",
                        width="medium",
                    ),
                    "Value": st.column_config.TextColumn(
                        "Value",
                        width="large",
                    ),
                },
            )

            csv_bytes = csv_table.to_csv(
                index=False,
            ).encode(
                "utf-8",
            )

            json_payload = {
                "file_name": uploaded_file.name,
                "gps": gps_result,
                "address": address_result,
                "vision": vision_result,
                "house_number_validation": final_result,
            }

            download_column_1, download_column_2 = st.columns(
                2,
            )

            with download_column_1:
                st.download_button(
                    label="Download CSV",
                    data=csv_bytes,
                    file_name="property_address_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with download_column_2:
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(
                        json_payload,
                        indent=2,
                        ensure_ascii=False,
                    ),
                    file_name="property_address_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if geocoding_warning:
                st.warning(
                    geocoding_warning,
                )

            with st.expander(
                "View Technical Details"
            ):
                st.write(
                    "**Cloud Vision full OCR text**"
                )

                st.code(
                    vision_result.get(
                        "full_text",
                    ) or "No text detected"
                )

                st.write(
                    "**Numeric OCR candidates**"
                )

                st.json(
                    vision_result.get(
                        "numeric_candidates",
                        [],
                    )
                )

                st.write(
                    "**Detected objects**"
                )

                st.json(
                    vision_result.get(
                        "detected_objects",
                        [],
                    )
                )

                st.write(
                    "**Gemini validation result**"
                )

                st.json(
                    final_result,
                )

    except Exception as exc:
        st.error(
            f"Processing failed: {exc}"
        )

        with st.expander(
            "Show Error Details"
        ):
            st.exception(
                exc,
            )


st.divider()

st.caption(
    "Testing application: low-confidence outputs are sent for manual review "
    "instead of being guessed."
)