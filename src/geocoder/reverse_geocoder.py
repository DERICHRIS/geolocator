from __future__ import annotations

import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


GOOGLE_GEOCODING_URL = (
    "https://maps.googleapis.com/maps/api/geocode/json"
)


class ReverseGeocodingError(Exception):
    """Raised when a reverse-geocoding request fails."""


def _get_component(
    address_components: list[dict[str, Any]],
    component_type: str,
    *,
    use_short_name: bool = False,
) -> str | None:
    """
    Find one address component returned by Google.

    Examples of component types:
    - street_number
    - route
    - locality
    - administrative_area_level_1
    - postal_code
    - country
    """

    for component in address_components:
        component_types = component.get("types", [])

        if component_type in component_types:
            key = "short_name" if use_short_name else "long_name"
            return component.get(key)

    return None


def _get_city(
    address_components: list[dict[str, Any]],
) -> str | None:
    """
    Google does not always use 'locality' for the city.

    Depending on the location, it may return:
    - locality
    - postal_town
    - administrative_area_level_2
    - sublocality
    """

    city_component_types = [
        "locality",
        "postal_town",
        "administrative_area_level_2",
        "sublocality_level_1",
        "sublocality",
    ]

    for component_type in city_component_types:
        value = _get_component(
            address_components,
            component_type,
        )

        if value:
            return value

    return None


def reverse_geocode(
    latitude: float,
    longitude: float,
    api_key: str | None = None,
    timeout_seconds: int = 15,
) -> dict[str, Any]:
    """
    Convert GPS coordinates into structured address information.

    This function does not treat the Google Maps house number as
    visually verified. The visible house number will later be
    determined through OCR and AI Vision.
    """

    if not -90 <= latitude <= 90:
        raise ValueError(
            "Latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:
        raise ValueError(
            "Longitude must be between -180 and 180."
        )

    resolved_api_key = (
        api_key or os.getenv("GOOGLE_MAPS_API_KEY")
    )

    if not resolved_api_key:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY is missing. "
            "Add it to the .env file."
        )

    parameters = {
        "latlng": f"{latitude},{longitude}",
        "key": resolved_api_key,
    }

    try:
        response = requests.get(
            GOOGLE_GEOCODING_URL,
            params=parameters,
            timeout=timeout_seconds,
        )

        response.raise_for_status()

    except requests.Timeout as error:
        raise ReverseGeocodingError(
            "Google Maps request timed out."
        ) from error

    except requests.RequestException as error:
        raise ReverseGeocodingError(
            f"Google Maps request failed: {error}"
        ) from error

    response_data = response.json()
    status = response_data.get("status", "UNKNOWN_ERROR")

    if status == "ZERO_RESULTS":
        return {
            "latitude": latitude,
            "longitude": longitude,
            "formatted_address": None,
            "map_house_number": None,
            "street": None,
            "neighborhood": None,
            "city": None,
            "district": None,
            "state": None,
            "state_code": None,
            "postal_code": None,
            "country": None,
            "country_code": None,
            "place_id": None,
            "location_type": None,
            "geocoding_status": "Not Available",
            "error": None,
        }

    if status != "OK":
        error_message = response_data.get(
            "error_message",
            "No error message returned.",
        )

        raise ReverseGeocodingError(
            f"Google Geocoding API returned "
            f"{status}: {error_message}"
        )

    results = response_data.get("results", [])

    if not results:
        raise ReverseGeocodingError(
            "Google returned OK but no address results."
        )

    # Google normally places the most precise result first.
    best_result = results[0]

    address_components = best_result.get(
        "address_components",
        [],
    )

    geometry = best_result.get("geometry", {})

    return {
        "latitude": latitude,
        "longitude": longitude,
        "formatted_address": best_result.get(
            "formatted_address"
        ),

        # This is only a map-derived candidate.
        # OCR/Vision will later verify the visible house number.
        "map_house_number": _get_component(
            address_components,
            "street_number",
        ),

        "street": _get_component(
            address_components,
            "route",
        ),

        "neighborhood": (
            _get_component(
                address_components,
                "neighborhood",
            )
            or _get_component(
                address_components,
                "sublocality_level_2",
            )
            or _get_component(
                address_components,
                "sublocality_level_1",
            )
        ),

        "city": _get_city(address_components),

        "district": _get_component(
            address_components,
            "administrative_area_level_2",
        ),

        "state": _get_component(
            address_components,
            "administrative_area_level_1",
        ),

        "state_code": _get_component(
            address_components,
            "administrative_area_level_1",
            use_short_name=True,
        ),

        "postal_code": _get_component(
            address_components,
            "postal_code",
        ),

        "country": _get_component(
            address_components,
            "country",
        ),

        "country_code": _get_component(
            address_components,
            "country",
            use_short_name=True,
        ),

        "place_id": best_result.get("place_id"),

        "location_type": geometry.get(
            "location_type"
        ),

        "geocoding_status": "Available",
        "error": None,
    }