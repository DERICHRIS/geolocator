from pprint import pprint

from src.geocoder.reverse_geocoder import (
    ReverseGeocodingError,
    reverse_geocode,
)


LATITUDE = 10.8072194
LONGITUDE = 78.6770306


def main() -> None:
    print("Reverse geocoding coordinates...")
    print(f"Latitude : {LATITUDE}")
    print(f"Longitude: {LONGITUDE}")
    print("-" * 60)

    try:
        address = reverse_geocode(
            latitude=LATITUDE,
            longitude=LONGITUDE,
        )

    except (
        ValueError,
        ReverseGeocodingError,
    ) as error:
        print(f"Reverse geocoding failed: {error}")
        return

    pprint(address, sort_dicts=False)


if __name__ == "__main__":
    main()