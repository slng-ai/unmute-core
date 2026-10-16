"""How code says what the place tools return: ratings, prices, today's hours, addresses.

The model never words a price or an opening time. These functions do, from the tool's own
fields, so nothing is said that the tool did not return.
"""

from datetime import date

from nosh.places import Place, PlaceInfo

PRICES = {
    "PRICE_LEVEL_FREE": "free to get in",
    "PRICE_LEVEL_INEXPENSIVE": "budget-friendly",
    "PRICE_LEVEL_MODERATE": "mid-range",
    "PRICE_LEVEL_EXPENSIVE": "on the pricier side",
    "PRICE_LEVEL_VERY_EXPENSIVE": "quite upscale, fine-dining territory",
}
"""Google's price buckets in words. A bucket not listed here is not said."""
MANY_REVIEWS = 1000


def rating(value: float | None) -> str:
    """Say a rating as "a 4.5 out of five", or nothing when there is none."""
    return f"a {value:g} out of five" if value is not None else ""


def reviews(count: int | None) -> str:
    """Say "lots of reviews" for a well-known spot, the count otherwise, or nothing."""
    if not count:
        return ""
    return "lots of reviews" if count >= MANY_REVIEWS else f"{count} reviews"


def address(text: str) -> str:
    """Say an address as directions: street and number, then the area, without the country.

    A numeric postcode is dropped too: "Carrer de la Unió, 17, Ciutat Vella, Barcelona".
    """
    parts = text.split(", ")[:-1] or [text]
    kept = [
        " ".join(w for w in part.split() if not (w.isdigit() and len(w) >= 4)) for part in parts
    ]
    return ", ".join(part for part in kept if part)


def today(lines: list[str] | None, day: date) -> str:
    """Say today's hours from the weekday lines ("Monday: 9:00 AM to 1:00 AM", with a dash), or nothing."""
    line = next((line for line in lines or [] if line.startswith(f"{day:%A}:")), "")
    if not line:
        return ""
    hours = " ".join(line.split(":", 1)[1].replace("\u2013", " to ").split())
    if hours.lower() == "closed":
        return "They're closed today."
    if "open 24 hours" in hours.lower():
        return "They're open all day today."
    return f"Today they're open {hours}."


def kind(place: Place | PlaceInfo) -> str:
    """Say what a place is, like "a tapas restaurant", or nothing."""
    label = (place.primary_type_display_name or "").lower()
    return f"{'an' if label[:1] in 'aeiou' else 'a'} {label}" if label else ""


def short(place: Place) -> str:
    """Say one search result in a breath: name, kind, rating, reviews. Never price or hours."""
    said = [kind(place), rating(place.rating) and f"with {rating(place.rating)}"]
    if (many := reviews(place.user_rating_count)) and place.rating is not None:
        said.append(f"from {many}")
    return ", ".join([place.name, " ".join(s for s in said if s)]).rstrip(", ")


def listed(places: list[Place]) -> str:
    """Say up to three results: "A; B; and C"."""
    said = [short(place) for place in places[:3]]
    return said[0] if len(said) == 1 else f"{'; '.join(said[:-1])}; and {said[-1]}"


def details(info: PlaceInfo, day: date) -> tuple[str, str, str]:
    """Return what code says about one place, with the price and today's hours in words.

    Returns:
        The spoken lines, the price in words, and today's hours in words.
    """
    price = PRICES.get(info.price_level or "", "")
    hours = today(info.opening_hours, day)
    about = ", and ".join(s for s in (rating(info.rating), price) if s)
    have = " and ".join(
        s for s, there in (("their number", info.phone), ("a website", info.website)) if there
    )
    spoken = " ".join(
        s
        for s in (
            f"{info.name} is on {address(info.address)}.",
            f"It's {about}." if about else "",
            hours,
            f"I've got {have} too, if you'd like them." if have else "",
            "Want to hear about another one, or is this the one?",
        )
        if s
    )
    return spoken, price, hours
