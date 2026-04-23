"""
memory/user_profile.py
----------------------
In-memory user profile store for PITAMBAR / FasalMitra.
Profiles are keyed by session_id and updated progressively as the
user reveals details through natural conversation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class UserProfile:
    session_id: str
    name: str = ""
    location: str = ""
    primary_crops: list[str] = field(default_factory=list)
    language: str = "en"
    land_acres: float = 0.0


# ---------------------------------------------------------------------------
# Keyword maps for extract_profile_hints
# ---------------------------------------------------------------------------

# Indian states, districts, and common city names relevant to agriculture.
# Extend freely — this is intentionally broad.
_LOCATION_KEYWORDS: dict[str, str] = {
    # Maharashtra
    "pune": "Pune, Maharashtra",
    "nagpur": "Nagpur, Maharashtra",
    "nashik": "Nashik, Maharashtra",
    "aurangabad": "Aurangabad, Maharashtra",
    "solapur": "Solapur, Maharashtra",
    "kolhapur": "Kolhapur, Maharashtra",
    "latur": "Latur, Maharashtra",
    "amravati": "Amravati, Maharashtra",
    "vidarbha": "Vidarbha, Maharashtra",
    "marathwada": "Marathwada, Maharashtra",
    # Punjab / Haryana
    "punjab": "Punjab",
    "haryana": "Haryana",
    "ludhiana": "Ludhiana, Punjab",
    "amritsar": "Amritsar, Punjab",
    "chandigarh": "Chandigarh",
    "hisar": "Hisar, Haryana",
    "karnal": "Karnal, Haryana",
    # UP / MP
    "lucknow": "Lucknow, UP",
    "agra": "Agra, UP",
    "bhopal": "Bhopal, MP",
    "indore": "Indore, MP",
    "jabalpur": "Jabalpur, MP",
    # Rajasthan
    "rajasthan": "Rajasthan",
    "jaipur": "Jaipur, Rajasthan",
    "jodhpur": "Jodhpur, Rajasthan",
    # Gujarat
    "gujarat": "Gujarat",
    "ahmedabad": "Ahmedabad, Gujarat",
    "surat": "Surat, Gujarat",
    # Karnataka / AP / Telangana
    "bangalore": "Bangalore, Karnataka",
    "bengaluru": "Bengaluru, Karnataka",
    "hyderabad": "Hyderabad, Telangana",
    "warangal": "Warangal, Telangana",
    "kurnool": "Kurnool, AP",
    # Bihar / Jharkhand / Odisha / WB
    "patna": "Patna, Bihar",
    "ranchi": "Ranchi, Jharkhand",
    "bhubaneswar": "Bhubaneswar, Odisha",
    "kolkata": "Kolkata, West Bengal",
}

# Crops — map common variants / transliterations to canonical names.
_CROP_KEYWORDS: dict[str, str] = {
    # Cereals
    "wheat": "wheat",
    "gehu": "wheat",
    "gehun": "wheat",
    "rice": "rice",
    "paddy": "rice",
    "dhan": "rice",
    "chawal": "rice",
    "maize": "maize",
    "corn": "maize",
    "makka": "maize",
    "sorghum": "sorghum",
    "jowar": "sorghum",
    "bajra": "pearl millet",
    "millet": "pearl millet",
    "ragi": "finger millet",
    "barley": "barley",
    "jau": "barley",
    # Pulses
    "soybean": "soybean",
    "soya": "soybean",
    "soyabean": "soybean",
    "chickpea": "chickpea",
    "chana": "chickpea",
    "gram": "chickpea",
    "lentil": "lentil",
    "masoor": "lentil",
    "pigeon pea": "pigeon pea",
    "tur": "pigeon pea",
    "arhar": "pigeon pea",
    "toor": "pigeon pea",
    "mung": "green gram",
    "moong": "green gram",
    "urad": "black gram",
    "black gram": "black gram",
    # Cash crops
    "cotton": "cotton",
    "kapas": "cotton",
    "sugarcane": "sugarcane",
    "ganna": "sugarcane",
    "tobacco": "tobacco",
    "groundnut": "groundnut",
    "peanut": "groundnut",
    "mungfali": "groundnut",
    "sunflower": "sunflower",
    "surajmukhi": "sunflower",
    "mustard": "mustard",
    "sarson": "mustard",
    "turmeric": "turmeric",
    "haldi": "turmeric",
    "ginger": "ginger",
    "adrak": "ginger",
    "onion": "onion",
    "pyaz": "onion",
    "tomato": "tomato",
    "potato": "potato",
    "aloo": "potato",
    "garlic": "garlic",
    "lahsun": "garlic",
    "chilli": "chilli",
    "mirch": "chilli",
}

# Language detection — simple but effective for Indian multilingual context.
_LANGUAGE_KEYWORDS: dict[str, str] = {
    # Hindi triggers
    "namaskar": "hi",
    "namaste": "hi",
    "kya": "hi",
    "meri": "hi",
    "mera": "hi",
    "mere": "hi",
    "fasal": "hi",
    "paani": "hi",
    "khet": "hi",
    "bhoomi": "hi",
    "zaroorat": "hi",
    "hum": "hi",
    "aap": "hi",
    "yahan": "hi",
    "kheti": "hi",
    # Marathi triggers
    "majha": "mr",
    "mazha": "mr",
    "shetat": "mr",
    "pikat": "mr",
    "pani": "mr",
}

# Land-size patterns: "5 acres", "2.5 bigha", "3 hectares", etc.
_LAND_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:acre|acres|bigha|bighas|hectare|hectares|ha\b)",
    re.IGNORECASE,
)

# Simple name extraction: "my name is X" / "I am X" / "mera naam X hai"
_NAME_PATTERN = re.compile(
    r"(?:my name is|i am|i'm|mera naam|mera naam hai|naam hai)\s+([A-Za-z]+)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_PROFILES: dict[str, UserProfile] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_profile(session_id: str) -> UserProfile:
    """Return existing profile or create a default one."""
    if session_id not in _PROFILES:
        _PROFILES[session_id] = UserProfile(session_id=session_id)
    return _PROFILES[session_id]


def update_profile(session_id: str, **kwargs: Any) -> UserProfile:
    """
    Update arbitrary profile fields by keyword argument.

    - For list fields (primary_crops) kwargs value is *merged* (no duplicates).
    - For scalar fields the value is replaced directly.

    Example
    -------
    update_profile("s1", name="Raju", location="Pune", land_acres=4.5)
    update_profile("s1", primary_crops=["wheat", "onion"])
    """
    profile = get_profile(session_id)
    for key, value in kwargs.items():
        if not hasattr(profile, key):
            raise AttributeError(f"UserProfile has no field {key!r}")
        if key == "primary_crops" and isinstance(value, list):
            existing = set(profile.primary_crops)
            for crop in value:
                existing.add(crop)
            profile.primary_crops = sorted(existing)
        else:
            setattr(profile, key, value)
    return profile


def extract_profile_hints(query: str, current_profile: UserProfile) -> UserProfile:
    """
    Parse *query* for implicit profile signals and mutate *current_profile*
    in-place (also persists to the store).

    Detected signals
    ----------------
    * Location  — city / state keyword match
    * Crops     — crop keyword match (adds, never removes)
    * Language  — Hindi / Marathi keyword triggers
    * Land size — numeric pattern before acre/bigha/hectare
    * Name      — "my name is X" patterns
    """
    q_lower = query.lower()
    tokens = re.findall(r"[a-z]+", q_lower)

    # --- Name ---
    name_match = _NAME_PATTERN.search(query)
    if name_match and not current_profile.name:
        current_profile.name = name_match.group(1).capitalize()

    # --- Location (multi-word first, then single-word) ---
    if not current_profile.location:
        # Check two-word combos (e.g. "pigeon pea" style locations aren't here,
        # but "vidarbha region" style mentions need this)
        for kw, canonical in sorted(_LOCATION_KEYWORDS.items(), key=lambda x: -len(x[0])):
            if kw in q_lower:
                current_profile.location = canonical
                break

    # --- Crops ---
    # Multi-word crops first (e.g. "pigeon pea")
    new_crops: set[str] = set(current_profile.primary_crops)
    for kw, canonical in sorted(_CROP_KEYWORDS.items(), key=lambda x: -len(x[0])):
        if kw in q_lower and canonical not in new_crops:
            new_crops.add(canonical)
    current_profile.primary_crops = sorted(new_crops)

    # --- Language ---
    if current_profile.language == "en":
        for kw, lang in _LANGUAGE_KEYWORDS.items():
            if kw in tokens:
                current_profile.language = lang
                break

    # --- Land size (take the first numeric match) ---
    if current_profile.land_acres == 0.0:
        land_match = _LAND_PATTERN.search(query)
        if land_match:
            raw_value = float(land_match.group(1))
            # Normalise bigha → acres (1 bigha ≈ 0.62 acres, varies by region)
            if re.search(r"bigha", land_match.group(0), re.IGNORECASE):
                raw_value *= 0.62
            # Normalise hectare → acres (1 ha = 2.471 acres)
            elif re.search(r"hectare|ha\b", land_match.group(0), re.IGNORECASE):
                raw_value *= 2.471
            current_profile.land_acres = round(raw_value, 2)

    # Persist changes back to store
    _PROFILES[current_profile.session_id] = current_profile
    return current_profile


# ---------------------------------------------------------------------------
# __main__ smoke-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("user_profile.py — smoke tests")
    print("=" * 60)

    SID = "profile-test-001"

    # 1. get_profile — default creation
    p = get_profile(SID)
    assert p.session_id == SID
    assert p.name == ""
    assert p.primary_crops == []
    assert p.language == "en"
    assert p.land_acres == 0.0
    print(f"[PASS] get_profile (default) → {p}")

    # 2. get_profile — idempotent
    p2 = get_profile(SID)
    assert p is p2
    print("[PASS] get_profile is idempotent")

    # 3. update_profile — scalar fields
    p = update_profile(SID, name="Raju", location="Nagpur, Maharashtra", land_acres=5.0)
    assert p.name == "Raju"
    assert p.location == "Nagpur, Maharashtra"
    assert p.land_acres == 5.0
    print(f"[PASS] update_profile (scalars) → name={p.name}, location={p.location}, acres={p.land_acres}")

    # 4. update_profile — crops merge (no duplicates)
    update_profile(SID, primary_crops=["wheat", "onion"])
    update_profile(SID, primary_crops=["wheat", "cotton"])  # wheat already present
    assert _PROFILES[SID].primary_crops == ["cotton", "onion", "wheat"]
    print(f"[PASS] update_profile (crops merge) → {_PROFILES[SID].primary_crops}")

    # 5. update_profile — bad field
    try:
        update_profile(SID, alien_field="oops")
        print("[FAIL] Should have raised AttributeError")
    except AttributeError as exc:
        print(f"[PASS] AttributeError on bad field → {exc}")

    # 6. extract_profile_hints — location detection
    SID2 = "profile-test-002"
    p2 = get_profile(SID2)
    extract_profile_hints("I farm near Pune and grow wheat", p2)
    assert p2.location == "Pune, Maharashtra", f"Got: {p2.location}"
    assert "wheat" in p2.primary_crops
    print(f"[PASS] extract_profile_hints (location+crop) → location={p2.location}, crops={p2.primary_crops}")

    # 7. extract_profile_hints — Hindi language + crop
    SID3 = "profile-test-003"
    p3 = get_profile(SID3)
    extract_profile_hints("Meri fasal mein gehu aur sarson hai", p3)
    assert p3.language == "hi", f"Got: {p3.language}"
    assert "wheat" in p3.primary_crops
    assert "mustard" in p3.primary_crops
    print(f"[PASS] extract_profile_hints (Hindi) → lang={p3.language}, crops={p3.primary_crops}")

    # 8. extract_profile_hints — land size in acres
    SID4 = "profile-test-004"
    p4 = get_profile(SID4)
    extract_profile_hints("I have 3.5 acres of land", p4)
    assert p4.land_acres == 3.5, f"Got: {p4.land_acres}"
    print(f"[PASS] extract_profile_hints (acres) → land_acres={p4.land_acres}")

    # 9. extract_profile_hints — land size in hectares
    SID5 = "profile-test-005"
    p5 = get_profile(SID5)
    extract_profile_hints("My farm is 2 hectares", p5)
    assert p5.land_acres == 4.94, f"Got: {p5.land_acres}"
    print(f"[PASS] extract_profile_hints (hectares → acres) → land_acres={p5.land_acres}")

    # 10. extract_profile_hints — name extraction
    SID6 = "profile-test-006"
    p6 = get_profile(SID6)
    extract_profile_hints("My name is Arjun and I grow cotton in Vidarbha", p6)
    assert p6.name == "Arjun", f"Got: {p6.name}"
    assert p6.location == "Vidarbha, Maharashtra", f"Got: {p6.location}"
    assert "cotton" in p6.primary_crops
    print(f"[PASS] extract_profile_hints (name+location+crop) → name={p6.name}, loc={p6.location}, crops={p6.primary_crops}")

    # 11. extract_profile_hints — multi-word crop "pigeon pea"
    SID7 = "profile-test-007"
    p7 = get_profile(SID7)
    extract_profile_hints("I grow arhar and chana in my khet", p7)
    assert "pigeon pea" in p7.primary_crops, f"Got: {p7.primary_crops}"
    assert "chickpea" in p7.primary_crops
    assert p7.language == "hi"
    print(f"[PASS] extract_profile_hints (pigeon pea / Hindi) → crops={p7.primary_crops}, lang={p7.language}")

    # 12. location not overwritten once set
    extract_profile_hints("I also sometimes farm near Nashik", p2)  # p2 already has Pune
    assert p2.location == "Pune, Maharashtra"
    print("[PASS] Location not overwritten once set")

    print("\nAll user_profile tests passed ✓")