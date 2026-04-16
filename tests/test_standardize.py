from biobot.data.standardize import parse_timestamp, to_float


def test_parse_french_local_timestamp_to_utc():
    parsed = parse_timestamp("20-09-2024 02:46:27")

    assert parsed.timestamp_utc == "2024-09-20T00:46:27Z"
    assert parsed.timestamp_assumption == "localized_europe_paris"


def test_parse_explicit_timezone_timestamp_to_utc():
    parsed = parse_timestamp("2026-03-26T00:03:42+0100")

    assert parsed.timestamp_utc == "2026-03-25T23:03:42Z"
    assert parsed.timestamp_assumption == "explicit_timezone"


def test_to_float_handles_nan_strings():
    assert to_float("NaN") is None
    assert to_float("nan") is None
    assert to_float("22,5") == 22.5

