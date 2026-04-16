from biobot.data.clean_aggregate import normalize_iot_sensor_id


def test_normalize_iot_sensor_id_keeps_14_digits():
    assert normalize_iot_sensor_id("20240313101500") == "20240313101500"


def test_normalize_iot_sensor_id_removes_date_separators():
    assert normalize_iot_sensor_id("2024-09-20 08:30:00") == "20240920083000"


def test_normalize_iot_sensor_id_rejects_incomplete_ids():
    assert normalize_iot_sensor_id("2024-09-20") == "unknown_sensor"
    assert normalize_iot_sensor_id("") == "unknown_sensor"

