from scripts.convert_h16_predictions import safe_key


def test_h16_key_is_stable_and_split_specific():
    assert safe_key("test", 2, "a.wav") == safe_key("test", 2, "a.wav")
    assert safe_key("test", 2, "a.wav") != safe_key("train", 2, "a.wav")
    assert safe_key("test", 2, "a.wav") != safe_key("test", 3, "a.wav")
