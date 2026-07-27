from __future__ import annotations

import re
import unicodedata


SUBREGION_TO_PROVINCES = {
    "Northwest": {
        "DienBien", "LaiChau", "SonLa", "HoaBinh",
    },
    "Northeast": {
        "HaGiang", "CaoBang", "BacKan", "TuyenQuang", "LaoCai", "YenBai",
        "ThaiNguyen", "LangSon", "BacGiang", "PhuTho", "QuangNinh",
    },
    "RedRiverDelta": {
        "HaNoi", "HaiPhong", "VinhPhuc", "BacNinh", "HaiDuong", "HungYen",
        "ThaiBinh", "HaNam", "NamDinh", "NinhBinh",
    },
    "NorthCentral": {
        "ThanhHoa", "NgheAn", "HaTinh", "QuangBinh", "QuangTri", "ThuaThienHue",
        "Hue",
    },
    "SouthCentralCoast": {
        "DaNang", "QuangNam", "QuangNgai", "BinhDinh", "PhuYen", "KhanhHoa",
        "NinhThuan", "BinhThuan",
    },
    "CentralHighlands": {
        "KonTum", "GiaLai", "DakLak", "DacLak", "DakNong", "DacNong", "LamDong",
    },
    "Southeast": {
        "HoChiMinh", "HCM", "BinhPhuoc", "TayNinh", "BinhDuong", "DongNai",
        "BaRiaVungTau",
    },
    "MekongDelta": {
        "LongAn", "TienGiang", "BenTre", "TraVinh", "VinhLong", "DongThap",
        "AnGiang", "KienGiang", "CanTho", "HauGiang", "SocTrang", "BacLieu",
        "CaMau",
    },
}

SUBREGION_TO_REGION = {
    "Northwest": "North",
    "Northeast": "North",
    "RedRiverDelta": "North",
    "NorthCentral": "Central",
    "SouthCentralCoast": "Central",
    "CentralHighlands": "Central",
    "Southeast": "South",
    "MekongDelta": "South",
}


def normalize_place(value: object) -> str:
    text = unicodedata.normalize("NFD", str(value))
    text = "".join(character for character in text if unicodedata.category(character) != "Mn")
    return re.sub(r"[^A-Za-z0-9]", "", text)


_PROVINCE_TO_SUBREGION = {
    normalize_place(province): subregion
    for subregion, provinces in SUBREGION_TO_PROVINCES.items()
    for province in provinces
}


def province_to_subregion(province_name: object) -> str:
    key = normalize_place(province_name)
    if key not in _PROVINCE_TO_SUBREGION:
        raise KeyError(
            f"Province '{province_name}' is absent from the configured subregion taxonomy"
        )
    return _PROVINCE_TO_SUBREGION[key]


def validate_hierarchy(region: str, subregion: str) -> None:
    expected = SUBREGION_TO_REGION[subregion]
    if region != expected:
        raise ValueError(f"Inconsistent hierarchy: {region=} but {subregion=} belongs to {expected}")

