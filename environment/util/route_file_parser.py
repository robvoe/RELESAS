from pathlib import Path
from typing import Dict
from xml.etree import ElementTree


def get_vehicle_departs(route_file_path: Path) -> Dict[str, float]:
    """Returns the "depart" fields of all "vehicle" nodes inside a route-XML-file."""
    assert route_file_path.is_file()
    _tree = ElementTree.parse(route_file_path)
    _root = _tree.getroot()
    assert _root.tag.lower() == "routes"

    vehicles__depart = {child.get("id"): float(child.get("depart", default=None)) for child in _root
                        if child.tag.lower() == "vehicle" and "id" in child.keys()}
    assert all(depart is not None for depart in vehicles__depart.values()), \
        f"At least one of the nodes in {route_file_path} does not provide a 'depart' attribute!"
    return vehicles__depart


def test_get_vehicle_departs():
    script_path = Path(__file__).parent
    vehicles__depart = get_vehicle_departs(script_path.parent / "unit_test_data" / "test.rou.xml")
    assert len(vehicles__depart) == 5
    assert vehicles__depart == {"0": 0, "1": 9, "2": 14, "3": 27, "4": 34}
