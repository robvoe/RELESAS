import queue
import re
import select
import socket
import sys
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Dict, Union, Tuple, Type
import queue


@dataclass
class TripInfo:
    """
    Represents the parsed contents of a 'tripinfos' XML object.
    For more information on the single fields, refer to  https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html
    """
    # Those metrics that come from 'tripinfo' tag. Descriptions below stem from SUMO website above
    id: str
    depart: float  # The real departure time [s] (the time the vehicle was inserted into the network)
    departDelay: float  # The time [s] the vehicle had to wait before it could start his journey
    duration: float  # The time [s] the vehicle needed to accomplish the route
    waitingTime: float  # The time [s] in which the vehicle speed was below or equal 0.1 m/s
    waitingCount: float  # The number of times the vehicle speed went below or equal 0.1 m/s
    timeLoss: float  # The time loss [s] due to driving below the ideal speed.
    routeLength: float  # The length [m] of the vehicle's route

    # Those metrics that come from 'emissions' tag
    CO_abs: float  # Complete amount of CO [mg] during this trip
    CO2_abs: float  # Complete amount of CO₂ [mg] during this trip
    HC_abs: float  # Complete amount of HC [mg] during this trip
    PMx_abs: float  # Complete amount of PMₓ [mg] during this trip
    NOx_abs: float  # Complete amount of NOₓ [mg] during this trip
    fuel_abs: float  # Complete amount of fuel [ml] during this trip

    @property
    def emissions_dict(self) -> Dict[str, float]:
        """Returns all emissions of this trip as dictionary.  Important: The trailing '_abs' gets removed beforehand!"""
        return {f_name.rstrip("_abs"): self.__dict__[f_name] for f_name in _EMISSIONS_TAG_FIELDS.keys()}


_TRIPINFO_TAG_FIELDS: Dict[str, Type] = {f.name: f.type for f in fields(TripInfo) if not f.name.lower().endswith("_abs")}
_EMISSIONS_TAG_FIELDS: Dict[str, Type] = {f.name: f.type for f in fields(TripInfo) if f.name.lower().endswith("_abs")}


_PATTERN_TEMPLATE = r"\s{fieldname}\s*=\s*\"([-a-zA-Z._0-9:]+)\"[\s/]"
_TRIPINFO_EXTRACTION_PATTERNS: Dict[str, re.Pattern] = {
    n: re.compile(_PATTERN_TEMPLATE.format(fieldname=n), flags=re.IGNORECASE) for n in _TRIPINFO_TAG_FIELDS.keys()}
_EMISSIONS_EXTRACTION_PATTERNS: Dict[str, re.Pattern] = {
    n: re.compile(_PATTERN_TEMPLATE.format(fieldname=n), flags=re.IGNORECASE) for n in _EMISSIONS_TAG_FIELDS.keys()}


_PATTERN_IS_TRIPINFO_START_TAG = re.compile(f"^\s*<tripinfo .+>\s*$", flags=re.IGNORECASE)
_PATTERN_IS_EMISSIONS_TAG = re.compile(f"^\s*<emissions .+/>\s*$", flags=re.IGNORECASE)


class TripInfoRecorder:
    """
    This class provides a TCP connection to SUMO and records vehicles' trip-info data.

    Inspiration for this code was taken from:
    https://github.com/eclipse/sumo/blob/main/tests/complex/sumo/socketout/runner.py
    """
    def __init__(self, verbose: bool = False):
        self._port_was_set, self._terminate_flag, self._do_clear, self._was_cleared, self._no_more_pending = \
            threading.Event(), threading.Event(), threading.Event(), threading.Event(), threading.Event()
        self._verbose = verbose
        self.recorded_trip_info_objects: List[TripInfo] = []
        self._thread = threading.Thread(target=self._main, daemon=True)
        self._thread.start()

        # Wait until the port variable is available
        self._port_was_set.wait(timeout=10)
        assert self._port_was_set.is_set(), "The receiver thread did not start properly!"

    @property
    def port(self) -> int:
        return self._port

    def close(self) -> None:
        self._terminate_flag.set()

    def clear(self) -> None:
        self._do_clear.set()

        # Wait until all variables were properly cleared
        self._was_cleared.wait(timeout=10)
        assert self._was_cleared.is_set(), "The sub-thread didn't clear the queue!"
        self._was_cleared = threading.Event()

    def wait_for_pending_messages(self) -> None:
        """Blocks the (calling) main thread, until all pending messages were parsed."""
        self._no_more_pending = threading.Event()
        self._no_more_pending.wait(20)
        assert self._no_more_pending.is_set(), "Not all pending messages were processed by worker thread!"

    def _main(self):
        if self._verbose is True:
            sys.stdout.flush()
            sys.stdout.write(f"Thread '{threading.current_thread().name}' started")
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(("localhost", 0))  # Makes the OS select the port
        self._port = int(server_socket.getsockname()[1])
        self._port_was_set.set()
        server_socket.listen()
        connection, _ = server_socket.accept()
        connection.settimeout(1)
        _byte_buffer = b""
        while self._terminate_flag.is_set() is False:
            if self._do_clear.is_set():
                _byte_buffer = b""
                self.recorded_trip_info_objects = []
                self._do_clear = threading.Event()
                self._was_cleared.set()
            ready = select.select((connection,), (), (), 0.2)
            if ready[0]:
                _byte_buffer += connection.recv(512)
                _byte_messages = _byte_buffer.split(b"\n")
                _trip_info_objects, _deferred_byte_messages = _process_received_messages(_byte_messages[:-1])
                _byte_buffer = b"\n".join(_deferred_byte_messages + [_byte_messages[-1]])
                if self._verbose is True and len(_trip_info_objects) > 0:
                    for _msg in _byte_messages[:-1]:
                        sys.stdout.write(_msg.decode("utf-8") + "\n")
                    for _t in _trip_info_objects:
                        sys.stdout.write("     --> " + str(_t) + "\n")
                self.recorded_trip_info_objects.extend(_trip_info_objects)
            else:
                self._no_more_pending.set()
        if self._verbose is True:
            sys.stdout.write("Thread exited")


def _process_received_messages(messages: Union[List[bytes], List[str]]) -> Tuple[List[TripInfo], List[bytes]]:
    """
    Processes received messages. Returns TripInfo objects, and those messages that need completion by later messages.
    """
    trip_info_objects = []
    _str_messages = [m.strip().decode("utf-8") if isinstance(m, bytes) else m.strip() for m in messages]
    i = 0
    while i < len(messages):
        if _PATTERN_IS_TRIPINFO_START_TAG.match(_str_messages[i]) is None:  # <tripinfo ...>
            i += 1
            continue
        if i >= len(messages) - 2:  # Make sure there are at least 2 next messages. If not, keep those we have for later
            return trip_info_objects, messages[i:]
        if _PATTERN_IS_EMISSIONS_TAG.match(_str_messages[i+1]) is None or _str_messages[i+2].lower() != "</tripinfo>":
            i += 1
            continue
        _tripinfo_message, _emissions_message = _str_messages[i], _str_messages[i+1]
        try:
            trip_info_fields = {name: _type(_TRIPINFO_EXTRACTION_PATTERNS[name].search(_tripinfo_message).group(1))
                                for name, _type in _TRIPINFO_TAG_FIELDS.items()}
            emissions_fields = {name: _type(_EMISSIONS_EXTRACTION_PATTERNS[name].search(_emissions_message).group(1))
                                for name, _type in _EMISSIONS_TAG_FIELDS.items()}
            trip_info_objects += [TripInfo(**trip_info_fields, **emissions_fields)]
        except Exception as e:
            raise RuntimeError(f"An error occurred while parsing trip-info messages\n"
                               f"{_tripinfo_message}\n"
                               f"{_emissions_message}") from e
        i += 3
    return trip_info_objects, []


def from_xml_file(file_path: Path) -> List[TripInfo]:
    assert file_path.is_file(), f"File '{file_path}' does not exist or is no file!"
    with open(file_path, mode="r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    trip_info_objects, _ = _process_received_messages(lines)
    return trip_info_objects


def test_process_received_messages():
    messages = [
        """<tripinfo id="flow_ns.1" depart="1.00" departLane="n_t_1" departPos="5.10" departSpeed="15.22" departDelay="0.00" arrival="22.00" arrivalLane="t_s_1" arrivalPos="141.95" arrivalSpeed="14.46" duration="21.00" routeLength="294.90" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="0.85" rerouteNo="0" devices="tripinfo_flow_ns.1 emissions_flow_ns.1" vType="DEFAULT_VEHTYPE" speedFactor="1.09" vaporized="">""",
        """    <emissions CO_abs="310.186499" CO2_abs="55072.057896" HC_abs="2.511690" PMx_abs="0.679782" NOx_abs="19.233711" fuel_abs="17565.506308" electricity_abs="0"/>""",
        """</tripinfo>""",

        """<tripinfo id="flow_ns.0" depart="0.00" departLane="n_t_0" departPos="5.10" departSpeed="12.42" departDelay="0.00" arrival="26.00" arrivalLane="t_s_0" arrivalPos="141.95" arrivalSpeed="11.15" duration="26.00" routeLength="294.90" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="1.40" rerouteNo="0" devices="tripinfo_flow_ns.0 emissions_flow_ns.0" vType="DEFAULT_VEHTYPE" speedFactor="0.89" vaporized="">""",
        """    <emissions CO_abs="403.679991" CO2_abs="53575.026207" HC_abs="2.955078" PMx_abs="0.584128" NOx_abs="18.978321" fuel_abs="17088.158289" electricity_abs="0"/>""",
        """</tripinfo>""",

        """<tripinfo id="flow_ns.2" depart="8.00" departLane="n_t_1" departPos="5.10" departSpeed="13.77" departDelay="0.00" arrival="31.00" arrivalLane="t_s_1" arrivalPos="141.95" arrivalSpeed="13.13" duration="23.00" routeLength="294.90" waitingTime="0.00" waitingCount="0" stopTime="0.00" timeLoss="1.29" rerouteNo="0" devices="tripinfo_flow_ns.2 emissions_flow_ns.2" vType="DEFAULT_VEHTYPE" speedFactor="0.99" vaporized="">""",
        """    <emissions CO_abs="414.171060" CO2_abs="51523.072809" HC_abs="2.959159" PMx_abs="0.684452" NOx_abs="18.968221" fuel_abs="16433.541212" electricity_abs="0"/>"""
        #  --> </tripinfo> end tag intentionally missing here, to demonstrate message deferral
    ]
    trip_info_objects, _deferred_messages = _process_received_messages(messages)
    assert _deferred_messages == messages[-2:]
    assert TripInfo(id="flow_ns.1", depart=1, departDelay=0, duration=21, waitingTime=0, waitingCount=0, timeLoss=0.85,
                    CO_abs=310.186499, CO2_abs=55072.057896, HC_abs=2.511690, PMx_abs=0.679782,
                    NOx_abs=19.233711, fuel_abs=17565.506308) in trip_info_objects
    assert TripInfo(id="flow_ns.0", depart=0, departDelay=0, duration=26, waitingTime=0, waitingCount=0, timeLoss=1.40,
                    CO_abs=403.679991, CO2_abs=53575.026207, HC_abs=2.955078, PMx_abs=0.584128,
                    NOx_abs=18.978321, fuel_abs=17088.158289) in trip_info_objects


def test_grid4x4_tripinfo_xml_file():
    file_path = Path(__file__).parent.parent / "unit_test_data" / "grid4x4_tripinfo_1.xml"
    trip_info_objects = from_xml_file(file_path)
    assert len(trip_info_objects) == 319
