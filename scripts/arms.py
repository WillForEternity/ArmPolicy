"""Map SO-101 arms to serial ports by the bus adapter's USB serial number.

macOS renames /dev/cu.usbmodem* ports when cables move, but the adapter's serial number is fixed,
so roles are stored as {role: serial} in arms.json and resolved to a port at run time.

    arms.py list            show connected Feetech adapters and their assigned roles
    arms.py assign ROLE     assign the single unassigned adapter to ROLE, print its port
    arms.py port ROLE       print the port of ROLE (fails if that arm is unplugged)
"""

import json
import sys
from pathlib import Path

from serial.tools import list_ports

ROLES = ("follower", "leader")
FEETECH_VID = 0x1A86  # WCH CH34x chip on the Waveshare/Feetech bus servo adapter
MAP_FILE = Path(__file__).resolve().parent.parent / "arms.json"


def adapters() -> dict[str, str]:
    return {p.serial_number: p.device for p in list_ports.comports() if p.vid == FEETECH_VID and p.serial_number}


def load() -> dict[str, str]:
    return json.loads(MAP_FILE.read_text()) if MAP_FILE.exists() else {}


def die(msg: str) -> None:
    sys.exit(f"error: {msg}")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "assign", "port"):
        die(__doc__)
    cmd = sys.argv[1]
    connected, roles = adapters(), load()

    if cmd == "list":
        if not connected:
            print("no arm adapters connected")
        for serial, port in connected.items():
            role = next((r for r, s in roles.items() if s == serial), "unassigned")
            print(f"{port}  serial={serial}  role={role}")
        return

    role = sys.argv[2] if len(sys.argv) > 2 else ""
    if role not in ROLES:
        die(f"role must be one of {ROLES}")

    if cmd == "port":
        serial = roles.get(role)
        if serial is None:
            die(f"no adapter assigned to {role}; run calibrate.sh {role} first")
        if serial not in connected:
            die(f"{role} arm (adapter {serial}) is not plugged in")
        print(connected[serial])
        return

    # assign: keep an existing assignment if that adapter is plugged in
    if roles.get(role) in connected:
        print(connected[roles[role]])
        return
    taken = {s for r, s in roles.items() if r != role}
    free = [s for s in connected if s not in taken]
    if len(free) != 1:
        die(f"expected exactly 1 unassigned arm adapter plugged in, found {len(free)}. "
            f"Plug in only the {role} arm's USB (or unplug the other arm) and retry.")
    roles[role] = free[0]
    MAP_FILE.write_text(json.dumps(roles, indent=2) + "\n")
    print(connected[free[0]])


if __name__ == "__main__":
    main()
