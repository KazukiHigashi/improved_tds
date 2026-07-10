from __future__ import annotations

from improved_tds.tools.hardware import HardwareCommand, HardwareState


class UnconfiguredHardwareAdapter:
    """Fail-closed base for real adapters whose transport/safety data is not supplied."""

    platform_name = "unconfigured"

    def _missing(self) -> RuntimeError:
        return RuntimeError(
            f"{self.platform_name} adapter is a stub: configure transport, sensor mapping, "
            "units, limits, watchdog, and emergency release before use"
        )

    def read_state(self) -> HardwareState:
        raise self._missing()

    def write_command(self, command: HardwareCommand) -> None:
        del command
        raise self._missing()

    def emergency_stop(self) -> None:
        raise self._missing()

    def close(self) -> None:
        pass


class SprayBottleAdapterStub(UnconfiguredHardwareAdapter):
    platform_name = "spray bottle"


class RemoteControlAdapterStub(UnconfiguredHardwareAdapter):
    platform_name = "remote control"

