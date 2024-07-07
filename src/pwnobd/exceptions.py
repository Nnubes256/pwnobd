from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Device


class PwnObdException(Exception):
    pass


class CurrentConnectionUndefinedException(PwnObdException):
    pass


class ValidationMandatoryFailedException(PwnObdException):
    def __init__(self, prop_name: str) -> None:
        super().__init__(f"Missing mandatory property: {prop_name}")
        self.prop_name = prop_name


class ValidationDevicesInvalidException(PwnObdException):
    def __init__(self, validation_errors: dict["Device", list[type]]) -> None:
        s = f"One or more devices don't match with this attack's requirements"
        for device, missing_subclasses in validation_errors.items():
            s += f"\n  - Device {device.device_name()} ({device.device_type()}) does not implement: "
            s += ", ".join([cls.__name__ for cls in missing_subclasses])
        self.validation_errors = validation_errors


class AttackNotFoundException(PwnObdException):
    pass
