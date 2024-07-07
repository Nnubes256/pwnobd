from __future__ import annotations
from contextlib import contextmanager, asynccontextmanager

from .exceptions import *
from .cli import command, Command, _calculate_min_row_width
from enum import Enum
from argparse import ArgumentParser
from pathlib import Path
from os.path import dirname, abspath
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.completion import (
    Completer,
    CompleteEvent,
    Completion,
    WordCompleter,
    PathCompleter,
    NestedCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
import typeguard
import sys
import asyncio
from asyncio import CancelledError
import traceback
import inspect
import copy
import re
from abc import ABC, abstractmethod

import typing
from typing import TYPE_CHECKING, Generic, TypeVar, TypedDict, Union, Optional

if TYPE_CHECKING:
    from .cli import CommandContext, Namespace
    from concurrent.futures import Future
    from typing import Any, Iterable, AsyncGenerator, Literal

    T = TypeVar("T")


class Context:
    def __init__(self):
        self._currently_resolving: set[type] = set()
        self.managers: dict[type, Any] = {}

    def manager(self, ty: type[T]) -> T:
        existing_manager = self.managers.get(ty)
        if existing_manager is not None:
            assert isinstance(existing_manager, ty)
            return existing_manager

        if ty in self._currently_resolving:
            raise Exception(
                f"Circular dependency detected! {self._currently_resolving}"
            )

        self._currently_resolving.add(ty)
        new_instance = ty(self)
        self.managers[ty] = new_instance
        return new_instance


class Device:
    def __init__(self):
        self._task_lock = asyncio.Lock()
        self._task_locked_by = None

    def device_name(self) -> str:
        return "(none)"

    def device_type(self) -> str:
        return "(not specified)"

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def handle(self):
        pass

    def log_info(self, *msg: Any):
        print_formatted_text(
            HTML(
                '[<style fg="#00ff00">+</style>] (<style fg="#009000">{}</style>)'
            ).format(self.device_name()),
            *msg,
        )

    def log_debug(self, *msg: Any):
        # pass
        print_formatted_text(
            HTML(
                '[<style fg="#0000ff">*</style>] (<style fg="#009000">{}</style>)'
            ).format(self.device_name()),
            *msg,
        )

    def log_error(self, *msg: Any):
        print_formatted_text(
            HTML(
                '[<style fg="#ff0000">!</style>] (<style fg="#009000">{}</style>)'
            ).format(self.device_name()),
            *msg,
        )


class SendCan(ABC):
    @abstractmethod
    async def send_can(self, arbitration_id: int, data: bytes):
        pass

    async def send_can_multiple(self, packets: list[tuple[int, bytes]]):
        responses = []
        for arbitration_id, data in packets:
            response = await self.send_can(arbitration_id, data)
            responses.append(response)
        return responses


class RecvCan:
    def on_receive_can(self, data: bytes):
        if hasattr(self, "_recvcan_collectors"):
            for collector in self._recvcan_collectors:
                collector._on_receive_can(data)

    def register_can_collector(self, collector: "DeviceManager"):
        if not hasattr(self, "_recvcan_collectors"):
            self._recvcan_collectors: list[DeviceManager] = []
        self._recvcan_collectors.append(collector)


class Reset(ABC):
    @abstractmethod
    async def reset(self):
        pass


class ScannedDevice:
    def name(self) -> str:
        return "DUMMY"

    def device_type(self) -> str:
        return "DUMMY"

    def create_device(self, *args, **kwargs) -> Device:
        return None


TScanResult = TypeVar("TScanResult")
TScannedDevice = TypeVar("TScannedDevice", bound=ScannedDevice)


class Scanner(Generic[TScanResult]):
    def __init__(self):
        pass

    async def scan(self, ctx: ScanContext) -> TScanResult:
        pass


class LeafScanner(Scanner[TScannedDevice]):
    def __init__(self):
        pass

    async def scan(self, ctx: ScanContext) -> list[TScannedDevice]:
        pass


class ScannerProxy(Scanner[TScanResult]):
    def __init__(self, ctx: ScanContext, fut: Future = None):
        self.ctx = ctx
        self.fut = fut

    async def scan(self, _ctx) -> list[TScanResult]:
        return copy.deepcopy(await self.fut)


TScanner = TypeVar("TScanner", bound=Scanner)


class ScanContext:
    def __init__(self):
        self.scans: dict[type, Future] = {}

    def get_scanner(self, ty: type[TScanner]) -> TScanner:
        task = self.scans.get(ty)
        if task is not None:
            # XXX: missing deadlock detection
            return ScannerProxy(self, task)

        task = asyncio.create_task(SCANNERS[ty].scan(self))
        self.scans[ty] = task
        return ScannerProxy(self, task)


SCANNERS: dict[type, Scanner] = {}
LEAF_SCANNERS: dict[str, LeafScanner] = {}


def scanner(name=""):
    def _wrapper(cls):
        nonlocal name
        assert issubclass(cls, Scanner)
        if issubclass(cls, LeafScanner):
            assert type(name) == str and name != ""
            LEAF_SCANNERS[name] = cls()
        else:
            SCANNERS[cls] = cls()
        return cls

    return _wrapper


DRIVERS: dict[str, type[Device]] = {}


def driver(name):
    def _wrapper(cls):
        nonlocal name
        assert issubclass(cls, Device)
        DRIVERS[name] = cls
        return cls

    return _wrapper


class WorkTaskStatus(Enum):
    PENDING = 10
    RUNNING = 20
    BLOCKED_ON_DEVICE = 21
    COMPLETED = 30
    FAILED = 40
    CANCELLED = 50


TASK_STATUS_COLORS = {
    WorkTaskStatus.PENDING: "#666666",
    WorkTaskStatus.RUNNING: "#4444ee",
    WorkTaskStatus.BLOCKED_ON_DEVICE: "#ee44ee",
    WorkTaskStatus.COMPLETED: "#22ff22",
    WorkTaskStatus.FAILED: "#ff0000",
    WorkTaskStatus.CANCELLED: "#996666",
}


class WorkTaskable:
    NAME: str = "(dummy)"

    async def run(self, ctx: WorkTaskContext):
        pass


class WorkTask:
    def __init__(self, taskable: WorkTaskable, task_id: int, ctx: Context):
        self.taskable: WorkTaskable = taskable
        self.task_id = task_id
        self.ctx = ctx
        self.async_task = None
        super().__setattr__("status", WorkTaskStatus.PENDING)

    def _create_context(self):
        return WorkTaskContext(self)

    async def _run_task(self):
        self.status = WorkTaskStatus.RUNNING
        try:
            await self.taskable.run(self._create_context())
            if self.status == WorkTaskStatus.RUNNING:
                self.status = WorkTaskStatus.COMPLETED
        except CancelledError:
            self.status = WorkTaskStatus.CANCELLED
            raise
        except:
            self.status = WorkTaskStatus.FAILED
            self._log_error(
                HTML("Task had an exception, now printing...\n{}").format(
                    traceback.format_exc()
                )
            )

    def start_task(self):
        assert self.status == WorkTaskStatus.PENDING
        self.async_task = asyncio.create_task(self._run_task())

    def cancel_task(self):
        self.async_task.cancel()

    def _log_info(self, *msg: Any):
        print_formatted_text(
            HTML('|Task {}| [<style fg="#00ff00">+</style>]').format(self.task_id), *msg
        )

    def _log_debug(self, *msg: Any):
        print_formatted_text(
            HTML('|Task {}| [<style fg="#0000ff">*</style>]').format(self.task_id), *msg
        )

    def _log_error(self, *msg: Any):
        print_formatted_text(
            HTML('|Task {}| [<style fg="#ff0000">!</style>]').format(self.task_id), *msg
        )

    def __setattr__(self, name: str, value: Any):
        if name == "status":
            self._log_info(
                "Task status changed:",
                FormattedText(
                    [
                        (TASK_STATUS_COLORS[self.status], self.status.name),
                        ("", " -> "),
                        (TASK_STATUS_COLORS[value], value.name),
                    ]
                ),
            )
        super().__setattr__(name, value)


class WorkTaskContext:
    def __init__(self, task_obj: WorkTask):
        self.task = task_obj

    @contextmanager
    async def device(self, connection_id: int, display_to_user=True):
        device_manager = self.task.ctx.manager(DeviceManager)
        device = device_manager.connections.get(connection_id)
        if device is None:
            raise CurrentConnectionUndefinedException
        if device._task_lock.locked():
            self.log_info(
                HTML("Now waiting for task {} to free up connection {} ({})").format(
                    device._task_locked_by, connection_id, device.device_name()
                )
            )
        if display_to_user:
            self.task.status = WorkTaskStatus.BLOCKED_ON_DEVICE
        async with device._task_lock:
            device._task_locked_by = self.task.task_id
            self.task.status = WorkTaskStatus.RUNNING
            self.log_info(
                HTML(
                    "Connection {} ({}) now locked", connection_id, device.device_name()
                )
            )
            yield device
            self.log_info(
                HTML(
                    "Finished work with connection {} ({}) now locked, unlocking...",
                    connection_id,
                    device.device_name(),
                )
            )
            device._task_locked_by = None

    def log_info(self, *msg: Any):
        return self.task._log_info(*msg)

    def log_debug(self, *msg: Any):
        return self.task._log_debug(*msg)

    def log_error(self, *msg: Any):
        return self.task._log_error(*msg)


TOptionTypeConcreteSingle = Union[
    int,
    float,
    str,
]
TOptionTypeConcrete = Union[TOptionTypeConcreteSingle, list[TOptionTypeConcreteSingle]]
TOptionType = TypeVar("TOptionType", bound=TOptionTypeConcrete)


class AttackOptions(TypedDict, Generic[TOptionType]):
    type: type[TOptionType]
    hint: Literal["path"] | None
    help: str
    mandatory: bool
    default: Optional[TOptionType]


class AttackOptionsStore:
    def __init__(self, attack_ty: type[Attack]):
        self.attack = attack_ty
        self.schema = attack_ty.OPTIONS
        self._store: dict[str, Any] = {}
        self.reset_to_defaults()

    def reset_to_defaults(self):
        for name, schema_item in self.schema.items():
            self._store.__setitem__(name, schema_item.get("default"))

    def validate_all(self, devices: Optional[Iterable[Device]] = None):
        if devices is not None:
            self.validate_devices(list(map(type, devices)))
        for name, schema_item in self.schema.items():
            value = self._store.get(name)
            if schema_item["mandatory"] and value is None:
                raise ValidationMandatoryFailedException(name)
            if value is None:
                continue
            self._validate_value_against_type(name, value, schema_item["type"])

    def validate_all_and_build_attack(
        self, devices: Optional[Iterable[Device]] = None
    ) -> Attack:
        self.validate_all(devices)
        return self.attack(**self._store)

    def _validate_value_against_type(self, key: str, value: Any, ty: type[TOptionType]):
        typeguard.check_type(value, ty)
        current_opts_with_new = {**self._store, key: value}
        self.attack.precheck(**current_opts_with_new)

    def is_list(self, key: str):
        schema_item = self.schema.get(key)
        if schema_item is None:
            return False
        return typing.get_origin(schema_item["type"]) == list

    def expected_type(self, key: str):
        schema_item = self.schema.get(key)
        if schema_item is None:
            return False
        if self.is_list(key):
            return typing.get_args(schema_item["type"])[0]
        else:
            return typing.get_origin(schema_item["type"])

    def get_invalid_devices(self, devices: Iterable[type[Device]]):
        missing_requirements: dict[Device, list[type]] = {}
        requirements_set = set(self.attack.DEVICE_REQUIREMENTS)
        for device in devices:
            for requirement in requirements_set:
                if not issubclass(device, requirement):
                    reqs = missing_requirements.setdefault(device, [])
                    reqs.append(requirement)
                    missing_requirements[device] = reqs
        return missing_requirements

    def validate_devices(self, devices: Iterable[type[Device]]):
        validation_errors = self.get_invalid_devices(devices)
        if len(validation_errors) > 0:
            raise ValidationDevicesInvalidException(validation_errors)

    def _smart_cast_single(self, ty: type, value: str):
        if ty == int:
            return int(value)
        elif ty == float:
            return float(value)
        elif ty == str:
            return str(value)
        else:
            raise PwnObdException(f'Unsupported casting target "{ty.__name__}"')

    def _smart_cast(self, schema: AttackOptions, value: str):
        if typing.get_origin(schema["type"]) == list:
            assert type(value) == list
            return [
                self._smart_cast_single(typing.get_args(schema["type"])[0], x)
                for x in value
            ]
        else:
            return self._smart_cast_single(schema["type"], value)

    def __getitem__(self, key: str, /):
        return self._store.__getitem__(key)

    def get(self, key: str, /):
        return self._store.get(key)

    def __setitem__(self, key: str, value: Any, /):
        item_schema = self.schema.get(key)
        if item_schema is None:
            raise KeyError(f"Option {key} not found in schema!")
        casted_value = self._smart_cast(item_schema, value)
        self._validate_value_against_type(key, casted_value, item_schema["type"])
        return self._store.__setitem__(key, casted_value)

    def __delitem__(self, key: str, /):
        self._store.__delitem__(key)


class AttackTaskContext(WorkTaskContext):
    def __init__(self, task_obj: AttackTask, prelocked_conns: dict[int, Device]):
        super().__init__(task_obj)
        self.prelocked_conns = prelocked_conns

    @contextmanager
    async def device(self, connection_id: int, display_to_user=True):
        prelocked_conn = self.prelocked_conns.get(connection_id)
        if prelocked_conn is not None:
            yield prelocked_conn
        else:
            async with super().device(connection_id, display_to_user) as device:
                yield device


class AttackTask(WorkTask):
    def __init__(
        self, taskable: Attack, task_id: int, ctx: Context, connection_ids: set[int]
    ):
        super().__init__(taskable, task_id, ctx)
        self.connection_ids = connection_ids

    def _create_context(self, prelocked_conns: dict[int, Device]):
        return AttackTaskContext(self, prelocked_conns)

    @asynccontextmanager
    async def _prelock_devices(self):
        async def _lock(device: Device):
            await device._task_lock.acquire()
            device._task_locked_by = self.task_id

        def _unlock(device: Device):
            assert device._task_locked_by == self.task_id
            device._task_locked_by = None
            device._task_lock.release()

        device_manager = self.ctx.manager(DeviceManager)
        devices = [device_manager.connections.get(id) for id in self.connection_ids]
        if any(map(lambda x: x is None, devices)):
            raise CurrentConnectionUndefinedException
        try:
            tasks = [_lock(device) for device in devices]
            self.status = WorkTaskStatus.BLOCKED_ON_DEVICE
            await asyncio.gather(*tasks)
            yield dict(zip(self.connection_ids, devices))
        finally:
            tasks = []
            for device in devices:
                _unlock(device)

    async def _run_task(self):
        self._log_debug(f"Prelocking devices: {self.connection_ids}")
        async with self._prelock_devices() as devices:
            self._log_debug(f"Prelocked devices: {self.connection_ids}")
            self.status = WorkTaskStatus.RUNNING
            try:
                await self.taskable.run(self._create_context(devices), devices)
                if self.status == WorkTaskStatus.RUNNING:
                    self.status = WorkTaskStatus.COMPLETED
            except CancelledError:
                self.status = WorkTaskStatus.CANCELLED
                raise
            except:
                self.status = WorkTaskStatus.FAILED
                self._log_error(
                    HTML(
                        "<b><ansired>Task had an exception, now printing...</ansired></b>\n{}"
                    ).format(traceback.format_exc())
                )

    def start_task(self):
        assert self.status == WorkTaskStatus.PENDING
        self.async_task = asyncio.create_task(self._run_task())


class Attack(WorkTaskable):
    NAME: str = ""
    HELP: str = ""
    OPTIONS: dict[str, AttackOptions] = {}
    DEVICE_REQUIREMENTS: list[type] = []

    def precheck(**kwargs):
        pass

    def __init__(self, **kwargs):
        pass

    async def setup(self):
        pass

    async def run(self, ctx: WorkTaskContext, devices: dict[int, Device]):
        pass


ATTACKS: dict[str, type[Attack]] = {}
ATTACKS_BUILTIN_ROOT_DIR = Path(dirname(abspath(__file__)), "modules")


def attack(cls=None):
    def _wrapper(cls: type):
        assert issubclass(cls, Attack)
        attack_filepath = Path(sys.modules[cls.__module__].__file__)
        rel_filepath = attack_filepath.relative_to(ATTACKS_BUILTIN_ROOT_DIR)
        if cls.NAME is None or cls.NAME == "":
            raise Exception(
                f"Attack name empty or None! Please make sure to set {cls.__name__}.NAME"
            )
        identifier = f'{".".join(rel_filepath.parts).removesuffix(".py")}.{cls.NAME}'
        if identifier in ATTACKS:
            raise Exception(f'Attack name "{identifier}" was already registered!')
        ATTACKS[identifier] = cls
        return cls

    if cls is None:
        return _wrapper

    return _wrapper(cls)


class DeviceManager:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.drivers = DRIVERS
        driver_info, _driver_argparsers = self._read_driver_info()
        self.driver_info = driver_info
        self._driver_argparsers: dict[str, ArgumentParser] = _driver_argparsers

        self.next_connection_id = 1
        self.connections: dict[int, Device] = {}
        self.last_scan_results: list["ScannedDevice"] = []
        self.current_connection = None

    def _read_driver_info(self):
        driver_info = {}
        driver_argparsers = {}
        for name, driver in self.drivers.items():
            constructor_signature = inspect.signature(driver.__init__)
            params_info = {}
            for i, (param_name, parameter) in enumerate(
                constructor_signature.parameters.items()
            ):
                if i == 0:  # 'self'
                    continue

                param_info = {}
                param_info["name"] = param_name
                if parameter.default != inspect._empty:
                    param_info["default"] = parameter.default
                if parameter.annotation != inspect._empty and inspect.isclass(
                    parameter.annotation
                ):
                    param_info["type"] = parameter.annotation
                params_info[param_name] = param_info

            constructor_docs = driver.__init__.__doc__
            if constructor_docs is not None:
                for doc in constructor_docs.strip().split("\n"):
                    try_param = doc.strip().split(":")
                    if len(try_param) == 2 and try_param[0].strip() in param_info:
                        params_info[try_param[0]]["help"] = try_param[1].strip()

            parser = ArgumentParser(prog="", add_help=False)
            parser.exit = lambda _1, _2: None

            for param_name, param_info in params_info.items():
                param_type = None
                if "type" in param_info:
                    param_type = param_info["type"]
                param_required = "default" not in param_info
                parser.add_argument(
                    f"--{param_name}", type=param_type, required=param_required
                )

            driver_info[name] = {"params": params_info}
            driver_argparsers[name] = parser
        return driver_info, driver_argparsers

    def register_last_scan_results(self, results: list["ScannedDevice"]):
        self.last_scan_results = results

    async def launch_leaf_scanners(
        self, scanners: Iterable[LeafScanner] = LEAF_SCANNERS.values()
    ) -> list[ScannedDevice]:
        tasks = []
        scan_ctx = ScanContext()
        for scanner in scanners:
            task = asyncio.create_task(scanner.scan(scan_ctx))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        return [device for result_list in results for device in result_list]

    async def connect_to_scan_result(self, idx: int):
        if idx < 0 or idx >= len(self.last_scan_results):
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no such scan result index")
            )
            return

        scan_result = self.last_scan_results[idx]
        device = scan_result.create_device()
        return await self._setup_device(device)

    async def connect_using_driver(self, driver_name: str, **kwargs):
        driver = self.drivers.get(driver_name)
        if driver is None:
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> driver '{}' does not exist"
                ).format(driver_name)
            )
            return
        device = driver(**kwargs)
        return await self._setup_device(device)

    async def _setup_device(self, device: Device):
        if isinstance(device, RecvCan):
            self.register_can_receiver(device)

        try:
            await device.connect()
        except Exception as e:
            traceback.print_exc()
            print_formatted_text(
                HTML("<ansired>Couldn't connect to</ansired> {}").format(
                    device.device_name()
                )
            )
            return None

        print_formatted_text(
            HTML("<ansigreen>Connected to</ansigreen> {}").format(device.device_name())
        )

        device_idx = self.next_connection_id
        self.next_connection_id += 1
        self.connections[device_idx] = device

        async def _handle_device_connection():
            print_formatted_text(
                HTML("<b><ansigreen>Created connection</ansigreen></b> {} ({})").format(
                    device_idx, device.device_name()
                )
            )
            try:
                await device.handle()
            except Exception:
                traceback.print_exc()
            finally:
                print_formatted_text(
                    HTML(
                        "<b><ansired>Destroyed connection</ansired></b> {} ({})"
                    ).format(device_idx, device.device_name())
                )
                if self.current_connection == device_idx:
                    self.current_connection = None
                if device_idx in self.connections:
                    del self.connections[device_idx]

        asyncio.create_task(_handle_device_connection())
        return device_idx

    def register_can_receiver(self, receiver: RecvCan):
        receiver.register_can_collector(self)

    def _on_receive_raw(self, data: bytes):
        # print(f"RAW: {data}")
        pass

    def _on_receive_can(self, data: bytes):
        print(f"CAN: {data.hex()}")


class TaskManager:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.work_tasks: dict[int, WorkTask] = {}
        self.next_task_id = 1
        self.attack_options: dict[str, AttackOptionsStore] = {}
        self.attacks: dict[str, type[Attack]] = ATTACKS
        self.current_attack_id = None

    def start_task(self, taskable: WorkTaskable):
        task_id = self.next_task_id
        self.next_task_id += 1
        task = WorkTask(taskable, task_id, self.ctx)
        self.work_tasks[task_id] = task
        task.start_task()
        print_formatted_text(
            HTML("<b><ansigreen>Started task</ansigreen></b> {} ({})").format(task_id)
        )

    def change_current_attack(self, attack_id: str):
        attack_ty = self.attacks.get(attack_id)
        if attack_ty is None:
            raise AttackNotFoundException()
        self.current_attack_id = attack_id

    def get_attack_shorthand(self, attack_id: str):
        return attack_id.removeprefix("attacks.")  # WIP

    def get_attack_options(self, attack_id: str):
        attack_ty = self.attacks.get(attack_id)
        if attack_ty is None:
            raise AttackNotFoundException()
        opts = self.attack_options.get(attack_id)
        if opts is None:
            opts = AttackOptionsStore(attack_ty)
            self.attack_options[attack_id] = opts
        return opts

    async def launch_attack(self, options: AttackOptionsStore):
        device_manager = self.ctx.manager(DeviceManager)
        if device_manager.current_connection is None:
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> no connection currently selected"
                )
            )
            print_formatted_text(
                HTML(
                    "<b>Hint:</b> when you connect to a device, it will be selected automatically."
                )
            )
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> check the "<b>list</b>" of connections, then run "<b>device <u>connection_id</u></b>" to select it.\n'
                )
            )
            return None
        devices = [device_manager.connections[device_manager.current_connection]]
        try:
            attack_instance = options.validate_all_and_build_attack(devices=devices)
            await attack_instance.setup()
        except PwnObdException as e:
            print_formatted_text(HTML("<b><ansired>Error:</ansired></b> {}").format(e))
            return None
        task_id = self.next_task_id
        self.next_task_id += 1
        task = AttackTask(
            attack_instance, task_id, self.ctx, [device_manager.current_connection]
        )
        self.work_tasks[task_id] = task
        task.start_task()
        return task_id


@command
class DeviceConnectCommand(Command):
    HELP = "Connect to a device."
    NAMES = ["connect"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)
        parser.add_argument("--scanned", type=str)
        parser.add_argument("--driver", type=str),

    async def run(self, ctx: CommandContext, kwargs: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
        scanned_device: str | None = kwargs.scanned
        driver_name: str | None = kwargs.driver

        if driver_name is not None:
            driver_argparser = device_manager._driver_argparsers.get(driver_name)
            if driver_argparser is None:
                print_formatted_text(
                    HTML(
                        "<b><ansired>Error:</ansired></b> driver '{}' does not exist"
                    ).format(driver_name)
                )
                return
            assert driver_argparser is not None
            try:
                failed = False

                def set_failed(_a, _b):
                    nonlocal failed
                    failed = True

                parser = copy.deepcopy(driver_argparser)
                parser.prog = f"--driver {driver_name}"
                parser.exit = set_failed
                kwargs = vars(parser.parse_args(ctx.unknown_arguments))
                kwargs = {k: v for k, v in iter(kwargs.items()) if v is not None}
            except Exception as e:
                print(e)
                driver_argparser.print_help()
                return

            device_id = await device_manager.connect_using_driver(driver_name, **kwargs)
            if device_id is not None:
                device_manager.current_connection = device_id
            return

        if scanned_device is None or len(scanned_device.strip()) == 0:
            ctx.parser.print_help()
            return

        if driver_name is not None:
            print_formatted_text(
                HTML(
                    f"<b><ansired>Error:</ansired></b> you cannot specify a custom driver to execute against a scan result"
                )
            )
            return

        m = re.fullmatch("^([0-9])$", scanned_device)
        if m is None:
            print_formatted_text(
                HTML(f"<b><ansired>Error:</ansired></b> invalid scan result index")
            )
            return
        scan_id = int(m.groups()[0], 10)
        if scan_id < 0 or scan_id >= len(device_manager.last_scan_results):
            print_formatted_text(
                HTML(f"<b><ansired>Error:</ansired></b> no such scan result index")
            )
            return

        device_id = await device_manager.connect_to_scan_result(scan_id)
        if device_id is not None:
            device_manager.current_connection = device_id


class ConnectionCompleter(Completer):
    def __init__(self, ctx: Context) -> None:
        super().__init__()
        self.ctx = ctx

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        This should be a generator that yields :class:`.Completion` instances.

        If the generation of completions is something expensive (that takes a
        lot of time), consider wrapping this `Completer` class in a
        `ThreadedCompleter`. In that case, the completer algorithm runs in a
        background thread and completions will be displayed as soon as they
        arrive.

        :param document: :class:`~prompt_toolkit.document.Document` instance.
        :param complete_event: :class:`.CompleteEvent` instance.
        """
        device_manager = self.ctx.manager(DeviceManager)
        for id, connection in device_manager.connections.items():
            yield Completion(
                text=f"{id}",
                display=f"#{id} ({connection.device_type()}, {connection.device_name()})",
            )


@command
class DeviceDisconnectCommand(Command):
    HELP = "Disconnect from the current device."
    NAMES = ["disconnect"]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
        device_idx = device_manager.current_connection
        if device_idx is None:
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> no connection currently selected"
                )
            )
            print_formatted_text(
                HTML(
                    "<b>Hint:</b> when you connect to a device, it will be selected automatically."
                )
            )
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> check the "<b>list</b>" of connections, then run "<b>device <u>connection_id</u></b>" to select it.\n'
                )
            )
            return

        device = device_manager.connections.get(device_idx)
        if device is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no such connection")
            )
            return

        if not isinstance(device, Reset):
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> device does not support resetting."
                )
            )
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> <style fg="#555555"><i>Try plugging it off and on again?</i></style>'
                )
            )
            return

        print_formatted_text(
            HTML("Disconnecting from <b>{}</b>...").format(device.device_name())
        )
        try:
            await device.disconnect()
            print_formatted_text(
                HTML("Disconnect OK for <b>{}</b> !").format(device.device_name())
            )
        finally:
            device_manager.current_connection = None


@command
class DeviceUseCommand(Command):
    HELP = "Select the connection in which to execute further commands."
    NAMES = ["device"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)
        parser.add_argument("device", type=int)
        return ConnectionCompleter(ctx)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_idx = args.device
        device_manager = ctx.manager(DeviceManager)
        if device_manager.connections.get(device_idx) is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no such connection")
            )
            return
        device_manager.current_connection = args.device


@command
class DeviceResetCommand(Command):
    HELP = "Reset the device behind the given connection."
    NAMES = ["reset"]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
        device_idx = device_manager.current_connection
        if device_idx is None:
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> no connection currently selected"
                )
            )
            print_formatted_text(
                HTML(
                    "<b>Hint:</b> when you connect to a device, it will be selected automatically."
                )
            )
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> check the "<b>list</b>" of connections, then run "<b>device <u>connection_id</u></b>" to select it.\n'
                )
            )
            return

        device = device_manager.connections.get(device_idx)
        if device is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no such connection")
            )
            return

        if not isinstance(device, Reset):
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> device does not support resetting."
                )
            )
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> <style fg="#555555"><i>Try plugging it off and on again?</i></style>'
                )
            )
            return

        print_formatted_text(
            HTML("Resetting <b>{}</b>...").format(device.device_name())
        )
        await device.reset()
        print_formatted_text(
            HTML("Reset OK for <b>{}</b> !").format(device.device_name())
        )


@command
class ConnectionListCommand(Command):
    HELP = "List all open connections."
    NAMES = ["list"]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
        print_formatted_text(HTML(f"\n<b>Connections</b>\n{'='*20}\n"))
        for i, device in device_manager.connections.items():
            print_formatted_text(
                HTML("  <ansigreen>{}</ansigreen>\t\t{}\t\t{}").format(
                    i, device.device_name(), device.device_type()
                )
            )
        print()


@command
class AttackUseCommand(Command):
    HELP = "Select the attack to use."
    NAMES = ["use"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)
        parser.add_argument("attack", type=str)
        task_manager = ctx.manager(TaskManager)
        return WordCompleter(lambda: set(task_manager.attacks.keys()))

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        attack_id = args.attack
        task_manager = ctx.manager(TaskManager)
        task_manager.change_current_attack(attack_id)


@command
class TaskListCommand(Command):
    HELP = "List the currently-running and completed tasks."
    NAMES = ["tasks"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        task_manager = ctx.manager(TaskManager)
        status_column_pad = _calculate_min_row_width(WorkTaskStatus._member_names_)
        for id, task in task_manager.work_tasks.items():
            print_formatted_text(
                HTML("  {}\t").format(id),
                FormattedText(
                    [
                        (
                            TASK_STATUS_COLORS[task.status],
                            task.status.name.ljust(status_column_pad),
                        )
                    ]
                ),
                HTML("\t<b>{}</b>").format(task.taskable.NAME),
            )


class TaskCompleter(Completer):
    def __init__(self, ctx: Context, filter=lambda task: True) -> None:
        super().__init__()
        self.ctx = ctx
        self.filter = filter

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        This should be a generator that yields :class:`.Completion` instances.

        If the generation of completions is something expensive (that takes a
        lot of time), consider wrapping this `Completer` class in a
        `ThreadedCompleter`. In that case, the completer algorithm runs in a
        background thread and completions will be displayed as soon as they
        arrive.

        :param document: :class:`~prompt_toolkit.document.Document` instance.
        :param complete_event: :class:`.CompleteEvent` instance.
        """
        task_manager = self.ctx.manager(TaskManager)
        for id, task in task_manager.work_tasks.items():
            if not self.filter(task):
                continue
            yield Completion(
                text=f"{id}",
                display=FormattedText(
                    [
                        ("", f"#{id} ({task.taskable.NAME}, "),
                        (TASK_STATUS_COLORS[task.status], task.status.name),
                        ("", ")"),
                    ]
                ),
            )


@command
class TaskCancelCommand(Command):
    HELP = "Cancel the provided task."
    NAMES = ["cancel"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)
        parser.add_argument("task", type=int)
        return TaskCompleter(ctx, self.task_cancellable)

    def task_cancellable(self, task: WorkTask):
        return task.status in [
            WorkTaskStatus.PENDING,
            WorkTaskStatus.BLOCKED_ON_DEVICE,
            WorkTaskStatus.RUNNING,
        ]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        task_id = args.task
        task_manager = ctx.manager(TaskManager)
        task = task_manager.work_tasks.get(task_id)
        if task is None:
            print_formatted_text(HTML("<b><ansired>Error:</ansired></b> no such task"))
            return
        if not self.task_cancellable(task):
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> task has state"),
                FormattedText([(TASK_STATUS_COLORS[task.status], task.status.name)]),
                HTML(", it cannot be cancelled."),
            )
            return
        # TODO move some validation here
        task.cancel_task()


@command
class AttackListCommand(Command):
    HELP = "List the attacks available."
    NAMES = ["attacks"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        task_manager = ctx.manager(TaskManager)
        id_column_pad = _calculate_min_row_width(task_manager.attacks.keys())
        help_column_pad = _calculate_min_row_width(
            map(lambda atk: atk.HELP, task_manager.attacks.values())
        )
        for id, attack_ty in task_manager.attacks.items():
            print_formatted_text(
                HTML("  {}\t{}").format(
                    str(id).ljust(id_column_pad), attack_ty.HELP.ljust(help_column_pad)
                )
            )


class AttackOptionsCompleter(Completer):
    def __init__(self, ctx: Context) -> None:
        super().__init__()
        self.ctx = ctx
        self.cached_attack_completer: dict[str, Completer] = {}

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        """
        This should be a generator that yields :class:`.Completion` instances.

        If the generation of completions is something expensive (that takes a
        lot of time), consider wrapping this `Completer` class in a
        `ThreadedCompleter`. In that case, the completer algorithm runs in a
        background thread and completions will be displayed as soon as they
        arrive.

        :param document: :class:`~prompt_toolkit.document.Document` instance.
        :param complete_event: :class:`.CompleteEvent` instance.
        """
        task_manager = self.ctx.manager(TaskManager)
        selected_attack_id = task_manager.current_attack_id
        if selected_attack_id is None:
            return
        cached_completer = self.cached_attack_completer.get(selected_attack_id)
        if cached_completer is not None:
            yield from cached_completer.get_completions(document, complete_event)
            return
        selected_attack = task_manager.attacks.get(selected_attack_id)
        if selected_attack is None:
            return
        processed_options = {}
        for name, descriptor in selected_attack.OPTIONS.items():
            processed_options[name] = None
            if descriptor["type"] == str and descriptor.get("hint") == "path":
                processed_options[name] = PathCompleter()
        finished_completer = NestedCompleter.from_nested_dict(processed_options)
        self.cached_attack_completer[selected_attack_id] = finished_completer
        yield from finished_completer.get_completions(document, complete_event)


@command
class AttackGetOptionsCommand(Command):
    HELP = "List the options set for the current attack."
    NAMES = ["options"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        task_manager = ctx.manager(TaskManager)
        if task_manager.current_attack_id is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no attack has been selected")
            )
            return
        opts = task_manager.get_attack_options(task_manager.current_attack_id)
        name_column_pad = _calculate_min_row_width(opts.schema.keys())
        for name, schema_item in opts.schema.items():
            value = opts.get(name)
            ty = schema_item["type"]
            help = schema_item["help"]
            print_formatted_text(
                HTML("  {}\t{}\t{}\t{}").format(
                    name.ljust(name_column_pad), ty.__name__, value, help
                )
            )


@command
class AttackSetOptionsCommand(Command):
    HELP = "Set the value of a given option within the current attack."
    NAMES = ["set"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)
        parser.add_argument("key", type=str)
        parser.add_argument("values", nargs="*")
        return AttackOptionsCompleter(ctx)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        key = args.key
        value = args.values
        task_manager = ctx.manager(TaskManager)
        if task_manager.current_attack_id is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no attack has been selected.")
            )
            return
        opts = task_manager.get_attack_options(task_manager.current_attack_id)
        if len(value) == 0:
            del opts[key]
            print_formatted_text(HTML('<b>Cleared</b> parameter "{}".').format(key))
            return
        if not opts.is_list(key):
            if len(value) > 1:
                print_formatted_text(
                    HTML(
                        '<b><ansired>Error:</ansired></b> parameter "{}" only accepts a single value.'
                    ).format(key)
                )
                return
            value = value[0]
        opts[key] = value
        print_formatted_text(HTML("<b>{}</b> = {}").format(key, opts[key]))


@command
class AttackSetOptionsCommand(Command):
    HELP = "Launch the selected attack."
    NAMES = ["launch"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        task_manager = ctx.manager(TaskManager)
        if task_manager.current_attack_id is None:
            print_formatted_text(
                HTML("<b><ansired>Error:</ansired></b> no attack has been selected.")
            )
            return
        opts = task_manager.get_attack_options(task_manager.current_attack_id)
        task_id = await task_manager.launch_attack(opts)
        if task_id is not None:
            print_formatted_text(
                HTML("<b><ansigreen>Created task</ansigreen></b> {}").format(task_id)
            )
