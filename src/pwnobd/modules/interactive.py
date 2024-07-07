from abc import abstractmethod, ABC
from argparse import ArgumentParser, Namespace

from prompt_toolkit import HTML, PromptSession, print_formatted_text
from prompt_toolkit.completion import (
    Completer,
    CompleteEvent,
    Completion,
    FuzzyWordCompleter,
    PathCompleter,
    NestedCompleter,
)
from prompt_toolkit.document import Document
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style
from typing import TYPE_CHECKING, Iterable

from pwnobd.cli import Command, CommandContext, _calculate_min_row_width, command
from pwnobd.core import Context, DeviceManager, Device, Reset
from pwnobd.exceptions import PwnObdException


class Interactive(ABC):
    @abstractmethod
    async def request_interactive(self, command: str):
        raise PwnObdException("Not implemented")


if TYPE_CHECKING:

    class _DeviceWithInteractive(Device, Interactive):
        pass


async def _stub(_ctx, _device):
    pass


async def cmd_help(ctx: CommandContext, device: "_DeviceWithInteractive"):
    command_name_pad = _calculate_min_row_width(META_COMMANDS.keys())
    for name, (_, help_text) in META_COMMANDS.items():
        print_formatted_text(
            HTML(
                '  <style fg="#555555">.</style><ansigreen>{}</ansigreen>    {}'
            ).format(name.ljust(command_name_pad), help_text)
        )
    print()


async def cmd_device_reset(ctx: CommandContext, device: "_DeviceWithInteractive"):
    if not isinstance(device, Reset):
        print_formatted_text(
            HTML("<b><ansired>Error:</ansired></b> device does not support resetting.")
        )
        print_formatted_text(
            HTML(
                '<b>Hint:</b> <style fg="#555555"><i>Try plugging it off and on again?</i></style>'
            )
        )
        return

    print_formatted_text(HTML("Resetting <b>{}</b>...").format(device.device_name()))
    await device.reset()
    print_formatted_text(HTML("Reset OK for <b>{}</b> !").format(device.device_name()))


META_COMMANDS = {
    "help": (cmd_help, "Show this help."),
    "quit": (_stub, "Return to the primary pwnobd prompt."),
    "exit": (_stub, "Return to the primary pwnobd prompt."),
    "reset": (cmd_device_reset, "Reset the device."),
}


class InteractiveShellCompleter(Completer):
    def __init__(self) -> None:
        super().__init__()
        self.meta_completer = FuzzyWordCompleter(META_COMMANDS)

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
        if document.current_line.startswith("."):
            meta_command = document.current_line.removeprefix(".")
            new_doc = Document(meta_command, document.cursor_position - 1)
            yield from self.meta_completer.get_completions(new_doc, complete_event)


@command
class InteractiveShellCommand(Command):
    HELP = "Open an interactive prompt for sending direct commands to the device."
    NAMES = ["interactive"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
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
        device = device_manager.connections[device_manager.current_connection]
        assert device is not None
        if not isinstance(device, Interactive):
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> device does not support interactive mode"
                )
            )
            print_formatted_text(
                HTML(
                    "<b>Hint:</b> for a device to support interactive mode, its driver must implement the <u>Interactive</u> interface."
                )
            )
            return
        if device._task_lock.locked():
            print_formatted_text(
                HTML(
                    "<b><ansired>Error:</ansired></b> device is currently in use by task {}."
                ).format(device._task_locked_by)
            )
            if device._task_locked_by is not None:
                print_formatted_text(
                    HTML(
                        "<b>Hint:</b> you may cancel this task with the command <b>cancel <u>{}</u></b>"
                    ).format(device._task_locked_by)
                )
            return
        async with device._task_lock:
            device._task_locked_by = "(interactive mode)"
            await self.interactive_loop(ctx, device)

    async def interactive_loop(
        self, ctx: CommandContext, device: "_DeviceWithInteractive"
    ):
        tui = PromptSession(
            style=Style.from_dict(
                {
                    # User input (default text).
                    #'':          '#ff0066',
                    # Prompt.
                    "pwnobd": "#ffcc22",
                    "device": "#00ff00",
                    "attack": "#cc33cc",
                    "other_chars": "",
                    "bottom-toolbar": "",
                }
            ),
            completer=InteractiveShellCompleter(),
        )
        while True:
            prompt_message = [
                ("class:pwnobd", device.device_type()),
                ("", " ("),
                ("class:device", device.device_name()),
                ("", ")> "),
            ]

            with patch_stdout():
                user_input: str = await tui.prompt_async(prompt_message)

            if user_input.startswith("."):
                meta_command = user_input.removeprefix(".").rstrip()
                if meta_command == "quit" or meta_command == "exit":
                    break
                meta_command_entry = META_COMMANDS.get(meta_command)
                if meta_command_entry is None:
                    print_formatted_text(
                        HTML("<b><ansired>Error:</ansired></b> command not found")
                    )
                    continue
                command_fn, _ = meta_command_entry
                await command_fn(ctx, device)
            else:
                response = await device.request_interactive(user_input)
                print(response.replace("\r", "\n"))
