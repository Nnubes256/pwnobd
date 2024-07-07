from __future__ import annotations
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.completion import FuzzyCompleter, NestedCompleter, Completer
from prompt_toolkit.document import Document
import copy
import heapq
from argparse import ArgumentParser
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Context
    from datetime import timedelta
    from argparse import Namespace
    from typing import Any, Dict, Union, Iterable, TypeVar, Set

    T = TypeVar("T")
    CommandDict = Dict[str, Union["Command", "Subcategory", "CommandDict"]]

COMMANDS = {}

_get_command_path_cache = {}


# https://stackoverflow.com/a/49947100
def _get_command_path(mydict, keyspec):
    global _get_command_path_cache
    if keyspec == "" or keyspec == []:
        return mydict, mydict, ""
    try:
        spec = _get_command_path_cache[keyspec]
    except KeyError:
        spec = tuple(keyspec.split("."))
        _get_command_path_cache[keyspec] = spec

    parent = mydict
    last_key = ""
    for key in spec:
        try:
            parent = mydict
            last_key = key
            mydict = mydict[key]
        except KeyError:
            mydict[key] = {}
    return mydict, parent, last_key


def command(cls: "Command" = None):
    def _wrapped(cls: "Command"):
        subcategory, _, _ = _get_command_path(COMMANDS, cls.PATH)
        names = copy.deepcopy(cls.NAMES)
        if isinstance(names, str):
            names = [names]
        for name in names:
            subcategory[name] = cls()
        return cls

    if cls is None:
        return _wrapped

    return _wrapped(cls)


def subcategory(cls: "Subcategory" = None):
    def _wrapped(cls: "Subcategory"):
        _, parent, key = _get_command_path(COMMANDS, cls.PATH)
        assert isinstance(parent[key], dict)
        parent[key] = cls(parent[key])
        return cls

    if cls is None:
        return _wrapped

    return _wrapped(cls)


class CommandContext:
    def __init__(self, ctx: Context, parser: ArgumentParser, unknown_args: list[str]):
        self.ctx = ctx
        self.parser = parser
        self.unknown_arguments = unknown_args
        self.quit = False

    def manager(self, ty: type[T]) -> T:
        return self.ctx.manager(ty)


class Command:
    HELP: str = ""
    PATH: str | Iterable[str] = ""
    NAMES: str | Iterable[str] = []

    def customize(
        self, ctx: Context, parser: ArgumentParser
    ) -> Union[Any, Set[str], None, Completer]:
        pass

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        pass


class Subcategory:
    HELP = ""
    PATH: str | Iterable[str] = ""

    def __init__(self, subcommands: CommandDict):
        self.subcommands = subcommands

    def __getitem__(self, *args, **kwargs):
        return self.subcommands.__getitem__(*args, **kwargs)

    def __setitem__(self, *args, **kwargs):
        return self.subcommands.__setitem__(*args, **kwargs)


class MainHelpCommand(Command):
    HELP = "Show the help menu."
    NAMES = ["help"]

    def __init__(self, commands: CommandDict):
        self.commands = commands

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        print_formatted_text(HTML(f"\n<b>Commands</b>\n{'='*20}\n"))
        command_name_pad = _calculate_min_row_width(self.commands.keys())
        for name, command in self.commands.items():
            if isinstance(command, dict):
                print_formatted_text(HTML("  <ansigreen>{}</ansigreen>").format(name))
            else:
                print_formatted_text(
                    HTML("  <ansigreen>{}</ansigreen>    {}").format(
                        name.ljust(command_name_pad), command.HELP
                    )
                )
        print()


class HelpCommand(Command):
    HELP = "Show the help menu."
    NAMES = ["help"]

    def __init__(self, commands: CommandDict, current_chain: list[str]):
        self.chain = current_chain[1:]
        self.commands = commands

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        full_parent_command = " ".join(self.chain)
        print_formatted_text(
            HTML("<u>Usage:</u> <b>{}</b> [subcommand] ...\n").format(
                full_parent_command
            )
        )
        print_formatted_text(HTML(f"\n<b>Commands</b>\n{'='*20}\n"))
        command_name_pad = _calculate_min_row_width(self.commands.keys())
        for name, command in self.commands.items():
            if isinstance(command, dict):
                print_formatted_text(
                    HTML(
                        '  <style fg="#555555">{}</style> <ansigreen>{}</ansigreen>'
                    ).format(full_parent_command, name)
                )
            else:
                print_formatted_text(
                    HTML(
                        '  <style fg="#555555">{}</style> <ansigreen>{}</ansigreen>    {}'
                    ).format(
                        full_parent_command, name.ljust(command_name_pad), command.HELP
                    )
                )
        print()


@command
class QuitCommand(Command):
    HELP = "Quit pwnobd."
    NAMES = ["exit", "quit"]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        ctx.quit = True


def _build_prompt_recursive(
    ctx: Context,
    actions: CommandDict,
    parser: ArgumentParser,
    current_chain: list[str] = ["cmd"],
):
    completions = {}
    parsers = {"_self": parser}
    if current_chain == ["cmd"]:
        actions["help"] = MainHelpCommand(actions)
    else:
        actions["help"] = HelpCommand(actions, current_chain)
    subparser = parser.add_subparsers(dest=".".join(current_chain))
    for key, value in actions.items():
        if isinstance(value, dict):
            value = Subcategory(value)
        if isinstance(value, Subcategory):
            child_parser = subparser.add_parser(
                key, prog=" ".join(current_chain[1:]), help=value.HELP, add_help=False
            )
            child_parser.exit = lambda _1, _2: None
            chain = current_chain.copy()
            chain.append(key)
            completions[key], parsers[key] = _build_prompt_recursive(
                ctx, value.subcommands, child_parser, current_chain=chain
            )
        elif isinstance(value, Command):
            child_parser = subparser.add_parser(
                key,
                prog=f"{' '.join(current_chain[1:])} {key}",
                help=value.HELP,
                add_help=False,
            )
            child_parser.exit = lambda _1, _2: None
            completions[key] = value.customize(ctx, child_parser)
            parsers[key] = child_parser
    return completions, parsers


def build_prompt(ctx: Context):
    parser = ArgumentParser(prog="", add_help=False)
    parser.exit = lambda _1, _2: None
    # side effect: COMMANDS is modified with the "help" commands
    completions, parsers = _build_prompt_recursive(ctx, COMMANDS, parser)
    return (
        FuzzyCompleter(NestedCompleter.from_nested_dict(completions)),
        parsers["_self"],
        parsers,
    )


def retrieve_command(
    parsed_args: Namespace,
    command_branch=COMMANDS,
    command_chain="cmd",
    subparsers_branch={},
) -> tuple[Command, ArgumentParser]:
    if parsed_args[command_chain] is None:
        return command_branch["help"], subparsers_branch["_self"]

    subcommand = parsed_args[command_chain]
    command_obj = command_branch[subcommand]
    parser_obj = subparsers_branch[subcommand]
    assert command_obj is not None
    command_obj_has_subcommands = isinstance(command_obj, dict) or isinstance(
        command_obj, Subcategory
    )
    parser_obj_has_subcommands = isinstance(parser_obj, dict)
    assert command_obj_has_subcommands == parser_obj_has_subcommands

    if command_obj_has_subcommands:
        return retrieve_command(
            parsed_args,
            command_obj,
            command_chain=f"{command_chain}.{subcommand}",
            subparsers_branch=parser_obj,
        )
    else:
        return command_obj, parser_obj


def _calculate_min_row_width(s: Iterable[str]):
    sl = list(s)
    return len(sl[heapq.nlargest(1, range(len(sl)), key=lambda x: len(sl[x]))[0]])


def format_timedelta(timedelta: timedelta) -> str:
    """
    Return hh:mm:ss, or mm:ss if the amount of hours is zero.
    """
    result = f"{timedelta}".split(".")[0]
    if result.startswith("0:"):
        result = result[2:]
    return result
