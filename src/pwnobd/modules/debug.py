from argparse import Namespace
import asyncio

from typing import Iterable

from pwnobd.cli import Command, command, CommandContext, subcategory, Subcategory


@subcategory
class DebugSubcategory(Subcategory):
    HELP = "Commands for debugging."
    PATH = "debug"


@command
class DebugAsyncTasksCommand(Command):
    HELP = "List the asyncio tasks and their stacks."
    PATH: str | Iterable[str] = "debug"
    NAMES = ["reactor"]

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        for task in asyncio.all_tasks():
            task.print_stack()
