from __future__ import annotations
from pwnobd.cli import command, subcategory, Command, Subcategory
from pwnobd import core
from pwnobd.core import Context, DeviceManager
from prompt_toolkit import HTML, print_formatted_text

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cli import CommandContext, Namespace
    from ..core import ScannedDevice
    from argparse import ArgumentParser


@subcategory
class ReconSubcategory(Subcategory):
    HELP = "Commands related to OBD device reconnaissance."
    PATH = "recon"


@command
class DeviceScanCommand(Command):
    HELP = "Find nearby devices using multiple scanners."
    PATH = "recon"
    NAMES = ["scan"]

    def customize(self, ctx: Context, parser: ArgumentParser):
        return super().customize(ctx, parser)

    async def run(self, ctx: CommandContext, args: Namespace) -> str | None:
        device_manager = ctx.manager(DeviceManager)
        print_formatted_text(
            HTML("Scanning using <b>{}</b> scanners...").format(len(core.LEAF_SCANNERS))
        )
        total_results = await device_manager.launch_leaf_scanners()

        if len(total_results) == 0:
            print("Found no devices :(")
        else:
            print_formatted_text(
                HTML("Found <b>{}</b> devices!\n").format(len(total_results))
            )

        for i, result in enumerate(total_results):
            print_formatted_text(
                HTML(
                    "  <b><ansibrightred>{}</ansibrightred></b>\t<ansigreen>{}</ansigreen>\t\t\t{}"
                ).format(i, result.name(), result.device_type())
            )

        print("")
        if len(total_results) != 0:
            print_formatted_text(
                HTML(
                    '<b>Hint:</b> you may now connect to one of these devices like so:\n\n  connect --scanned <ansibrightred>0</ansibrightred>\t<style fg="#555555">(connects to {})</style>\n'
                ).format(total_results[0].name())
            )

        device_manager.register_last_scan_results(total_results)
