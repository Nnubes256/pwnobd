import importlib.metadata
import importlib.util

from . import core, cli
from .exceptions import *

import os
import asyncio
import importlib
import traceback
from pathlib import Path
from os.path import dirname, abspath, isfile
from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText

MODULE_ROOT_DIR = dirname(abspath(__file__))
LOGO = """   ___  __    __    __  ___  ___    ___ 
  / _ \/ / /\ \ \/\ \ \/___\/ __\  /   \\
 / /_)/\ \/  \/ /  \/ //  //__\// / /\ /
/ ___/  \  /\  / /\  / \_// \/  \/ /_// 
\/       \/  \/\_\ \/\___/\_____/___,'"""
LOGO_GRADIENT = ["#ff0000", "#cc0000", "#aa0000", "#880000", "#660000"]


def get_modules_py_files():
    cwd = Path(MODULE_ROOT_DIR, "modules/")
    return [
        f".modules.{'.'.join(Path(x).relative_to(cwd).parts).removesuffix('.py')}"
        for x in cwd.rglob("*.py")
        if isfile(x)
    ]


def dynamic_import(module_name):
    try:
        importlib.import_module(module_name, "pwnobd")
        print(f"OK {module_name}")
    except Exception:
        traceback.print_exc()
        print(f"KO {module_name}")


def load_all_modules():
    from concurrent.futures.thread import ThreadPoolExecutor

    modules = get_modules_py_files()
    with ThreadPoolExecutor(max_workers=4) as executor:
        for name in modules:
            executor.submit(dynamic_import, name)


def show_logo_and_help():
    print()
    logo = LOGO.split("\n")
    logo_formatted = []
    for line, color in zip(logo, LOGO_GRADIENT):
        logo_formatted.append((color, line + "\n"))
    print_formatted_text(
        FormattedText(logo_formatted),
        HTML("v{}\n").format(importlib.metadata.version("pwnobd")),
    )
    print()
    print_formatted_text(
        HTML("Loaded {} drivers, {} attacks.").format(
            len(core.DRIVERS), len(core.ATTACKS)
        )
    )
    print_formatted_text(HTML('Run "<b><u>help</u></b>" for command list.\n'))


async def run():
    print("Loading...")
    load_all_modules()

    show_logo_and_help()

    ctx = core.Context()
    device_manager = ctx.manager(core.DeviceManager)
    task_manager = ctx.manager(core.TaskManager)

    completer, parser, subparsers = cli.build_prompt(ctx)

    home_dir = os.environ["HOME"]

    bottom_bar_text = None
    tui = PromptSession(
        auto_suggest=AutoSuggestFromHistory(),
        history=FileHistory(f"{home_dir}/.pwnobd_history"),
        completer=completer,
        style=Style.from_dict(
            {
                # User input (default text).
                #'':          '#ff0066',
                # Prompt.
                "pwnobd": "#ff0000",
                "device": "#00ff00",
                "attack": "#cc33cc",
                "other_chars": "",
                "bottom-toolbar": "",
            }
        ),
    )

    while True:
        prompt_message = [
            ("class:pwnobd", "pwnobd"),
        ]
        if (
            device_manager.current_connection is not None
            or task_manager.current_attack_id is not None
        ):
            prompt_message.append(("class:other_chars", " ("))
            if device_manager.current_connection is not None:
                prompt_message.append(
                    (
                        "class:device",
                        device_manager.connections[
                            device_manager.current_connection
                        ].device_name(),
                    )
                )
                if task_manager.current_attack_id is not None:
                    prompt_message.append(("", ", "))
            if task_manager.current_attack_id is not None:
                prompt_message.append(
                    (
                        "class:attack",
                        task_manager.get_attack_shorthand(
                            task_manager.current_attack_id
                        ),
                    )
                )
            prompt_message.append(("class:other_chars", ")"))
        prompt_message.append(("class:other_chars", "> "))

        # UNUSED
        if task_manager.current_attack_id is not None:
            if device_manager.current_connection is None:
                bottom_bar_text = HTML(
                    "<ansired>Missing a device to launch the attack on!</ansired>"
                )
            else:
                device = device_manager.connections[device_manager.current_connection]
                opts = task_manager.get_attack_options(task_manager.current_attack_id)
                try:
                    opts.validate_all([device])
                    # bottom_bar_text = HTML("<ansigreen><b>Attack configured and good to go!</b></ansigreen>")
                    bottom_bar_text = None
                except ValidationDevicesInvalidException as e:
                    bottom_bar_text = HTML(
                        "<ansired>Device <b>{}</b> is not compatible! (missing: {})</ansired>"
                    ).format(
                        device.device_name(), ",".join(e.validation_errors[device])
                    )
                except ValidationMandatoryFailedException as e:
                    bottom_bar_text = HTML(
                        '<ansired>Need a value for parameter "{}"</ansired>'
                    ).format(e.prop_name)
                except PwnObdException as e:
                    bottom_bar_text = str(e)

        with patch_stdout():
            user_input = await tui.prompt_async(prompt_message)
        user_input: list[str] = user_input.split()
        # print(f"{user_input}")
        parsed, unknown_args = parser.parse_known_args(user_input)
        # print(f"{vars(parsed)}")
        # print(f"{unknown}")
        # print(subparsers)
        command, subparser = cli.retrieve_command(
            vars(parsed), subparsers_branch=subparsers
        )
        # print(f"{command}")
        ctx = cli.CommandContext(ctx, subparser, unknown_args)

        try:
            await command.run(ctx, parsed)
        except asyncio.CancelledError:
            print_formatted_text(HTML("<b><ansired>Cancelled</ansired></b>"))
        except PwnObdException as e:
            print_formatted_text(HTML("<b><ansired>Error:</ansired> {}</b>").format(e))
        except KeyboardInterrupt:
            pass
        except Exception:
            traceback.print_exc()

        if ctx.quit:
            break


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
