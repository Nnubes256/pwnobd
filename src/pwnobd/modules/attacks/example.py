from prompt_toolkit import HTML
from pwnobd.core import WorkTaskContext, attack, Attack, Device
import asyncio


@attack
class MyTestAttack(Attack):
    NAME = "example_attack"
    HELP = "Example attack"
    OPTIONS = {
        "sleep_sec": {
            "mandatory": True,
            "help": "Number of seconds to sleep",
            "type": float,
            "default": 3.0,
        },
        "important_flag": {
            "mandatory": True,
            "help": "Set this to be able to run the thing",
            "type": int,
        },
        "string_to_print": {
            "mandatory": False,
            "help": "String to print",
            "type": str,
        },
    }
    DEVICE_REQUIREMENTS: list[type] = []

    def precheck(**kwargs):
        pass

    def __init__(self, sleep_sec: float, important_flag: int, string_to_print: str):
        self.sleep_sec = sleep_sec
        self.important_flag = important_flag
        self.string_to_print = string_to_print

    async def setup(self):
        pass

    async def run(self, ctx: WorkTaskContext, devices: dict[int, Device]):
        for id, device in devices.items():
            ctx.log_info(f"({id}, {device.device_name()})")
        ctx.log_info(HTML("Sleeping for <b>{}</b> seconds").format(self.sleep_sec))
        await asyncio.sleep(self.sleep_sec)
        ctx.log_info(
            HTML("Sleeping done, here's your string: {}").format(self.string_to_print)
        )
        pass
