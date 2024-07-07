from pwnobd.core import (
    Device,
    ScanContext,
    driver,
    SendCan,
    RecvCan,
    Reset,
    scanner,
    LeafScanner,
    ScannedDevice,
)
from pwnobd.modules import bluetooth
from pwnobd.modules.bluetooth import BluetoothScanner
from prompt_toolkit import HTML

import asyncio
import binascii
from asyncio import Queue

from pwnobd.modules.interactive import Interactive


class BluetoothElm327ScanResult(ScannedDevice):
    def __init__(self, name: str, address: str):
        self.bt_name = name
        self.address = address

    def name(self) -> str:
        return f"{self.bt_name} ({self.address})"

    def device_type(self) -> str:
        return "ELM327"

    def create_device(self, *args, **kwargs):
        return BluetoothElm327Device(address=self.address, port=1, *args, **kwargs)


@scanner("elm327")
class BluetoothElm327Scanner(LeafScanner):
    async def scan(self, ctx: ScanContext):
        devices = await ctx.get_scanner(BluetoothScanner).scan(ctx)
        scan_results = []
        for _, (device, _) in devices.items():
            if device.name != "OBDII":
                continue

            scan_results.append(
                BluetoothElm327ScanResult(name=device.name, address=device.address)
            )
        return scan_results


@driver("elm327")
class BluetoothElm327Device(Device, SendCan, RecvCan, Reset, Interactive):
    def __init__(self, address: str, port: int = 1):
        """
        address: BLE address to which connect.
        port: BLE port to which connect (default 1).
        """
        super().__init__()
        self.addr = address
        self.port = port
        self.reader = None
        self.writer = None
        self.setup_completed = False
        self.dev_protocol = ""
        self.send_queue: Queue[tuple[bytes, asyncio.Future[bytes]]] = Queue()

    def device_name(self):
        return f"{self.addr}"

    def device_type(self):
        return f"ELM327 over Bluetooth Serial"

    async def connect(self):
        self.reader, self.writer = await bluetooth.bt_open_connection(
            self.addr, self.port
        )
        self.log_info(HTML("Connected successfully to <b>{}</b>!").format(self.addr))
        await self.reset()
        self.setup_completed = True

    async def disconnect(self):
        self.writer.close()
        self.setup_completed = False

    async def request_raw(self, data: bytes) -> bytes:
        if not self.setup_completed:
            return await self._request(data)

        future: asyncio.Future[bytes] = asyncio.Future()
        await self.send_queue.put((data, future))
        return await future

    async def request_raw_multiple(self, data: list[bytes]) -> list[bytes]:
        if not self.setup_completed:
            # Default (safe) implementation; do them one by one
            responses = []
            for cmd in data:
                response = await self.request_raw(cmd)
                responses.append(response)
            return responses

        futures = []
        for cmd in data:
            future: asyncio.Future[bytes] = asyncio.Future()
            futures.append(future)
            await self.send_queue.put((cmd, future))

        return await asyncio.gather(*futures)

    async def _request(self, command: bytes, existing_recv_task=None) -> bytes:
        await self.send_raw(command)
        if existing_recv_task is None:
            existing_recv_task = self._recv_raw()

        return (await existing_recv_task).removesuffix(b">")

    async def _request_multiple(self, commands: list[bytes], existing_recv_task=None):
        for command in commands:
            await self.send_raw(command)
            task = existing_recv_task
            if task is None:
                task = self._recv_raw()
            yield (await task).removesuffix(b">")
            if existing_recv_task is not None:
                existing_recv_task = None

    async def handle(self):
        assert self.reader is not None and self.writer is not None
        while not self.reader.at_eof():
            reader_task = asyncio.create_task(self._recv_raw())
            writer_queue_task = asyncio.create_task(self.send_queue.get())
            done, pending = await asyncio.wait(
                [writer_queue_task, reader_task], return_when=asyncio.FIRST_COMPLETED
            )

            if reader_task in done and writer_queue_task in pending:
                writer_queue_task.cancel()
            elif writer_queue_task in done:
                send_data, future = writer_queue_task.result()
                existing_recv_task = None
                if not reader_task.done() and not reader_task.cancelled():
                    existing_recv_task = reader_task
                response = await self._request(send_data, existing_recv_task)
                try:
                    future.set_result(response)
                except asyncio.InvalidStateError:
                    pass
                # query the queue again, make sure there aren't more pending requests
                while not self.reader.at_eof():
                    try:
                        send_data, future = self.send_queue.get_nowait()
                        # there's more stuff in the queue, so handle it
                        response = await self._request(send_data)
                        try:
                            future.set_result(response)
                        except asyncio.InvalidStateError:
                            pass
                    except asyncio.QueueEmpty:
                        # nothing else, loop back
                        break

    async def reset(self):
        await self.request_raw_multiple(
            [
                b"ATZ\r",  # reset all
                b"ATE0\r",  # echo off
                b"ATE0\r",  # (2x)
                b"ATM0\r",  # memory off
                b"ATL0\r",  # linefeeds off
                b"ATS0\r",  # spaces off
            ]
        )
        dev_description = await self.request_raw(b"AT@1\r")  # display device version
        self.log_info(f"Device description: {dev_description.decode().strip()}")
        dev_id = await self.request_raw(b"ATI\r")  # display device version
        self.log_info(f"Device ID: {dev_id.decode().strip()}")
        await self.request_raw_multiple(
            [
                b"ATAT1\r",  # adaptive timing Auto1
                b"ATDPN\r",  # get current protocol
                b"ATSP0\r",  # set protocol to 0
                b"0100\r",
                b"ATH1\r",  # headers on
            ]
        )
        self.dev_protocol = (
            await self.request_raw(b"ATDPN\r")
        ).decode()  # get current protocol
        self.dev_protocol = self.dev_protocol.strip().removeprefix("A")
        self.log_info(f"Protocol ID: {self.dev_protocol}")
        await self.request_raw_multiple(
            [
                b"0100\r",
                b"ATH0\r",  # headers off
            ]
        )

    async def _recv_raw(self):
        try:
            data = await self.reader.readuntil(b">")
        except Exception as e:
            if not self.reader.at_eof():
                self.log_error(HTML("Read error: {}").format(e))
            return b""
        self.log_debug(HTML('[raw] <style fg="#8800ff">&lt;-</style> {}').format(data))
        return data

    async def send_raw(self, data: bytes):
        assert self.writer is not None
        data = bytes(data)
        self.log_debug(HTML('[raw] <style fg="#ff8800">-&gt;</style> {}').format(data))
        try:
            self.writer.write(data)
        except Exception as e:
            if not self.reader.at_eof():
                self.log_error(HTML("Write error: {}").format(e))

    async def _pre_can_reset(self):
        # See ELM Electronics' AN07 - Sending Arbitrary CAN Messages
        self.log_info("Configuring device for CAN communications")
        await self.request_raw_multiple(
            [
                b"ATZ\r",  # reset all
                b"ATE0\r",  # echo off
                b"ATE0\r",  # (2x)
                b"ATM0\r",  # memory off
                b"ATL0\r",  # linefeeds off
                b"ATS0\r",  # spaces off
                b"ATSP"
                + str(self.dev_protocol).encode("utf-8")
                + b"\r",  # disable automatic protocol search (use previously-discovered protocol)
                b"ATAL\r",  # allow long messages (for compatibility with older ELM327 chipsets)
                b"ATCEA\r",  # turn off CAN extended addressing
                b"ATCAF0\r",  # turn off CAN automatic formatting
                b"ATV1\r",  # use variable DLC
                b"ATBI\r",  # bypass initialization
            ]
        )

    async def _post_can_reset(self):
        self.log_info("Performing post-CAN communication self-reset")
        await self.reset()

    async def send_can(self, arbitration_id: int, data: bytes):
        try:
            await self._pre_can_reset()
            await self.request_raw(
                b"ATSH " + hex(arbitration_id).encode("utf-8")[2:] + b"\r"
            )
            await self.request_raw(binascii.hexlify(data).upper())
        finally:
            await self._post_can_reset()

    async def send_can_multiple(self, packets: list[tuple[int, bytes]]):
        await self._pre_can_reset()
        self.log_info(f"Now sending {len(packets)} CAN packets...")
        try:
            last_arbitration_id = None
            for packet in packets:
                self.log_info(
                    f"Arbitration ID: {hex(packet[0])[2:]}, Data: {binascii.hexlify(packet[1]).upper()}"
                )
                if last_arbitration_id is None or packet[0] != last_arbitration_id:
                    await self.request_raw(
                        b"ATSH " + hex(packet[0]).encode("utf-8")[2:] + b"\r"
                    )
                await self.request_raw(binascii.hexlify(packet[1]).upper() + b"\r")
        finally:
            await self._post_can_reset()

    async def request_interactive(self, command: str):
        final_command = command.strip().encode("utf-8") + b"\r"
        raw_response = await self.request_raw(final_command)
        return raw_response.decode()
