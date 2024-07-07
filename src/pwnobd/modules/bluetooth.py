from ..core import scanner, Scanner, ScanContext
import asyncio
import socket
from bleak import BleakScanner
from bleak.backends.client import BLEDevice
from bleak.backends.scanner import AdvertisementData


_DEFAULT_LIMIT = 2**16  # 64 KiB


async def bt_open_connection(addr=None, port=None, *, limit=_DEFAULT_LIMIT, **kwds):
    """A wrapper for create_connection() returning a (reader, writer) pair.

    The reader returned is a StreamReader instance; the writer is a
    StreamWriter instance.

    The arguments are all the usual arguments to create_connection()
    except protocol_factory; most common are positional host and port,
    with various optional keyword arguments following.

    Additional optional keyword arguments are loop (to set the event loop
    instance to use) and limit (to set the buffer limit passed to the
    StreamReader).

    (If you want to customize the StreamReader and/or
    StreamReaderProtocol classes, just copy the code -- there's
    really nothing special here except some convenience.)
    """
    loop = asyncio.events.get_running_loop()
    reader = asyncio.StreamReader(limit=limit, loop=loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=loop)
    waiter = loop.create_future()
    sock = socket.socket(
        family=socket.AF_BLUETOOTH, type=socket.SOCK_STREAM, proto=socket.BTPROTO_RFCOMM
    )
    sock.setblocking(False)
    await loop.sock_connect(sock, (addr, port))
    transport = loop._make_socket_transport(sock, protocol, waiter)
    writer = asyncio.StreamWriter(transport, protocol, reader, loop)
    return reader, writer


@scanner()
class BluetoothScanner(Scanner[dict[str, tuple[BLEDevice, AdvertisementData]]]):
    async def scan(self, ctx: ScanContext):
        return await BleakScanner.discover(timeout=20, return_adv=True)
