import pyshark
from typing import AsyncGenerator


class AsyncFileCapture(pyshark.FileCapture):
    def __init__(
        self,
        input_file=None,
        keep_packets=True,
        display_filter=None,
        only_summaries=False,
        decryption_key=None,
        encryption_type="wpa-pwk",
        decode_as=None,
        disable_protocol=None,
        tshark_path=None,
        override_prefs=None,
        use_json=False,
        use_ek=False,
        output_file=None,
        include_raw=False,
        eventloop=None,
        custom_parameters=None,
        debug=False,
    ):
        super().__init__(
            input_file,
            keep_packets,
            display_filter,
            only_summaries,
            decryption_key,
            encryption_type,
            decode_as,
            disable_protocol,
            tshark_path,
            override_prefs,
            use_json,
            use_ek,
            output_file,
            include_raw,
            eventloop,
            custom_parameters,
            debug,
        )
        self._packet_generator: AsyncGenerator = (
            self._go_through_packets_from_fd_generator()
        )

    async def _go_through_packets_from_fd_generator(
        self, packet_count=None, close_tshark=True
    ):
        """A generator which goes through a stream and calls a given callback for each XML packet seen in it."""
        fd = (await self._get_tshark_process(packet_count=packet_count)).stdout
        try:
            packets_captured = 0
            self._log.debug("Starting to go through packets (generator)")

            parser = self._setup_tshark_output_parser()
            data = b""

            while True:
                try:
                    packet, data = await parser.get_packets_from_stream(
                        fd, data, got_first_packet=packets_captured > 0
                    )
                except EOFError:
                    self._log.debug("EOF reached")
                    self._eof_reached = True
                    break

                if packet:
                    packets_captured += 1
                    yield packet

                if packet_count and packets_captured >= packet_count:
                    break
        finally:
            if close_tshark:
                await self.close_async()

    def __aiter__(self):
        return self

    async def __anext__(self):
        """Returns the next packet in the cap.

        If the capture's keep_packets flag is True, will also keep it in the internal packet list.
        """
        if not self.keep_packets:
            return await self._packet_generator.asend(None)
        elif self._current_packet >= len(self._packets):
            packet = await self._packet_generator.asend(None)
            self._packets += [packet]
        return super(pyshark.FileCapture, self).next_packet()
