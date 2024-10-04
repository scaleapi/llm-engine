# [katie] modified from the aiohttp_sse_client library's https://github.com/rtfol/aiohttp-sse-client/blob/master/aiohttp_sse_client/client.py

# -*- coding: utf-8 -*-
"""Main module."""
# import asyncio
# import logging
from datetime import timedelta

import attr

# from aiohttp import hdrs, ClientSession, ClientConnectionError
from aiohttp import ClientResponse  # [katie]

# from typing import Optional, Dict, Any


# from multidict import MultiDict
# from yarl import URL

# READY_STATE_CONNECTING = 0
# READY_STATE_OPEN = 1
# READY_STATE_CLOSED = 2

# DEFAULT_RECONNECTION_TIME = timedelta(seconds=5)
# DEFAULT_MAX_CONNECT_RETRY = 5
# DEFAULT_MAX_READ_RETRY = 10

# CONTENT_TYPE_EVENT_STREAM = 'text/event-stream'
# LAST_EVENT_ID_HEADER = 'Last-Event-Id'

# _LOGGER = logging.getLogger(__name__)


@attr.s(slots=True, frozen=True)
class MessageEvent:
    """Represent DOM MessageEvent Interface

    .. seealso:: https://www.w3.org/TR/eventsource/#dispatchMessage section 4
    .. seealso:: https://developer.mozilla.org/en-US/docs/Web/API/MessageEvent
    """

    type = attr.ib(type=str)
    message = attr.ib(type=str)
    data = attr.ib(type=str)
    origin = attr.ib(type=str)
    last_event_id = attr.ib(type=str)


class EventSource:
    """Represent EventSource Interface as an async context manager.

    .. code-block:: python

        from aiohttp_sse_client import client as sse_client

        async with sse_client.EventSource(
            'https://stream.wikimedia.org/v2/stream/recentchange'
        ) as event_source:
            try:
                async for event in event_source:
                    print(event)
            except ConnectionError:
                pass

    .. seealso:: https://www.w3.org/TR/eventsource/#eventsource
    """

    def __init__(
        self,
        response: ClientResponse,
        #  url: str,
        #  option: Optional[Dict[str, Any]] = None,
        #  reconnection_time: timedelta = DEFAULT_RECONNECTION_TIME,
        #  max_connect_retry: int = DEFAULT_MAX_CONNECT_RETRY,
        #  session: Optional[ClientSession] = None,
        #  on_open=None,
        #  on_message=None,
        #  on_error=None,
        #  **kwargs
    ):
        """Construct EventSource instance.

        :param url: specifies the URL to which to connect
        :param option: specifies the settings, if any,
            in the form of an Dict[str, Any]. Accept the "method" key for
            specifying the HTTP method with which connection
            should be established
        :param reconnection_time: wait time before try to reconnect in case
            connection broken
        :param session: specifies a aiohttp.ClientSession, if not, create
            a default ClientSession
        :param on_open: event handler for open event
        :param on_message: event handler for message event
        :param on_error: event handler for error event
        :param kwargs: keyword arguments will pass to underlying
            aiohttp request() method.
        """
        # self._url = URL(url)
        # self._ready_state = READY_STATE_CONNECTING

        # if session is not None:
        #     self._session = session
        #     self._need_close_session = False
        # else:
        #     self._session = ClientSession()
        #     self._need_close_session = True

        # self._on_open = on_open
        # self._on_message = on_message
        # self._on_error = on_error

        # self._reconnection_time = reconnection_time
        # self._orginal_reconnection_time = reconnection_time
        # self._max_connect_retry = max_connect_retry
        self._last_event_id = ""
        # self._kwargs = kwargs
        # if 'headers' not in self._kwargs:
        #     self._kwargs['headers'] = MultiDict()

        self._event_id = ""
        self._event_type = ""
        self._event_data = ""

        self._origin = None
        self._response = response  # [katie]

        # self._method = 'GET' if option is None else option.get('method', 'GET')

    def __enter__(self):
        """Use async with instead."""
        raise TypeError("Use async with instead")

    def __exit__(self, *exc):
        """Should exist in pair with __enter__ but never executed."""
        pass  # pragma: no cover

    async def __aenter__(self) -> "EventSource":
        """Connect and listen Server-Sent Event."""
        # [katie] remove aiohttp
        # await self.connect(self._max_connect_retry)
        return self

    async def __aexit__(self, *exc):
        """Close connection and session if need."""
        # [katie] remove aiohttp
        # await self.close()
        # if self._need_close_session:
        #     await self._session.close()
        pass

    # @property
    # def url(self) -> URL:
    #     """Return URL to which to connect."""
    #     return self._url

    # @property
    # def ready_state(self) -> int:
    #     """Return ready state."""
    #     return self._ready_state

    def __aiter__(self):
        """Return"""
        return self

    async def __anext__(self) -> MessageEvent:
        """Process events"""
        if not self._response:
            raise ValueError

        # async for ... in StreamReader only split line by \n
        # while self._response.status != 204: [katie]
        async for line_in_bytes in self._response.content:
            line = line_in_bytes.decode("utf8")  # type: str
            line = line.rstrip("\n").rstrip("\r")

            if line == "":
                # empty line
                event = self._dispatch_event()
                if event is not None:
                    return event
                continue

            if line[0] == ":":
                # comment line, ignore
                continue

            if ":" in line:
                # contains ':'
                fields = line.split(":", 1)
                field_name = fields[0]
                field_value = fields[1].lstrip(" ")
                self._process_field(field_name, field_value)
            else:
                self._process_field(line, "")
            # [katie] remove aiohttp
            # self._ready_state = READY_STATE_CONNECTING
            # if self._on_error:
            #     self._on_error()
            # self._reconnection_time *= 2
            # _LOGGER.debug('wait %s seconds for retry',
            #               self._reconnection_time.total_seconds())
            # await asyncio.sleep(
            #     self._reconnection_time.total_seconds())
            # await self.connect()
        raise StopAsyncIteration

    # async def connect(self, retry=0):
    #     """Connect to resource."""
    #     _LOGGER.debug('connect')
    #     headers = self._kwargs['headers']

    #     # For HTTP connections, the Accept header may be included;
    #     # if included, it must contain only formats of event framing that are
    #     # supported by the user agent (one of which must be text/event-stream,
    #     # as described below).
    #     headers[hdrs.ACCEPT] = CONTENT_TYPE_EVENT_STREAM

    #     # If the event source's last event ID string is not the empty string,
    #     # then a Last-Event-Id HTTP header must be included with the request,
    #     # whose value is the value of the event source's last event ID string,
    #     # encoded as UTF-8.
    #     if self._last_event_id != '':
    #         headers[LAST_EVENT_ID_HEADER] = self._last_event_id

    #     # User agents should use the Cache-Control: no-cache header in
    #     # requests to bypass any caches for requests of event sources.
    #     headers[hdrs.CACHE_CONTROL] = 'no-cache'

    #     try:
    #         response = await self._session.request(
    #             self._method,
    #             self._url,
    #             **self._kwargs
    #         )
    #     except ClientConnectionError:
    #         if retry <= 0 or self._ready_state == READY_STATE_CLOSED:
    #             await self._fail_connect()
    #             raise
    #         else:
    #             self._ready_state = READY_STATE_CONNECTING
    #             if self._on_error:
    #                 self._on_error()
    #             self._reconnection_time *= 2
    #             _LOGGER.debug('wait %s seconds for retry',
    #                           self._reconnection_time.total_seconds())
    #             await asyncio.sleep(
    #                 self._reconnection_time.total_seconds())
    #             await self.connect(retry - 1)
    #         return

    #     if response.status >= 400 or response.status == 305:
    #         error_message = 'fetch {} failed: {}'.format(
    #             self._url, response.status)
    #         _LOGGER.error(error_message)

    #         await self._fail_connect()

    #         if response.status in [305, 401, 407]:
    #             raise ConnectionRefusedError(error_message)
    #         raise ConnectionError(error_message)

    #     if response.status != 200:
    #         error_message = 'fetch {} failed with wrong response status: {}'. \
    #             format(self._url, response.status)
    #         _LOGGER.error(error_message)
    #         await self._fail_connect()
    #         raise ConnectionAbortedError(error_message)

    #     if response.content_type != CONTENT_TYPE_EVENT_STREAM:
    #         error_message = \
    #           'fetch {} failed with wrong Content-Type: {}'.format(
    #               self._url, response.headers.get(hdrs.CONTENT_TYPE))
    #         _LOGGER.error(error_message)

    #         await self._fail_connect()
    #         raise ConnectionAbortedError(error_message)

    #     # only status == 200 and content_type == 'text/event-stream'
    #     await self._connected()

    #     self._response = response
    #     self._origin = str(response.real_url.origin())

    # async def close(self):
    #     """Close connection."""
    #     _LOGGER.debug('close')
    #     self._ready_state = READY_STATE_CLOSED
    #     if self._response is not None:
    #         self._response.close()
    #         self._response = None

    # async def _connected(self):
    #     """Announce the connection is made."""
    #     if self._ready_state != READY_STATE_CLOSED:
    #         self._ready_state = READY_STATE_OPEN
    #         if self._on_open:
    #             self._on_open()
    #     self._reconnection_time = self._orginal_reconnection_time

    # async def _fail_connect(self):
    #     """Announce the connection is failed."""
    #     if self._ready_state != READY_STATE_CLOSED:
    #         self._ready_state = READY_STATE_CLOSED
    #         if self._on_error:
    #             self._on_error()
    #     pass

    def _dispatch_event(self):
        """Dispatch event."""
        self._last_event_id = self._event_id

        if self._event_data == "":
            self._event_type = ""
            return

        self._event_data = self._event_data.rstrip("\n")

        message = MessageEvent(
            type=self._event_type if self._event_type != "" else None,
            message=self._event_type,
            data=self._event_data,
            origin=self._origin,
            last_event_id=self._last_event_id,
        )
        # _LOGGER.debug(message)
        # if self._on_message:
        #     self._on_message(message)

        self._event_type = ""
        self._event_data = ""
        return message

    def _process_field(self, field_name, field_value):
        """Process field."""
        if field_name == "event":
            self._event_type = field_value

        elif field_name == "data":
            self._event_data += field_value
            self._event_data += "\n"

        elif field_name == "id" and field_value not in ("\u0000", "\x00\x00"):
            self._event_id = field_value

        elif field_name == "retry":
            try:
                retry_in_ms = int(field_value)
                self._reconnection_time = timedelta(milliseconds=retry_in_ms)
            except ValueError:
                # _LOGGER.warning('Received invalid retry value %s, ignore it',
                #                 field_value)
                pass

        pass
