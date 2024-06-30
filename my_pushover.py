#!/usr/bin/env python

from http.client import HTTPSConnection
from urllib.parse import urlencode
import json
import os

class PushoverError(Exception): pass

class PushoverMessage:
    """
    Used for storing message specific data.
    """

    def __init__(self, message):
        """
        Creates a PushoverMessage object.
        """
        self.vars = {}
        self.vars['message'] = message

    def set(self, key, value):
        """
        Sets the value of a field "key" to the value of "value".
        """
        if value is not None:
            self.vars[key] = value

    def get(self):
        """
        Returns a dictionary with the values for the specified message.
        """
        return self.vars

    def user(self, user_token, user_device=None):
        """
        Sets a single user to be the recipient of this message with token "user_token" and device "user_device".
        """
        self.set('user', user_token)
        self.set('device', user_device)

    def __str__(self):
        return "PushoverMessage: " + str(self.vars)

class Pushover:
    """
    Creates a Pushover handler.

    Usage:

        po = Pushover("My App Token")
        po.user("My User Token", "My User Device Name")

        msg = po.msg("Hello, World!")

        po.send(msg)

    """

    PUSHOVER_SERVER = "api.pushover.net:443"
    PUSHOVER_ENDPOINT = "/1/messages.json"
    PUSHOVER_CONTENT_TYPE = { "Content-type": "application/x-www-form-urlencoded"}

    def __init__(self, token=None):
        """
        Creates a Pushover object.
        """

        if token is None:
            raise PushoverError("No token supplied.")
        else:
            self.token = token
            self.user_token = None
            self.user_device = None
            self.messages = []

    def msg(self, message):
        """
        Creates a PushoverMessage object. Takes one "message" parameter (the message to be sent).
        Returns with PushoverMessage object (msg).
        """

        message = PushoverMessage(message)
        self.messages.append(message)
        return message

    def send(self, message):
        """
        Sends a specified message with id "message" or as object.
        """
        if type(message) is PushoverMessage:
            return self._send(message)
        else:
            raise PushoverError("Wrong type passed to Pushover.send()!")

    def sendall(self):
        """
        Sends all PushoverMessage's owned by the Pushover object.
        """

        response = []
        for message in self.messages:
            response.append(self._send(message))
        return response

    def user(self, user_token, user_device=None):
        """
        Sets a single user to be the recipient of all messages created with this Pushover object.
        """

        self.user_token = user_token
        self.user_device = user_device

    def _send(self, message):
        """
        Sends the specified PushoverMessage object via the Pushover API.
        """

        kwargs = message.get()
        kwargs['token'] = self.token

        assert 'message' in kwargs
        assert self.token is not None

        if not 'user' in kwargs:
            if self.user is not None:
                kwargs['user'] = self.user_token
                if self.user_device is not None:
                    kwargs['device'] = self.user_device
            else:
                kwargs['user'] = os.environ['PUSHOVER_USER']

        data = urlencode(kwargs)
        conn = HTTPSConnection(Pushover.PUSHOVER_SERVER)
        conn.request("POST", Pushover.PUSHOVER_ENDPOINT, data, Pushover.PUSHOVER_CONTENT_TYPE)
        output = conn.getresponse().read().decode('utf-8')
        data = json.loads(output)

        if data['status'] != 1:
            raise PushoverError(output)
        else:
            return True
