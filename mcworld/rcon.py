"""Соединение RCON - то же, что mcrcon.MCRcon, но без паузы после команды.

mcrcon после КАЖДОЙ команды спит 3 мс: "time.sleep(0.003)  # MC-72390
workaround". Это обход давнего бага сервера, из-за которого частые
команды по RCON могли уронить сервер; в ванильном сервере он исправлен в
1.14.3. Мост шлёт сотни команд в секунду строго по одной (следующая -
только после ответа на предыдущую), и эти 3 мс - большая часть цены
команды: в гонке трёх мух шлюз упирался в 230-236 команд в секунду, а это
ровно 1 / (3 мс паузы + ~1.2 мс ответа сервера).
"""

import time

from .constants import RCON_PAUSE_S


def open_rcon(host, password, port):
    """Соединение с методами connect/command/disconnect, как у MCRcon."""
    from mcrcon import MCRcon

    if not hasattr(MCRcon, "_send"):
        return MCRcon(host, password, port=port)     # заглушка смоука

    class QuickRcon(MCRcon):
        def command(self, text):
            answer = self._send(2, text)
            if RCON_PAUSE_S > 0:
                time.sleep(RCON_PAUSE_S)
            return answer

    return QuickRcon(host, password, port=port)
