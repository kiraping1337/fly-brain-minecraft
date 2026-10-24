"""Связь с сервером: команды под общим замком, чтение счёта, учёт команд."""

import re
import threading
import time

from . import gateway, ident


def command_kind(text):
    """К какому опросу относится команда - для счётчика."""
    if text.startswith("tp "):
        return "телепорт"
    if "#sight" in text:
        return "видимость"
    if "#probe" in text or "minecraft:water" in text or "minecraft:lava" in text:
        return "рельеф"
    if re.search(r"#p\d", text) or "#pass" in text:
        return "досмотр"
    if "#hp" in text or text.endswith(" Health"):
        return "здоровье"
    if "#food" in text or "setblock" in text or "summon item" in text:
        return "еда"
    if "type=item" in text and "Item:" in text:
        return "запах"
    if "inGround" in text:
        return "снаряды"
    if "@p" in text:
        return "игрок"
    if "tag=!" in text:
        return "существа"
    if text.startswith("damage"):
        return "голод"
    return "прочее"


class CommandStats:
    """Сколько команд уходит, на что, и сколько сервер на них отвечает."""

    def __init__(self):
        self.lock = threading.Lock()
        self.kinds = {}
        self.total = 0
        self.busy_s = 0.0

    def add(self, text, seconds):
        kind = command_kind(text)
        with self.lock:
            self.kinds[kind] = self.kinds.get(kind, 0) + 1
            self.total += 1
            self.busy_s += seconds

    def snapshot(self):
        with self.lock:
            return dict(self.kinds), self.total, self.busy_s


class FairLock:
    """Замок с очередью: кто раньше пришёл, тот раньше и получит."""

    def __init__(self):
        self._cond = threading.Condition(threading.Lock())
        self._next = 0
        self._serving = 0

    def __enter__(self):
        with self._cond:
            ticket = self._next
            self._next += 1
            while ticket != self._serving:
                self._cond.wait()
        return self

    def __exit__(self, *exc):
        with self._cond:
            self._serving += 1
            self._cond.notify_all()
        return False


COMMAND_STATS = CommandStats()
RCON_TURN = FairLock()            # одна команда к серверу за раз, см. command()


# Правила, которые глушат вывод команд: (имя с 1.21.11, прежнее имя).
QUIET_RULES = (("send_command_feedback", "sendCommandFeedback"),
               ("log_admin_commands", "logAdminCommands"))


class LinkPart:
    """Часть MinecraftWorld: связь с сервером: команды под общим замком, чтение счёта, учёт команд."""

    def command(self, text):
        turn = self._talking if gateway.GATEWAY is not None else RCON_TURN
        with turn:
            started = time.perf_counter()
            answer = self._connection.command(text)
            COMMAND_STATS.add(text, time.perf_counter() - started)
            return answer

    def close(self):
        try:
            self._connection.disconnect()
        except Exception:
            pass

    def quiet(self):
        """Убрать спам от команд. Возвращает правила, которые сервер принял."""
        accepted = []
        for names in QUIET_RULES:
            for name in names:
                self.command(f"gamerule {name} false")
                state = self.command(f"gamerule {name}") or ""
                if "false" in state.lower():
                    accepted.append(name)
                    break
        return accepted

    def read_score(self, counter):
        """Прочитать счёт. None, если ответ пришёл не на нашу команду."""
        holder = counter.split()[0]
        answer = self.command(f"scoreboard players get {counter}") or ""
        self.last_answer = answer
        found = re.search(rf"{re.escape(holder)}[^\d-]*(-?\d+)", answer)
        if not found:
            self.mismatched += 1
            if len(self.bad_answers) < 12:
                self.bad_answers.append(answer[:80])
            return None
        return int(found.group(1))

    def _ensure_scoreboard(self):
        if not self._scoreboard_ready:
            self.command(f"scoreboard objectives add {ident.PROBE_OBJECTIVE} dummy")
            self._scoreboard_ready = True
