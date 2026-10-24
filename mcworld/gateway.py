"""Шлюз гонки: одна связь с сервером на всех мух."""

import datetime
import itertools
import os
import queue
import threading
import time
from multiprocessing.connection import Client, Listener

# Ответ сервера дольше этого - замирание: обычная команда стоит около 4 мс.
STALL_S = 0.15

# (хост, порт) шлюза, если муха работает в гонке; None - своё соединение.
GATEWAY = None
KEY_ENV = "FLYRACE_KEY"


class GatewayConnection:
    """Соединение мухи со шлюзом - тот же набор методов, что у MCRcon."""

    def __init__(self, address, name=""):
        self.address, self.name = tuple(address), name
        self._conn = None

    def connect(self):
        key = os.environ.get(KEY_ENV)
        if not key:
            raise RuntimeError(f"нет ключа шлюза в {KEY_ENV}: муху с --gateway "
                               f"запускает demos/race.py")
        self._conn = Client(self.address, authkey=key.encode())
        self._conn.send(("hello", self.name))
        return self

    def command(self, text):
        self._conn.send(text)
        status, answer = self._conn.recv()
        if status != "ok":
            raise RuntimeError(f"шлюз: {answer}")
        return answer

    def shared(self, last_seq, timeout):
        """Общий снимок новее last_seq: ("ok", снимок или None) или ("off", None)."""
        self._conn.send(("shared", last_seq, timeout))
        return self._conn.recv()

    def fact(self, key, timeout):
        """Общий факт: ("ok", значение) или ("compute", None) - выясни и отдай."""
        self._conn.send(("fact", key, timeout))
        return self._conn.recv()

    def fact_set(self, key, value):
        self._conn.send(("fact_set", key, value))
        return self._conn.recv()

    def disconnect(self):
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None


class GatewayServer:
    """Принимает мух и выполняет их команды по одной через одно соединение."""

    def __init__(self, rcon, key, priority=True):
        self.rcon = rcon
        self.priority = priority
        self.listener = Listener(("127.0.0.1", 0), authkey=key.encode())
        self.address = self.listener.address
        self._queue = queue.PriorityQueue()
        self._order = itertools.count()
        self.stats_lock = threading.Lock()
        self.per_fly = {}
        self.total, self.busy_s = 0, 0.0

        self.waits = {"телепорт": [0.0, 0, 0.0], "прочее": [0.0, 0, 0.0]}
        self.errors = []

        self.stalls = []
        self._closed = False
        self.observers = []
        self.shared = None
        self._facts = {}
        self._facts_busy = set()
        self._facts_cond = threading.Condition()
        threading.Thread(target=self._work, daemon=True).start()
        threading.Thread(target=self._accept, daemon=True).start()

    def command(self, text, who="шлюз"):
        """Команда от самого шлюза или от мухи: встаёт в очередь и ждёт ответа."""
        urgent = text.startswith("tp ")
        slot = {"done": threading.Event(), "came": time.perf_counter()}
        rank = 0 if (urgent and self.priority) else 1
        self._queue.put((rank, next(self._order), text, who, slot))
        slot["done"].wait()
        waited = time.perf_counter() - slot["came"]
        with self.stats_lock:
            entry = self.waits["телепорт" if urgent else "прочее"]
            entry[0] += waited
            entry[1] += 1
            entry[2] = max(entry[2], waited)
        if "error" in slot:
            raise slot["error"]
        return slot["answer"]

    def _work(self):
        while True:
            _, _, text, who, slot = self._queue.get()
            if text is None:
                return
            started = time.perf_counter()
            try:
                slot["answer"] = self.rcon.command(text)
            except Exception as exc:
                slot["error"] = exc
            spent = time.perf_counter() - started
            if "answer" in slot:
                for hook in self.observers:
                    try:
                        hook(text, slot["answer"])
                    except Exception as exc:
                        self.errors.append(f"наблюдатель: {type(exc).__name__}: {exc}")
            with self.stats_lock:
                if spent >= STALL_S:
                    clock = datetime.datetime.now().strftime("%H:%M:%S")
                    self.stalls.append((clock, spent, text.split(" ")[0]))
                entry = self.per_fly.setdefault(who, [0, 0.0])
                entry[0] += 1
                entry[1] += spent
                self.total += 1
                self.busy_s += spent
            slot["done"].set()

    def snapshot(self):
        with self.stats_lock:
            return ({k: tuple(v) for k, v in self.per_fly.items()},
                    self.total, self.busy_s)

    def take_stalls(self):
        """Замирания с прошлого вызова: [(часы, секунд, первое слово команды)]."""
        with self.stats_lock:
            out, self.stalls = self.stalls, []
        return out

    def take_waits(self):
        """Ожидание в шлюзе с прошлого вызова: класс -> (среднее с, максимум с, число)."""
        with self.stats_lock:
            out = {k: (v[0] / v[1] if v[1] else 0.0, v[2], v[1])
                   for k, v in self.waits.items()}
            self.waits = {k: [0.0, 0, 0.0] for k in self.waits}
        return out

    def close(self):
        self._closed = True
        self._queue.put((2, next(self._order), None, None, None))
        try:
            self.listener.close()
        except OSError:
            pass

    def _accept(self):
        while not self._closed:
            try:
                conn = self.listener.accept()
            except (OSError, EOFError):
                if self._closed:
                    return
                continue
            except Exception as exc:
                self.errors.append(f"приём: {type(exc).__name__}: {exc}")
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _fact_reply(self, key, timeout):
        """Факт готов - отдать; выясняет другая муха - ждать; никто - "compute"."""
        deadline = time.perf_counter() + timeout
        with self._facts_cond:
            while key not in self._facts and key in self._facts_busy:
                left = deadline - time.perf_counter()
                if left <= 0:
                    return ("compute", None)
                self._facts_cond.wait(left)
            if key in self._facts:
                return ("ok", self._facts[key])
            self._facts_busy.add(key)
            return ("compute", None)

    def _shared_reply(self, last_seq, timeout):
        sight = self.shared
        if sight is None:
            return ("off", None)
        snapshot = sight.wait_newer(last_seq, timeout)
        return ("ok", None if snapshot is None else sight.package(snapshot))

    def _serve(self, conn):
        try:
            hello = conn.recv()
            who = hello[1] if isinstance(hello, tuple) and hello[0] == "hello" else "?"
        except (EOFError, OSError):
            return
        while True:
            try:
                text = conn.recv()
            except (EOFError, OSError):
                return
            if isinstance(text, tuple) and text and text[0] in ("shared", "fact",
                                                                  "fact_set"):
                if text[0] == "shared":
                    reply = self._shared_reply(*text[1:])
                elif text[0] == "fact":
                    reply = self._fact_reply(*text[1:])
                else:
                    with self._facts_cond:
                        self._facts[text[1]] = text[2]
                        self._facts_busy.discard(text[1])
                        self._facts_cond.notify_all()
                    reply = ("ok", None)
                try:
                    conn.send(reply)
                except (OSError, EOFError):
                    return
                continue
            try:
                reply = ("ok", self.command(text, who))
            except Exception as exc:
                self.errors.append(f"{who}: {type(exc).__name__}: {exc}")
                reply = ("error", f"{type(exc).__name__}: {exc}")
            try:
                conn.send(reply)
            except (OSError, EOFError):
                return
