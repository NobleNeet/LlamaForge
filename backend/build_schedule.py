"""Daily, idle-only updates; independent of the browser and disabled by default."""
from contextlib import contextmanager
from datetime import datetime
import re
import threading


def valid_time(value):
    return (value if isinstance(value, str)
            and re.fullmatch(r"(?:[01][0-9]|2[0-3]):[0-5][0-9]", value) else None)


class BuildSchedule:
    def __init__(self, load, save, run):
        self.load, self.save, self.run = load, save, run
        self.lock = threading.Lock()
        self.active = 0
        self.updating = False

    @contextmanager
    def activity(self):
        """Admit requests/background mutations unless maintenance owns the server."""
        with self.lock:
            admitted = not self.updating
            if admitted:
                self.active += 1
        try:
            yield admitted
        finally:
            if admitted:
                with self.lock:
                    self.active -= 1

    def tick(self, now=None):
        now = now or datetime.now().astimezone()
        with self.lock:
            c = self.load()
            day = now.date().isoformat()
            if (c.get("build_auto_update_enabled") is not True
                    or valid_time(c.get("build_auto_update_time")) != now.strftime("%H:%M")
                    or c.get("build_auto_update_last_date", "") >= day
                    or self.updating):
                return
            # Persist before acting: a restart or repeated DST minute cannot retry.
            self.save({"build_auto_update_last_date": day,
                       "build_auto_update_status": "Checking idle state"})
            if self.active:
                self.save({"build_auto_update_status": "Skipped: requests or background work active"})
                return
            self.updating = True
        try:
            result = self.run()
            self.save({"build_auto_update_status": result})
        except Exception as exc:
            self.save({"build_auto_update_status": "Failed: " + str(exc)})
        finally:
            with self.lock:
                self.updating = False

    def loop(self, stop, report):
        while not stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                report("build.schedule.error", error=str(exc))
            stop.wait(10)
