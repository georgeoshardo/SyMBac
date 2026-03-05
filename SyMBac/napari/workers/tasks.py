from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class CancelToken:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


@dataclass
class WorkerHandle:
    worker: Any
    token: CancelToken | None = None

    def cancel(self) -> None:
        if self.token is not None:
            self.token.cancel()
        if hasattr(self.worker, "quit"):
            self.worker.quit()


def _default_error_handler(exc: Exception) -> None:
    from napari.utils.notifications import show_error

    show_error(str(exc))


def start_worker(
    fn: Callable[..., Any],
    *args,
    on_return: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    **kwargs,
) -> WorkerHandle:
    from napari.qt.threading import thread_worker

    worker = thread_worker(fn)(*args, **kwargs)
    if on_return is not None:
        worker.returned.connect(on_return)
    worker.errored.connect(on_error or _default_error_handler)
    worker.start()
    return WorkerHandle(worker=worker)


def start_cancellable_worker(
    fn: Callable[..., Any],
    *args,
    on_return: Callable[[Any], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    **kwargs,
) -> WorkerHandle:
    from napari.qt.threading import thread_worker

    token = CancelToken()

    worker = thread_worker(fn)(token, *args, **kwargs)
    if on_return is not None:
        worker.returned.connect(on_return)
    worker.errored.connect(on_error or _default_error_handler)
    worker.start()
    return WorkerHandle(worker=worker, token=token)
