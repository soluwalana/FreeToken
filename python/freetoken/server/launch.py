from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from freetoken.distributed import DistributedInfo
from freetoken.utils import init_logger

if TYPE_CHECKING:
    from .args import ServerArgs
    from .supervisor import BackendHandle


def _report_startup_error(ack_queue: mp.Queue, exc: BaseException) -> None:
    """Tell the parent WHY this worker is dying — push an ("error", reason) ack before it exits,
    so the supervisor reports the real cause (e.g. a config ValueError) instead of the generic
    "backend worker … exited during load". Best-effort and flushed (close + join_thread), since
    the process is about to terminate; a failure to report must never mask the original error."""
    try:
        ack_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        ack_queue.close()
        ack_queue.join_thread()
    except Exception:  # noqa: BLE001 -- reporting is a nicety; never shadow the real exception
        pass


def _detach_process_group() -> None:
    """Shell mode only: move this worker out of the terminal's foreground process group.

    The shell binds ^C to "cancel this turn", but a terminal delivers SIGINT to the whole
    foreground group — which, with the engine running in this same process, includes the
    workers. They would take the same ^C and exit (``_run_scheduler`` below stops gracefully on
    KeyboardInterrupt), leaving the shell chatting with a dead engine. Nothing depends on the
    signal reaching them: the parent tears the workers down explicitly on every stop path
    (uvicorn's lifespan, plus the SIGTERM/SIGHUP handler and the reap backstop in api_server).

    ``ft serve`` keeps the default — uvicorn owns ^C there, and the group-wide delivery is part
    of how it stops."""
    try:
        os.setpgrp()
    except OSError:  # no job control (already a group leader / unusual environment)
        pass


def _run_tokenize_worker(detach: bool, **kwargs) -> None:
    """Module-level so it survives the spawn pickle; exists only to detach the group first."""
    if detach:
        _detach_process_group()
    from freetoken.tokenizer import tokenize_worker

    tokenize_worker(**kwargs)


def scheduler_rank_args(server_args: ServerArgs, rank: int) -> ServerArgs:
    """Args for one scheduler worker. Expert parallel keeps ``tp_info`` at (0, 1)."""
    if server_args.ep_info.size > 1:
        world = server_args.ep_info.size
        return replace(
            server_args, tp_info=DistributedInfo(0, 1), ep_info=DistributedInfo(rank, world)
        )
    world = server_args.tp_info.size
    return replace(server_args, tp_info=DistributedInfo(rank, world))


def _parallel_placement(args: ServerArgs) -> tuple[int, int, bool]:
    """``(rank, world, owns_sockets)``. Sockets belong to tensor rank 0 and expert rank 0."""
    if args.ep_info.size > 1:
        rank, world = args.ep_info.rank, args.ep_info.size
    else:
        rank, world = args.tp_info.rank, args.tp_info.size
    return rank, world, args.tp_info.is_primary() and args.ep_info.is_primary()


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    if args.shell_mode:
        _detach_process_group()

    # published (not bound) here: the engine binds it after the allocator setup
    from freetoken.gpu_select import set_assigned_gpu

    rank, world, owns_sockets = _parallel_placement(args)
    # resolved UUIDs when we have them, the raw --gpu entries when NVML could not resolve them, else one CUDA ordinal per rank
    targets = args.gpu_assigned or args.gpu or tuple(str(r) for r in range(world))
    set_assigned_gpu(targets[rank])

    import torch
    from freetoken.scheduler import Scheduler

    if owns_sockets:
        from freetoken.utils.progress import set_progress_sink

        set_progress_sink(
            lambda desc, done, total: ack_queue.put(("progress", desc, done, total))
        )

    with torch.inference_mode():
        try:
            scheduler = Scheduler(args)
            scheduler.sync_all_ranks()
        except Exception as exc:  # noqa: BLE001 -- surface the reason, then let it propagate
            # A startup failure (bad config, OOM, corrupt weights) would otherwise reach the
            # parent only as a dead process -> a generic "exited during load". Push the real
            # reason first so the supervisor (and the desktop failure modal) can surface it;
            # the traceback still prints and the process still exits non-zero.
            _report_startup_error(ack_queue, exc)
            raise

        if owns_sockets:
            # Report the real per-unit cache VRAM costs (KV/expert/mamba), the device-wide free
            # VRAM, and the per-pool rebuild floors before the ready ack, so the supervisor has
            # them (and the desktop's slider bounds) by the time the gate flips. Optional +
            # best-effort: a failure here must never keep the model from serving, and older
            # consumers ignore ("meta", …).
            try:
                from freetoken.kvcache.cache_status import compute_cache_status_meta

                meta = compute_cache_status_meta(scheduler.engine)
                # the parent must not touch CUDA to learn this
                meta["gpus"] = scheduler.gpus
                ack_queue.put(("meta", meta))
            except Exception:  # noqa: BLE001 -- metadata is a nicety; readiness is not
                pass
            ack_queue.put("Scheduler is ready")
            # The supervisor stops draining ack_queue once ready, so uninstall the sink:
            # runtime cache rebuilds re-run the graph capture (which emits progress) and
            # would otherwise push onto a queue nobody reads for the server's lifetime.
            set_progress_sink(None)

        if args.silent_output:
            logging.disable(logging.INFO)

        try:
            scheduler.run_forever()
        except KeyboardInterrupt:
            logger = init_logger(__name__)
            if owns_sockets:
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            scheduler.shutdown()


def launch_server(
    run_shell: bool = False,
    argv: list[str] | None = None,
    prog: str | None = None,
) -> None:
    from .api_server import run_api_server
    from .args import parse_args

    server_args, run_shell = parse_args(
        sys.argv[1:] if argv is None else argv,
        run_shell,
        prog=prog,
    )
    logger = init_logger(__name__, "initializer")

    if server_args.gpu:
        # resolve here so a typo is one clear error before any worker spawns
        from freetoken.gpu_select import resolve_gpu_uuids

        try:
            server_args = replace(server_args, gpu_assigned=resolve_gpu_uuids(server_args.gpu))
        except ValueError as exc:
            raise SystemExit(f"{prog or 'ft serve'}: error: {exc}") from exc
        logger.info(
            f"--gpu {','.join(server_args.gpu)} -> "
            f"{', '.join(server_args.gpu_assigned) if server_args.gpu_assigned else 'resolved at CUDA init (no NVML)'}"
        )

    def start_subprocess() -> "BackendHandle":
        import multiprocessing as mp

        from .supervisor import BackendHandle

        mp.set_start_method("spawn", force=True)
        detach = server_args.shell_mode  # see _detach_process_group

        world_size = server_args.ep_info.size if server_args.ep_info.size > 1 else server_args.tp_info.size
        ack_queue: mp.Queue = mp.Queue()
        processes: list[mp.Process] = []
        kind = "EP" if server_args.ep_info.size > 1 else "TP"

        for i in range(world_size):
            new_args = scheduler_rank_args(server_args, i)
            p = mp.Process(
                target=_run_scheduler,
                args=(new_args, ack_queue),
                daemon=False,
                name=f"freetoken-{kind}{i}-scheduler",
            )
            p.start()
            processes.append(p)

        num_tokenizers = server_args.num_tokenizer
        p = mp.Process(
            target=_run_tokenize_worker,
            kwargs={
                "detach": detach,
                "tokenizer_path": server_args.model_path,
                "addr": server_args.zmq_detokenizer_addr,
                "backend_addr": server_args.zmq_backend_addr,
                "frontend_addr": server_args.zmq_frontend_addr,
                "local_bs": 1,
                "mm": server_args.mm,
                "create": server_args.tokenizer_create_addr,
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="freetoken-detokenizer-0",
        )
        p.start()
        processes.append(p)
        for i in range(num_tokenizers):
            p = mp.Process(
                target=_run_tokenize_worker,
                kwargs={
                    "detach": detach,
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "mm": server_args.mm,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"freetoken-tokenizer-{i}",
            )
            p.start()
            processes.append(p)

        # Expected ready acks: 1 primary scheduler + num_tokenizers + 1 detokenizer.
        return BackendHandle(
            ack_queue=ack_queue,
            processes=processes,
            expected_acks=num_tokenizers + 2,
        )

    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    launch_server()
