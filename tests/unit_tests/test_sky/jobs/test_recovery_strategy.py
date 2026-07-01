"""Unit tests for sky.jobs.recovery_strategy helpers."""
import asyncio
import types
from unittest import mock

import pytest

from sky import exceptions
from sky.jobs import recovery_strategy
from sky.jobs import scheduler as scheduler_module


def test_is_oom_failure_detects_oomkilled():
    exc = RuntimeError(
        'Failed to run setup commands on an instance. (exit code 1). '
        'Pod p terminated: OOMKilled (exit code 137).')
    assert recovery_strategy._is_oom_failure(exc) is True


def test_is_oom_failure_detects_out_of_memory_phrase():
    assert recovery_strategy._is_oom_failure(
        RuntimeError('The container ran out of memory.')) is True


def test_is_oom_failure_is_case_insensitive():
    assert recovery_strategy._is_oom_failure(
        RuntimeError('reason: oomkilled')) is True


def test_is_oom_failure_false_for_unrelated():
    assert recovery_strategy._is_oom_failure(
        RuntimeError('/bin/bash: line 1: conda: command not found')) is False


# ---------------------------------------------------------------------------
# Parked launch request handling (yield the launch slot while the underlying
# launch request is WAITING).
# ---------------------------------------------------------------------------


def _make_bare_executor():
    executor = recovery_strategy.StrategyExecutor.__new__(
        recovery_strategy.StrategyExecutor)
    return executor


def _request_payload(status: str, status_msg=None):
    return types.SimpleNamespace(status=status, status_msg=status_msg)


@pytest.mark.asyncio
async def test_await_launch_request_returns_on_stream_completion(monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        return 'result'

    api_status = mock.AsyncMock()
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)

    assert await executor._await_launch_request('req-1') is None
    api_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_await_launch_request_propagates_stream_exception(monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        raise ValueError('launch failed')

    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status',
                        mock.AsyncMock())

    with pytest.raises(ValueError, match='launch failed'):
        await executor._await_launch_request('req-1')


@pytest.mark.asyncio
async def test_await_launch_request_parks_on_waiting_status(monkeypatch):
    executor = _make_bare_executor()
    stream_cancelled = asyncio.Event()

    async def fake_stream_and_get(request_id, **kwargs):
        try:
            await asyncio.Event().wait()  # Block forever.
        except asyncio.CancelledError:
            stream_cancelled.set()
            raise

    api_status = mock.AsyncMock(return_value=[
        _request_payload('WAITING', 'Workload is pending on queue foo.')
    ])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_LAUNCH_REQUEST_STATUS_POLL_SECONDS', 0.01)

    with pytest.raises(recovery_strategy._LaunchRequestParked) as exc_info:
        await executor._await_launch_request('req-1')
    assert exc_info.value.request_id == 'req-1'
    assert exc_info.value.status_msg == 'Workload is pending on queue foo.'
    # The stream task must have been cancelled (no leaked stream). The
    # cancellation is delivered asynchronously; give the loop a tick.
    await asyncio.sleep(0.01)
    assert stream_cancelled.is_set()


@pytest.mark.asyncio
async def test_await_launch_request_tolerates_poll_failures(monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        await asyncio.sleep(0.05)
        return 'result'

    api_status = mock.AsyncMock(side_effect=RuntimeError('transient'))
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_LAUNCH_REQUEST_STATUS_POLL_SECONDS', 0.01)

    # Should not raise despite the status poll failing.
    assert await executor._await_launch_request('req-1') is None
    assert api_status.await_count >= 1


@pytest.mark.asyncio
async def test_await_launch_request_tolerates_unknown_request(monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        await asyncio.sleep(0.05)
        return 'result'

    api_status = mock.AsyncMock(return_value=[])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_LAUNCH_REQUEST_STATUS_POLL_SECONDS', 0.01)

    assert await executor._await_launch_request('req-1') is None


@pytest.mark.asyncio
async def test_wait_for_parked_request_returns_on_resume(monkeypatch):
    executor = _make_bare_executor()
    api_status = mock.AsyncMock(side_effect=[
        [_request_payload('WAITING')],
        [_request_payload('RUNNING')],
    ])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_PARKED_POLL_INITIAL_BACKOFF_SECONDS', 0.01)

    assert await executor._wait_for_parked_request('req-1') == 'req-1'
    assert api_status.await_count == 2


@pytest.mark.asyncio
async def test_wait_for_parked_request_relaunches_when_request_gone(
        monkeypatch):
    executor = _make_bare_executor()
    executor._cancel_launch_request = mock.AsyncMock()
    api_status = mock.AsyncMock(return_value=[])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_PARKED_POLL_INITIAL_BACKOFF_SECONDS', 0.01)

    assert await executor._wait_for_parked_request('req-1') is None
    # Multiple consecutive misses are required before concluding the request
    # is gone (a single miss can be a transient server hiccup), and the old
    # request is best-effort cancelled before falling back to a fresh launch.
    assert (api_status.await_count ==
            recovery_strategy._PARKED_POLL_MAX_CONSECUTIVE_MISSING)
    executor._cancel_launch_request.assert_awaited_once_with('req-1')


@pytest.mark.asyncio
async def test_wait_for_parked_request_tolerates_transient_missing(monkeypatch):
    executor = _make_bare_executor()
    api_status = mock.AsyncMock(side_effect=[
        [],
        [_request_payload('WAITING')],
        [_request_payload('RUNNING')],
    ])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_PARKED_POLL_INITIAL_BACKOFF_SECONDS', 0.01)

    assert await executor._wait_for_parked_request('req-1') == 'req-1'


@pytest.mark.asyncio
async def test_wait_for_parked_request_gives_up_on_persistent_poll_errors(
        monkeypatch):
    executor = _make_bare_executor()
    executor._cancel_launch_request = mock.AsyncMock()
    api_status = mock.AsyncMock(side_effect=RuntimeError('server down'))
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_PARKED_POLL_INITIAL_BACKOFF_SECONDS', 0.01)

    assert await executor._wait_for_parked_request('req-1') is None
    assert (api_status.await_count ==
            recovery_strategy._PARKED_POLL_MAX_CONSECUTIVE_ERRORS)
    executor._cancel_launch_request.assert_awaited_once_with('req-1')


@pytest.mark.asyncio
async def test_await_launch_request_reconnects_on_stream_interruption(
        monkeypatch):
    executor = _make_bare_executor()
    stream_tails = []

    async def fake_stream_and_get(request_id, tail=None, **kwargs):
        stream_tails.append(tail)
        if len(stream_tails) == 1:
            raise exceptions.RequestInterruptedError('stream interrupted')
        return 'result'

    api_status = mock.AsyncMock(return_value=[_request_payload('RUNNING')])
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_STREAM_RECONNECT_INITIAL_BACKOFF_SECONDS', 0.01)

    assert await executor._await_launch_request('req-1') is None
    # Reconnected once, skipping already-relayed lines on the reconnect.
    assert stream_tails == [None, 1]


@pytest.mark.asyncio
async def test_await_launch_request_fetches_result_if_finished_during_blip(
        monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        raise ConnectionError('connection reset')

    api_status = mock.AsyncMock(return_value=[_request_payload('FAILED')])
    get_mock = mock.AsyncMock(side_effect=ValueError('launch failed'))
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'get', get_mock)
    monkeypatch.setattr(recovery_strategy,
                        '_STREAM_RECONNECT_INITIAL_BACKOFF_SECONDS', 0.01)

    # The request finished while the stream was down: the request's own
    # exception is surfaced, not the stream error.
    with pytest.raises(ValueError, match='launch failed'):
        await executor._await_launch_request('req-1')
    get_mock.assert_awaited_once_with('req-1')


@pytest.mark.asyncio
async def test_await_launch_request_raises_when_server_unreachable(monkeypatch):
    executor = _make_bare_executor()

    async def fake_stream_and_get(request_id, **kwargs):
        raise ConnectionError('connection reset')

    api_status = mock.AsyncMock(side_effect=RuntimeError('server down'))
    monkeypatch.setattr(recovery_strategy.sdk_async, 'stream_and_get',
                        fake_stream_and_get)
    monkeypatch.setattr(recovery_strategy.sdk_async, 'api_status', api_status)
    monkeypatch.setattr(recovery_strategy,
                        '_STREAM_RECONNECT_INITIAL_BACKOFF_SECONDS', 0.001)

    # The stream interruption is re-raised as a launch failure when the
    # request status cannot be determined.
    with pytest.raises(ConnectionError):
        await executor._await_launch_request('req-1')
    assert (api_status.await_count ==
            recovery_strategy._STREAM_RECONNECT_MAX_STATUS_FAILURES)


def _make_launch_executor():
    """Build a minimally-initialized StrategyExecutor for _launch tests."""
    executor = _make_bare_executor()
    executor.job_id = 1
    executor.task_id = 0
    executor.pool = None
    executor.cluster_name = 'test-cluster'
    executor.dag = mock.MagicMock()
    executor.file_mounts_blob_id = None
    executor.starting = set()
    lock = asyncio.Lock()
    executor.starting_lock = lock
    executor.starting_signal = asyncio.Condition(lock)
    executor.RETRY_INIT_GAP_SECONDS = 0.01
    executor._cleanup_cluster = mock.MagicMock()
    executor._wait_until_job_starts_on_cluster = mock.AsyncMock(
        return_value=123.45)
    return executor


def _patch_launch_environment(monkeypatch):
    """Patch the scheduler/state/sdk plumbing used by _launch."""
    monkeypatch.setattr(scheduler_module.state, 'get_pool_from_job_id',
                        lambda job_id: None)
    monkeypatch.setattr(scheduler_module.file_content_utils,
                        'get_job_dag_content', lambda job_id: None)
    monkeypatch.setattr(scheduler_module.state, 'scheduler_set_launching_async',
                        mock.AsyncMock())
    monkeypatch.setattr(scheduler_module.state, 'scheduler_set_alive_async',
                        mock.AsyncMock())
    set_restarting = mock.AsyncMock()
    set_backoff_pending = mock.AsyncMock()
    monkeypatch.setattr(recovery_strategy.state, 'set_restarting_async',
                        set_restarting)
    monkeypatch.setattr(recovery_strategy.state, 'set_backoff_pending_async',
                        set_backoff_pending)
    monkeypatch.setattr(recovery_strategy.sdk, 'api_start', mock.MagicMock())
    sdk_launch = mock.MagicMock(return_value='req-123')
    monkeypatch.setattr(recovery_strategy.sdk, 'launch', sdk_launch)
    monkeypatch.setattr(recovery_strategy.global_user_state,
                        'get_handle_from_cluster_name', lambda name: None)
    return types.SimpleNamespace(sdk_launch=sdk_launch,
                                 set_restarting=set_restarting,
                                 set_backoff_pending=set_backoff_pending)


@pytest.mark.asyncio
async def test_launch_parks_and_reattaches_without_teardown(monkeypatch):
    """A parked launch request releases the slot and re-attaches on resume."""
    executor = _make_launch_executor()
    patches = _patch_launch_environment(monkeypatch)

    slot_free_while_parked = asyncio.Event()

    async def fake_wait_for_parked_request(request_id):
        # While parked, the job must not hold a launch slot.
        if executor.job_id not in executor.starting:
            slot_free_while_parked.set()
        return request_id

    executor._wait_for_parked_request = mock.AsyncMock(
        side_effect=fake_wait_for_parked_request)
    executor._await_launch_request = mock.AsyncMock(side_effect=[
        recovery_strategy._LaunchRequestParked(
            'req-123', 'Workload is pending on queue foo.'),
        None,
    ])

    result = await executor._launch(max_retry=1, raise_on_failure=True)

    assert result == 123.45
    # Only one sky.launch was submitted; the second attempt re-attached to
    # the same request.
    assert patches.sdk_launch.call_count == 1
    assert executor._await_launch_request.await_args_list == [
        mock.call('req-123', reattach=False),
        mock.call('req-123', reattach=True),
    ]
    # The launch slot was released while parked.
    assert slot_free_while_parked.is_set()
    # The task was set back to PENDING with the park reason while parked.
    patches.set_backoff_pending.assert_awaited_once()
    assert ('Workload is pending on queue foo.'
            in patches.set_backoff_pending.await_args.kwargs['reason'])
    # The task was set back to STARTING on resume.
    patches.set_restarting.assert_awaited_once_with(1, 0, False)
    # Parking must NOT tear down the (partially provisioned) cluster.
    executor._cleanup_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_launch_parking_does_not_consume_retry_budget(monkeypatch):
    """Park cycles must not count against max_retry."""
    executor = _make_launch_executor()
    patches = _patch_launch_environment(monkeypatch)

    executor._wait_for_parked_request = mock.AsyncMock(return_value='req-123')
    # One park followed by two real failures, with max_retry=2: the park must
    # not consume a retry, so both real failures should be attempted before
    # giving up.
    executor._await_launch_request = mock.AsyncMock(side_effect=[
        recovery_strategy._LaunchRequestParked('req-123', 'pending'),
        RuntimeError('boom'),
        RuntimeError('boom'),
    ])

    with pytest.raises(exceptions.ManagedJobReachedMaxRetriesError):
        await executor._launch(max_retry=2, raise_on_failure=True)

    assert executor._await_launch_request.await_count == 3
    # The first attempt launched, the reattach reused the request, and the
    # third attempt launched fresh (the failure tore the cluster down).
    assert patches.sdk_launch.call_count == 2


@pytest.mark.asyncio
async def test_launch_relaunches_when_parked_request_vanishes(monkeypatch):
    """If the parked request disappears, a fresh launch attempt is made."""
    executor = _make_launch_executor()
    patches = _patch_launch_environment(monkeypatch)

    # Request vanishes while parked.
    executor._wait_for_parked_request = mock.AsyncMock(return_value=None)
    executor._await_launch_request = mock.AsyncMock(side_effect=[
        recovery_strategy._LaunchRequestParked('req-123', 'pending'),
        None,
    ])

    result = await executor._launch(max_retry=1, raise_on_failure=True)

    assert result == 123.45
    # A fresh sky.launch was submitted for the second attempt.
    assert patches.sdk_launch.call_count == 2
    assert executor._await_launch_request.await_args_list[1] == mock.call(
        'req-123', reattach=False)
    # No teardown happened on the park path.
    executor._cleanup_cluster.assert_not_called()


@pytest.mark.asyncio
async def test_launch_cancel_while_parked_cancels_request(monkeypatch):
    """Cancelling the job while parked cancels the outstanding request."""
    executor = _make_launch_executor()
    _patch_launch_environment(monkeypatch)
    executor._cancel_launch_request = mock.AsyncMock()

    parked = asyncio.Event()

    async def wait_forever(request_id):
        parked.set()
        await asyncio.Event().wait()  # Block until cancelled.

    executor._wait_for_parked_request = mock.AsyncMock(side_effect=wait_forever)
    executor._await_launch_request = mock.AsyncMock(
        side_effect=recovery_strategy._LaunchRequestParked(
            'req-123', 'pending'))

    task = asyncio.create_task(
        executor._launch(max_retry=1, raise_on_failure=True))
    await asyncio.wait_for(parked.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    executor._cancel_launch_request.assert_awaited_once_with('req-123')
