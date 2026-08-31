import asyncio

import pytest

from kdcube_ai_app.apps.chat.emitters import ChatRelayCommunicator
from kdcube_ai_app.infra.orchestration.app import communicator as communicator_module
from kdcube_ai_app.infra.orchestration.app.communicator import ServiceCommunicator


_LISTENER_STOP = object()


class _FakePubSub:
    def __init__(self, owner: "_FakeRedis") -> None:
        self.owner = owner
        self.messages: asyncio.Queue[object] = asyncio.Queue()
        self.subscribed: set[str] = set()
        self.patterns: set[str] = set()
        self.closed = False

    async def subscribe(self, *channels: str) -> None:
        self.subscribed.update(channels)

    async def psubscribe(self, *channels: str) -> None:
        self.patterns.update(channels)

    async def unsubscribe(self, *channels: str) -> None:
        self.subscribed.difference_update(channels)

    async def punsubscribe(self, *channels: str) -> None:
        self.patterns.difference_update(channels)

    async def listen(self):
        while True:
            message = await self.messages.get()
            if message is _LISTENER_STOP:
                return
            yield message

    async def emit(self, payload: dict) -> None:
        await self.messages.put({"type": "message", "data": payload})

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.owner.active_count -= 1
        await self.messages.put(_LISTENER_STOP)


class _FakeRedis:
    _kdcube_shared = True

    def __init__(self) -> None:
        self.pubsubs: list[_FakePubSub] = []
        self.active_count = 0
        self.max_active_count = 0

    def pubsub(self) -> _FakePubSub:
        pubsub = _FakePubSub(self)
        self.pubsubs.append(pubsub)
        self.active_count += 1
        self.max_active_count = max(self.max_active_count, self.active_count)
        return pubsub


@pytest.fixture
async def relay_transport(monkeypatch):
    redis = _FakeRedis()
    monkeypatch.setattr(
        communicator_module,
        "get_async_redis_client",
        lambda _redis_url: redis,
    )
    comm = ServiceCommunicator(
        redis_url="redis://unused",
        orchestrator_identity="test.relay",
    )
    relay = ChatRelayCommunicator(comm=comm)
    try:
        yield relay, comm, redis
    finally:
        await relay.unsubscribe()


@pytest.mark.asyncio
async def test_final_session_release_stops_listener_and_closes_pubsub(relay_transport):
    relay, comm, redis = relay_transport

    await relay.acquire_session_channel("s1", tenant="t", project="p")
    pubsub = redis.pubsubs[-1]
    listener = comm._listen_task

    assert listener is not None
    assert comm.listener_alive()
    assert redis.active_count == 1

    await relay.release_session_channel("s1", tenant="t", project="p")

    assert listener.done()
    assert comm._listen_task is None
    assert comm._pubsub is None
    assert pubsub.closed
    assert redis.active_count == 0
    assert len(redis.pubsubs) == 1
    assert relay._listener_started is False


@pytest.mark.asyncio
async def test_repeated_acquire_release_cycles_do_not_accumulate_pubsubs(relay_transport):
    relay, _comm, redis = relay_transport

    for index in range(5):
        session_id = f"s{index}"
        await relay.acquire_session_channel(session_id, tenant="t", project="p")
        assert redis.active_count == 1

        await relay.release_session_channel(session_id, tenant="t", project="p")
        assert redis.active_count == 0

    assert len(redis.pubsubs) == 5
    assert all(pubsub.closed for pubsub in redis.pubsubs)
    assert redis.max_active_count == 1


@pytest.mark.asyncio
async def test_acquire_after_teardown_restarts_listener_and_delivers(relay_transport):
    relay, comm, redis = relay_transport

    await relay.acquire_session_channel("s1", tenant="t", project="p")
    first_pubsub = redis.pubsubs[-1]
    await relay.release_session_channel("s1", tenant="t", project="p")

    received: list[dict] = []
    delivered = asyncio.Event()

    async def on_message(message: dict) -> None:
        received.append(message)
        delivered.set()

    await relay.acquire_session_channel(
        "s2",
        tenant="t",
        project="p",
        callback=on_message,
    )
    second_pubsub = redis.pubsubs[-1]

    assert second_pubsub is not first_pubsub
    assert not second_pubsub.closed
    assert comm.listener_alive()

    payload = {"event": "chat_step", "data": {"text": "after restart"}}
    await second_pubsub.emit(payload)
    await asyncio.wait_for(delivered.wait(), timeout=1.0)

    assert received == [payload]

    await relay.release_session_channel("s2", tenant="t", project="p")
    assert second_pubsub.closed
    assert redis.active_count == 0


@pytest.mark.asyncio
async def test_transport_stays_open_until_session_and_project_refs_are_released(relay_transport):
    relay, comm, redis = relay_transport

    await relay.acquire_session_channel("s1", tenant="t", project="p")
    await relay.acquire_session_channel("s2", tenant="t", project="p")
    await relay.acquire_project_channel(tenant="t", project="p")
    pubsub = redis.pubsubs[-1]

    await relay.release_session_channel("s1", tenant="t", project="p")
    assert not pubsub.closed
    assert comm.listener_alive()

    await relay.release_session_channel("s2", tenant="t", project="p")
    assert not pubsub.closed
    assert comm.listener_alive()

    await relay.release_project_channel(tenant="t", project="p")
    assert pubsub.closed
    assert redis.active_count == 0


@pytest.mark.asyncio
async def test_legacy_subscription_keeps_transport_open(relay_transport):
    relay, comm, redis = relay_transport

    async def on_message(_message: dict) -> None:
        return None

    await relay.subscribe(on_message)
    await relay.acquire_session_channel("s1", tenant="t", project="p")
    pubsub = redis.pubsubs[-1]

    await relay.release_session_channel("s1", tenant="t", project="p")

    assert not pubsub.closed
    assert comm.listener_alive()
    assert comm._subscribed_channels == ["test.relay.chat.events"]

    await relay.unsubscribe()
    assert pubsub.closed
    assert redis.active_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "project"])
async def test_reconcile_to_empty_releases_transport(relay_transport, scope):
    relay, comm, redis = relay_transport

    if scope == "session":
        await relay.acquire_session_channel("s1", tenant="t", project="p")
    else:
        await relay.acquire_project_channel(tenant="t", project="p")
    pubsub = redis.pubsubs[-1]

    if scope == "session":
        await relay.reconcile_sessions({}, reason="test")
    else:
        await relay.reconcile_project_channels({}, reason="test")

    assert pubsub.closed
    assert comm._pubsub is None
    assert not comm.listener_alive()
    assert redis.active_count == 0
    assert len(redis.pubsubs) == 1
