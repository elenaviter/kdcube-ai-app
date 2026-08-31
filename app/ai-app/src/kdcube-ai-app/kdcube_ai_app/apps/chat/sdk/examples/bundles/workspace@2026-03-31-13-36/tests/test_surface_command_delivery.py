from __future__ import annotations

from pathlib import Path


def _scene_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "ui"
        / "scene"
        / "src"
        / "main.tsx"
    ).read_text(encoding="utf-8")


def test_cold_surface_waits_for_listener_readiness_before_first_delivery():
    source = _scene_source()

    assert "SCENE_SURFACE_READY" in source
    assert "if (spec.ready?.type === 'message') return Boolean(surfaceReadyRef.current[spec.alias])" in source
    assert "frameLoadedRef.current[spec.alias]" in source
    assert "if (type === SCENE_SURFACE_READY)" in source
    assert "surfaceReadyRef.current[sourceAlias] = true" in source
    assert "surfaces.forEach((surface) => sceneRuntime.flushSurface(surface))" in source

    component_load = source.split("armFrameFocusRaise(spec.alias)", 1)[1].split(
        "</FloatingWindow>", 1
    )[0]
    assert "window.setTimeout" in component_load
    assert "surfaceReadyRef.current[spec.alias] = true" in component_load
    assert "MESSAGE_READY_FALLBACK_MS = 6000" in source
    assert "MEMORY_STATUS_FALLBACK_MS = 1800" in source
    assert "after_ms: fallbackDelayMs" in component_load
    assert "explicit surface readiness not received; using load fallback" in component_load
    assert "if (spec.ready?.type === 'message')" in component_load
    assert "if (usesStatusReadiness)" in component_load
    fallback = component_load.split("window.setTimeout", 1)[1]
    assert fallback.index("surfaceReadyRef.current[spec.alias] = true") < fallback.index(
        "sceneRuntime.flushSurface(surface)"
    )
