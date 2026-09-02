#!/usr/bin/env python3
"""Resolve an installed KDCube application's browser surface."""

from __future__ import annotations

import argparse
from pathlib import Path

from kdcube_cli.control import (
    ApplicationRef,
    LocalDeploymentTarget,
    SurfaceKind,
    SurfaceSelector,
    select_local_target,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--bundle-id", default="connection-hub@1-0")
    parser.add_argument("--widget", default="connections_settings")
    args = parser.parse_args()

    reference = select_local_target(args.workdir)
    target = LocalDeploymentTarget(reference)
    surface = target.resolve_surface(
        ApplicationRef(args.bundle_id),
        SurfaceSelector(kind=SurfaceKind.WIDGET, alias=args.widget),
    )
    print(surface.url)


if __name__ == "__main__":
    main()
