# cleanup_old_aggregates.py
import asyncio
from typing import Iterable, List

from aiobotocore.session import get_session


AGG_SUFFIXES = (".daily", ".monthly", ".yearly", ".hourly")


def _has_agg_folder_segment(key: str) -> bool:
    """
    Return True if any path segment looks like '2025.10.17.daily',
    '2025.10.monthly', etc.

    Raw data lives under segments like '2025.10.17', without suffix,
    so it is preserved.
    """
    segments = key.split("/")
    for seg in segments:
        if any(seg.endswith(sfx) for sfx in AGG_SUFFIXES):
            return True
    return False


async def cleanup_old_aggregates(
        bucket: str,
        root_prefix: str,
        *,
        dry_run: bool = True,
) -> None:
    """
    Delete all old aggregate files under <root_prefix> that are inside
    *.daily/*.monthly/*.yearly/*.hourly "folders".

    Example:
      bucket      = "nestlogic-llm-benchmark"
      root_prefix = "data/kdcube/ai-app/cisoteria/accounting/allciso/cisoteria-ciso/"

    - Deletes:
        .../2025.10.17.daily/aggregate.json
        .../2025.10.17.daily/whatever-else.json
      Keeps:
        .../2025.10.17/llm/...raw...
    """
    session = get_session()
    deleted_total = 0
    candidates_total = 0

    async with session.create_client("s3") as s3:
        paginator = s3.get_paginator("list_objects_v2")

        async for page in paginator.paginate(Bucket=bucket, Prefix=root_prefix):
            contents = page.get("Contents", [])
            keys_to_delete: List[str] = []

            for obj in contents:
                key = obj["Key"]
                if _has_agg_folder_segment(key):
                    candidates_total += 1
                    keys_to_delete.append(key)

            if not keys_to_delete:
                continue

            if dry_run:
                for k in keys_to_delete:
                    print(f"[DRY-RUN] Would delete: s3://{bucket}/{k}")
                continue

            # delete_objects allows max 1000 keys at once, but our batch is small per page anyway
            resp = await s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": k} for k in keys_to_delete]},
            )

            deleted = len(resp.get("Deleted", []))
            deleted_total += deleted
            print(f"Deleted {deleted} objects from this page.")

    mode = "DRY RUN" if dry_run else "REAL DELETE"
    print(f"[{mode}] Done. Candidates: {candidates_total}, deleted: {deleted_total}.")


if __name__ == "__main__":
    # Adapt these to your case:
    BUCKET = "nestlogic-llm-benchmark"
    ROOT_PREFIX = "data/kdcube/ai-app/cisoteria/accounting/allciso/cisoteria-ciso/"

    # 1) First run with dry_run=True to see what would be removed
    # 2) Then set dry_run=False when you’re confident
    asyncio.run(
        cleanup_old_aggregates(
            bucket=BUCKET,
            root_prefix=ROOT_PREFIX,
            dry_run=False,   # change to False to actually delete
        )
    )
