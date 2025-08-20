# data_model/processing.py
import functools
import os
from typing import Any, Optional, Dict

from pydantic import BaseModel, Field
from datetime import datetime

class ProcessingResult(BaseModel):
    timestamp_started: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="When processing started"
    )
    timestamp_completed: Optional[str] = Field(None)
    duration_seconds: Optional[float] = Field(None)

def record_timing(func):
    """
    Decorator for any function that returns either:
      - a single ProcessingResult (or subclass), or
      - a list of ProcessingResult (or subclasses).
    It will:
      1. Record `start = now()` before calling the function.
      2. Call the function.
      3. Record `end = now()` after it returns.
      4. Compute `duration = (end - start).total_seconds()`.
      5. Overwrite the returned object's .timestamp_started,
         .timestamp_completed, and .duration_seconds fields.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_dt = datetime.now()
        result = func(*args, **kwargs)
        end_dt = datetime.now()
        duration = (end_dt - start_dt).total_seconds()
        started_iso = start_dt.isoformat()
        completed_iso = end_dt.isoformat()

        # Helper to patch a single ProcessingResult instance:
        def _patch_one(obj: ProcessingResult):
            obj.timestamp_started = started_iso
            obj.timestamp_completed = completed_iso
            obj.duration_seconds = duration

        # If the function returned a list of ProcessingResult subclasses:
        if isinstance(result, list):
            for item in result:
                if isinstance(item, ProcessingResult):
                    _patch_one(item)
        # If it returned exactly one ProcessingResult subclass:
        elif isinstance(result, ProcessingResult):
            _patch_one(result)

        return result

    return wrapper


class DataSourceExtractionResult(ProcessingResult):
    """Result of data extraction from a source."""
    content: str
    metadata: Dict = Field(default_factory=dict)

    def to_markdown(self, output_path: str) -> str:
        """Save content as markdown and return the path."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.content)
        return output_path
