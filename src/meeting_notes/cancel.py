"""The one exception a pipeline raises when it was asked to stop."""

from __future__ import annotations


class PipelineCancelled(Exception):
    """Raised between steps (never mid-step) when the caller's ``should_cancel`` turned true.

    A speech model is mid-way through a C++ call and cannot be interrupted; what CAN be done is to stop at the
    next boundary, which for a long recording is the next part (a few minutes), not the end of the meeting.
    """
