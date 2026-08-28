"""Provides the assembly-geometry record through which an acquisition system reports the shape one session's assembly
job takes.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AssemblyGeometry:
    """Describes the shape one session's assembly job takes, as the system that assembles the session reports it.

    Notes:
        An assembly holds two families of arrays at once, and they are counted on different clocks. The frame it
        builds carries one column per assembled value at the height of the reference clock, while the sources it reads
        arrive at whatever clock each of them was sampled on. A fast camera writes far more samples than the reference
        clock the assembly settles on, so the two families diverge by the ratio between those rates and neither one
        bounds the other.

        This record states sample counts alone. What a sample of the assembled frame costs, and what a sample of a
        source costs, are properties of the assembly stage rather than of the data, so the sizing pass applies them.
        The system that donates this record answers only for its own data.
    """

    reference_samples: int
    """The samples the reference clock holds, which is the height of the frame the assembly job builds."""
    source_samples: tuple[int, ...]
    """The samples each source the assembly reads holds on that source's own clock, one entry per source. A source
    the assembly reads through several files of equal height contributes one entry, since those files share a clock.
    """
