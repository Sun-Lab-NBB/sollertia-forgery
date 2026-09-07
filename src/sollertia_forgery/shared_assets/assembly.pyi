from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class AssemblyGeometry:
    reference_samples: int
    source_samples: tuple[int, ...]
