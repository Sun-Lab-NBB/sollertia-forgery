from pathlib import Path

from ataraxis_base_utilities import console


def forge_dataset(processed_data_root: Path, dataset_precursor: Path) -> None:
    """Forges an analysis dataset by expanding the dataset precursor to include additional data."""
    # Extracts the list of animals to process.
    animals = [candidate for candidate in dataset_precursor.glob("*") if candidate.is_dir()]

    # Processes each animal's data sequentially.
    for animal in animals:
        # Extracts the list of sessions to process for each animal.
        sessions = [candidate for candidate in animal.glob("*") if candidate.is_dir()]

        # Extracts the necessary data for each session
        for session in sessions:
            # Searches the processed data root for the specific session's data directory.
            session_candidates = [
                candidate for candidate in processed_data_root.rglob(f"**/{animal.stem}/{session.stem}")
            ]
            if len(session_candidates) != 1:
                message = (
                    f"Unable to forge the analysis dataset. More than a single data source directory found for the "
                    f"session {session.stem} performed by the animal {animal.stem}. The processed data root directory "
                    f"must only contain a single directory for each unique animal and session combination."
                )
                console.error(message=message, error=ValueError)
                # Fallback to appease mypy, should not be reachable
                raise ValueError(message)  # pragma: no cover

            # Extracts the path to the session's data directory.
            session_directory = session_candidates.pop()
