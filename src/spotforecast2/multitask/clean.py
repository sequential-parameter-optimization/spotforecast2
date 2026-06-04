# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Cache-cleaning task — removes all cached pipeline artefacts.

Re-exports ``execute_clean`` from the safe package.  Redefines ``CleanTask``
as a thin subclass of the sf2 ``BaseTask`` and preserves the historical
``show=True`` default for this package.
"""

from pathlib import Path
from typing import Any, Dict, Optional

# Re-export execute_clean from the safe package.
from spotforecast2_safe.multitask.clean import execute_clean  # noqa: F401

from spotforecast2.multitask.base import BaseTask

__all__ = ["execute_clean", "CleanTask"]


class CleanTask(BaseTask):
    """Cache-cleaning task — removes all cached data from the pipeline cache.

    CleanTask deletes the entire cache directory configured for the pipeline,
    including saved models, tuning results, and any other cached artefacts
    written by training or tuning tasks.

    Unlike training or prediction tasks, CleanTask does not require
    ``prepare_data()`` to be called before ``run()``.  It operates purely on
    the file system and can be used as a standalone reset mechanism between
    experiments or deployments.

    Passing ``dry_run=True`` to ``run()`` reports what would be deleted
    without actually removing anything, which is useful for inspecting cache
    contents before committing to removal.

    Examples:
        ```{python}
        from spotforecast2.multitask import CleanTask

        task = CleanTask(data_frame_name="demo10")
        print(f"Task: {task.TASK}")
        print(f"Task name: {task._task_name}")
        ```
    """

    _task_name = "clean"

    def run(
        self,
        dry_run: bool = False,
        cache_home: Optional[Path] = None,
        show: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Remove all cached data from the pipeline cache directory.

        Does not require ``prepare_data()`` to be called first.

        Args:
            dry_run: If ``True``, report what would be deleted without
                actually removing anything.  Useful for inspecting the
                cache before committing to removal.
            cache_home: Override the directory to clean.  ``None`` uses
                the directory configured on this instance via
                ``get_cache_home()``.
            show: Accepted for API consistency with other tasks.  Not used
                by the clean task.

        Returns:
            Dict with keys:
                status: ``"success"``, ``"dry_run"``, or ``"empty"``.
                cache_dir: The Path targeted for cleaning.
                deleted_items: Names of top-level items removed (or that
                    would have been removed in ``dry_run`` mode).

        Raises:
            RuntimeError: If the cache directory cannot be removed due to
                a permissions error or OS-level failure.

        Examples:
            ```{python}
            import tempfile
            from pathlib import Path
            from spotforecast2.multitask import CleanTask

            with tempfile.TemporaryDirectory() as tmp:
                task = CleanTask(cache_home=Path(tmp) / "test_cache")
                result = task.run(dry_run=True)
                print(result["status"])
            ```
        """
        return execute_clean(self, cache_home=cache_home, dry_run=dry_run)
