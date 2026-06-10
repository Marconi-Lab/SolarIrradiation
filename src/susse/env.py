"""Process-wide environment loading for notebooks and scripts.

Single entry point for the ``.env``-loading idiom every notebook needs.
Notebooks (and any ad-hoc script) call :func:`load_project_env` once
during setup, after which the standard ``os.environ`` carries any
credentials defined in the project's ``.env`` file — ``CAMS_EMAIL``,
``WANDB_API_KEY``, ``EARTHDATA_USERNAME`` / ``EARTHDATA_PASSWORD``,
etc.

Library code that needs credentials (e.g. :class:`CamsClient`,
:func:`_Earthdata.from_env`) calls :func:`dotenv.load_dotenv` directly
at the point of need; notebooks call this helper at the top of their
setup cell so subsequent reads of ``os.environ`` work regardless of
whether the user exported the variable in their shell or wrote it
into ``.env``.
"""

from __future__ import annotations

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load ``.env`` from the working directory or any parent into ``os.environ``.

    Wraps :func:`dotenv.load_dotenv` so notebooks have a single
    project-level import to remember. Idempotent: existing
    ``os.environ`` values are preserved (``load_dotenv`` defaults to
    ``override=False``), so a shell-exported variable always wins
    over a ``.env`` entry.
    """
    load_dotenv()
