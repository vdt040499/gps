"""Import this module before ``torch`` / ``transformers`` to hide noisy third-party FutureWarnings."""

import warnings


def _apply() -> None:
    # Message-based: works even when the warning is attributed to torch vs transformers
    for pat in (
        r".*_register_pytree_node.*",
        r".*resume_download.*",
    ):
        warnings.filterwarnings("ignore", category=FutureWarning, message=pat)
    for mod in (r"^transformers", r"^huggingface_hub", r"^torch"):
        warnings.filterwarnings("ignore", category=FutureWarning, module=mod)


_apply()
