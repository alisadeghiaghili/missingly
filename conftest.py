# Root conftest.py — prevent pytest from collecting source files as test modules.
# The missingly/ package contains functions whose parameters happen to match
# pytest fixture names (e.g. ``frame``). Adding the package dir to
# collect_ignore_glob ensures pytest never tries to collect them.

collect_ignore_glob = ["missingly/*.py"]
