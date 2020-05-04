
import os


def asset_path(relative_path: str) -> str:

    return os.path.join(
        os.path.dirname(__file__),
        *relative_path.split("/")
    )
