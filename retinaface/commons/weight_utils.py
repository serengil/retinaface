import os
from pathlib import Path
from typing import List, Optional

import gdown
import requests  # type: ignore[import-untyped]

from retinaface.commons.logger import Logger

logger = Logger(module="retinaface/commons/weight_utils.py")


def download_weights_if_necessary(file_name: str, source_urls: List[str]) -> str:
    """
    Download the pre-trained weights from external sources if not downloaded yet.
    Args:
        file_name (str): target file name with extension
        source_urls (list of str): source urls to be downloaded. the urls are tried in
            order and the next one is used as a backup when downloading from the
            previous one fails
    Returns:
        target_file (str): exact path for the target file
    """
    home = str(os.getenv("DEEPFACE_HOME", default=str(Path.home())))
    weights_dir = os.path.join(home, ".deepface", "weights")
    target_file = os.path.join(weights_dir, file_name)

    if os.path.isfile(target_file):
        logger.debug(f"{file_name} is already available at {target_file}")
        return target_file

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
        logger.info(f"Directory {weights_dir} created")

    last_err: Optional[Exception] = None
    for idx, url in enumerate(source_urls):
        try:
            logger.info(f"{file_name} will be downloaded from the url {url}")
            ensure_source_is_reachable(url)
            gdown.download(url, target_file, quiet=False)
            if not os.path.isfile(target_file):
                raise ValueError(f"{file_name} could not be downloaded from {url}")
            last_err = None
            break
        except Exception as err:  # pylint: disable=broad-except
            last_err = err
            # do not let a partially downloaded file be used by the next attempt
            if os.path.isfile(target_file):
                os.remove(target_file)
            if idx < len(source_urls) - 1:
                logger.warn(
                    f"Downloading {file_name} from {url} failed ({err}). "
                    f"Trying the backup source {source_urls[idx + 1]}..."
                )

    if last_err is not None:
        raise ValueError(
            f"Pre-trained weight could not be loaded! An exception occurred while "
            f"downloading {file_name} from {', '.join(source_urls)}. "
            f"You might try to download it manually and copy it to {target_file}."
        ) from last_err

    return target_file


def ensure_source_is_reachable(url: str) -> None:
    """
    gdown does not raise for http errors of non google drive urls, and saves the error
    page (e.g. "Not Found") as if it were the weight file. check the status in advance
    to fail fast, so that the backup source can be tried.
    Args:
        url (str): source url to be downloaded
    """
    response = requests.head(url, allow_redirects=True, timeout=30)
    # some servers do not allow head requests, let gdown decide for them
    if response.status_code >= 400 and response.status_code != 405:
        raise ValueError(f"{url} responded with status code {response.status_code}")
