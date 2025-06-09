import argparse
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

from shardcast import ClientNode

POLL_INTERVAL = 5
WINDOW_SIZE = int(os.environ.get("SHARDCAST_WINDOW_SIZE", 2))  # Default window size is 2
logger = logging.getLogger(__name__)


def main(servers: list[str], output_dir: Path, versions_to_keep: int = -1, backlog_version: int = -1, window_size: int = WINDOW_SIZE):
    """
    Download the latest versions of the model from the servers and delete expired versions.
    Versions will be saved as v{version}.safetensors in the output directory.

    Args:
        servers: list of servers to download from
        output_dir: directory to save the downloaded files
        versions_to_keep: number of versions to keep
        backlog_version: version to attempt to get first
        window_size: number of latest versions to always try to prefetch
    """
    client = ClientNode(servers, str(output_dir))
    logged_versions: set[int] = set()  # Track versions we've logged messages for
    missing_versions: set[int] = set()  # Track missing versions to retry

    while True:
        available_versions = sorted([int(x[1:]) for x in client.list_available_versions().keys()])
        if not available_versions:
            if -1 not in logged_versions:
                logger.warning("No versions available")
                logged_versions.add(-1)
            time.sleep(POLL_INTERVAL)
            continue
        # Sliding window: always try to fetch the latest window_size versions
        target_versions = set(available_versions[-window_size:])
        if backlog_version != -1:
            target_versions.add(backlog_version)
            backlog_version += 1
        # Add missing versions to the target set
        target_versions.update(missing_versions)
        for version in sorted(target_versions):
            safetensors_filepath = output_dir / f"step_{version}/model.safetensors"
            if version not in available_versions:
                if version not in logged_versions:
                    logger.warning(f"Version {version} not found on server")
                    logged_versions.add(version)
                missing_versions.add(version)
                continue
            if safetensors_filepath.exists():
                if version not in logged_versions:
                    logger.info(f"Version {version} already exists locally")
                    logged_versions.add(version)
                if version in missing_versions:
                    missing_versions.remove(version)
                continue
            logger.info(f"Downloading version {version}")
            start = time.time()
            try:
                filepath = client.download_version(f"v{version}", str(safetensors_filepath))
                logger.info(f"Downloaded version {version} in {time.time() - start} seconds")
                if filepath is not None:
                    (output_dir / f"step_{version}/stable").touch()
                    if versions_to_keep != -1:
                        try:
                            logger.info(f"Deleting expired version {version - versions_to_keep}")
                            expired_version = output_dir / f"step_{version - versions_to_keep}/model.safetensors"
                            expired_version.unlink()
                            (output_dir / f"step_{version - versions_to_keep}/stable").unlink()
                        except FileNotFoundError:
                            logger.warning(f"Expired version {version - versions_to_keep} not found")
                        except Exception as e:
                            logger.warning(f"Error deleting expired version {version - versions_to_keep}: {e}")
                    if version in missing_versions:
                        missing_versions.remove(version)
            except Exception as e:
                logger.warning(f"Error downloading version {version}: {e}")
                missing_versions.add(version)
        logger.info(f"Current available_versions: {available_versions}, target_versions: {sorted(target_versions)}, missing_versions: {sorted(missing_versions)}")
        time.sleep(POLL_INTERVAL)


def run_main_bg(servers: list[str], output_dir: Path, versions_to_keep: int = -1, backlog_version: int = -1, window_size: int = WINDOW_SIZE) -> mp.Process:
    """
    Run the main function in a background process.

    Args:
        servers: list of servers to download from
        output_dir: directory to save the downloaded files
        versions_to_keep: number of versions to keep
        backlog_version: version to attempt to get first
        window_size: number of latest versions to always try to prefetch

    Returns:
        mp.Process: The created process running the main function
    """
    output_dir = Path(output_dir)
    logger.info(f"Starting shardcast downloader with {servers=}, {output_dir=}, {versions_to_keep=}, {backlog_version=}, {window_size=}")
    process = mp.Process(target=main, args=(servers, output_dir, versions_to_keep, backlog_version, window_size))
    process.start()
    return process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--servers", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--versions-to-keep", type=int, default=-1)
    parser.add_argument("--backlog-version", type=int, default=-1)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()
    servers = args.servers.split(",")
    main(servers, Path(args.output_dir), args.versions_to_keep, args.backlog_version, args.window_size)
