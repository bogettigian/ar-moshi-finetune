from __future__ import annotations

import logging
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)


class S3Store:
    def __init__(self, bucket: str | None, prefix: str = "") -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = None
        if not bucket:
            return

        session = boto3.Session()
        identity = session.client("sts").get_caller_identity()
        logger.info("aws account %s as %s", identity["Account"], identity["Arn"])
        self._client = session.client("s3")

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def upload(self, local: Path, key: str) -> None:
        if not self._client:
            return
        self._client.upload_file(str(local), self.bucket, self._key(key))
        logger.debug("uploaded s3://%s/%s", self.bucket, self._key(key))

    def download(self, key: str, local: Path) -> None:
        if not self._client:
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, self._key(key), str(local))

    def list_keys(self, subprefix: str = "") -> set[str]:
        if not self._client:
            return set()
        full = self._key(subprefix)
        keys: set[str] = set()
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if self.prefix:
                    key = key[len(self.prefix) + 1 :]
                keys.add(key)
        logger.info("s3://%s/%s: %d existing object(s)", self.bucket, full, len(keys))
        return keys
