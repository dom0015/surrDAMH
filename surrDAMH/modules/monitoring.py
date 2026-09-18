#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for writing sampler outputs to CSV files.

This module intentionally stays separate from ``modules/algorithms.py`` so that
sampling logic is not mixed with file-management details.

TODO:
- If post-processing/output grows further, consider separating the low-level CSV
  writer from the higher-level sampling-output organization.
- Consider a more explicit output schema object instead of free-form row lists.
"""

from __future__ import annotations

import csv
import os
from typing import Any


class CsvWriter:
    """Simple CSV writer wrapper used by sampling output monitors.

    ``header``, when given, is written as the first row of the freshly opened file
    (output format v2: every CSV carries a header row).
    """

    def __init__(self, dirname: str, basename: str, header: list[str] | None = None) -> None:
        path = os.path.join(dirname, basename)
        os.makedirs(dirname, exist_ok=True)
        # buffering=1 == line-buffered (G6): every completed row reaches the file system as
        # soon as it is written, so a run that is still going (or was killed) can be inspected
        # and post-processed. Content is unchanged; the cost is one write() syscall per row.
        self._file = open(path, "w", buffering=1)
        self._writer = csv.writer(self._file)
        if header is not None:
            self._writer.writerow(header)

    def writerow(self, row: list[Any]) -> None:
        self._writer.writerow(row)

    def close_file(self) -> None:
        self._file.close()


class SamplingOutputMonitor:
    """
    Lazily creates CSV writers for different categories of sampling output.

    Each ``data_name`` gets its own file in:
    ``<output_dir>/sampling_output/<data_name>/<stage.name>/<basename>``.

    A header registered with :meth:`set_header` is written as the first row of that file
    when it is created. Files are still created LAZILY, on the first row that is actually
    written, so registering a header does not create an output file for a stage (or a
    ``data_name``) that never writes anything.
    """

    def __init__(self, output_dir: str, stage: Any, basename: str) -> None:
        self.output_dir = output_dir
        self.stage = stage
        self.basename = basename
        self.writers: dict[str, CsvWriter] = {}
        self.headers: dict[str, list[str]] = {}

    def set_header(self, data_name: str, header: list[str]) -> None:
        """Register the header row written when the file for ``data_name`` is created."""
        self.headers[data_name] = list(header)

    def add_writer(self, data_name: str) -> None:
        dirname = os.path.join(self.output_dir, "sampling_output", data_name, self.stage.name)
        writer = CsvWriter(dirname=dirname, basename=self.basename, header=self.headers.get(data_name))
        self.writers[data_name] = writer

    def __call__(self, data_name: str, row: list[Any], condition: bool = True) -> None:
        if not condition:
            return
        if data_name not in self.writers:
            self.add_writer(data_name)
        self.writers[data_name].writerow(row)

    def close_files(self) -> None:
        for writer in self.writers.values():
            writer.close_file()


# Backward-friendly aliases in case old names are still referenced internally.
Writer = CsvWriter
Monitor = SamplingOutputMonitor


__all__ = ["CsvWriter", "SamplingOutputMonitor", "Writer", "Monitor"]
