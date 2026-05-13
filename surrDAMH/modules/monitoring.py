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
    """Simple CSV writer wrapper used by sampling output monitors."""

    def __init__(self, dirname: str, basename: str) -> None:
        path = os.path.join(dirname, basename)
        os.makedirs(dirname, exist_ok=True)
        self._file = open(path, "w")
        self._writer = csv.writer(self._file)

    def writerow(self, row: list[Any]) -> None:
        self._writer.writerow(row)

    def close_file(self) -> None:
        self._file.close()


class SamplingOutputMonitor:
    """
    Lazily creates CSV writers for different categories of sampling output.

    Each ``data_name`` gets its own file in:
    ``<output_dir>/sampling_output/<data_name>/<stage.name>/<basename>``.
    """

    def __init__(self, output_dir: str, stage: Any, basename: str) -> None:
        self.output_dir = output_dir
        self.stage = stage
        self.basename = basename
        self.writers: dict[str, CsvWriter] = {}

    def add_writer(self, data_name: str) -> None:
        dirname = os.path.join(self.output_dir, "sampling_output", data_name, self.stage.name)
        writer = CsvWriter(dirname=dirname, basename=self.basename)
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
