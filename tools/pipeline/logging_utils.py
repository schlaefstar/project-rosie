"""Shared logging configuration for pipeline scripts."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

DEFAULT_LOG_NAME = "pipeline.log"
LLM_LOG_NAME = "logs/llm_prompts.log"
PIPELINE_LOGGER_NAME = "pipeline"


def _root_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def log_path(filename: str = DEFAULT_LOG_NAME) -> Path:
    return _root_dir() / filename


def configure_logging(
    *,
    level: int = logging.INFO,
    log_files: Iterable[str] = (DEFAULT_LOG_NAME,),
    console: bool = True,
) -> None:
    logger = logging.getLogger()
    if getattr(logger, "_pipeline_logging_configured", False):
        return

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    for filename in log_files:
        path = log_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    llm_logger = logging.getLogger("tools.pipeline.llm")
    if not getattr(llm_logger, "_pipeline_llm_logging_configured", False):
        llm_path = log_path(LLM_LOG_NAME)
        llm_path.parent.mkdir(parents=True, exist_ok=True)
        llm_handler = logging.FileHandler(llm_path)
        llm_handler.setLevel(logging.DEBUG)
        llm_handler.setFormatter(formatter)
        llm_logger.addHandler(llm_handler)
        llm_logger._pipeline_llm_logging_configured = True  # type: ignore[attr-defined]

    logger._pipeline_logging_configured = True  # type: ignore[attr-defined]


__all__ = ["configure_logging", "log_path", "PIPELINE_LOGGER_NAME"]

