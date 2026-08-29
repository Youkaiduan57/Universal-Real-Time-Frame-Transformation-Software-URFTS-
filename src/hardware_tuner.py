"""OpenCV thread-count benchmarking."""

from __future__ import annotations

import logging
import statistics
import time

import cv2

from config import OPENCV_THREAD_CANDIDATES

logger = logging.getLogger(__name__)


class HardwareTuner:
    """Benchmark candidate OpenCV thread counts and select the best one."""

    def __init__(
        self,
        thread_candidates: tuple[int, ...] = OPENCV_THREAD_CANDIDATES,
        rounds: int = 3,
    ):
        if not thread_candidates:
            raise ValueError("thread_candidates must not be empty.")

        self.thread_candidates = thread_candidates
        self.rounds = rounds

    @staticmethod
    def _percentile(values, percentile_value):
        ordered_values = sorted(values)

        index = round(
            (percentile_value / 100)
            * (len(ordered_values) - 1)
        )

        return ordered_values[index]

    def _benchmark_thread_count(
        self,
        test_frame,
        output_width,
        output_height,
        thread_count,
    ):
        if thread_count <= 0:
            raise ValueError("thread_count must be greater than zero.")

        cv2.setNumThreads(thread_count)

        for _ in range(30):
            cv2.resize(
                test_frame,
                (output_width, output_height),
                interpolation=cv2.INTER_CUBIC,
            )

        round_scores = []

        for _ in range(self.rounds):
            timings = []

            for _ in range(150):
                start = time.perf_counter()

                cv2.resize(
                    test_frame,
                    (output_width, output_height),
                    interpolation=cv2.INTER_CUBIC,
                )

                end = time.perf_counter()

                timings.append(
                    (end - start) * 1000
                )

            median_ms = statistics.median(timings)
            percentile_95_ms = self._percentile(
                timings,
                95,
            )

            round_score = (
                median_ms
                + (0.5 * percentile_95_ms)
            )

            round_scores.append(round_score)

        final_score = statistics.median(round_scores)

        return final_score

    def find_best_opencv_threads(
        self,
        test_frame,
        output_width,
        output_height,
    ):
        logger.info("Testing OpenCV thread configurations...")

        results = {}

        for thread_count in self.thread_candidates:
            score = self._benchmark_thread_count(
                test_frame=test_frame,
                output_width=output_width,
                output_height=output_height,
                thread_count=thread_count,
            )

            results[thread_count] = score

            logger.info("Threads %s: final score %.3f", thread_count, score)

        best_thread_count = min(
            results,
            key=results.get,
        )

        cv2.setNumThreads(best_thread_count)

        logger.info("Selected OpenCV threads: %s", best_thread_count)

        return best_thread_count