import statistics
import time

import cv2
import numpy as np


def measure(name, operation, iterations=300):
    values = []
    for _ in range(20):
        operation()
    for _ in range(iterations):
        started = time.perf_counter()
        operation()
        values.append((time.perf_counter() - started) * 1000.0)
    print(f"{name}: median={statistics.median(values):.3f} mean={statistics.mean(values):.3f} p95={np.percentile(values, 95):.3f} ms")


rng = np.random.default_rng(42)
a = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
b = rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8)
small_a = cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA)
small_b = cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA)
generated = cv2.addWeighted(small_a, 0.45, small_b, 0.55, 0)
mask = rng.integers(0, 2, (90, 160), dtype=np.uint8) * 255


def resize_pair():
    cv2.resize(a, (160, 90), interpolation=cv2.INTER_AREA)
    cv2.resize(b, (160, 90), interpolation=cv2.INTER_AREA)


def motion_checks():
    first = cv2.cvtColor(small_a, cv2.COLOR_BGR2GRAY)
    second = cv2.cvtColor(small_b, cv2.COLOR_BGR2GRAY)
    motion = cv2.absdiff(first, second)
    cv2.mean(motion)[0]
    cv2.countNonZero(cv2.compare(motion, 2, cv2.CMP_GT))
    difference = cv2.absdiff(small_a, small_b)
    active_motion = cv2.max(cv2.max(difference[..., 0], difference[..., 1]), difference[..., 2])
    cv2.countNonZero(cv2.compare(active_motion, 12, cv2.CMP_GT))


def current_inputs():
    for frame in (small_a, small_b):
        image = frame[..., ::-1].astype(np.float32) / np.float32(255.0)
        image = np.pad(image, ((0, 6), (0, 0), (0, 0)), mode="constant")
        np.ascontiguousarray(np.transpose(image, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)


def direct_inputs():
    for frame in (small_a, small_b):
        tensor = np.zeros((1, 3, 96, 160), dtype=np.float32)
        np.multiply(frame[..., 2], np.float32(1.0 / 255.0), out=tensor[0, 0, :90])
        np.multiply(frame[..., 1], np.float32(1.0 / 255.0), out=tensor[0, 1, :90])
        np.multiply(frame[..., 0], np.float32(1.0 / 255.0), out=tensor[0, 2, :90])


def current_full_post():
    output = cv2.resize(generated, (1280, 720), interpolation=cv2.INTER_LINEAR)
    midpoint = cv2.addWeighted(a, 0.5, b, 0.5, 0.0)
    full_mask = cv2.resize(mask, (1280, 720), interpolation=cv2.INTER_NEAREST)
    cv2.copyTo(midpoint, full_mask, output)


measure("resize pair", resize_pair)
measure("motion checks", motion_checks)
measure("current tensor pair", current_inputs)
measure("direct tensor pair", direct_inputs)
measure("full-resolution post", current_full_post)
