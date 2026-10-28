import os
import time
import statistics
import requests

URL = os.getenv("BENCHMARK_URL", "http://localhost:8082/predict")
IMAGE = "test_digit.png"

WARMUP = 10
RUNS = 100


def send_request():
    with open(IMAGE, "rb") as f:
        files = {
            "file": ("test_digit.png", f, "image/png")
        }

        start = time.perf_counter()

        response = requests.post(
            URL,
            files=files,
            timeout=10
        )

        total_ms = (
            time.perf_counter() - start
        ) * 1000

    response.raise_for_status()

    return response.json(), total_ms


print("=" * 60)
print("K3S MNIST INFERENCE BENCHMARK")
print("=" * 60)

print(f"\nWarm-up requests: {WARMUP}")

for _ in range(WARMUP):
    result, _ = send_request()

print("Warm-up complete.")

api_latencies = []
end_to_end_latencies = []

print(f"\nBenchmark requests: {RUNS}")

for i in range(RUNS):

    result, total_ms = send_request()

    api_latencies.append(
        result["latency_ms"]
    )

    end_to_end_latencies.append(
        total_ms
    )

    if (i + 1) % 20 == 0:
        print(
            f"Completed {i + 1}/{RUNS}"
        )


def percentile(values, percentile_value):

    values = sorted(values)

    index = int(
        percentile_value / 100
        * (len(values) - 1)
    )

    return values[index]


print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print("\nModel:")
print(result["model_version"])

print("\nPrediction:")
print(result["prediction"])

print("\nONNX inference latency")

print(
    "Mean   :",
    round(statistics.mean(api_latencies), 3),
    "ms"
)

print(
    "Median :",
    round(statistics.median(api_latencies), 3),
    "ms"
)

print(
    "P95    :",
    round(percentile(api_latencies, 95), 3),
    "ms"
)

print(
    "Min    :",
    round(min(api_latencies), 3),
    "ms"
)

print(
    "Max    :",
    round(max(api_latencies), 3),
    "ms"
)


print("\nEnd-to-end HTTP latency")

print(
    "Mean   :",
    round(statistics.mean(end_to_end_latencies), 3),
    "ms"
)

print(
    "Median :",
    round(statistics.median(end_to_end_latencies), 3),
    "ms"
)

print(
    "P95    :",
    round(percentile(end_to_end_latencies, 95), 3),
    "ms"
)

print(
    "Min    :",
    round(min(end_to_end_latencies), 3),
    "ms"
)

print(
    "Max    :",
    round(max(end_to_end_latencies), 3),
    "ms"
)

print("=" * 60)