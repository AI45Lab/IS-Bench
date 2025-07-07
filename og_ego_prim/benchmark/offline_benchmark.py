from og_ego_prim.benchmark.base_benchmark import Benchmark


__all__ = ['OFFLINE_BENCHMARKS']


class OfflineBenchmark(Benchmark):
    pass


class OfflineBehaviorBenchmark(OfflineBenchmark):
    pass


OFFLINE_BENCHMARKS = {
    'BehaviorTask': OfflineBehaviorBenchmark,
}