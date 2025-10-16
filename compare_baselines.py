"""
Comparison script for cage generation baselines.

Tests RandomWalk, MCTS, and Constructive generators on (3,5)-cage.
"""

from ai.cage import RandomWalkGenerator, MCTSGenerator, ConstructiveGenerator
import statistics
import time

def test_generator(GeneratorClass, name, k, g, trials=10, max_steps=5000):
    """Test a generator class and return statistics."""
    print(f"\n{name}:")
    print(f"  Testing {trials} trials, max {max_steps} steps each...")
    
    results = []
    start_time = time.time()
    
    for trial in range(trials):
        gen = GeneratorClass(k, g)
        for i in range(max_steps):
            gen.step()
            if gen.is_complete:
                results.append((i+1, gen.elapsed_time()))
                break
        else:
            results.append((None, None))
    
    total_time = time.time() - start_time
    
    successes = [r for r in results if r[0] is not None]
    failures = len(results) - len(successes)
    
    print(f"  Successes: {len(successes)}/{trials} ({len(successes)*100//trials}%)")
    print(f"  Failures: {failures}/{trials}")
    
    if successes:
        steps = [r[0] for r in successes]
        times = [r[1] for r in successes]
        print(f"  Steps: min={min(steps)}, max={max(steps)}, avg={statistics.mean(steps):.0f}, median={statistics.median(steps):.0f}")
        print(f"  Time per success: avg={statistics.mean(times)*1000:.1f}ms, median={statistics.median(times)*1000:.1f}ms")
    
    print(f"  Total benchmark time: {total_time:.2f}s")
    
    return {
        'name': name,
        'successes': len(successes),
        'trials': trials,
        'success_rate': len(successes) / trials if trials > 0 else 0,
        'avg_steps': statistics.mean(steps) if successes else None,
        'median_steps': statistics.median(steps) if successes else None,
        'avg_time_ms': statistics.mean(times) * 1000 if successes else None,
    }


if __name__ == '__main__':
    k, g = 3, 5
    print(f"=" * 60)
    print(f"Cage Generation Baseline Comparison: ({k},{g})-cage")
    print(f"=" * 60)
    
    # Test all three generators
    results = []
    
    results.append(test_generator(
        RandomWalkGenerator,
        "Baseline 1: Random Walk",
        k, g,
        trials=10,
        max_steps=10000
    ))
    
    results.append(test_generator(
        MCTSGenerator,
        "Baseline 2: MCTS (Tree Search)",
        k, g,
        trials=10,
        max_steps=5000
    ))
    
    results.append(test_generator(
        ConstructiveGenerator,
        "Baseline 3: Constructive (Smart Start)",
        k, g,
        trials=20,
        max_steps=2000
    ))
    
    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  Success Rate: {r['success_rate']*100:.0f}% ({r['successes']}/{r['trials']})")
        if r['avg_steps']:
            print(f"  Median Steps: {r['median_steps']:.0f}")
            print(f"  Median Time: {r['avg_time_ms']:.1f}ms")
    
    # Find best
    successful = [r for r in results if r['success_rate'] > 0]
    if successful:
        best = min(successful, key=lambda x: x['median_steps'] or float('inf'))
        print(f"\n🏆 Best performer: {best['name']}")
        print(f"   {best['success_rate']*100:.0f}% success, ~{best['median_steps']:.0f} steps median")
