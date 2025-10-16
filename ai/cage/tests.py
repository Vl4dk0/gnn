"""
Test suite for cage graph generators.

This module provides reusable test cases to evaluate different generators
on various cage graph targets without recreating test code each time.
"""

import sys
import time
from typing import Dict, List, Tuple
import numpy as np

from ai.cage.random_walk import RandomWalkGenerator
from ai.cage.bruteforce import BruteforceGenerator
from ai.cage.astar import AStarGenerator
from utils.graph_utils import is_valid_cage, compute_girth, moore_bound


class CageTest:
    """Test configuration for a specific cage graph."""
    
    def __init__(self, k: int, g: int, name: str = None): # type: ignore
        self.k = k
        self.g = g
        self.name = name or f"({k},{g})-cage"
        self.moore_bound = moore_bound(k, g)
        
    def __repr__(self):
        return f"CageTest({self.name}, Moore={self.moore_bound})"


# Standard test suite
SMALL_CAGES = [
    CageTest(3, 5, "Petersen graph"),  # n=10
    CageTest(3, 6, "Heawood graph"),   # n=14
    CageTest(4, 5, "(4,5)-cage"),      # n=19
]

MEDIUM_CAGES = [
    CageTest(3, 7, "McGee graph"),     # n=24
    CageTest(3, 8, "Tutte-Coxeter"),   # n=30
    CageTest(5, 5, "(5,5)-cage"),      # n=30
]

LARGE_CAGES = [
    CageTest(3, 9, "(3,9)-cage"),      # n=58
    CageTest(4, 6, "(4,6)-cage"),      # n=26
]


class GeneratorTestResult:
    """Results from testing a generator on a specific cage."""
    
    def __init__(self, test: CageTest, generator_name: str):
        self.test = test
        self.generator_name = generator_name
        self.success = False
        self.steps = 0
        self.final_n = 0
        self.final_girth = 0
        self.time_seconds = 0.0
        self.error = None
        
    def __repr__(self):
        if self.success:
            return f"✅ {self.generator_name} on {self.test.name}: {self.steps} steps, {self.time_seconds:.2f}s"
        else:
            reason = self.error or f"Failed at n={self.final_n}, g={self.final_girth}"
            return f"❌ {self.generator_name} on {self.test.name}: {reason}"


def run_single_test(generator_class, test: CageTest, max_steps: int = 10000, 
                   verbose: bool = False) -> GeneratorTestResult:
    """
    Run a single test case with the given generator.
    
    Args:
        generator_class: Generator class to instantiate
        test: CageTest configuration
        max_steps: Maximum steps before giving up
        verbose: Print progress updates
        
    Returns:
        GeneratorTestResult with outcome
    """
    result = GeneratorTestResult(test, generator_class.__name__)
    
    try:
        # Create generator
        gen = generator_class(test.k, test.g)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Testing {generator_class.__name__} on {test.name}")
            print(f"Target: k={test.k}, g={test.g}, Moore bound={test.moore_bound}")
            print(f"Starting with n={gen.graph.number_of_nodes()}")
            print(f"{'='*60}")
        
        start_time = time.time()
        
        # Run generation
        for step in range(max_steps):
            if gen.is_complete:
                result.success = True
                result.steps = step
                result.final_n = gen.graph.number_of_nodes()
                result.final_girth = compute_girth(gen.graph)  # type: ignore
                result.time_seconds = time.time() - start_time
                
                if verbose:
                    print(f"\n✅ SUCCESS at step {step}")
                    print(f"   Final graph: n={result.final_n}, g={result.final_girth}")
                    print(f"   Time: {result.time_seconds:.2f}s")
                    
                    # Verify it's actually valid
                    if is_valid_cage(gen.graph, test.k, test.g):
                        print(f"   ✓ Verified as valid ({test.k},{test.g})-cage")
                    else:
                        print(f"   ⚠️  WARNING: Failed validation check!")
                        result.error = "Failed validation"  # type: ignore
                        result.success = False
                
                break
                
            gen.step()
            
            # Progress updates
            if verbose and step % 1000 == 0 and step > 0:
                n = gen.graph.number_of_nodes()
                m = gen.graph.number_of_edges()
                current_g = compute_girth(gen.graph)
                print(f"   Step {step}: n={n}, m={m}, girth={current_g}")
        
        else:
            # Max steps reached
            result.steps = max_steps
            result.final_n = gen.graph.number_of_nodes()
            result.final_girth = compute_girth(gen.graph)  # type: ignore
            result.time_seconds = time.time() - start_time
            result.error = f"Timeout at {max_steps} steps"  # type: ignore
            
            if verbose:
                print(f"\n❌ FAILED: Reached {max_steps} steps without success")
                print(f"   Final state: n={result.final_n}, girth={result.final_girth}")
                
    except Exception as e:
        result.error = str(e) # type: ignore
        result.time_seconds = time.time() - start_time
        
        if verbose:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return result


def run_test_suite(generator_class, tests: List[CageTest], max_steps: int = 10000,
                  verbose: bool = False) -> List[GeneratorTestResult]:
    """
    Run multiple tests on a generator.
    
    Args:
        generator_class: Generator class to test
        tests: List of CageTest configurations
        max_steps: Maximum steps per test
        verbose: Print detailed progress
        
    Returns:
        List of GeneratorTestResult objects
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running test suite: {generator_class.__name__}")
    print(f"Tests: {len(tests)}, Max steps per test: {max_steps}")
    print(f"{'='*70}")
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{len(tests)}] Testing {test.name}...")
        result = run_single_test(generator_class, test, max_steps, verbose)
        results.append(result)
        
        # Quick summary
        if result.success:
            print(f"   ✅ Success in {result.steps} steps ({result.time_seconds:.2f}s)")
        else:
            print(f"   ❌ Failed: {result.error}")
    
    return results


def print_summary(results: List[GeneratorTestResult]):
    """Print a summary table of test results."""
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    # Group by generator
    by_generator = {}
    for r in results:
        if r.generator_name not in by_generator:
            by_generator[r.generator_name] = []
        by_generator[r.generator_name].append(r)
    
    for gen_name, gen_results in by_generator.items():
        print(f"\n{gen_name}:")
        print(f"  {'Test':<25} {'Result':<10} {'Steps':<10} {'Time':<10}")
        print(f"  {'-'*60}")
        
        successes = 0
        total_steps = []
        
        for r in gen_results:
            status = "✅ SUCCESS" if r.success else "❌ FAILED"
            steps = str(r.steps) if r.success else "-"
            time_str = f"{r.time_seconds:.2f}s" if r.success else "-"
            
            print(f"  {r.test.name:<25} {status:<10} {steps:<10} {time_str:<10}")
            
            if r.success:
                successes += 1
                total_steps.append(r.steps)
        
        # Statistics
        success_rate = (successes / len(gen_results)) * 100
        print(f"  {'-'*60}")
        print(f"  Success rate: {successes}/{len(gen_results)} ({success_rate:.1f}%)")
        
        if total_steps:
            median_steps = np.median(total_steps)
            mean_steps = np.mean(total_steps)
            print(f"  Steps (successful): median={median_steps:.0f}, mean={mean_steps:.0f}")


def compare_generators(generator_classes: List, tests: List[CageTest], 
                      max_steps: int = 10000, verbose: bool = False):
    """
    Compare multiple generators on the same test suite.
    
    Args:
        generator_classes: List of generator classes to compare
        tests: List of CageTest configurations
        max_steps: Maximum steps per test
        verbose: Print detailed progress
    """
    all_results = []
    
    for gen_class in generator_classes:
        results = run_test_suite(gen_class, tests, max_steps, verbose)
        all_results.extend(results)
    
    print_summary(all_results)


# Convenient test runners
def test_small_cages(generator_class, max_steps: int = 10000, verbose: bool = False):
    """Test on small cages: (3,5), (3,6), (4,5)"""
    results = run_test_suite(generator_class, SMALL_CAGES, max_steps, verbose)
    print_summary(results)
    return results


def test_medium_cages(generator_class, max_steps: int = 10000, verbose: bool = False):
    """Test on medium cages: (3,7), (3,8), (5,5)"""
    results = run_test_suite(generator_class, MEDIUM_CAGES, max_steps, verbose)
    print_summary(results)
    return results


def test_all_cages(generator_class, max_steps: int = 10000, verbose: bool = False):
    """Test on all known cages"""
    all_tests = SMALL_CAGES + MEDIUM_CAGES + LARGE_CAGES
    results = run_test_suite(generator_class, all_tests, max_steps, verbose)
    print_summary(results)
    return results


if __name__ == "__main__":
    """
    Run tests from command line:
        python ai/cage/tests.py small
        python ai/cage/tests.py small --generator bruteforce
        python ai/cage/tests.py small --generator astar
        python ai/cage/tests.py medium --steps 5000 --verbose
    """
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Test cage generators")
    parser.add_argument("suite", choices=["small", "medium", "large", "all"],
                       help="Which test suite to run")
    parser.add_argument("--generator", choices=["randomwalk", "bruteforce", "astar", "all"],
                       default="randomwalk",
                       help="Which generator to test (default: randomwalk)")
    parser.add_argument("--steps", type=int, default=10000,
                       help="Max steps per test (default: 10000)")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed progress")
    
    args = parser.parse_args()
    
    # Select generators
    generators = []
    if args.generator in ["randomwalk", "all"]:
        generators.append(RandomWalkGenerator)
    if args.generator in ["bruteforce", "all"]:
        generators.append(BruteforceGenerator)
    if args.generator in ["astar", "all"]:
        generators.append(AStarGenerator)
    
    # Select test suite
    if args.suite == "small":
        tests = SMALL_CAGES
    elif args.suite == "medium":
        tests = MEDIUM_CAGES
    elif args.suite == "large":
        tests = LARGE_CAGES
    else:
        tests = SMALL_CAGES + MEDIUM_CAGES + LARGE_CAGES
    
    # Run tests
    if len(generators) == 1:
        results = run_test_suite(generators[0], tests, args.steps, args.verbose)
        print_summary(results)
    else:
        compare_generators(generators, tests, args.steps, args.verbose)
