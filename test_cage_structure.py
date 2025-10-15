"""
Test script to verify the restructured cage generation code works.
"""

from ai.cage import RandomWalkGenerator

def test_random_walk():
    """Test Random Walk generator."""
    print("Testing Random Walk Generator...")
    print("-" * 50)
    
    # Test (3,5)-cage generation
    k, g = 3, 5
    generator = RandomWalkGenerator(k, g, max_steps=100)
    
    print(f"Target: ({k},{g})-cage")
    print(f"Starting generation...\n")
    
    while not generator.is_complete and generator.step_count < 100:
        generator.step()
        
        if generator.step_count % 20 == 0:
            print(f"Step {generator.step_count}: "
                  f"nodes={len(generator.graph.nodes())}, "
                  f"edges={len(generator.graph.edges())}, "
                  f"regular={generator.is_regular()}")
    
    print(f"\n{'='*50}")
    print(f"Generation {'SUCCEEDED' if generator.success else 'FAILED'}")
    print(f"Final stats:")
    print(f"  Steps: {generator.step_count}")
    print(f"  Nodes: {len(generator.graph.nodes())}")
    print(f"  Edges: {len(generator.graph.edges())}")
    print(f"  Regular: {generator.is_regular()}")
    print(f"  Time: {generator.elapsed_time():.2f}s")
    print(f"{'='*50}")

if __name__ == "__main__":
    test_random_walk()
