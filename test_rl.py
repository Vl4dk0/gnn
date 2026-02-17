from ai.cage import RLGenerator
import networkx as nx


def test_rl_gen():
    print("Testing RL Generator...")
    gen = RLGenerator(k=3, g=5, model_type="gin")
    print("Generator initialized.")

    for i in range(50):
        gen.step()
        if gen.is_complete:
            print(f"Complete at step {i}")
            break

    print(
        f"Final Graph: {gen.graph.number_of_nodes()} nodes, {gen.graph.number_of_edges()} edges"
    )
    print(f"Success: {gen.success}")


if __name__ == "__main__":
    test_rl_gen()
