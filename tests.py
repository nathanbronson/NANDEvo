from evolve import evolve
from datasets import Addition

if __name__ == "__main__":
    ad = Addition(0, 7)
    evolve(ad.x, ad.y, ad.scheme, logging=5, initial_population=["a"])