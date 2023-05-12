import gc

def create_large_object():
    return list(range(10**6))

# Create a large object
my_object = create_large_object()

# Delete the reference to the large object
my_object = None

# Run garbage collection
collected = gc.collect()

# Check the result
print(f"Garbage collector collected {collected} objects.")
