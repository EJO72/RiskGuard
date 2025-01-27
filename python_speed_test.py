import time

start = time.perf_counter() # start timer

a = 0

for i in range(10000):
    for j in range(10000):
        a = j

end = time.perf_counter() # end counter
print(f'computation time: {(end - start):.4f}')