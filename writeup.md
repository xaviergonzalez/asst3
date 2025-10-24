Xavier Gonzalez: xavier18

# Part 1

## Question 1

In both settings, I use a N=20 million.

When I run saxpy from Assign1 using ISPC, it takes 11 ms with a bandwidth of around 27 GB/s.

When I run saxpy on GPU in this assignment, the kernel takes around 1 ms, but the total process (including memory to and frrom GPU) takes around 40 ms, for a bandwidth of around 5.5 GB/s.

This exercise shows just how expensive moving memory to and from GPU is (which we are doing in this assignment, but weren't doing in Assign1). Even though the GPU is much faster at computing the saxpy operation, the memory transfer overhead makes the overall process slower than the CPU version.

# Part 2

# Part 3