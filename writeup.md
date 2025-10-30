Xavier Gonzalez: xavier18

# Part 1

## Question 1

In both settings, I use a N=20 million.

When I run saxpy from Assign1 using ISPC, it takes 11 ms with a bandwidth of around 27 GB/s.

When I run saxpy on GPU in this assignment, the kernel takes around 1 ms, but the total process (including memory to and frrom GPU) takes around 40 ms, for a bandwidth of around 5.3 GB/s.

This exercise shows just how expensive moving memory to and from GPU is (which we are doing in this assignment, but weren't doing in Assign1). Even though the GPU is much faster at computing the saxpy operation, the memory transfer overhead makes the overall process slower than the CPU version.

## Question 2

Again, for N=20 million.

When I run saxpy on GPU in this assignment, the kernel takes around 1 ms, but the total process (including memory to and frrom GPU) takes around 40 ms, for a bandwidth of around 5.3 GB/s. This shows that so much more time in saxpy is spent moving memory to and from the GPU than in the actual computation.

According to the provided specs, the T4 can move memory from host (CPU) to device (GPU) at a peak rate of 32 GB/s, where as we are only achieving around 5.3 GB/s. This is likely due to overhead in setting up the transfers, and jives with the expected memory bandwidth on AWS.

# Part 2

I ran parts 2 and 3 on an NVIDIA T4 GPU on AWS.

-------------------------
Scan Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 0.626           | 0.478           | 1.25            |
| 10000000        | 9.028           | 7.857           | 1.25            |
| 20000000        | 17.823          | 15.631          | 1.25            |
| 40000000        | 35.285          | 31.292          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------

-------------------------
Find_repeats Score Table:
-------------------------
-------------------------------------------------------------------------
| Element Count   | Ref Time        | Student Time    | Score           |
-------------------------------------------------------------------------
| 1000000         | 1.021           | 0.846           | 1.25            |
| 10000000        | 13.003          | 10.833          | 1.25            |
| 20000000        | 21.573          | 20.222          | 1.25            |
| 40000000        | 42.597          | 39.075          | 1.25            |
-------------------------------------------------------------------------
|                                   | Total score:    | 5.0/5.0         |
-------------------------------------------------------------------------


# Part 3

## Solution Performance (question 2)

I ran parts 2 and 3 on an NVIDIA T4 GPU on AWS.

Here's the print out from the call to nvidia-smi

|=========================================+========================+======================|
|   0  NVIDIA T4G                     On  |   00000000:00:1F.0 Off |                    0 |
| N/A   34C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |

------------
Score table:
------------
--------------------------------------------------------------------------
| Scene Name      | Ref Time (T_ref) | Your Time (T)   | Score           |
--------------------------------------------------------------------------
| rgb             | 0.2655           | 0.201           | 9               |
| rand10k         | 3.0601           | 3.0411          | 9               |
| rand100k        | 29.3956          | 29.9609         | 9               |
| pattern         | 0.4068           | 0.3095          | 9               |
| snowsingle      | 19.556           | 6.6541          | 9               |
| biglittle       | 15.1625          | 30.3684         | 6               |
| rand1M          | 220.5381         | 125.2941        | 9               |
| micro2M         | 420.8556         | 156.4806        | 9               |
--------------------------------------------------------------------------
|                                    | Total score:    | 69/72           |
--------------------------------------------------------------------------

## Solution Overview and Decomposition (question 3)

In my solution, I edit `render` and have it call a new kernel I wrote, called `kernelRenderPixelsPerTile`.

At a high level, the idea is that I divide work on the threads into different tiles, each of which are size 32x32 pixels. Each tile is given to a thread block, also of size 32x32 threads. 

So, each tile is being worked on in parallel by a different thread block.

Within each tile, I then use the threads to parallelize computation over the circles (in effect using the thread to effect a `map` on the circles). For each circle, I want to know whether it intersects the tile for now (I use `circleInBoxConservative` for this task). Once I get this mask (for each circle, does it intersect the title or not), I use the exclusive scan provided in `sharedMemExclusiveScan` to implement a `gather` to get a list of the circles that need to be considerd for a specific tile.

Then, finally, for each pixel in the tile  (i.e., parallelized over the pixels, one thread in the thread block per pixel in the tile) , I loop over the circles that intersect the tile and compute the color for that pixel.

## Sychronization and Communication (q4)

I call `syncthread` a lot in my kernel.

First, I use shared memory in the tile to accelerate data fetching (this is a place I reduced communication requirements). However, as a result, I need to initialize the shared memory. I incur slight worker imbalance by making thread0 in the block initialize the shared memory, and then call `syncthreads` to make sure all threads see the initialized shared memory.

Then, when I use the threads to `map` over the circles to get the mask of which circles intersect the tile, I need to call `syncthreads` again to make sure all threads have finished writing to shared memory before I call the exclusive scan.

In my implementation of the `gather` (using the exclusive scan), I actually call `syncthreads` three times. 

First, I use the different threads to initialize the prefixSumInput (in shared memory, a requirement of the exclusive scan/an optimization of it), so I call syncthreads once I've loaded up the initialization of `prefixSumInput.` Then, I syncthreads after running the exclusive scan. Finally, I incur a bit of worker imbalance by making thread0 update some shared variables, and again sync after. 

I had to do multiple loops because the pscan only works with size max 1024, and in general we could have far more circles than this.

And then I throw in a final syncthreads at the end of my kernel, though I'm not sure that was strictly needed.

Definitely using the shared memory ws a win for reducing badnwidth requriements. Also, because I can only launch so much shared memory per thread block, I loop through the tiles as well, filling up the thread block with as much shared memory as I can (4096 ints worth of memory).

## Description of process/journey (question 6)

I started by thinking of the atomicity of updating the pixels, but then reealized that cuda's `atomicCAS` isn't implemented for floats, and so wouldn't work.

Then, I saw that hint aboout switching the axes of parallelism first, and then not needing atomics.

The starter code tried to parallelize over circles first, which was hard because the order of the circles needs to be respected.

So, in my first attempt, I tried to parallelize over pixels. 

My original idea was really beautiful. I was going to give a thread to each pixel, and then for each pixel, compute the contributions of the circle to it. If you look at the pixel shading formula, you realize that it's actually a linear dynamical system, for which their exists a pscan (but you need to do some multiplication)! I figured out a beautiful way to do this with the primitives we were given would be to take logs of the opacities and then add them with the pscan we were given.

However, I couldn't actually impelment this idea because if I was giving one thread per pixel, I couldn't use the pscan (which requies many parallel processors, i.e. a thread block).

In hindisght, I could have assigned a thread block to each pixel.

In any case, what I actually tried was writing a kernel called `kernelRenderPixels`, which was a compromise. I was parallelized over pixels, but then in each pixel, I did a for loop over circles. This approach was correct but too slow.

I knew we were given the tile checker functions for a reason, and so thinking about how to use them, I realized I could do a compromise: assign a thread block to each tile (instead of to each pixel). Then, I could make a mask of which circles intersect the title. I realized I could `gather` from this mask using the pscan (conveniently using all the threads in the thread block). I had to write a loop because the pscan can only handle `SCAN_BLOCK_DIM` circles at a time, but this was a prettty good solution.

The one problem was that I was originally passing `2 * numCircles` of shared memory to each kernel. So, I had a good implementation for the two smallest tasks (rgb and pattern), but on all the large tasks I failed with black images and super fast runtimes. I realized that on the other tasks, `numCircles` was gigantic, and so was I was asking for way too much shared memory for a thread block! So the kernel launched were failing and not doing anything. 

I played around and figured that as long as I asked for 4096 ints of shared memory per thread block, I was fine. So, I wrote a loop to loop over tiles, filling up the thread block with as much shared memory as possible (4096 ints worth of memory). This worked pretty well, was was my final solution.