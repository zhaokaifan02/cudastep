福特架构之前

**单指令-多线程** 

single instruction multiple threads.

指的就是同一个线程束中，各个分支的次序是顺序执行的

也就是说，如果这个线程束中的32个线程，他们的代码当然是一样的。

但是里面有一个if。

比如

```
__global__ void add(const double *x,const double*y,double *z)
{
    int index = threadIdx.x + blockDim.x+ blockIdx.x;
    
    if(index%2 == 0)
    {
        z[index] = x[index] + y[index];
    } else 
    {
        z[index] = x[index] - y[index];
    }
}
```

虽然明面上是32个线程并发执行，但是同时只能执行if中的某一个，比如这里32个线程中，16个index为even，16个index为odd。所以本质上是16个线程并行。

这种缺陷叫做**分支发散 branch divergence**

所以写代码时要避免分治发散。

但是实际上的逻辑中分治发散是不可避免的。因为逻辑就在那里。

一种好的解决策略是：**分支发散的另一个分支，不进行任何操作，这样对整体也没什么影响**



福特架构之后

每个线程有自己的寄存器了，这样会导致线程寄存器少一点



## 线程束内同步函数

回忆一下以前的线程块内同步函数，在重复读取同一块内存时，我们希望一个读取写入操作完成后，再对这块内存进行另一次读取。

为了实现这种目的，我们有两种思路。

1. 使用原子操作，一口气执行完避免同时读取
2. 使用线程同步机制

首先是线程块内的同步。因为我们都知道**线程块本质上是一个逻辑组织**，因为这个逻辑组织内本质上还是以线程束来执行的。逻辑的含金量就在可以share_memory中可以减少对global_memory的访问。

但是_syncthreads() 这个很慢，因为线程块是以warp放在SM中执行的。同一个线程块的线程必须在一个SM中。

也就是说一个block可以由多个wrap组成。但如果一个线程块刚好是32个线程，可以被一个warp完美运行。那么我们就可以用warp同步机制了。比如

__warpthreads() 这个函数

```
__warpthreads(unsigned mask = 0xffffffff);
```

这个参数是一个32位的二进制，每一个1表示这个warp内的对应线程参与同步。遮蔽线程块内同步要快很多。

因为哪怕线程块也是借助warp来并行执行的。



## 更多线程束内基本函数



### 束内指标

我们都知道，线程是按线程束执行的。标记一个线程，由blockIdx 和 threadIdx共同决定的。

同样的，对于一个线程来说，他在warp内也有独立的id。比如这个block有128个线程，

对于线程id位 34的线程，它位于第二个warp，束内ID为

```
int lane_id = threadIdx.x % wapr_size;
```

也可以更加推荐的按位与指标

```
int lane_id = threadIdx.x % (w_size-1); 
```

比如线程块大小16，手动设置的w为8

则有如下对应关系

```
线程指标 0  1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  
束内指标 0  1  2  3  4  5  6  7  0  1   2   3   4   5   6   7
```

还有一个量叫做 MASK掩码。他是上面提到过的32位二进制，标明了这一个warp内的32的线程。

当对应的二进制为1时，表示参与了计算。



### 经典函数

#### 投票函数

__ballot_sync(mask, predicate)。

返回无符号整数，如果线程中的第n个线程参与运算，切predicate不为0，则返回的第n为二进制为1.

前面的mask表示参与投票（计算）的线程，和前面的MASK掩码一样。

相当于从一个mask获得了一个mask

__all_sync(mask,predicate);

返回值为0和1.

参与现成的predicate都不为0时才返回1。就是都为真就是真，predicate是一种判断

__any_sync(mask,predicate);

返回值0和1.

只要有一个不为0就返回1.



#### 线程束洗牌函数

说白了就是在线程束内交换某些数据。

T __shfl_sync(mask, v, srcLane, w)

mask中为1的参与线程，获得线程束内线程编号为srcLane的变量v的值.

理解为一种广播机制： 把某个线程的某个值，广播到线程束内的所有值。





T __shfl_up_sync (mask, v , d, w)

对于一个标号为t的线程，获得标号为t-d的线程中v的值。

对于t-d <0的线程获得自身的v。

比如当线程束大小为8时。设置d=2；

则2号线程获得 0号线程的v的值。

3号贤臣获得 1号线程的值。

0号线程和1号线程的值还是他本身。

实现了一种把某个数据向上运输的功能。

也可以理解为从0开始到 w-d-1为止，每个线程都把自己的v往上运d个单位。

不过这种理解不太适合获得某个值的思考方式。



T __shfl_down _ sync (mask, v, d, w)

标号为t的线程获得t+d线程的v的值，把数据从上往下拿，和up相反



T __shfl _ xor _ sync(mask, v , laneMask, w);

标号为t的线程获得 t异或laneMask的结果。比如我们取laneMask为2，w为8 二进制为 0010。

则束内坐标分别是0 1 2 3 4 5 6 7

异或后的结果分别是 2 3 0 1 6 7 4 5。

发现什么了吗？用xor交换，可以实现上面的三个交换！！奇妙desuwa

只要我们合理设计laneMask







# 总结

block内的线程，可以通过share_memory进行交互。

但是，memory他毕竟是memory。肯定没有寄存器register块啊。

所以在warp内，可以用寄存器交互。

然后warp还可以画出逻辑的子warp，w

通过三个选择函数，来标出哪些w中的线程要处理。

然后利用那四个shuffle操作，实现快速的访问。



## 协作组

前面讲过，warp内的线程，可以通过ballot+shuffle实现交互。

现在，我想实现，线程块内部间的通信，和线程块之间的通信，以及多GPU的设备间的写作

```
#include <cooperative_groups.h>
using namespace cooperative_groups;

```



这个很简单啊，就是从使用者的角度来整体管理block了



```c
__global__ void warp_reduce_kernel(int *data) {
    thread_block block = this_thread_block(); //获得这个线程所在的block
    thread_block_tile<32> warp = tiled_partition<32>(block); // 将线程块划分为32线程的组

    int local_val = data[block.thread_index]; //获得这个线程id
    // Warp级别的归约
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        int remote_val = warp.shfl_down(local_val, offset);
        local_val += remote_val;
    }
    // 仅第一个线程存储结果
    if (warp.thread_rank() == 0) {
        data[blockIdx.x] = local_val;
    }
}
```



这些函数和前面的投票函数与洗牌函数是一样的，只不过不能指定mask了，因为分片来的partition中的每一个都需要参与运算。

这些人为划分出来的线程片，也具有前面shfl交换的功能。



# 合并内存访问



在 CUDA 中，内存访问合并（coalescing）是通过合理的内存访问模式来提高性能的关键。以下是为什么让相邻的线程访问相邻的数据能保证内存访问合并的原因：

### 1. **内存访问模式**

- **全局内存的结构**：在 GPU 中，全局内存是以 32 字节（或 64 字节）为单位进行访问的。当多个线程同时访问相邻的内存地 址时，它们的访问可以被合并为一个内存事务。
- **线程和内存地址的关系**：如果线程 0 访问 `data[0]`，线程 1 访问 `data[1]`，以此类推，那么访问的内存地址是连续的。GPU 能够在同一时间将所有这些请求合并为一次大的内存访问。

### 2. **合并访问的示例**

假设有一个数组 `data`，其元素是连续存储的。以下是一个简单的示例：

```
__global__ void kernel(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // 每个线程处理相邻的数据
        data[tid] *= 2.0f;
    }
}
```

在这个例子中：

- 线程 0 访问 `data[0]`
- 线程 1 访问 `data[1]`
- 线程 2 访问 `data[2]`

因为这些访问是相邻的，GPU 可以将这些请求合并成一个内存事务，从而提高内存访问效率。

### 3. **非合并访问的情况**

如果线程访问的内存地址不相邻，例如：

```
__global__ void kernel(float *data) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        // 非相邻的访问模式
        data[tid * 10] *= 2.0f; // 线程 0 访问 data[0], 线程 1 访问 data[10]
    }
}
```

在这种情况下：

- 线程 0 访问 `data[0]`
- 线程 1 访问 `data[10]`

由于访问地址相隔较远，GPU 无法将这些访问合并，会造成多个独立的内存事务，从而降低性能。

### 4. **内存带宽的利用**

合并访问可以显著提高内存带宽的利用率。通过让相邻线程访问相邻的数据，可以减少内存访问的延迟，并充分利用可用的带宽。

### 总结

让相邻线程访问相邻的数据可以确保内存访问是合并的，因为：

- 内存访问模式是连续的，允许 GPU 将多个访问合并为一次大事务。
- 减少了内存访问的延迟，提高了带宽利用率。

这种设计在编写高性能的 CUDA 代码时至关重要。





