# record time
```
#include <chrono>
auto start = std::chrono::high_resolution_clock::now();
//code

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end -start);
std::cout << "run time: " << duration.count() << " us" << std::endl;
```
