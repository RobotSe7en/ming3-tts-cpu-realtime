#include "trial.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

int main() try {
    for (int steps : {5, 6, 7, 10}) {
        const auto grid = ming_cpu::trial::time_grid(steps);
        if (grid.size() != static_cast<size_t>(steps + 1) || grid.front() != 0.0f || grid.back() != 1.0f) {
            throw std::runtime_error("EPSS must span the complete [0, 1] interval");
        }
        for (size_t i = 0; i < grid.size(); ++i) {
            uint32_t bits;
            std::memcpy(&bits, &grid[i], sizeof(bits));
            if (!std::isfinite(grid[i]) || (bits & 0xffffU) || (i && grid[i] <= grid[i - 1])) {
                throw std::runtime_error("EPSS nodes must be ordered finite BF16 values");
            }
        }
    }
    bool rejected = false;
    try { ming_cpu::trial::time_grid(8); } catch (const std::runtime_error &) { rejected = true; }
    if (!rejected) throw std::runtime_error("Unsupported time grid was accepted");
    std::cout << "Four complete EPSS schedules passed invariant checks\n";
    return 0;
} catch (const std::exception & error) {
    std::cerr << error.what() << '\n';
    return 1;
}
