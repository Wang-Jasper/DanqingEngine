// Test entry point: run all PHYS_TEST cases and exit non-zero on failure.

#include "test_framework.h"

int main()
{
    int failed = phys::RunAllTests();
    return failed == 0 ? 0 : 1;
}
