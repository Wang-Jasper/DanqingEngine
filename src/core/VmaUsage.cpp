// VMA is a single-header library: VMA_IMPLEMENTATION must be defined in exactly
// one translation unit. Zero → undefined-symbol link errors; two or more →
// duplicate definitions. Other files include vk_mem_alloc.h without the define.
#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"
