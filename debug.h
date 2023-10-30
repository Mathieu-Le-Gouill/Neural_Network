#pragma once
#include <cassert>

#define DEBUG 1

#if DEBUG
	#define debug_assert(condition) assert(condition)
#else
	#define debug_assert(condition)
#endif


