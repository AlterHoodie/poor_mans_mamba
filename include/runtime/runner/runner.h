#pragma once

#include <cstdint>
#include <span>

#include "core/status.h"

struct Sequence {
    int seq_id;
};

class Runner {
   public:
    virtual ~Runner() = default;

    virtual Status prefill(Sequence&, std::span<const int32_t> tokens);
    virtual Status decode(Sequence&, const int32_t token);
};