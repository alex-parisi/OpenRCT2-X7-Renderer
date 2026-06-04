/// mock_embree.h — Configurable mock state for Embree API functions.

#pragma once

#include <cstdint>
#include <embree4/rtcore.h>

namespace MockEmbree {
    struct State {
        bool device_init_fails = false;
        bool geometry_alloc_fails = false;
        bool buffer_alloc_fails = false;
        bool vertex_buffer_fails = false;
        bool normal_buffer_fails = false;
        bool index_buffer_fails = false;
        RTCErrorFunction error_callback = nullptr;
        void* error_callback_user_ptr = nullptr;
    };

    State& state();
    void reset();
} // namespace MockEmbree
