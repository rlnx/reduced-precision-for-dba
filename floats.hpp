#pragma once

#include <cstdint>
#include <CL/sycl/half_type.hpp>

inline float round_to_f16(float x) {
    return float(sycl::half(x));
}

inline float round_to_bf16(float x) {
    std::uint32_t y = ((*reinterpret_cast<std::uint32_t *>(&x)) >> 16) << 16;
    return *reinterpret_cast<float*>(&y);
}

struct f32_tag {};
struct f16_tag {};
struct bf16_tag {};

template <typename F>
inline float round_to(float x) {
    return x;
}

template <>
inline float round_to<f16_tag>(float x) {
    return round_to_f16(x);
}

template <>
inline float round_to<bf16_tag>(float x) {
    return round_to_bf16(x);
}
