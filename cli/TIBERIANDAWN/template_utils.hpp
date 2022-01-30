#pragma once

#include <type_traits>
#include <algorithm>

template<typename Ty>
typename std::add_pointer_t<std::add_const_t<std::remove_extent_t<Ty>>> copy_array(const Ty& x, Ty& y)
{
    return std::copy_n(
        static_cast<std::add_pointer_t<std::add_const_t<std::remove_extent_t<Ty>>>>(x),
        std::extent<Ty>::value,
        static_cast<std::add_pointer_t<std::remove_extent_t<Ty>>>(y)
    );
}
