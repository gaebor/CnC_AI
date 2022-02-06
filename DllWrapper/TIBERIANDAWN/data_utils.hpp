#pragma once
#include <unordered_map>
#include <string>

extern const float cost_lookup_table[];

extern const std::unordered_map<std::string, std::int32_t> static_tile_names;
extern const std::unordered_map<std::string, std::int32_t> dynamic_object_names;

extern std::int32_t HouseColorMap[256];

std::int32_t ConvertMask(const unsigned int per_house_mask);
