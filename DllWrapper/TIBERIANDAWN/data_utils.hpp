#pragma once
#include <unordered_map>
#include <string>

extern const std::unordered_map<std::string, int> buildables;
extern unsigned char HouseColorMap[256];

unsigned char ConvertSelectedMask(const unsigned int per_house_mask);
