#pragma once
#include <unordered_map>
#include <string>

extern const std::unordered_map<std::string, std::int32_t> static_tile_names;
extern const std::unordered_map<std::string, std::int32_t> dynamic_object_names;

extern std::int32_t HouseColorMap[256];

std::int32_t ConvertMask(const unsigned int per_house_mask);

template<typename T>
void step_buffer(T*& buffer_ptr, size_t& buffer_size, size_t size)
{
	static_assert(sizeof(T) == 1, "this function only works for single byte types");
	buffer_ptr += size;
	buffer_size -= size;
}

template <typename T>
bool CopyToBuffer(const T&, unsigned char*& buffer, size_t& buffer_size);

template <typename T>
bool CopyToBuffer(const T& t, unsigned char*& buffer, size_t& buffer_size) {
	static_assert(std::is_trivially_copyable<T>::value, "template argument must be trivially copyable");
	if (buffer_size < sizeof(T))
		return false;
	memcpy_s(buffer, buffer_size, &t, sizeof(T));
	step_buffer(buffer, buffer_size, sizeof(T));
	return true;
}

template <typename T>
bool CopyToBuffer(const std::vector<T>& t, unsigned char*& buffer, size_t& buffer_size) {
	static_assert(std::is_trivially_copyable<T>::value, "argument's value_type must be trivially copyable");
	const auto array_size = t.size() * sizeof(T);
	const std::int32_t Count = t.size();
	if (buffer_size < sizeof(Count) + array_size)
		return false;
	CopyToBuffer(Count, buffer, buffer_size);
	memcpy_s(buffer, buffer_size, t.data(), array_size);
	step_buffer(buffer, buffer_size, array_size);
	return true;
}
