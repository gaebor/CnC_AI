#pragma once
#include <stdlib.h>

#include <vector>

#include "DLLInterface.h"

struct StaticTile
{
    std::int32_t AssetName;
	std::int32_t ShapeIndex;

	StaticTile& operator=(const CNCStaticCellStruct&);
	StaticTile& operator=(const CNCDynamicMapEntryStruct&);
	StaticTile();
};

struct DynamicObject
{
	std::int32_t AssetName;
	std::int32_t ShapeIndex;
	std::int32_t Owner;
	std::int32_t Pips[MAX_OBJECT_PIPS];

	float PositionX;
	float PositionY;
	float Strength;

	float IsSelected;
	float IsRepairing;
	float Cloak;
	float ControlGroup;

	DynamicObject();
	void Assign(const CNCObjectStruct&);
	void Assign(const CNCDynamicMapEntryStruct&);
};


struct StaticMap {
	int	MapCellX;
	int	MapCellY;
	int	MapCellWidth;
	int	MapCellHeight;
	int	OriginalMapCellX;
	int	OriginalMapCellY;
	int	OriginalMapCellWidth;
	int	OriginalMapCellHeight;
	StaticTile StaticCells[62][62];

	StaticMap();
	StaticMap& operator=(const CNCMapDataStruct&);
};

struct SidebarEntry
{
    std::int32_t AssetName;
	float Progress;
	float Cost;
	float BuildTime;
	float Constructing;
	float ConstructionOnHold;
	float Busy;

	SidebarEntry();
	SidebarEntry& operator=(const CNCSidebarEntryStruct&);
};

struct SideBar {
	int Credits;
	int PowerProduced;
	int PowerDrained;
	bool RepairBtnEnabled; // this means that you HAVE a Repair button, not that it is toggled!
	bool SellBtnEnabled;
	bool RadarMapActive; // TODO implement radar map view
	std::vector<SidebarEntry> Entries;

	SideBar& operator=(const CNCSidebarStruct&);
};

struct SideBarView {
	float Credits;
	float PowerProduced;
	float PowerDrained;
	float RepairBtnEnabled; // this means that you HAVE a Repair button, not that it is toggled!
	float SellBtnEnabled;
	float RadarMapActive; // TODO implement radar map view
	size_t Count;
	const SidebarEntry* Entries;

	SideBarView& operator=(const SideBar&);
};


struct VectorRepresentation
{
	StaticMap map;
	std::vector<DynamicObject> dynamic_objects;
	void Render(const CNCMapDataStruct*, const CNCDynamicMapStruct*, const CNCObjectListStruct*);
};

struct VectorRepresentationView
{
	const StaticTile* map; // 62*62 but tiles may be empty
	size_t dynamic_objects_count;
	const DynamicObject* dynamic_objects;

	VectorRepresentationView& operator=(const VectorRepresentation&);
};

struct PlayerVectorRepresentationView : VectorRepresentationView
{
	SideBarView sidebar;
};


void RenderPOV(VectorRepresentation&, const VectorRepresentation&, const CNCShroudStruct* shroud, std::int32_t Owner);