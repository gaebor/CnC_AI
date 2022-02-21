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
	std::int32_t ControlGroup;
	std::int32_t Cloak;

	float PositionX;
	float PositionY;
	float Strength;

	float IsSelected;
	float IsRepairing;

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
	std::int32_t BuildableType; // only for internal use
	std::int32_t BuildableID; // only for internal use
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
	struct {
		float Credits;
		float PowerProduced;
		float PowerDrained;
		float RepairBtnEnabled; // this means that you HAVE a Repair button, not that it is toggled!
		float SellBtnEnabled;
		float RadarMapActive; // TODO implement radar map view
	} Members;
	std::vector<SidebarEntry> Entries;

	SideBar& operator=(const CNCSidebarStruct&);
};



struct VectorRepresentation
{
	StaticMap map;
	std::vector<DynamicObject> dynamic_objects;
	void Render(const CNCMapDataStruct*, const CNCDynamicMapStruct*, const CNCObjectListStruct*);
};

void RenderPOV(VectorRepresentation&, const VectorRepresentation&, const CNCShroudStruct* shroud, std::int32_t Owner);
