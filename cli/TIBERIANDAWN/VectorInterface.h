#pragma once
#include <stdlib.h>

#include "DLLInterface.h"

struct StaticTile
{
    char AssetName[CNC_OBJECT_ASSET_NAME_LENGTH];
    unsigned short ShapeIndex;

	StaticTile& operator=(const CNCStaticCellStruct&);
	StaticTile& operator=(const CNCDynamicMapEntryStruct&);
	StaticTile();
};

struct DynamicObject
{
	char 				AssetName[CNC_OBJECT_ASSET_NAME_LENGTH];
	int					PositionX;
	int					PositionY;
	int					Pips[MAX_OBJECT_PIPS];
	short				Strength;
	unsigned short		ShapeIndex;
	unsigned char		Owner;
	bool				IsSelected;
	bool				IsRepairing;
	unsigned char		Cloak;
	bool				IsPrimaryFactory;
	unsigned char		ControlGroup;

	DynamicObject();
	void Assign(const CNCObjectStruct&, const unsigned char House);
	void Assign(const CNCDynamicMapEntryStruct&, const unsigned char House);
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
	StaticTile* StaticCells;

	StaticMap();
	StaticMap(const CNCMapDataStruct&);
	StaticMap(const StaticMap&);
	~StaticMap();
};

struct VectorRepresentation
{
	StaticMap static_map;
	DynamicObject* dynamic_objects;
	int n_objects;
	VectorRepresentation();
	VectorRepresentation(const StaticMap&);
	void Render(const CNCDynamicMapStruct&, const CNCObjectListStruct&, const unsigned char House);
	~VectorRepresentation();
};
