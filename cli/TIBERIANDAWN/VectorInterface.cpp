#include "VectorInterface.h"

#include <string>
#include <algorithm>

#include "GamePlay.hpp"

StaticTile::StaticTile() 
{
    AssetName[0] = '\0';
}


StaticTile& StaticTile::operator=(const CNCStaticCellStruct& tile)
{
    ShapeIndex = (decltype(ShapeIndex))tile.IconNumber;
    for (int i = sizeof(tile.TemplateTypeName) - 1; i >= 0; --i)
    {
        if (tile.TemplateTypeName[i] == '_')
            strncpy_s(AssetName, tile.TemplateTypeName, i);
    }
    return *this;
}

StaticTile& StaticTile::operator=(const CNCDynamicMapEntryStruct& entry)
{
    std::copy(entry.AssetName, entry.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    ShapeIndex = (decltype(ShapeIndex))entry.ShapeIndex;
    return *this;
}

StaticMap::StaticMap() : StaticCells(nullptr)
{

}

StaticMap::StaticMap(const StaticMap& other)
{
    *this = other;
    StaticCells = new StaticTile[OriginalMapCellHeight * OriginalMapCellWidth];
    std::copy(other.StaticCells, other.StaticCells + OriginalMapCellHeight * OriginalMapCellWidth, StaticCells);
}

StaticMap::~StaticMap()
{
    delete[] StaticCells;
}

void StaticMap::Add(const CNCDynamicMapStruct& dynamic_map)
{
    const auto end_ptr = dynamic_map.Entries + dynamic_map.Count;
    for (auto entry = dynamic_map.Entries; entry != end_ptr; ++entry)
    {
        if (entry->IsOverlay)
        {
            if (entry->IsFlag)
                continue;
            if (entry->Type >= 1 && entry->Type <= 5) // wall
                continue;
            if (entry->Type >= 28) // crate
                continue;
        }
        const int i = (entry->CellY - OriginalMapCellY) * OriginalMapCellWidth + entry->CellX - OriginalMapCellX;
        strcpy_s(StaticCells[i].AssetName, entry->AssetName);
        StaticCells[i].ShapeIndex = entry->ShapeIndex;
    }
}

StaticMap::StaticMap(const CNCMapDataStruct& static_map)
    : MapCellX(static_map.MapCellX), MapCellY(static_map.MapCellY),
    MapCellWidth(static_map.MapCellWidth), MapCellHeight(static_map.MapCellHeight),
    OriginalMapCellX(static_map.OriginalMapCellX), OriginalMapCellY(static_map.OriginalMapCellY),
    OriginalMapCellWidth(static_map.OriginalMapCellWidth), OriginalMapCellHeight(static_map.OriginalMapCellHeight)
{
    const auto x_preset = OriginalMapCellX - MapCellX;
    const auto y_preset = OriginalMapCellY - MapCellY;
    
    const auto x_postset = MapCellWidth - OriginalMapCellWidth - x_preset;
    const auto y_postset = MapCellHeight - OriginalMapCellHeight - y_preset;

    StaticCells = new StaticTile[OriginalMapCellHeight * OriginalMapCellWidth];

    int source_i = y_preset * MapCellWidth;
    int dest_i = 0;
    for (int y = 0; y < OriginalMapCellHeight; ++y)
    {
        source_i += x_preset;
        for (int x = 0; x < OriginalMapCellWidth; ++x, ++source_i, ++dest_i)
        {
            StaticCells[dest_i] = static_map.StaticCells[source_i];
        }
        source_i += x_postset;
    }
}

DynamicObject::DynamicObject()
{
    AssetName[0] = '\0';
}

DynamicObject::DynamicObject(const CNCObjectStruct& object, const unsigned char House)
    : PositionX(object.PositionX), PositionY(object.PositionY),
    Strength(object.Strength),
    ShapeIndex(object.ShapeIndex),
    Owner(GamePlay::HouseColorMap[object.Owner]),
    IsSelected(object.IsSelectedMask & (1 << House)),
    IsRepairing(object.IsRepairing),
    Cloak(object.Cloak),
    IsPrimaryFactory(object.Owner == House ? object.IsPrimaryFactory : false),
    ControlGroup(object.ControlGroup)
{
    std::copy(object.AssetName, object.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    if (object.Owner == House)
    {
        std::copy(object.Pips, object.Pips + object.MaxPips, Pips);
        std::fill_n(Pips + object.MaxPips, MAX_OBJECT_PIPS - object.MaxPips, -1);
    }
    else // one cannot see the pips of other teams
        std::fill_n(Pips, MAX_OBJECT_PIPS, -1);
}

DynamicObject::DynamicObject(const CNCDynamicMapEntryStruct& entry, const unsigned char House)
    :PositionX(entry.PositionX), PositionY(entry.PositionY),
    ShapeIndex(entry.ShapeIndex),
    Strength(0),
    IsSelected(false),
    IsRepairing(false),
    Cloak(0),
    IsPrimaryFactory(false),
    ControlGroup(-1)
{
    if (entry.IsFlag)
        Owner = entry.Owner;
    else if (entry.IsSellable) // wall
        // One can distinguish only one's own wall, not even ally's walls
        Owner = entry.Owner == House ? House : 0xff;
    else
        Owner = 0xff;

    Owner = GamePlay::HouseColorMap[Owner];

    std::copy(entry.AssetName, entry.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    std::fill_n(Pips, MAX_OBJECT_PIPS, -1);
}

VectorRepresentation::VectorRepresentation()
    : static_map(), dynamic_objects(nullptr), n_objects(0)
{
}

VectorRepresentation::~VectorRepresentation()
{
    if (n_objects > 0)
        delete[] dynamic_objects;
}
