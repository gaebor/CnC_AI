#include "VectorInterface.h"

#include <string>
#include <algorithm>
#include <type_traits>

#include "GamePlay.hpp"

StaticTile::StaticTile()
{
    AssetName[0] = '\0';
}


StaticTile& StaticTile::operator=(const CNCStaticCellStruct& tile)
{
    ShapeIndex = (decltype(ShapeIndex))tile.IconNumber;
    for (int i = 0; i < std::extent<decltype(tile.TemplateTypeName)>::value; ++i)
    {
        if (tile.TemplateTypeName[i] == '_')
            strncpy_s(AssetName, tile.TemplateTypeName, i);
    }
    return *this;
}

StaticTile& StaticTile::operator=(const CNCDynamicMapEntryStruct& entry)
{
    std::copy(entry.AssetName, entry.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    ShapeIndex = entry.ShapeIndex;
    return *this;
}

StaticMap::StaticMap() 
  : MapCellX(0),
    MapCellY(0),
    MapCellWidth(0),
    MapCellHeight(0),
    OriginalMapCellX(0),
    OriginalMapCellY(0),
    OriginalMapCellWidth(0),
    OriginalMapCellHeight(0),
    StaticCells(nullptr)
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


StaticMap& StaticMap::operator=(const CNCMapDataStruct& static_map)
{
    MapCellX = static_map.MapCellX;
    MapCellY = static_map.MapCellY;
    MapCellWidth = static_map.MapCellWidth;
    MapCellHeight = static_map.MapCellHeight;
    OriginalMapCellX = static_map.OriginalMapCellX;
    OriginalMapCellY = static_map.OriginalMapCellY;
    OriginalMapCellWidth = static_map.OriginalMapCellWidth;
    OriginalMapCellHeight = static_map.OriginalMapCellHeight;

    if (StaticCells != nullptr)
        delete[] StaticCells;
    StaticCells = new StaticTile[OriginalMapCellHeight * OriginalMapCellWidth];

    auto dest = StaticCells;
    auto src = static_map.StaticCells + ((OriginalMapCellY - MapCellY) * MapCellWidth + OriginalMapCellX - MapCellX);
    for (int y = 0; y < OriginalMapCellHeight; ++y, src += OriginalMapCellWidth)
    {
        dest = std::copy(src, src + OriginalMapCellWidth, dest);
    }

    return *this;
}

DynamicObject::DynamicObject()
{
    AssetName[0] = '\0';
}

void DynamicObject::Assign(const CNCObjectStruct& object, const unsigned char House)
{
    std::copy(object.AssetName, object.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    
    PositionX = object.PositionX;
    PositionY = object.PositionY;
    Strength = object.Strength;
    ShapeIndex = object.ShapeIndex;
    IsRepairing = object.IsRepairing;

    Cloak = object.Cloak; // some other logic elsewhere

    Owner = GamePlay::GetHouseColorMap()[object.Owner];
    IsSelected = object.IsSelectedMask & (1 << House);

    if (object.Owner == House)
    {
        std::copy(object.Pips, object.Pips + object.MaxPips, Pips);
        std::fill_n(Pips + object.MaxPips, MAX_OBJECT_PIPS - object.MaxPips, -1);
        IsPrimaryFactory = object.IsPrimaryFactory;
        ControlGroup = object.ControlGroup;
    }
    else
    {
        std::fill_n(Pips, MAX_OBJECT_PIPS, -1);
        IsPrimaryFactory = false;
        ControlGroup = decltype(ControlGroup)(-1);
    }
}

void DynamicObject::Assign(const CNCDynamicMapEntryStruct& entry, const unsigned char House)
{
    if (entry.IsSellable) // wall
        // One can distinguish only one's own wall, not even ally's walls
        Owner = entry.Owner == House ? House : 0xff;
    else
        Owner = entry.Owner;

    Owner = GamePlay::GetHouseColorMap()[Owner];

    PositionX = entry.PositionX;
    PositionY = entry.PositionY;
    ShapeIndex = entry.ShapeIndex;
    Strength = 0;
    IsSelected = false;
    IsRepairing = false;
    Cloak = 0;
    IsPrimaryFactory = false;
    ControlGroup = decltype(ControlGroup)(-1);

    std::copy(entry.AssetName, entry.AssetName + CNC_OBJECT_ASSET_NAME_LENGTH, AssetName);
    std::fill_n(Pips, MAX_OBJECT_PIPS, -1);
}

VectorRepresentation::VectorRepresentation()
    : static_map(), dynamic_objects(nullptr), n_objects(0)
{
}

VectorRepresentation::VectorRepresentation(const StaticMap& static_map)
    : static_map(static_map), dynamic_objects(nullptr), n_objects(0)
{
}

void VectorRepresentation::Render(const CNCDynamicMapStruct& dynamic_map, const CNCObjectListStruct& layers, const unsigned char House)
{
    if (n_objects > 0)
        delete[] dynamic_objects;
    
    n_objects = 0;
    dynamic_objects = new DynamicObject[dynamic_map.Count + layers.Count]; // over-estimate

    const auto end_of_dynamic_map= dynamic_map.Entries + dynamic_map.Count;
    for (auto entry = dynamic_map.Entries; entry != end_of_dynamic_map; ++entry)
    {
        //      flag                                             wall                             crate
        if (entry->IsFlag || (entry->IsOverlay && ((entry->Type >= 1 && entry->Type <= 5) || entry->Type >= 28)))
        {
            dynamic_objects[n_objects++].Assign(*entry, House);
        }
        else
        {
            const int i = (entry->CellY - static_map.OriginalMapCellY) * static_map.OriginalMapCellWidth + entry->CellX - static_map.OriginalMapCellX;
            if (static_map.StaticCells[i].AssetName[0] == '\0' || entry->IsResource)
            {
                // this will prefer tiberium
                static_map.StaticCells[i] = *entry;
            }
        }
    }
    const auto end_of_layers = layers.Objects + layers.Count;
    for (auto object = layers.Objects; object != end_of_layers; ++object)
    {
        if (object->Cloak != 2 || House == object->Owner)
        {
            dynamic_objects[n_objects++].Assign(*object, House);
        }
    }
}


VectorRepresentation::~VectorRepresentation()
{
    if (n_objects > 0)
        delete[] dynamic_objects;
}

SidebarEntry::SidebarEntry()
    : Progress(0), Constructing(false), ConstructionOnHold(false), Busy(false)
{
    AssetName[0] = '\0';
}

SidebarEntry& SidebarEntry::operator=(const CNCSidebarEntryStruct& entry)
{
    std::copy(entry.AssetName, entry.AssetName + std::extent<decltype(entry.AssetName)>::value, AssetName);
    if (entry.Completed)
        Progress = 1;
    else
        Progress = entry.Progress;
    
    Constructing = entry.Constructing;
    ConstructionOnHold = entry.ConstructionOnHold;
    Busy = entry.Busy;

    return *this;
}

SideBar::SideBar():EntryCount(0), Entries(nullptr)
{
}

SideBar::~SideBar()
{
    if (Entries != nullptr)
        delete[] Entries;
}


SideBar& SideBar::operator=(const CNCSidebarStruct& sidebar)
{
    Credits = sidebar.Credits;
    PowerProduced = sidebar.PowerProduced;
    PowerDrained = sidebar.PowerDrained;

    RepairBtnEnabled = false;
    SellBtnEnabled = false;
    RadarMapActive = sidebar.RadarMapActive;
    EntryCount = sidebar.EntryCount[0] + sidebar.EntryCount[1];
    
    Entries = new SidebarEntry[EntryCount];
    std::copy(sidebar.Entries, sidebar.Entries + EntryCount, Entries);

    return *this;
}
