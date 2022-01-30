#include "VectorInterface.hpp"

#include <string>
#include <type_traits>
#include <algorithm>

#include "template_utils.hpp"
#include "data_utils.hpp"


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
    copy_array(entry.AssetName, AssetName);
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
    StaticCells()
{
}


StaticMap::~StaticMap()
{
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

    StaticCells.clear();
    StaticCells.resize(OriginalMapCellHeight * OriginalMapCellWidth);

    auto dest = StaticCells.begin();
    auto src = static_map.StaticCells + ((OriginalMapCellY - MapCellY) * MapCellWidth + OriginalMapCellX - MapCellX);
    for (int y = 0; y < OriginalMapCellHeight; ++y, src += MapCellWidth)
    {
        dest = std::copy_n(src, OriginalMapCellWidth, dest);
    }

    return *this;
}

DynamicObject::DynamicObject()
{
    AssetName[0] = '\0';
}

void DynamicObject::Assign(const CNCObjectStruct& object)
{
    copy_array(object.AssetName, AssetName);

    PositionX = object.PositionX;
    PositionY = object.PositionY;
    Strength = object.Strength;
    ShapeIndex = object.ShapeIndex;
    Owner = HouseColorMap[(unsigned char)object.Owner];
    IsSelected = ConvertMask(object.IsSelectedMask);
    IsRepairing = object.IsRepairing;
    Cloak = object.Cloak;
    if (object.IsPrimaryFactory)
    {
        Pips[0] = 2U; // PIP_PRIMARY // "Primary" building marker
        std::fill_n(Pips + 1, MAX_OBJECT_PIPS - 1, -1);
    }
    else
    {
        std::copy_n(object.Pips, object.MaxPips, Pips);
        std::fill_n(Pips + object.MaxPips, MAX_OBJECT_PIPS - object.MaxPips, -1);
    }
    ControlGroup = object.ControlGroup;
}

void DynamicObject::Assign(const CNCDynamicMapEntryStruct& entry)
{
    copy_array(entry.AssetName, AssetName);
    PositionX = entry.PositionX;
    PositionY = entry.PositionY;
    Strength = 0;
    ShapeIndex = entry.ShapeIndex;
    Owner = HouseColorMap[(unsigned char)entry.Owner];
    IsSelected = 0;
    IsRepairing = false;
    Cloak = 0;
    std::fill_n(Pips, MAX_OBJECT_PIPS, -1);
    ControlGroup = decltype(ControlGroup)(-1);
}

void VectorRepresentation::Render(
    const CNCMapDataStruct* static_map,
    const CNCDynamicMapStruct* dynamic_map,
    const CNCObjectListStruct* layers)
{
    map = *static_map;

    dynamic_objects.clear();
    
    const auto end_of_dynamic_map = dynamic_map->Entries + dynamic_map->Count;
    for (auto entry = dynamic_map->Entries; entry != end_of_dynamic_map; ++entry)
    {
        //      flag                                             wall                             crate
        if (entry->IsFlag || (entry->IsOverlay && ((entry->Type >= 1 && entry->Type <= 5) || entry->Type >= 28)))
        {
            dynamic_objects.emplace_back();
            dynamic_objects.back().Assign(*entry);
        }
        else
        {
            const auto x = entry->CellX - static_map->OriginalMapCellX;
            const auto y = entry->CellY - static_map->OriginalMapCellY;
            if (x >= 0 && x <= static_map->OriginalMapCellWidth && y >= 0 && y <= static_map->OriginalMapCellHeight)
            {
                const int i = y * static_map->OriginalMapCellWidth + x;
                if (map.StaticCells[i].AssetName[0] == '\0' || entry->IsResource)
                {
                    // this will prefer tiberium
                    map.StaticCells[i] = *entry;
                }
            }
        }
    }

    const auto end_of_layers = layers->Objects + layers->Count;
    for (auto object = layers->Objects; object != end_of_layers; ++object)
    {
        if (object->Type <= TERRAIN)
        {
            dynamic_objects.emplace_back();
            dynamic_objects.back().Assign(*object);
        }
    }
}


SidebarEntry::SidebarEntry()
    : Progress(0), Constructing(false), ConstructionOnHold(false), Busy(false)
{
    AssetName[0] = '\0';
}

SidebarEntry& SidebarEntry::operator=(const CNCSidebarEntryStruct& entry)
{
    copy_array(entry.AssetName, AssetName);
    if (entry.Completed)
        Progress = 1;
    else
        Progress = entry.Progress;
    
    Constructing = entry.Constructing;
    ConstructionOnHold = entry.ConstructionOnHold;
    Busy = entry.Busy;

    return *this;
}


SideBar& SideBar::operator=(const CNCSidebarStruct& sidebar)
{
    Credits = sidebar.Credits;
    PowerProduced = sidebar.PowerProduced;
    PowerDrained = sidebar.PowerDrained;

    RepairBtnEnabled = false;
    SellBtnEnabled = false;
    RadarMapActive = sidebar.RadarMapActive;
    Entries.resize(sidebar.EntryCount[0] + sidebar.EntryCount[1]);
    std::copy_n(sidebar.Entries, Entries.size(), Entries.begin());

    return *this;
}

StaticMapView& StaticMapView::operator=(const StaticMap& other)
{
    MapCellWidth = other.OriginalMapCellWidth;
    MapCellHeight = other.OriginalMapCellHeight;
    StaticCells = other.StaticCells.data();
    return *this;
}

SideBarView& SideBarView::operator=(const SideBar& other)
{
    Credits = other.Credits;
    PowerProduced = other.PowerProduced;
    PowerDrained = other.PowerDrained;
    RepairBtnEnabled = other.RepairBtnEnabled;
    SellBtnEnabled = other.SellBtnEnabled;
    RadarMapActive = other.RadarMapActive;
    
    Count = other.Entries.size();
    Entries = other.Entries.data();
    return *this;
}

VectorRepresentationView& VectorRepresentationView::operator=(const VectorRepresentation& other)
{
    map = other.map;
    dynamic_objects_count = other.dynamic_objects.size();
    dynamic_objects = other.dynamic_objects.data();

    return *this;
}

void RenderPOV(VectorRepresentation&, const VectorRepresentation&, const CNCShroudStruct* shroud, unsigned char Owner)
{
    /* TODO
    *  * conver IsSelectedMask to IsSelected (using color)
    *  * hide cloaked vehicles from non-owners
    *  * cover shroud
    *  * hide pips from non-owners
    */
}
