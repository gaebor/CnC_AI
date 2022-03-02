#include "VectorInterface.hpp"

#include <string>
#include <type_traits>
#include <algorithm>

#include "template_utils.hpp"
#include "data_utils.hpp"


StaticTile::StaticTile() : AssetName(0), ShapeIndex(0)
{
}


StaticTile& StaticTile::operator=(const CNCStaticCellStruct& tile)
{
    ShapeIndex = (decltype(ShapeIndex))tile.IconNumber;
    for (size_t i = 0; i < std::extent<decltype(tile.TemplateTypeName)>::value; ++i)
    {
        if (tile.TemplateTypeName[i] == '_')
            AssetName = static_tile_names.at(std::string(tile.TemplateTypeName, i));
    }
    return *this;
}

StaticTile& StaticTile::operator=(const CNCDynamicMapEntryStruct& entry)
{
    AssetName = static_tile_names.at(entry.AssetName);
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
    OriginalMapCellHeight(0)
{
    new (StaticCells) StaticTile[62 * 62];
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

    auto src = static_map.StaticCells + ((OriginalMapCellY - MapCellY) * MapCellWidth + OriginalMapCellX - MapCellX);
    for (int y = 0; y < OriginalMapCellHeight; ++y, src += MapCellWidth)
    {
        std::copy_n(src, OriginalMapCellWidth, StaticCells[y]);
    }

    return *this;
}

DynamicObject::DynamicObject() : AssetName(0), ShapeIndex(0)
{
}

void DynamicObject::Assign(const CNCObjectStruct& object)
{
    AssetName = dynamic_object_names.at(object.AssetName);

    PositionX = (float)object.PositionX;
    PositionY = (float)object.PositionY;
    Strength = (float)object.Strength;
    ShapeIndex = object.ShapeIndex;
    Owner = HouseColorMap[(unsigned char)object.Owner];
    IsSelected = (float)ConvertMask(object.IsSelectedMask);
    IsRepairing = (float)object.IsRepairing;
    Cloak = object.Cloak;
    if (object.IsPrimaryFactory)
    {
        Pips[0] = 3U; // PIP_PRIMARY + 1
        std::fill_n(Pips + 1, MAX_OBJECT_PIPS - 1, 0);
    }
    else
    {
        for (int i = 0; i < object.MaxPips; ++i)
            Pips[i] = object.Pips[i] + 1;
        std::fill_n(Pips + object.MaxPips, MAX_OBJECT_PIPS - object.MaxPips, 0);
    }
    ControlGroup = object.ControlGroup;
}

void DynamicObject::Assign(const CNCDynamicMapEntryStruct& entry)
{
    AssetName = dynamic_object_names.at(entry.AssetName);
    PositionX = (float)entry.PositionX;
    PositionY = (float)entry.PositionY;
    Strength = 0.f;
    ShapeIndex = entry.ShapeIndex;
    Owner = HouseColorMap[(unsigned char)entry.Owner];
    IsSelected = 0.f;
    IsRepairing = 0.f;
    Cloak = 0;
    std::fill_n(Pips, MAX_OBJECT_PIPS, 0);
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
                if (!(map.StaticCells[y][x].AssetName >= 232 && map.StaticCells[y][x].AssetName <= 243)) // tiberium
                {
                    map.StaticCells[y][x] = *entry;
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


SidebarEntry::SidebarEntry() : AssetName(0), Progress(0.f), Constructing(0.0f), ConstructionOnHold(0.0f), Busy(0.0f)
{
}

SidebarEntry& SidebarEntry::operator=(const CNCSidebarEntryStruct& entry)
{
    AssetName = dynamic_object_names.at(entry.AssetName);
    if (entry.Completed)
        Progress = 1.f;
    else
        Progress = entry.Progress;
    Cost = (float)entry.Cost;
    BuildTime = (float)entry.BuildTime;
    Constructing = entry.Constructing ? 1.f : 0.f;
    ConstructionOnHold = entry.ConstructionOnHold ? 1.f : 0.f;
    Busy = entry.Busy ? 1.f : 0.f;
    BuildableType = entry.BuildableType;
    BuildableID = entry.BuildableID;

    return *this;
}


SideBar& SideBar::operator=(const CNCSidebarStruct& sidebar)
{
    Members.Credits = (float)sidebar.Credits;
    Members.PowerProduced = (float)sidebar.PowerProduced;
    Members.PowerDrained = (float)sidebar.PowerDrained;

    Members.RepairBtnEnabled = sidebar.RepairBtnEnabled ? 1.0f : 0.0f;
    Members.SellBtnEnabled = sidebar.SellBtnEnabled ? 1.0f : 0.0f;
    Members.RadarMapActive = sidebar.RadarMapActive ? 1.0f : 0.0f;
    Entries.resize(sidebar.EntryCount[0] + sidebar.EntryCount[1]);
    std::copy_n(sidebar.Entries, Entries.size(), Entries.begin());

    return *this;
}


void RenderPOV(VectorRepresentation& target, const VectorRepresentation& source, const CNCShroudStruct* shroud, std::int32_t Owner)
{
    target.map.MapCellX = source.map.MapCellX;
    target.map.MapCellY = source.map.MapCellY;
    target.map.MapCellWidth = source.map.MapCellWidth;
    target.map.MapCellHeight = source.map.MapCellHeight;
    target.map.OriginalMapCellX = source.map.OriginalMapCellX;
    target.map.OriginalMapCellY = source.map.OriginalMapCellY;
    target.map.OriginalMapCellWidth = source.map.OriginalMapCellWidth;
    target.map.OriginalMapCellHeight = source.map.OriginalMapCellHeight;

    const auto offset = (source.map.OriginalMapCellY - source.map.MapCellY) * source.map.MapCellWidth + source.map.OriginalMapCellX - source.map.MapCellX;
    {
        const auto column_stepping = source.map.MapCellWidth - source.map.OriginalMapCellWidth;
        auto shroud_ptr = shroud->Entries + offset;

        for (int y = 0; y < target.map.OriginalMapCellHeight; ++y, shroud_ptr += column_stepping)
        {
            for (int x = 0; x < target.map.OriginalMapCellWidth; ++x, ++shroud_ptr)
            {
                if (shroud_ptr->IsVisible && shroud_ptr->ShadowIndex == (char)(-1))
                    target.map.StaticCells[y][x] = source.map.StaticCells[y][x];
            }
        }
    }
    target.dynamic_objects.clear();

    for (const auto& source_object : source.dynamic_objects)
    {
        const auto x = (int)source_object.PositionX / 24;
        const auto y = (int)source_object.PositionY / 24;
        const auto& shroud_ptr = shroud->Entries[offset + y * source.map.MapCellWidth + x];
        if (shroud_ptr.IsVisible && shroud_ptr.ShadowIndex == (char)(-1))
        {
            if (source_object.Cloak == 2U && source_object.Owner != Owner) // TODO one should see the cloaked allies too
                continue;
            target.dynamic_objects.emplace_back(source_object);
            auto& target_object = target.dynamic_objects.back();
            
            target_object.IsSelected = (((unsigned int)target_object.IsSelected) & (1 << Owner)) ? 1.f : 0.f;

            if (target_object.Owner != Owner)
            {
                std::fill_n(target_object.Pips, MAX_OBJECT_PIPS, 0);
                target_object.ControlGroup = 255U;
            }
        }
    }
}
