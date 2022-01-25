# What engine can retrieve from DLL
Via `CNC_Get_Game_State`

These calls basically copy a struct to caller's memory.

 * `GameStateRequestEnum` defines what to copy
 * each calls a `DLLExportClass::Get_[\w]+_State`
   * except `GAME_STATE_STATIC_MAP`
 * each request has a corresponding struct that is being copied

## Static Map
 * `GameStateRequestEnum::GAME_STATE_STATIC_MAP`
 * `CNCMapDataStruct`
 * _map width * map height_ of `CNCStaticCellStruct`
 * independent of player
 * contains tile asset name and icon number
 * does not change during a game
   * except when a bridge collapses
 
## Dynamic Map
 * `GameStateRequestEnum::GAME_STATE_DYNAMIC_MAP`
 * `DLLExportClass::Get_Dynamic_Map_State`
 * `CNCDynamicMapStruct`
 * contains variable length of `CNCDynamicMapEntryStruct`
 * looks like it is independent of player because doesn't use the variable `player_id`
 * content types:
   * smudges
     * craters, scorch
     * bibs: trampled grass under and in front of builds
   * overlays
     * walls
     * crates
     * tiberium
     * plant fields
     * road
   * flags
 * each entry contains
   * `HousesType Owner` stored as `char`
   * position, both in pixel and in cell
   * shape (?)
   * `Type` is `SmudgeType` or `OverlayType` stored as `short`
     * or `-1` for flag
   * `char AssetName[16]`
 
## Layers
 * `GameStateRequestEnum::GAME_STATE_LAYERS`
 * `DLLExportClass::Get_Layer_State`
 * `CNCObjectListStruct`
 * contains:
   * _"static"_ elements
     * trees
     * buildings
   * _"dynamic"_ elements
     * _"owned"_
       * infantry, units (naval and vehicles), aircrafts
     * "unowned"
       * there is an owner, but not with color
         * animations: gunfire, smoke
         * bullets, projectiles
 * independent of player
   * enemy units outside of shroud do show!
 * contains variable length of `CNCObjectStruct`
 * each contains:
   * `ID` for referencing it in game logic
   * `DllObjectTypeEnum Type`
   * position, shape
   * (`Rotation` stored as `unsigned char`)
   * `HousesType Owner` stored as `char`
   * `Strength` (`MaxStrength`) if it is destructible
   * pips: dots marking how full it is
   * in case of buildings: `short OccupyList[MAX_OCCUPY_CELLS]`
     * although, its better to retrieve the cells of the building from `Occupier`
   * trees can occupy more than one cell
     * but they don't have an `OccupyList`
     * it is worth lookung up those in `Occupier` as well
   * other types only occupy at most one cell right underneath
   * `unsigned int IsSelectedMask`: bit field with `Owner` bit shifts

## Occupier
 * `GameStateRequestEnum::GAME_STATE_OCCUPIER`
 * `DLLExportClass::Get_Occupier_State`
 * `CNCOccupierHeaderStruct` is kind of a vector of vector of `CNCOccupierObjectStruct`
 * independent of player
 * contains only `Type` and `ID`
 * These infos are contained in the "Layers"
   * here the same info is listed by map cell

## Shroud
 * `GameStateRequestEnum::GAME_STATE_SHROUD`
 * `DLLExportClass::Get_Shroud_State`
 * `CNCShroudStruct` is a vector of `CNCShroudEntryStruct`
   * size _map width * map height_
 * player dependent
 * contains visibility, jammed, mapped, shadow type

## Sidebar
 * `GameStateRequestEnum::GAME_STATE_SIDEBAR`
 * `DLLExportClass::Get_Sidebar_State`
 * `CNCSidebarStruct`
 * contains two columns of `CNCSidebarEntryStruct`
 * player dependent
 * each entry contains:
   * cost, time to build
   * build status
   * `Type` and `ID` to identify later
   * for buildings: `PowerProvided = Power - Drain`

## Placement
 * `GameStateRequestEnum::GAME_STATE_PLACEMENT`
 * `DLLExportClass::Get_Sidebar_State`
 * `CNCPlacementInfoStruct` contains a list of `CNCPlacementCellInfoStruct`
 * If a placement is started then it contains a list of all cells where a building could be placed
 * player dependent


 ## Player Info
 * `GameStateRequestEnum::GAME_STATE_PLAYER_INFO`
 * `DLLExportClass::Get_Player_Info_State`
 * `CNCPlayerInfoStruct`
