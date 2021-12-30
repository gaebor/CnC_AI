# CnCRemastered dll
## Init/setup
 * `CNC_Version`
 * `CNC_Init`
   * `command_line`: `-CD"%STEAM%\steamapps\common\CnCRemastered\DATA\CNCDATA\TIBERIAN_DAWN\CD1"`
     * this is the `content_directory` (IDK the encoding, not UTF-8)
   * `event_callback`: maybe it can be a lazy function
 * `CNC_Config`: I think this only affects AI players
 ```cpp
[
 {1.2f, 1.2f, 1.2f, 0.3f, 0.8f, 0.8f, 0.6f, 0.001f, 0.001f, false, true, true},
 {1f, 1f, 1f, 1f, 1f, 1f, 1f, 0.02f, 0.03f, true, true, true},
 {0.9f, 0.9f, 0.9f, 1.05f, 1.05f, 1f, 1f, 0.05f, 0.1f, true, true, true}
]
```
 * `CNC_Add_Mod_Path`
 * `CNC_Handle_Game_Request`
 * `CNC_Handle_Game_Settings_Request`: for some reason it is called two times
 * `CNC_Set_Difficulty`: only in singleplayer campaign
 * `CNC_Set_Multiplayer_Data`
 * `CNC_Set_Home_Cell` (?)

 ### 
 * `CNC_Start_Instance`
 * `CNC_Start_Instance_Variation`
 * `CNC_Start_Custom_Instance`: for custom defined maps (need to pass directory and filename of the map)

 These define the map. Also find starting positions randomly if `StartLocationIndex` was `127` in `CNC_Set_Multiplayer_Data`.

## engine requests data
 * `CNC_Get_Visible_Page`: copies image buffer of first player to caller's memory
    * only works if you play aganist AI in single player mode 
 * `CNC_Get_Palette`: copies the color palette of the legacy images
 * `CNC_Get_Game_State`: various queries about the game state
   * ```c++ DLLExportClass::Get_Layer_State```: `CNCObjectListStruct`
   * `DLLExportClass::Get_Sidebar_State`: `CNCSidebarStruct`
   * `DLLExportClass::Get_Placement_State`: `CNCPlacementInfoStruct`
   * `DLLExportClass::Get_Dynamic_Map_State`: `CNCDynamicMapStruct`
   * `DLLExportClass::Get_Shroud_State`: `CNCShroudStruct`
   * `DLLExportClass::Get_Occupier_State`
   * `DLLExportClass::Get_Player_Info_State`: basically copies the `CNCPlayerInfoStruct` of the required player
     * contains selection info too (`player_info->SelectedID`)
     * this can be used as `object_id` in `CNC_Handle_Structure_Request`
   * `STATIC_MAP`
 * `CNC_Get_Start_Game_Info`: In case the dll specified the starting position, this call copies the start position back to engine's memory.

## game logic
 * `CNC_Advance_Instance`
### user input
 * `CNC_Handle_Input`: mouse clicks, area
   * for example deploying MCV is registered as COMMAND_AT_POSITION
     * even if it is done by hotkey
   * `INPUT_REQUEST_SPECIAL_KEYS`
     * it is called each time any of the SHIFT+CTRL+ALT keys' state change
     * CTRL=1, ALT=2, SHIFT=4 bitfield
 * `CNC_Handle_Structure_Request`: repair/cell structures
   * `request_type`
     * `INPUT_STRUCTURE_REPAIR_START`: 1
   * **`object_id`** is the index of a `TFixedIHeapClass<BuildingClass> Buildings` global variable.
 * `CNC_Handle_Unit_Request`: stop/guard/scatter commands
   * `request_type`
     * `INPUT_UNIT_NONE`: 0
     * `INPUT_UNIT_SCATTER`: 1
     * `INPUT_UNIT_SELECT_NEXT`: 2
     * `INPUT_UNIT_SELECT_PREVIOUS`: 3
     * `INPUT_UNIT_GUARD_MODE`: 4
     * `INPUT_UNIT_STOP`: 5
 * `CNC_Handle_Sidebar_Request`
   * start/hold/cancel production (unit or building)
     * doesn't matter that you click on sidebar or press hotkey for a unit
   * start/cancel building placement
   * `buildable_type`: `RTTIType`
   * `buildable_id`: `UnitType`, `StructType`, `InfantryType`... depending on `buildable_type`
 * `CNC_Handle_SuperWeapon_Request` 
   * `buildable_type` should be `RTTI_SPECIAL`
   * `buildable_id` is `SpecialWeaponType`
 * `CNC_Handle_Beacon_Request`
 * `CNC_Handle_ControlGroup_Request`
   * `request_type`
     * `CONTROL_GROUP_REQUEST_CREATE`: 0
	   * `CONTROL_GROUP_REQUEST_TOGGLE`: 1
	   * `CONTROL_GROUP_REQUEST_ADDITIVE_SELECTION`: 2
   * `control_group_index`: 0-9
 * `CNC_Clear_Object_Selection` and `CNC_Select_Object` are not really reproducible, I can live without them. They are called when "select all units in the view" or "select all units on the map" or double click on a unit.
 *  not really reproducible

### unknown to DLL
 * move window
 * define/jump to map location bookmark
 * toggle sidebar
 * change sidebar tab
 * change radar map view
 * double click to select same units within your view
 * select all units within your view, select all of your units

CNC_Handle_Debug_Request
CNC_Read_INI
CNC_Save_Load
CNC_Handle_Player_Switch_To_AI
CNC_Handle_Human_Team_Wins
CNC_Start_Mission_Timer
