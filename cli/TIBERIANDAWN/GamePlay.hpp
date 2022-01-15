#pragma once
#include <Windows.h>

#include "DLLInterface.h"

#include <vector>
#include <string>

typedef void(__cdecl* CNC_Event_Callback_Type)(const EventCallbackStruct& event);
typedef unsigned __int64 uint64;
typedef __int64 int64;

class GamePlay
{
    const HMODULE dll_handle;
    std::vector<CNCPlayerInfoStruct> players;
    std::string content_directory;

    unsigned int(__cdecl* CNC_Version)(unsigned int version_in);
    void(__cdecl* CNC_Init)(const char* command_line, CNC_Event_Callback_Type event_callback);
    void(__cdecl* CNC_Config)(const CNCRulesDataStruct& rules);
    void(__cdecl* CNC_Add_Mod_Path)(const char* mod_path);
    bool(__cdecl* CNC_Get_Visible_Page)(unsigned char* buffer_in, unsigned int& width, unsigned int& height);
    bool(__cdecl* CNC_Get_Palette)(unsigned char(&palette_in)[256][3]);
    bool(__cdecl* CNC_Start_Instance)(int scenario_index, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
    bool(__cdecl* CNC_Start_Instance_Variation)(int scenario_index, int scenario_variation, int scenario_direction, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
    bool(__cdecl* CNC_Start_Custom_Instance)(const char* content_directory, const char* directory_path, const char* scenario_name, int build_level, bool multiplayer);
    bool(__cdecl* CNC_Advance_Instance)(uint64 player_id);
    bool(__cdecl* CNC_Get_Game_State)(GameStateRequestEnum state_type, uint64 player_id, unsigned char* buffer_in, unsigned int buffer_size);
    bool(__cdecl* CNC_Read_INI)(int scenario_index, int scenario_variation, int scenario_direction, const char* content_directory, const char* override_map_name, char* ini_buffer, int _ini_buffer_size);
    void(__cdecl* CNC_Set_Home_Cell)(int x, int y, uint64 player_id);
    void(__cdecl* CNC_Handle_Game_Request)(GameRequestEnum request_type);
    void(__cdecl* CNC_Handle_Game_Settings_Request)(int health_bar_display_mode, int resource_bar_display_mode);
    void(__cdecl* CNC_Handle_Input)(InputRequestEnum mouse_event, unsigned char special_key_flags, uint64 player_id, int x1, int y1, int x2, int y2);
    void(__cdecl* CNC_Handle_Structure_Request)(StructureRequestEnum request_type, uint64 player_id, int object_id);
    void(__cdecl* CNC_Handle_Unit_Request)(UnitRequestEnum request_type, uint64 player_id);
    void(__cdecl* CNC_Handle_Sidebar_Request)(SidebarRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, short cell_x, short cell_y);
    void(__cdecl* CNC_Handle_SuperWeapon_Request)(SuperWeaponRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, int x1, int y1);
    void(__cdecl* CNC_Handle_ControlGroup_Request)(ControlGroupRequestEnum request_type, uint64 player_id, unsigned char control_group_index);
    void(__cdecl* CNC_Handle_Debug_Request)(DebugRequestEnum debug_request_type, uint64 player_id, const char* object_name, int x, int y, bool unshroud, bool enemy);
    void(__cdecl* CNC_Handle_Beacon_Request)(BeaconRequestEnum beacon_request_type, uint64 player_id, int pixel_x, int pixel_y);
    bool(__cdecl* CNC_Set_Multiplayer_Data)(int scenario_index, CNCMultiplayerOptionsStruct& game_options, int num_players, CNCPlayerInfoStruct* player_list, int max_players);
    bool(__cdecl* CNC_Clear_Object_Selection)(uint64 player_id);
    bool(__cdecl* CNC_Select_Object)(uint64 player_id, int object_type_id, int object_to_select_id);
    bool(__cdecl* CNC_Save_Load)(bool save, const char* file_path_and_name, const char* game_type);
    void(__cdecl* CNC_Set_Difficulty)(int difficulty);
    void(__cdecl* CNC_Handle_Player_Switch_To_AI)(uint64 player_id);
    void(__cdecl* CNC_Handle_Human_Team_Wins)(uint64 player_id);
    void(__cdecl* CNC_Start_Mission_Timer)(int time);
    bool(__cdecl* CNC_Get_Start_Game_Info)(uint64 player_id, int& start_location_waypoint_index);

    std::vector<unsigned char> buffer;
    std::vector<unsigned char[256][3]> palette;

public:
    GamePlay(const TCHAR* dll_filename, const char* content_directory = "-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1");
    ~GamePlay();
    bool is_initialized()const;
    void add_player(const CNCPlayerInfoStruct& player);
    bool retrieve_players_info();
    bool init_palette();

    bool start_game(const CNCMultiplayerOptionsStruct& multiplayer_options, int scenario_index, int build_level = 7, int difficulty = 0);
    bool save_game(const char* filename);
    bool load_game(const char* filename);

    static unsigned char HouseColorMap[256];
};
