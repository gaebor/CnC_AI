#include <Windows.h>

#include "DLLInterface.h"

#include <vector>
#include <string>
#include <iostream>

typedef void(__cdecl* CNC_Event_Callback_Type)(const EventCallbackStruct& event);
typedef unsigned __int64 uint64;
typedef __int64 int64;



class GamePlay
{
    const HMODULE dll_handle;
    std::vector<CNCPlayerInfoStruct> players;
    const char* const content_directory;
    static const CNCRulesDataStruct rule_data_struct;
    
    unsigned int (__cdecl* CNC_Version)(unsigned int version_in);
    void (__cdecl* CNC_Init)(const char* command_line, CNC_Event_Callback_Type event_callback);
    void (__cdecl* CNC_Config)(const CNCRulesDataStruct& rules);
    void (__cdecl* CNC_Add_Mod_Path)(const char* mod_path);
    bool (__cdecl* CNC_Get_Visible_Page)(unsigned char* buffer_in, unsigned int& width, unsigned int& height);
    bool (__cdecl* CNC_Get_Palette)(unsigned char(&palette_in)[256][3]);
    bool (__cdecl* CNC_Start_Instance)(int scenario_index, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
    bool (__cdecl* CNC_Start_Instance_Variation)(int scenario_index, int scenario_variation, int scenario_direction, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
    bool (__cdecl* CNC_Start_Custom_Instance)(const char* content_directory, const char* directory_path, const char* scenario_name, int build_level, bool multiplayer);
    bool (__cdecl* CNC_Advance_Instance)(uint64 player_id);
    bool (__cdecl* CNC_Get_Game_State)(GameStateRequestEnum state_type, uint64 player_id, unsigned char* buffer_in, unsigned int buffer_size);
    bool (__cdecl* CNC_Read_INI)(int scenario_index, int scenario_variation, int scenario_direction, const char* content_directory, const char* override_map_name, char* ini_buffer, int _ini_buffer_size);
    void (__cdecl* CNC_Set_Home_Cell)(int x, int y, uint64 player_id);
    void (__cdecl* CNC_Handle_Game_Request)(GameRequestEnum request_type);
    void (__cdecl* CNC_Handle_Game_Settings_Request)(int health_bar_display_mode, int resource_bar_display_mode);
    void (__cdecl* CNC_Handle_Input)(InputRequestEnum mouse_event, unsigned char special_key_flags, uint64 player_id, int x1, int y1, int x2, int y2);
    void (__cdecl* CNC_Handle_Structure_Request)(StructureRequestEnum request_type, uint64 player_id, int object_id);
    void (__cdecl* CNC_Handle_Unit_Request)(UnitRequestEnum request_type, uint64 player_id);
    void (__cdecl* CNC_Handle_Sidebar_Request)(SidebarRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, short cell_x, short cell_y);
    void (__cdecl* CNC_Handle_SuperWeapon_Request)(SuperWeaponRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, int x1, int y1);
    void (__cdecl* CNC_Handle_ControlGroup_Request)(ControlGroupRequestEnum request_type, uint64 player_id, unsigned char control_group_index);
    void (__cdecl* CNC_Handle_Debug_Request)(DebugRequestEnum debug_request_type, uint64 player_id, const char* object_name, int x, int y, bool unshroud, bool enemy);
    void (__cdecl* CNC_Handle_Beacon_Request)(BeaconRequestEnum beacon_request_type, uint64 player_id, int pixel_x, int pixel_y);
    bool (__cdecl* CNC_Set_Multiplayer_Data)(int scenario_index, CNCMultiplayerOptionsStruct& game_options, int num_players, CNCPlayerInfoStruct* player_list, int max_players);
    bool (__cdecl* CNC_Clear_Object_Selection)(uint64 player_id);
    bool (__cdecl* CNC_Select_Object)(uint64 player_id, int object_type_id, int object_to_select_id);
    bool (__cdecl* CNC_Save_Load)(bool save, const char* file_path_and_name, const char* game_type);
    void (__cdecl* CNC_Set_Difficulty)(int difficulty);
    void (__cdecl* CNC_Handle_Player_Switch_To_AI)(uint64 player_id);
    void (__cdecl* CNC_Handle_Human_Team_Wins)(uint64 player_id);
    void (__cdecl* CNC_Start_Mission_Timer)(int time);
    bool (__cdecl* CNC_Get_Start_Game_Info)(uint64 player_id, int& start_location_waypoint_index);

    std::vector<unsigned char> buffer;
    std::vector<unsigned char[256][3]> palette;

public:
    GamePlay(const TCHAR* dll_filename, const char* content_directory = "-CDDATA\\CNCDATA\\TIBERIAN_DAWN\\CD1")
        : dll_handle(LoadLibrary(dll_filename)), content_directory(content_directory),
        CNC_Version((decltype(CNC_Version))GetProcAddress(dll_handle, "CNC_Version")),
        CNC_Init((decltype(CNC_Init))GetProcAddress(dll_handle, "CNC_Init")),
        CNC_Config((decltype(CNC_Config))GetProcAddress(dll_handle, "CNC_Config")),
        CNC_Add_Mod_Path((decltype(CNC_Add_Mod_Path))GetProcAddress(dll_handle, "CNC_Add_Mod_Path")),
        CNC_Get_Visible_Page((decltype(CNC_Get_Visible_Page))GetProcAddress(dll_handle, "CNC_Get_Visible_Page")),
        CNC_Get_Palette((decltype(CNC_Get_Palette))GetProcAddress(dll_handle, "CNC_Get_Palette")),
        CNC_Start_Instance((decltype(CNC_Start_Instance))GetProcAddress(dll_handle, "CNC_Start_Instance")),
        CNC_Start_Instance_Variation((decltype(CNC_Start_Instance_Variation))GetProcAddress(dll_handle, "CNC_Start_Instance_Variation")),
        CNC_Start_Custom_Instance((decltype(CNC_Start_Custom_Instance))GetProcAddress(dll_handle, "CNC_Start_Custom_Instance")),
        CNC_Advance_Instance((decltype(CNC_Advance_Instance))GetProcAddress(dll_handle, "CNC_Advance_Instance")),
        CNC_Get_Game_State((decltype(CNC_Get_Game_State))GetProcAddress(dll_handle, "CNC_Get_Game_State")),
        CNC_Read_INI((decltype(CNC_Read_INI))GetProcAddress(dll_handle, "CNC_Read_INI")),
        CNC_Set_Home_Cell((decltype(CNC_Set_Home_Cell))GetProcAddress(dll_handle, "CNC_Set_Home_Cell")),
        CNC_Handle_Game_Request((decltype(CNC_Handle_Game_Request))GetProcAddress(dll_handle, "CNC_Handle_Game_Request")),
        CNC_Handle_Game_Settings_Request((decltype(CNC_Handle_Game_Settings_Request))GetProcAddress(dll_handle, "CNC_Handle_Game_Settings_Request")),
        CNC_Handle_Input((decltype(CNC_Handle_Input))GetProcAddress(dll_handle, "CNC_Handle_Input")),
        CNC_Handle_Structure_Request((decltype(CNC_Handle_Structure_Request))GetProcAddress(dll_handle, "CNC_Handle_Structure_Request")),
        CNC_Handle_Unit_Request((decltype(CNC_Handle_Unit_Request))GetProcAddress(dll_handle, "CNC_Handle_Unit_Request")),
        CNC_Handle_Sidebar_Request((decltype(CNC_Handle_Sidebar_Request))GetProcAddress(dll_handle, "CNC_Handle_Sidebar_Request")),
        CNC_Handle_SuperWeapon_Request((decltype(CNC_Handle_SuperWeapon_Request))GetProcAddress(dll_handle, "CNC_Handle_SuperWeapon_Request")),
        CNC_Handle_ControlGroup_Request((decltype(CNC_Handle_ControlGroup_Request))GetProcAddress(dll_handle, "CNC_Handle_ControlGroup_Request")),
        CNC_Handle_Debug_Request((decltype(CNC_Handle_Debug_Request))GetProcAddress(dll_handle, "CNC_Handle_Debug_Request")),
        CNC_Handle_Beacon_Request((decltype(CNC_Handle_Beacon_Request))GetProcAddress(dll_handle, "CNC_Handle_Beacon_Request")),
        CNC_Set_Multiplayer_Data((decltype(CNC_Set_Multiplayer_Data))GetProcAddress(dll_handle, "CNC_Set_Multiplayer_Data")),
        CNC_Clear_Object_Selection((decltype(CNC_Clear_Object_Selection))GetProcAddress(dll_handle, "CNC_Clear_Object_Selection")),
        CNC_Select_Object((decltype(CNC_Select_Object))GetProcAddress(dll_handle, "CNC_Select_Object")),
        CNC_Save_Load((decltype(CNC_Save_Load))GetProcAddress(dll_handle, "CNC_Save_Load")),
        CNC_Set_Difficulty((decltype(CNC_Set_Difficulty))GetProcAddress(dll_handle, "CNC_Set_Difficulty")),
        CNC_Handle_Player_Switch_To_AI((decltype(CNC_Handle_Player_Switch_To_AI))GetProcAddress(dll_handle, "CNC_Handle_Player_Switch_To_AI")),
        CNC_Handle_Human_Team_Wins((decltype(CNC_Handle_Human_Team_Wins))GetProcAddress(dll_handle, "CNC_Handle_Human_Team_Wins")),
        CNC_Start_Mission_Timer((decltype(CNC_Start_Mission_Timer))GetProcAddress(dll_handle, "CNC_Start_Mission_Timer")),
        CNC_Get_Start_Game_Info((decltype(CNC_Get_Start_Game_Info))GetProcAddress(dll_handle, "CNC_Get_Start_Game_Info")),
        buffer(4 * 1024 * 1024), palette(1)
    {
        if (dll_handle == NULL)
            return;

        CNC_Init(content_directory, NULL);
        CNC_Config(rule_data_struct);

    }
    ~GamePlay() {
        if (dll_handle != NULL)
        {
            FreeLibrary(dll_handle);
        }
    }
    bool is_initialized()const { return dll_handle != NULL; }
    void add_player(const CNCPlayerInfoStruct& player)
    {
        players.emplace_back(player);
    }
    bool retrieve_players_info()
    {
        for (auto& player : players)
        {
            if (!CNC_Get_Game_State(GAME_STATE_PLAYER_INFO, player.GlyphxPlayerID, (unsigned char*)(&player), sizeof(player) + 33))
                return false;
        }
        return true;
    }
    bool init_palette()
    {
        if (CNC_Get_Palette(palette[0]))
        {
            for (auto& i : palette[0])
                for (auto& j : i)
                    j *= 4;
            return true;
        }
        else
            return false;
    }

    bool start_game(const CNCMultiplayerOptionsStruct& multiplayer_options, int scenario_index, int difficulty=0)
    {
        if (!is_initialized())
            return false;

        CNCMultiplayerOptionsStruct multiplayer_options_copy(multiplayer_options);
        if (!CNC_Set_Multiplayer_Data(
            scenario_index,
            multiplayer_options_copy,
            int(players.size()),
            players.data(),
            6
        ))
        {
            return false;
        }

        if (!CNC_Start_Instance_Variation(scenario_index, -1, 0, 7, "MULTI", "GAME_GLYPHX_MULTIPLAYER", content_directory+3, -1, ""))
        {
            return false;
        }

        CNC_Set_Difficulty(difficulty);

        for (auto& player : players)
        {
            if (!CNC_Get_Start_Game_Info(player.GlyphxPlayerID, player.StartLocationIndex))
                return false;
        }

        if (!retrieve_players_info())
            return false;

        if (!init_palette())
            return false;

        CNC_Handle_Game_Request(INPUT_GAME_LOADING_DONE);

        return true;
    }
};

const CNCRulesDataStruct GamePlay::rule_data_struct = { {
    {1.2f, 1.2f, 1.2f, 0.3f, 0.8f, 0.8f, 0.6f, 0.001f, 0.001f, false, true, true},
    {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.02f, 0.03f, true, true, true},
    {0.9f, 0.9f, 0.9f, 1.05f, 1.05f, 1.f, 1.f, 0.05f, 0.1f, true, true, true}
} };

int wmain(const int argc, const TCHAR* argv[])
{
    const TCHAR* dll_name = TEXT("TiberianDawn.dll");
    if (argc > 1)
    {
        dll_name = argv[1];
    }
    const TCHAR* install_directory = TEXT("C:\\Program Files (x86)\\Steam\\steamapps\\common\\CnCRemastered");
    if (argc > 2)
    {
        install_directory = argv[2];
    }
    SetCurrentDirectory(install_directory);
    GamePlay TD(dll_name);

    if (!TD.is_initialized())
        return 1;

    CNCPlayerInfoStruct player1 = { "gaebor",
        0, // GOOD
        0, // color
        271828182,
             0, // Team
            127, //StartLocationIndex
    };
    player1.IsAI = false;

    CNCPlayerInfoStruct player2 = { "ai1",
    1, // BAD  
    271828182,
         1,
        0,
        1,
        true,
        127 };
    CNCMultiplayerOptionsStruct multiplayer_options = { 2,
        1,
        5000,
        1,
        0,
         0,
         1,
         0,
         false,
         false,
         true,
         false,
         true,
         false,
         false,
         true };

    TD.add_player(player1);
    TD.add_player(player2);

    if (!TD.start_game(multiplayer_options, 50))
    {
        return 1;
    }

    return 0;
}