#include <Windows.h>

#include <string>
#include <vector>
#include <algorithm>

#include "DllInterface.h"
#include "data_utils.hpp"
#include "VectorInterface.hpp"

typedef void(__cdecl* CNC_Event_Callback_Type)(const EventCallbackStruct& event);
typedef unsigned __int64 uint64;
typedef __int64 int64;

// unsigned int(__cdecl* CNC_Version)(unsigned int version_in);
void(__cdecl* CNC_Init)(const char* command_line, CNC_Event_Callback_Type event_callback);
void(__cdecl* CNC_Config)(const CNCRulesDataStruct& rules);
// void(__cdecl* CNC_Add_Mod_Path)(const char* mod_path);
bool(__cdecl* CNC_Get_Visible_Page)(unsigned char* buffer_in, unsigned int& width, unsigned int& height);
bool(__cdecl* CNC_Get_Palette)(unsigned char(&palette_in)[256][3]);
// bool(__cdecl* CNC_Start_Instance)(int scenario_index, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
bool(__cdecl* CNC_Start_Instance_Variation)(int scenario_index, int scenario_variation, int scenario_direction, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
bool(__cdecl* CNC_Start_Custom_Instance)(const char* content_directory, const char* directory_path, const char* scenario_name, int build_level, bool multiplayer);
bool(__cdecl* CNC_Advance_Instance)(uint64 player_id);
bool(__cdecl* CNC_Get_Game_State)(GameStateRequestEnum state_type, uint64 player_id, unsigned char* buffer_in, unsigned int buffer_size);
// bool(__cdecl* CNC_Read_INI)(int scenario_index, int scenario_variation, int scenario_direction, const char* content_directory, const char* override_map_name, char* ini_buffer, int _ini_buffer_size);
// void(__cdecl* CNC_Set_Home_Cell)(int x, int y, uint64 player_id);
void(__cdecl* CNC_Handle_Game_Request)(GameRequestEnum request_type);
void(__cdecl* CNC_Handle_Game_Settings_Request)(int health_bar_display_mode, int resource_bar_display_mode);
void(__cdecl* CNC_Handle_Input)(InputRequestEnum mouse_event, unsigned char special_key_flags, uint64 player_id, int x1, int y1, int x2, int y2);
void(__cdecl* CNC_Handle_Structure_Request)(StructureRequestEnum request_type, uint64 player_id, int object_id);
void(__cdecl* CNC_Handle_Unit_Request)(UnitRequestEnum request_type, uint64 player_id);
void(__cdecl* CNC_Handle_Sidebar_Request)(SidebarRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, short cell_x, short cell_y);
void(__cdecl* CNC_Handle_SuperWeapon_Request)(SuperWeaponRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, int x1, int y1);
void(__cdecl* CNC_Handle_ControlGroup_Request)(ControlGroupRequestEnum request_type, uint64 player_id, unsigned char control_group_index);
// void(__cdecl* CNC_Handle_Debug_Request)(DebugRequestEnum debug_request_type, uint64 player_id, const char* object_name, int x, int y, bool unshroud, bool enemy);
void(__cdecl* CNC_Handle_Beacon_Request)(BeaconRequestEnum beacon_request_type, uint64 player_id, int pixel_x, int pixel_y);
bool(__cdecl* CNC_Set_Multiplayer_Data)(int scenario_index, CNCMultiplayerOptionsStruct& game_options, int num_players, CNCPlayerInfoStruct* player_list, int max_players);
// bool(__cdecl* CNC_Clear_Object_Selection)(uint64 player_id);
// bool(__cdecl* CNC_Select_Object)(uint64 player_id, int object_type_id, int object_to_select_id);
bool(__cdecl* CNC_Save_Load)(bool save, const char* file_path_and_name, const char* game_type);
void(__cdecl* CNC_Set_Difficulty)(int difficulty);
// void(__cdecl* CNC_Handle_Player_Switch_To_AI)(uint64 player_id);
// void(__cdecl* CNC_Handle_Human_Team_Wins)(uint64 player_id);
// void(__cdecl* CNC_Start_Mission_Timer)(int time);
bool(__cdecl* CNC_Get_Start_Game_Info)(uint64 player_id, int& start_location_waypoint_index);

HMODULE dll_handle;
std::vector<CNCPlayerInfoStruct> players;
std::vector<unsigned char> orginal_houses;

std::string content_directory;
std::vector<unsigned char> static_map_buffer(4 * 1024 * 1024);
std::vector<unsigned char> dynamic_map_buffer(4 * 1024 * 1024);
std::vector<unsigned char> layers_buffer(4 * 1024 * 1024);
std::vector<unsigned char> sidebar_buffer(4 * 1024 * 1024);

static const CNCRulesDataStruct rule_data_struct = { {
    {1.2f, 1.2f, 1.2f, 0.3f, 0.8f, 0.8f, 0.6f, 0.001f, 0.001f, false, true, true},
    {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.02f, 0.03f, true, true, true},
    {0.9f, 0.9f, 0.9f, 1.05f, 1.05f, 1.f, 1.f, 0.05f, 0.1f, true, true, true}
} };

std::wstring FromUtf8(const char* str) noexcept
{
    static std::wstring wstr;
    const int wchars_num = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
    if (wchars_num > 0)
    {
        wstr.resize(wchars_num);
        if (MultiByteToWideChar(CP_UTF8, 0, str, -1, &wstr.front(), wchars_num) > 0)
        {
            return wstr; // copy
        }
    }
    return std::wstring();
}

extern "C" __declspec(dllexport) bool __cdecl ChDir(const char* dirname_utf8)
{
    const auto dirname_unicode = FromUtf8(dirname_utf8);
    return SetCurrentDirectoryW(dirname_unicode.data()) != 0;
}

#define LoadSymbolFromDll(name) {name = (decltype(name))GetProcAddress(dll_handle, #name);\
                                if (name == NULL) return false;}

extern "C" __declspec(dllexport) bool __cdecl Init(
    const char* dll_filename_utf8, 
    const char* content_directory_ascii)
{
    {
        const auto dll_filename = FromUtf8(dll_filename_utf8);
        dll_handle = LoadLibraryW(dll_filename.data());
    }
    content_directory = content_directory_ascii + 3;
    if (dll_handle == NULL)
        return false;

    LoadSymbolFromDll(CNC_Init);
    LoadSymbolFromDll(CNC_Config);
    LoadSymbolFromDll(CNC_Start_Instance_Variation);
    LoadSymbolFromDll(CNC_Start_Custom_Instance);
    LoadSymbolFromDll(CNC_Advance_Instance);
    LoadSymbolFromDll(CNC_Get_Game_State);
    LoadSymbolFromDll(CNC_Handle_Game_Request);
    LoadSymbolFromDll(CNC_Handle_Game_Settings_Request);
    LoadSymbolFromDll(CNC_Handle_Input);
    LoadSymbolFromDll(CNC_Handle_Structure_Request);
    LoadSymbolFromDll(CNC_Handle_Unit_Request);
    LoadSymbolFromDll(CNC_Handle_Sidebar_Request);
    LoadSymbolFromDll(CNC_Handle_SuperWeapon_Request);
    LoadSymbolFromDll(CNC_Handle_ControlGroup_Request);
    LoadSymbolFromDll(CNC_Handle_Beacon_Request);
    LoadSymbolFromDll(CNC_Set_Multiplayer_Data);
    LoadSymbolFromDll(CNC_Save_Load);
    LoadSymbolFromDll(CNC_Set_Difficulty);
    LoadSymbolFromDll(CNC_Get_Start_Game_Info);
    LoadSymbolFromDll(CNC_Get_Palette);
    LoadSymbolFromDll(CNC_Get_Visible_Page);
    
    CNC_Init(content_directory_ascii, NULL);
    CNC_Config(rule_data_struct);

    return true;
}


extern "C" __declspec(dllexport) void __cdecl AddPlayer(const CNCPlayerInfoStruct& player)
{
    players.emplace_back(player);
    orginal_houses.emplace_back(player.House);
}

bool retrieve_players_info()
{
    for (auto& player : players)
    {
        if (!CNC_Get_Game_State(
            GAME_STATE_PLAYER_INFO,
            player.GlyphxPlayerID,
            reinterpret_cast<unsigned char*>(&player),
            sizeof(player) + 33))
        {
            return false;
        }
        HouseColorMap[player.House] = (std::remove_extent<decltype(HouseColorMap)>::type)player.ColorIndex;
    }
    return true;
}

extern "C" __declspec(dllexport) bool __cdecl StartGame(
    const CNCMultiplayerOptionsStruct & multiplayer_options,
    int scenario_index, int build_level, int difficulty)
{
    if (!CNC_Set_Multiplayer_Data(scenario_index, const_cast<CNCMultiplayerOptionsStruct&>(multiplayer_options), (int)players.size(), players.data(), 6))
    {
        return false;
    }
    if (!CNC_Start_Instance_Variation(
        scenario_index,
        -1, 0, build_level,
        "MULTI", "GAME_GLYPHX_MULTIPLAYER",
        content_directory.data(),
        -1, "")
        )
    {
        return false;
    }

    CNC_Set_Difficulty(difficulty);

    for (auto& player : players)
    {
        if (!CNC_Get_Start_Game_Info(player.GlyphxPlayerID, player.StartLocationIndex))
        {
            return false;
        }
    }
    
    CNC_Handle_Game_Request(INPUT_GAME_LOADING_DONE);

    return retrieve_players_info();
}

unsigned char CalculateScores()
{
    std::vector<int> scores(players.size(), 0);
    CommonVectorRepresentation game_state;

    if (!CNC_Get_Game_State(GAME_STATE_STATIC_MAP, 0, static_map_buffer.data(), static_map_buffer.size()))
        return 0xff;
    if (!CNC_Get_Game_State(GAME_STATE_DYNAMIC_MAP, 0, dynamic_map_buffer.data(), dynamic_map_buffer.size()))
        return 0xff;
    if (!CNC_Get_Game_State(GAME_STATE_LAYERS, 0, layers_buffer.data(), layers_buffer.size()))
        return 0xff;

    game_state.Render(
        reinterpret_cast<const CNCMapDataStruct*>(static_map_buffer.data()),
        reinterpret_cast<const CNCDynamicMapStruct*>(dynamic_map_buffer.data()),
        reinterpret_cast<const CNCObjectListStruct*>(layers_buffer.data())
    );

    for (const auto& unit : game_state.dynamic_objects)
    {
        const auto owner = unit.Owner;
        for (size_t i = 0; i < players.size(); ++i)
        {
            if (players[i].ColorIndex == owner)
            {
                scores[i] += buildables.at(unit.AssetName);
                break;
            }
        }
    }

    for (size_t i = 0; i < players.size(); ++i)
    {
        if (!CNC_Get_Game_State(GAME_STATE_SIDEBAR, players[i].GlyphxPlayerID, sidebar_buffer.data(), sidebar_buffer.size()))
            return 0xff;
        SideBar s;
        s = *reinterpret_cast<const CNCSidebarStruct*>(sidebar_buffer.data());
        // scores[i] += s.Credits;
    }

    unsigned char loser_mask = 0;
    auto max_score = scores[0];
    for (auto score : scores)
        if (score > max_score)
            max_score = score;
    for (size_t i = 0; i < players.size(); ++i)
    {
        loser_mask |= (scores[i] < max_score ? 1 : 0) << i;
    }
    return loser_mask;
}

extern "C" __declspec(dllexport) unsigned char __cdecl GetGameResult()
{
    retrieve_players_info();

    unsigned char loser_mask = 0;
    for (size_t i = 0; i < players.size(); ++i)
    {
        loser_mask |= (players[i].IsDefeated ? 1 : 0) << i;
    }
    if (loser_mask != 0U)
    {
        return loser_mask;
    }
    else
    {
        return CalculateScores();
    }
}


extern "C" __declspec(dllexport) void __cdecl FreeDll()
{
    FreeLibrary(dll_handle);
}

extern "C" __declspec(dllexport) bool __cdecl Advance()
{
    return CNC_Advance_Instance(0);
}
