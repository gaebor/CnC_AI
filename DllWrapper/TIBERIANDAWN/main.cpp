#include <Windows.h>

#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

#include "DllInterface.h"
#include "data_utils.hpp"
#include "VectorInterface.hpp"

typedef void(__cdecl* CNC_Event_Callback_Type)(const EventCallbackStruct& event);
typedef unsigned __int64 uint64;
typedef __int64 int64;

extern "C" {
    void(__cdecl* CNC_Init)(const char* command_line, CNC_Event_Callback_Type event_callback) = NULL;
    void(__cdecl * CNC_Config)(const CNCRulesDataStruct & rules) = NULL;
    // void(__cdecl* CNC_Add_Mod_Path)(const char* mod_path);
    // bool(__cdecl* CNC_Start_Instance)(int scenario_index, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name);
    bool(__cdecl * CNC_Start_Instance_Variation)(int scenario_index, int scenario_variation, int scenario_direction, int build_level, const char* faction, const char* game_type, const char* content_directory, int sabotaged_structure, const char* override_map_name) = NULL;
    bool(__cdecl * CNC_Start_Custom_Instance)(const char* content_directory, const char* directory_path, const char* scenario_name, int build_level, bool multiplayer) = NULL;
    bool(__cdecl * CNC_Advance_Instance)(uint64 player_id) = NULL;
    bool(__cdecl * CNC_Get_Game_State)(GameStateRequestEnum state_type, uint64 player_id, unsigned char* buffer_in, unsigned int buffer_size) = NULL;
    // bool(__cdecl* CNC_Read_INI)(int scenario_index, int scenario_variation, int scenario_direction, const char* content_directory, const char* override_map_name, char* ini_buffer, int _ini_buffer_size);
    // void(__cdecl* CNC_Set_Home_Cell)(int x, int y, uint64 player_id);
    void(__cdecl * CNC_Handle_Game_Request)(GameRequestEnum request_type) = NULL;
    void(__cdecl * CNC_Handle_Game_Settings_Request)(int health_bar_display_mode, int resource_bar_display_mode) = NULL;
    void(__cdecl * CNC_Handle_Input)(InputRequestEnum mouse_event, unsigned char special_key_flags, uint64 player_id, int x1, int y1, int x2, int y2) = NULL;
    void(__cdecl * CNC_Handle_Structure_Request)(StructureRequestEnum request_type, uint64 player_id, int object_id) = NULL;
    void(__cdecl * CNC_Handle_Unit_Request)(UnitRequestEnum request_type, uint64 player_id) = NULL;
    void(__cdecl * CNC_Handle_Sidebar_Request)(SidebarRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, short cell_x, short cell_y) = NULL;
    void(__cdecl * CNC_Handle_SuperWeapon_Request)(SuperWeaponRequestEnum request_type, uint64 player_id, int buildable_type, int buildable_id, int x1, int y1) = NULL;
    void(__cdecl * CNC_Handle_ControlGroup_Request)(ControlGroupRequestEnum request_type, uint64 player_id, unsigned char control_group_index) = NULL;
    // void(__cdecl* CNC_Handle_Debug_Request)(DebugRequestEnum debug_request_type, uint64 player_id, const char* object_name, int x, int y, bool unshroud, bool enemy);
    void(__cdecl * CNC_Handle_Beacon_Request)(BeaconRequestEnum beacon_request_type, uint64 player_id, int pixel_x, int pixel_y) = NULL;
    bool(__cdecl * CNC_Set_Multiplayer_Data)(int scenario_index, CNCMultiplayerOptionsStruct & game_options, int num_players, CNCPlayerInfoStruct * player_list, int max_players) = NULL;
    // bool(__cdecl* CNC_Clear_Object_Selection)(uint64 player_id);
    // bool(__cdecl* CNC_Select_Object)(uint64 player_id, int object_type_id, int object_to_select_id);
    bool(__cdecl * CNC_Save_Load)(bool save, const char* file_path_and_name, const char* game_type) = NULL;
    void(__cdecl * CNC_Set_Difficulty)(int difficulty) = NULL;
    // void(__cdecl* CNC_Handle_Player_Switch_To_AI)(uint64 player_id);
    // void(__cdecl* CNC_Handle_Human_Team_Wins)(uint64 player_id);
    // void(__cdecl* CNC_Start_Mission_Timer)(int time);
    bool(__cdecl * CNC_Get_Start_Game_Info)(uint64 player_id, int& start_location_waypoint_index) = NULL;
    bool(__cdecl * CNC_Get_Visible_Page)(unsigned char* buffer_in, unsigned int& width, unsigned int& height) = NULL;
    bool(__cdecl * CNC_Get_Palette)(unsigned char(&palette_in)[256][3]) = NULL;
}

extern "C" {
    __declspec(dllexport) bool __cdecl ChDir(const char* dirname_utf8);
    __declspec(dllexport) bool __cdecl Init(const char* dll_filename_utf8, const char* content_directory_ascii);
    __declspec(dllexport) void __cdecl AddPlayer(const CNCPlayerInfoStruct&);
    __declspec(dllexport) bool __cdecl StartGame(const CNCMultiplayerOptionsStruct&, int scenario_index, int build_level, int difficulty);
    __declspec(dllexport) unsigned char __cdecl GetGameResult();
    __declspec(dllexport) void __cdecl FreeDll();
    __declspec(dllexport) bool __cdecl CNC_Get_Visible_Page_(unsigned char* buffer_in, unsigned int& width, unsigned int& height)
    {
        return CNC_Get_Visible_Page(buffer_in, width, height);
    }
    __declspec(dllexport) bool __cdecl CNC_Get_Palette_(unsigned char(&palette_in)[256][3])
    {
        return CNC_Get_Palette(palette_in);
    }
    __declspec(dllexport) bool __cdecl Advance();
    __declspec(dllexport) bool __cdecl GetCommonVectorRepresentation(VectorRepresentationView&);
    __declspec(dllexport) bool __cdecl GetPlayersVectorRepresentation(PlayerVectorRepresentationView* output);
    
    __declspec(dllexport) void __cdecl HandleSidebarRequest(size_t player_id, SidebarRequestEnum requestType, std::int32_t assetNameIndex);
    __declspec(dllexport) void __cdecl HandleInputRequest(size_t player_id, InputRequestEnum requestType, int x1, int y1);
}

HMODULE dll_handle;
std::vector<CNCPlayerInfoStruct> players;
std::vector<unsigned char> orginal_houses;
VectorRepresentation game_state;
std::vector<VectorRepresentation> players_view;
std::vector<SideBar> players_sidebar;

std::string content_directory;
std::vector<unsigned char> static_map_buffer(4 * 1024 * 1024);
std::vector<unsigned char> dynamic_map_buffer(4 * 1024 * 1024);
std::vector<unsigned char> layers_buffer(4 * 1024 * 1024);
std::vector<unsigned char> general_buffer(4 * 1024 * 1024);

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

bool __cdecl ChDir(const char* dirname_utf8)
{
    const auto dirname_unicode = FromUtf8(dirname_utf8);
    return SetCurrentDirectoryW(dirname_unicode.data()) != 0;
}

#define LoadSymbolFromDll(name) {name = (decltype(name))GetProcAddress(dll_handle, #name);\
                                if (name == NULL) return false;}

bool __cdecl Init(const char* dll_filename_utf8, const char* content_directory_ascii)
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

void __cdecl AddPlayer(const CNCPlayerInfoStruct& player)
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

bool __cdecl StartGame(
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

unsigned char __cdecl GetGameResult()
{
    retrieve_players_info();

    unsigned char loser_mask = 0;
    for (size_t i = 0; i < players.size(); ++i)
    {
        loser_mask |= (players[i].IsDefeated ? 1 : 0) << i;
    }
    return loser_mask;
}

void __cdecl FreeDll()
{
    FreeLibrary(dll_handle);
}

bool __cdecl Advance()
{
    return CNC_Advance_Instance(0);
}

bool __cdecl GetCommonVectorRepresentation(VectorRepresentationView& view)
{
    if (!CNC_Get_Game_State(GAME_STATE_STATIC_MAP, 0, static_map_buffer.data(), static_map_buffer.size()))
        return false;
    if (!CNC_Get_Game_State(GAME_STATE_DYNAMIC_MAP, 0, dynamic_map_buffer.data(), dynamic_map_buffer.size()))
        return false;
    if (!CNC_Get_Game_State(GAME_STATE_LAYERS, 0, layers_buffer.data(), layers_buffer.size()))
        return false;
    try {
        game_state.Render(
            reinterpret_cast<const CNCMapDataStruct*>(static_map_buffer.data()),
            reinterpret_cast<const CNCDynamicMapStruct*>(dynamic_map_buffer.data()),
            reinterpret_cast<const CNCObjectListStruct*>(layers_buffer.data())
        );
    }
    catch (const std::out_of_range&)
    {
        return false;
    }
    view = game_state;

    return true;
}

bool __cdecl GetPlayersVectorRepresentation(PlayerVectorRepresentationView* output)
{
    {
        VectorRepresentationView dummy;
        if (!GetCommonVectorRepresentation(dummy))
            return false;
    }
    players_view.resize(players.size());
    players_sidebar.resize(players.size());

    for (size_t i = 0; i < players.size(); ++i, ++output)
    {
        if (!CNC_Get_Game_State(GAME_STATE_SHROUD, players[i].GlyphxPlayerID, general_buffer.data(), general_buffer.size()))
            return false;
        RenderPOV(players_view[i], game_state, (const CNCShroudStruct*)general_buffer.data(), (std::remove_extent<decltype(HouseColorMap)>::type)(players[i].ColorIndex));
        static_cast<VectorRepresentationView&>(*output) = (players_view[i]);

        if (!CNC_Get_Game_State(GAME_STATE_SIDEBAR, players[i].GlyphxPlayerID, general_buffer.data(), general_buffer.size()))
            return false;
        players_sidebar[i] = *(const CNCSidebarStruct*)(general_buffer.data());
        output->sidebar = players_sidebar[i];
    }
    return true;
}

void __cdecl HandleSidebarRequest(size_t player_id, SidebarRequestEnum requestType, std::int32_t assetNameIndex)
{
    if (player_id >= players.size())
        return;

    const auto& player = players[player_id];
    if (!CNC_Get_Game_State(GAME_STATE_SIDEBAR, player.GlyphxPlayerID, general_buffer.data(), general_buffer.size()))
        return;

    const auto& sidebar = players_sidebar[player_id];

    if (requestType == SIDEBAR_REQUEST_PLACE)
        return; // buildings are placed with click for now

    for (const auto& entry : sidebar.Entries)
    {
        if (entry.AssetName == assetNameIndex)
        {
            CNC_Handle_Sidebar_Request(requestType, player.GlyphxPlayerID, entry.BuildableType, entry.BuildableID, 0, 0);
            return;
        }
    }
}

void __cdecl HandleInputRequest(size_t player_id, InputRequestEnum requestType, int x1, int y1)
{
    if (player_id >= players.size())
        return;

    switch (requestType) {
        // these should be handled differently
        case INPUT_REQUEST_SPECIAL_KEYS:
        case INPUT_REQUEST_MOUSE_AREA:
        case INPUT_REQUEST_MOUSE_AREA_ADDITIVE:
        return;
    }
    CNC_Handle_Input(requestType, 0U, players[player_id].GlyphxPlayerID, x1, y1, 0, 0);
}
