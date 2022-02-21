#define NOMINMAX
#include <Windows.h>

#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <limits>

#include "DllInterface.h"
#include "data_utils.hpp"
#include "VectorInterface.hpp"
#include "../WebSocket.h"

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
    // void(__cdecl * CNC_Handle_Game_Settings_Request)(int health_bar_display_mode, int resource_bar_display_mode) = NULL;
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
    // bool(__cdecl * CNC_Get_Visible_Page)(unsigned char* buffer_in, unsigned int& width, unsigned int& height) = NULL;
    // bool(__cdecl * CNC_Get_Palette)(unsigned char(&palette_in)[256][3]) = NULL;
}

struct StartGameArgs
{
    const CNCMultiplayerOptionsStruct multiplayer_info;
    int scenario_index;
    int build_level;
    int difficulty;
};

struct StartGameCustomArgs
{
    const CNCMultiplayerOptionsStruct multiplayer_info;
    char directory_path[256];
    char scenario_name[256];
    int build_level;
};


bool ChDir(const char* dirname_utf8);
bool Init(const char* dll_filename_utf8, const char* content_directory_ascii);
void AddPlayer(const CNCPlayerInfoStruct*);
bool StartGame(const StartGameArgs*);
bool StartGameCustom(const StartGameCustomArgs*);
unsigned char GetGameResult();
void FreeDll();

bool StartGameCallback();

class ProcessGuard
{
public:
    ProcessGuard() {}
    ~ProcessGuard()
    {
        FreeDll();
        DestroyWebSocket();
    }
};

bool Advance();
bool GetCommonVectorRepresentation();
bool GetPlayersVectorRepresentation(void*& buffer, size_t& buffer_size);

struct SidebarRequestArgs
{
    std::uint32_t player_id;
    SidebarRequestEnum requestType;
    std::uint32_t assetNameIndex;
};


struct NoughtRequestArgs
{
    std::uint32_t player_id;
};

struct InputRequestArgs
{
    std::uint32_t player_id;
    InputRequestEnum requestType;
    float x1;
    float y1;
};

void HandleSidebarRequest(const SidebarRequestArgs*);
void HandleInputRequest(const InputRequestArgs*);

HMODULE dll_handle;
std::vector<CNCPlayerInfoStruct> players;
std::vector<unsigned char> orginal_houses;
VectorRepresentation game_state;
std::vector<SideBar> players_sidebar;

std::string content_directory;

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

bool ChDir(const char* dirname_utf8)
{
    const auto dirname_unicode = FromUtf8(dirname_utf8);
    return SetCurrentDirectoryW(dirname_unicode.data()) == TRUE;
}

#define LoadSymbolFromDll(name) {name = (decltype(name))GetProcAddress(dll_handle, #name);\
                                if (name == NULL) return false;}

bool Init(const char* dll_filename_utf8, const char* content_directory_ascii)
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
    
    CNC_Init(content_directory_ascii, NULL);
    CNC_Config(rule_data_struct);

    return true;
}

#undef LoadSymbolFromDll

void AddPlayer(const CNCPlayerInfoStruct* player)
{
    players.emplace_back(*player);
    orginal_houses.emplace_back(player->House);
    players_sidebar.resize(players.size());
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

bool StartGame(const StartGameArgs* info)
{
    if (!CNC_Set_Multiplayer_Data(info->scenario_index, const_cast<CNCMultiplayerOptionsStruct&>(info->multiplayer_info), (int)players.size(), players.data(), 6))
    {
        return false;
    }
    if (!CNC_Start_Instance_Variation(
        info->scenario_index,
        -1, 0, info->build_level,
        "MULTI", "GAME_GLYPHX_MULTIPLAYER",
        content_directory.data(),
        -1, "")
        )
    {
        return false;
    }

    CNC_Set_Difficulty(info->difficulty);

    return StartGameCallback();
}

bool StartGameCustom(const StartGameCustomArgs* info)
{
    if (!CNC_Set_Multiplayer_Data(-1, const_cast<CNCMultiplayerOptionsStruct&>(info->multiplayer_info), (int)players.size(), players.data(), 6))
    {
        return false;
    }
    if (!CNC_Start_Custom_Instance(
            content_directory.data(),
            info->directory_path, 
            info->scenario_name,
            info->build_level, true
    ))
    {
        return false;
    }

    return StartGameCallback();
}

unsigned char GetGameResult()
{
    retrieve_players_info();

    unsigned char loser_mask = 0;
    for (size_t i = 0; i < players.size(); ++i)
    {
        loser_mask |= (players[i].IsDefeated ? 1 : 0) << i;
    }
    return loser_mask;
}

void FreeDll()
{
    FreeLibrary(dll_handle);
}

bool StartGameCallback()
{
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

bool Advance()
{
    return CNC_Advance_Instance(0);
}

bool GetCommonVectorRepresentation()
{
    static std::vector<unsigned char> static_map_buffer(4 * 1024 * 1024);
    static std::vector<unsigned char> dynamic_map_buffer(4 * 1024 * 1024);
    static std::vector<unsigned char> layers_buffer(4 * 1024 * 1024);

    if (!CNC_Get_Game_State(GAME_STATE_STATIC_MAP, 0, static_map_buffer.data(), static_map_buffer.size()))
        return false;
    if (!CNC_Get_Game_State(GAME_STATE_DYNAMIC_MAP, 0, dynamic_map_buffer.data(), dynamic_map_buffer.size()))
        return false;
    if (!CNC_Get_Game_State(GAME_STATE_LAYERS, 0, layers_buffer.data(), layers_buffer.size()))
        return false;
    
    game_state.Render(
        reinterpret_cast<const CNCMapDataStruct*>(static_map_buffer.data()),
        reinterpret_cast<const CNCDynamicMapStruct*>(dynamic_map_buffer.data()),
        reinterpret_cast<const CNCObjectListStruct*>(layers_buffer.data())
    );
    return true;
}

bool GetPlayersVectorRepresentation(unsigned char*& buffer, size_t& buffer_size)
{
    if (!GetCommonVectorRepresentation())
        return false;

    static VectorRepresentation view;
    static std::vector<unsigned char> internal_buffer(4 * 1024 * 1024);

    for (size_t i = 0; i < players.size(); ++i)
    {
        if (!CNC_Get_Game_State(GAME_STATE_SHROUD, players[i].GlyphxPlayerID, internal_buffer.data(), internal_buffer.size()))
            return false;
        RenderPOV(view, game_state, (const CNCShroudStruct*)internal_buffer.data(), (std::remove_extent<decltype(HouseColorMap)>::type)(players[i].ColorIndex));
        if (!CopyToBuffer(view.map, buffer, buffer_size) || !CopyToBuffer(view.dynamic_objects, buffer, buffer_size))
            return false;

        if (!CNC_Get_Game_State(GAME_STATE_SIDEBAR, players[i].GlyphxPlayerID, internal_buffer.data(), internal_buffer.size()))
            return false;
        players_sidebar[i] = *(const CNCSidebarStruct*)(internal_buffer.data());
        if (!CopyToBuffer(players_sidebar[i].Members, buffer, buffer_size) || !CopyToBuffer(players_sidebar[i].Entries, buffer, buffer_size))
            return false;
    }
    return true;
}

void HandleSidebarRequest(const SidebarRequestArgs* args)
{
    if (args->player_id >= players.size())
        return;

    const auto& player = players[args->player_id];
    const auto& sidebar = players_sidebar[args->player_id];
    if (args->assetNameIndex < sidebar.Entries.size())
    {
        const auto& entry = sidebar.Entries[args->assetNameIndex];
        if (args->requestType == SIDEBAR_REQUEST_PLACE)
            return; // buildings are placed with click for now

        CNC_Handle_Sidebar_Request(args->requestType, player.GlyphxPlayerID, entry.BuildableType, entry.BuildableID, 0, 0);
    }
}

void HandleInputRequest(const InputRequestArgs* args)
{
    if (args->player_id >= players.size())
        return;

    switch (args->requestType)
    {
        // these should be handled differently
        case INPUT_REQUEST_SPECIAL_KEYS:
        case INPUT_REQUEST_MOUSE_AREA:
        case INPUT_REQUEST_MOUSE_AREA_ADDITIVE:
        return;
    }
    CNC_Handle_Input(args->requestType, 0U, players[args->player_id].GlyphxPlayerID, (int)(args->x1), (int)(args->y1), 0, 0);
}

enum WebsocketMessageType : std::uint32_t
{
    CHDIR,
    INIT,
    ADDPLAYER,
    STARTGAME,
    STARTGAMECUSTOM,
    INPUTREQUEST,
    SIDEBARREQUEST,
    NOUGHTREQUEST,
    LOAD,
};

bool init_loop()
{
    std::vector<unsigned char> buffer(4 * 1024 * 1024);
    size_t message_size;

    while (ReceiveOnSocket(buffer.data(), buffer.size(), (DWORD*)(&message_size)) == NO_ERROR)
    {
        const unsigned char* message_ptr = buffer.data();
        const auto message_type = *(const WebsocketMessageType*)message_ptr;
        step_buffer(message_ptr, message_size, sizeof(WebsocketMessageType));

        if (message_type == CHDIR)
        {
            if (message_size < 1)
                return false;
            const std::string dir_path = safe_str_copy(message_ptr, message_size);
            if (!ChDir(dir_path.c_str()))
                return false;
        }
        else if (message_type == INIT)
        {
            if (message_size < 2)
                return false;
            const std::string dll_path = safe_str_copy(message_ptr, message_size);
            const std::string content_dir = safe_str_copy(message_ptr, message_size);
            if (!Init(dll_path.c_str(), content_dir.c_str()))
                return false;
        }
        else if (message_type == ADDPLAYER)
        {
            if (message_size < sizeof(CNCPlayerInfoStruct))
                return false;

            AddPlayer((const CNCPlayerInfoStruct*)message_ptr);
        }
        else if (message_type == STARTGAME)
        {
            if (message_size < sizeof(StartGameArgs))
                return false;

            if (!StartGame((const StartGameArgs*)message_ptr))
                return false;
            return true;
        }
        else if (message_type == STARTGAMECUSTOM)
        {
            if (message_size < sizeof(StartGameCustomArgs))
                return false;

            if (!StartGameCustom((const StartGameCustomArgs*)message_ptr))
                return false;
            return true;
        }
        else if (message_type == LOAD)
        {
            if (message_size < 1)
                return false;
            const std::string filename = safe_str_copy(message_ptr, message_size);
            if (!CNC_Save_Load(false, filename.c_str(), "GAME_GLYPHX_MULTIPLAYER"))
                return false;
            StartGameCallback();
            return true;
        }
    }
    return false;
}

int main(int argc, const char* argv[])
{
    std::vector<unsigned char> buffer(4 * 1024 * 1024);
    size_t message_size;
    unsigned short port = 8888;
    if (argc > 1)
    {
        const auto parsed_int = atoi(argv[1]);
        if (parsed_int < std::numeric_limits<decltype(port)>::min() || parsed_int > std::numeric_limits<decltype(port)>::max())
        {
            return 1;
        }
        port = (decltype(port))parsed_int;
    }

    ProcessGuard process_guard;
    if (NO_ERROR != InitializeWebSocket(port))
        return 1;

    {
        const std::string startup_message = "READY";
        if (NO_ERROR != SendOnSocket((void*)startup_message.c_str(), startup_message.size() + 1))
            return 1;
    }

    if (!init_loop())
    {
        return 1;
    }

    while (Advance())
    {
        // send current state of the game
        unsigned char* output_ptr = buffer.data();
        size_t output_size = buffer.size();
        if (!GetPlayersVectorRepresentation(output_ptr, output_size))
            return 1;
        if (NO_ERROR != SendOnSocket(buffer.data(), buffer.size() - output_size))
            return 1;

        // receive instructions
        if (NO_ERROR != ReceiveOnSocket(buffer.data(), buffer.size(), (DWORD*)(&message_size)))
            return 1;
        // play instructions per player
        const unsigned char* message_ptr = buffer.data();
        for (size_t i = 0; i < players.size() && message_size > sizeof(WebsocketMessageType); ++i)
        {
            const auto message_type = *(const WebsocketMessageType*)message_ptr;
            step_buffer(message_ptr, message_size, sizeof(WebsocketMessageType));
            switch (message_type)
            {
            case INPUTREQUEST:
                HandleInputRequest((const InputRequestArgs*)message_ptr);
                step_buffer(message_ptr, message_size, sizeof(InputRequestArgs));
                break;
            case SIDEBARREQUEST:
                HandleSidebarRequest((const SidebarRequestArgs*)message_ptr);
                step_buffer(message_ptr, message_size, sizeof(SidebarRequestArgs));
                break;
            case NOUGHTREQUEST:
                step_buffer(message_ptr, message_size, sizeof(NoughtRequestArgs));
                break;
            }
        }
    }
    
    buffer[0] = GetGameResult();
    SendOnSocket(buffer.data(), 1);
    
    return 0;
}
