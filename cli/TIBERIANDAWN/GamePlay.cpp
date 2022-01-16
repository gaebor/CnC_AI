#include "GamePlay.hpp"

#include <type_traits>

static const CNCRulesDataStruct rule_data_struct = { {
    {1.2f, 1.2f, 1.2f, 0.3f, 0.8f, 0.8f, 0.6f, 0.001f, 0.001f, false, true, true},
    {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 0.02f, 0.03f, true, true, true},
    {0.9f, 0.9f, 0.9f, 1.05f, 1.05f, 1.f, 1.f, 0.05f, 0.1f, true, true, true}
} };

GamePlay::GamePlay(const TCHAR* dll_filename, const char* content_directory)
    : dll_handle(LoadLibrary(dll_filename)), content_directory(content_directory),
    buffer(4 * 1024 * 1024), palette(1)
{
    if (dll_handle == NULL)
        return;

    CNC_Version = (decltype(CNC_Version))GetProcAddress(dll_handle, "CNC_Version");
    CNC_Init = (decltype(CNC_Init))GetProcAddress(dll_handle, "CNC_Init");
    CNC_Config = (decltype(CNC_Config))GetProcAddress(dll_handle, "CNC_Config");
    CNC_Add_Mod_Path = (decltype(CNC_Add_Mod_Path))GetProcAddress(dll_handle, "CNC_Add_Mod_Path");
    CNC_Get_Visible_Page = (decltype(CNC_Get_Visible_Page))GetProcAddress(dll_handle, "CNC_Get_Visible_Page");
    CNC_Get_Palette = (decltype(CNC_Get_Palette))GetProcAddress(dll_handle, "CNC_Get_Palette");
    CNC_Start_Instance = (decltype(CNC_Start_Instance))GetProcAddress(dll_handle, "CNC_Start_Instance");
    CNC_Start_Instance_Variation = (decltype(CNC_Start_Instance_Variation))GetProcAddress(dll_handle, "CNC_Start_Instance_Variation");
    CNC_Start_Custom_Instance = (decltype(CNC_Start_Custom_Instance))GetProcAddress(dll_handle, "CNC_Start_Custom_Instance");
    CNC_Advance_Instance = (decltype(CNC_Advance_Instance))GetProcAddress(dll_handle, "CNC_Advance_Instance");
    CNC_Get_Game_State = (decltype(CNC_Get_Game_State))GetProcAddress(dll_handle, "CNC_Get_Game_State");
    CNC_Read_INI = (decltype(CNC_Read_INI))GetProcAddress(dll_handle, "CNC_Read_INI");
    CNC_Set_Home_Cell = (decltype(CNC_Set_Home_Cell))GetProcAddress(dll_handle, "CNC_Set_Home_Cell");
    CNC_Handle_Game_Request = (decltype(CNC_Handle_Game_Request))GetProcAddress(dll_handle, "CNC_Handle_Game_Request");
    CNC_Handle_Game_Settings_Request = (decltype(CNC_Handle_Game_Settings_Request))GetProcAddress(dll_handle, "CNC_Handle_Game_Settings_Request");
    CNC_Handle_Input = (decltype(CNC_Handle_Input))GetProcAddress(dll_handle, "CNC_Handle_Input");
    CNC_Handle_Structure_Request = (decltype(CNC_Handle_Structure_Request))GetProcAddress(dll_handle, "CNC_Handle_Structure_Request");
    CNC_Handle_Unit_Request = (decltype(CNC_Handle_Unit_Request))GetProcAddress(dll_handle, "CNC_Handle_Unit_Request");
    CNC_Handle_Sidebar_Request = (decltype(CNC_Handle_Sidebar_Request))GetProcAddress(dll_handle, "CNC_Handle_Sidebar_Request");
    CNC_Handle_SuperWeapon_Request = (decltype(CNC_Handle_SuperWeapon_Request))GetProcAddress(dll_handle, "CNC_Handle_SuperWeapon_Request");
    CNC_Handle_ControlGroup_Request = (decltype(CNC_Handle_ControlGroup_Request))GetProcAddress(dll_handle, "CNC_Handle_ControlGroup_Request");
    CNC_Handle_Debug_Request = (decltype(CNC_Handle_Debug_Request))GetProcAddress(dll_handle, "CNC_Handle_Debug_Request");
    CNC_Handle_Beacon_Request = (decltype(CNC_Handle_Beacon_Request))GetProcAddress(dll_handle, "CNC_Handle_Beacon_Request");
    CNC_Set_Multiplayer_Data = (decltype(CNC_Set_Multiplayer_Data))GetProcAddress(dll_handle, "CNC_Set_Multiplayer_Data");
    CNC_Clear_Object_Selection = (decltype(CNC_Clear_Object_Selection))GetProcAddress(dll_handle, "CNC_Clear_Object_Selection");
    CNC_Select_Object = (decltype(CNC_Select_Object))GetProcAddress(dll_handle, "CNC_Select_Object");
    CNC_Save_Load = (decltype(CNC_Save_Load))GetProcAddress(dll_handle, "CNC_Save_Load");
    CNC_Set_Difficulty = (decltype(CNC_Set_Difficulty))GetProcAddress(dll_handle, "CNC_Set_Difficulty");
    CNC_Handle_Player_Switch_To_AI = (decltype(CNC_Handle_Player_Switch_To_AI))GetProcAddress(dll_handle, "CNC_Handle_Player_Switch_To_AI");
    CNC_Handle_Human_Team_Wins = (decltype(CNC_Handle_Human_Team_Wins))GetProcAddress(dll_handle, "CNC_Handle_Human_Team_Wins");
    CNC_Start_Mission_Timer = (decltype(CNC_Start_Mission_Timer))GetProcAddress(dll_handle, "CNC_Start_Mission_Timer");
    CNC_Get_Start_Game_Info = (decltype(CNC_Get_Start_Game_Info))GetProcAddress(dll_handle, "CNC_Get_Start_Game_Info");

    CNC_Init(content_directory, NULL);
    CNC_Config(rule_data_struct);
}

GamePlay::~GamePlay() {
    if (dll_handle != NULL)
    {
        FreeLibrary(dll_handle);
    }
}

bool GamePlay::is_initialized() const { return dll_handle != NULL; }

void GamePlay::add_player(const SimplePlayerInfoStruct& player)
{
    players.emplace_back();
    auto& player_info = players.back();
    std::copy(player.Name, player.Name + 64, player_info.Name);
    player_info.House = player.House;
    player_info.ColorIndex = player.ColorIndex;
    player_info.GlyphxPlayerID = player.GlyphxPlayerID;
    player_info.Team = player.Team;
    player_info.StartLocationIndex = player.StartLocationIndex;
    player_info.IsAI = player.IsAI;
}

bool GamePlay::retrieve_players_info()
{
    if (!is_initialized())
        return false;

    for (auto& player : players)
    {
        if (!CNC_Get_Game_State(GAME_STATE_PLAYER_INFO, player.GlyphxPlayerID, (unsigned char*)(&player), sizeof(player) + 33))
            return false;
        HouseColorMap[player.House] = std::remove_extent<decltype(HouseColorMap)>::type(player.ColorIndex);    
    }
    return true;
}

bool GamePlay::init_palette()
{
    if (!is_initialized())
        return false;

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

bool GamePlay::start_game(const CNCMultiplayerOptionsStruct& multiplayer_options, int scenario_index, int build_level, int difficulty)
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

    if (!CNC_Start_Instance_Variation(scenario_index, -1, 0, build_level, "MULTI", "GAME_GLYPHX_MULTIPLAYER", content_directory.data() + 3, -1, ""))
    {
        return false;
    }

    CNC_Set_Difficulty(difficulty);

    //for (auto& player : players)
    //{
    //    if (!CNC_Get_Start_Game_Info(player.GlyphxPlayerID, player.StartLocationIndex))
    //        return false;
    //}

    return prepare();
}

bool GamePlay::save_game(const char* filename)
{
    if (!is_initialized())
        return false;

    return CNC_Save_Load(true, filename, "GAME_GLYPHX_MULTIPLAYER");
}

bool GamePlay::load_game(const char* filename)
{
    if (!is_initialized())
        return false;

    if (!CNC_Save_Load(false, filename, "GAME_GLYPHX_MULTIPLAYER"))
        return false;

    return prepare();
}

bool GamePlay::prepare()
{
    if (!retrieve_players_info())
        return false;

    if (!init_palette())
        return false;

    if (!retrieve_satic_map())
        return false;

    CNC_Handle_Game_Request(INPUT_GAME_LOADING_DONE);
    
    return true;
}

bool GamePlay::retrieve_satic_map()
{
    if (!is_initialized())
        return false;

    if (!CNC_Get_Game_State(GAME_STATE_STATIC_MAP, 0, buffer.data(), buffer.size()))
        return false;

    static_map = *((const CNCMapDataStruct*)buffer.data());

    return true;
}

const unsigned char* GamePlay::GetHouseColorMap()
{
    return HouseColorMap;
}

// based on HousesType and PlayerColorType
unsigned char GamePlay::HouseColorMap[] = {
        0, 2, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
};