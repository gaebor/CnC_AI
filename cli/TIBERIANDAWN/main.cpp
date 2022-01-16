#include <Windows.h>

#include "GamePlay.hpp"

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

    SimplePlayerInfoStruct player1 = { "gaebor",
        0, // GOOD (GDI)
        0, // color
        314159265,
        0, // Team
        127, //StartLocationIndex
        false // IsAI
    };

    SimplePlayerInfoStruct player2 = { "ai1",
        1, // BAD (NOD)
        2, // color
        271828182,
        1, // Team
        127, //StartLocationIndex
        true // IsAI
    };

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