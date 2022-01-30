#include "data_utils.hpp"

const std::unordered_map<std::string, int> buildables = { 
    {"FACT", 5000},
    {"FACTMAKE", 5000},
    {"NUKE", 300},
    {"NUKEMAKE", 300},
    {"NUK2", 700},
    {"NUK2MAKE", 700},
    {"PROC", 2000},
    {"PROCMAKE", 2000},
    {"SILO", 150},
    {"SILOMAKE", 150},
    {"PYLE", 300},
    {"PYLEMAKE", 300},
    {"HAND", 300},
    {"HANDMAKE", 300},
    {"GTWR", 500},
    {"GTWRMAKE", 500},
    {"ATWR", 1000},
    {"ATWRMAKE", 1000},
    {"GUN", 600},
    {"GUNMAKE", 600},
    {"OBLI", 1500},
    {"OBLIMAKE", 1500},
    {"AFLD", 2000},
    {"AFLDMAKE", 2000},
    {"WEAP", 2000},
    {"WEAPMAKE", 2000},
    {"HQ", 1000},
    {"HQMAKE", 1000},
    {"EYE", 2800},
    {"EYEMAKE", 2800},
    {"TMPL", 3000},
    {"TMPLMAKE", 3000},
    {"FIX", 1200},
    {"FIXMAKE", 1200},
    {"HPAD", 1500},
    {"HPADMAKE", 1500},
    {"SAM", 750},
    {"SAMMAKE", 750},

    {"SBAG", 50},
    {"BRIK", 100},
    {"CYCL", 75},
    {"E1", 100},
    {"E2", 160},
    {"E3", 300},
    {"E4", 200},
    {"E5", 300},
    {"E6", 500},
    {"RMBO", 1000},
    {"LTNK", 600},
    {"MTNK", 800},
    {"HTNK", 1500},
    {"FTNK", 800},
    {"STNK", 900},
    {"HARV", 1400},
    {"JEEP", 400},
    {"APC", 700},
    {"ARTY", 450},
    {"BGGY", 300},
    {"BIKE", 500},
    {"MCV", 5000},
    {"MLRS", 750},
    {"MSAM", 800},
    {"HELI", 1200},
    {"ORCA", 1200},
    {"TRAN", 1500}
};

unsigned char HouseColorMap[] = {
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
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff
};

unsigned char ConvertMask(const unsigned int per_house_mask)
{
    unsigned char per_color_mask = 0;
    for (int i = 4; i < 10; ++i) // HOUSE_MULTI1 ... HOUSE_MULTI6
    {
        if (per_house_mask & (1 << i))
            per_color_mask |= 1 << HouseColorMap[i];
    }
    return per_color_mask;
}
