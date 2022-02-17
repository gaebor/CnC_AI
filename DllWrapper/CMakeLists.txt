cmake_minimum_required (VERSION 3.12)
project (DllWrapper)

add_compile_options(/W4 /Zp1)
add_compile_definitions(WINDOWS_IGNORE_PACKING_MISMATCH)

foreach(target TIBERIANDAWN)
    foreach(header DLLInterface.h DLLInterfaceVersion.h)
        file(DOWNLOAD https://raw.githubusercontent.com/electronicarts/CnC_Remastered_Collection/master/${target}/${header} ${CMAKE_SOURCE_DIR}/${target}/${header})
    endforeach(header)

    file(GLOB SOURCE_FILES
        "${CMAKE_SOURCE_DIR}/${target}/*.cpp"
        "${CMAKE_SOURCE_DIR}/${target}/*.c"
        "${CMAKE_SOURCE_DIR}/WebSocket.c"
    )
    file(GLOB HEADER_FILES
        "${CMAKE_SOURCE_DIR}/${target}/*.h"
        "${CMAKE_SOURCE_DIR}/${target}/*.hpp"
        "${CMAKE_SOURCE_DIR}/WebSocket.h"
    )

    add_executable("${target}_wrapper" ${SOURCE_FILES} ${HEADER_FILES})
    target_include_directories("${target}_wrapper" PRIVATE ${CMAKE_SOURCE_DIR}/${target})
    TARGET_LINK_LIBRARIES("${target}_wrapper" winhttp)

endforeach(target)
