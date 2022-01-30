cmake_minimum_required (VERSION 3.12)
project (cnc_cli)

add_compile_options(/W4 /Zp1)
add_compile_definitions(WINDOWS_IGNORE_PACKING_MISMATCH)

foreach(target TIBERIANDAWN)
    foreach(header DLLInterface.h DLLInterfaceVersion.h)
        file(DOWNLOAD https://raw.githubusercontent.com/electronicarts/CnC_Remastered_Collection/master/${target}/${header} ${CMAKE_SOURCE_DIR}/${target}/${header})
    endforeach(header)

    file(GLOB SOURCE_FILES "${CMAKE_SOURCE_DIR}/${target}/*.cpp")
    file(GLOB HEADER_FILES "${CMAKE_SOURCE_DIR}/${target}/*.h" "${CMAKE_SOURCE_DIR}/${target}/*.hpp")

    add_library("${target}_wrapper" SHARED ${SOURCE_FILES} ${HEADER_FILES})
    target_include_directories("${target}_wrapper" PRIVATE ${CMAKE_SOURCE_DIR}/${target})
    
endforeach(target)
