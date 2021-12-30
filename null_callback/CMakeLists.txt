cmake_minimum_required (VERSION 3.1)
project (null_callback)

add_compile_options( /W4 /D_CRT_SECURE_NO_WARNINGS )

foreach(target TIBERIANDAWN REDALERT)
    make_directory(${CMAKE_SOURCE_DIR}/${target})
    foreach(header DLLInterface.h DLLInterfaceVersion.h)
        file(DOWNLOAD https://raw.githubusercontent.com/electronicarts/CnC_Remastered_Collection/master/${target}/${header} ${CMAKE_SOURCE_DIR}/${target}/${header})
    endforeach(header)

    add_library("${target}_null_callback" SHARED null_callback.cpp)
    target_include_directories("${target}_null_callback" PRIVATE ${CMAKE_SOURCE_DIR}/${target})
  
endforeach(target)
