cmake_minimum_required(VERSION 3.10)
project(go2_rl_lcm CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

#####################################################
# 项目相对路径
#####################################################
get_filename_component(PROJ_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

# LCM (由 deploy/install_deploy.sh 下载并编译到 thirdparty/lcm)
set(LCM_ROOT  ${PROJ_ROOT}/thirdparty/lcm)
set(LCM_BUILD ${LCM_ROOT}/build)
set(LCM_LIBRARY ${LCM_BUILD}/lcm/liblcm.so)

if(NOT EXISTS "${LCM_ROOT}/lcm/lcm-cpp.hpp" OR NOT EXISTS "${LCM_LIBRARY}")
    message(FATAL_ERROR
        "LCM was not found under ${LCM_ROOT}. "
        "Run deploy/install_deploy.sh first to download and build LCM."
    )
endif()

# Unitree SDK2
set(UNITREE_DIR        ${CMAKE_CURRENT_SOURCE_DIR}/unitree_sdk2_bin)
set(UNITREE_SDK2_ROOT  ${UNITREE_DIR}/library/unitree_sdk2)

# 子路径
set(UNITREE_INCLUDE        ${UNITREE_SDK2_ROOT}/include)
set(UNITREE_THIRDPARTY     ${UNITREE_SDK2_ROOT}/thirdparty)
set(UNITREE_THIRDPARTY_INC ${UNITREE_THIRDPARTY}/include)
set(UNITREE_LIB_X86        ${UNITREE_SDK2_ROOT}/lib/x86_64)
set(UNITREE_THIRDPARTY_LIB ${UNITREE_THIRDPARTY}/lib/x86_64)

# 静态主库
set(UNITREE_STATIC_LIB ${UNITREE_LIB_X86}/libunitree_sdk2.a)

#####################################################
# 头文件搜索路径
#####################################################
include_directories(
    # LCM
    ${LCM_ROOT}/lcm
    ${LCM_BUILD}/lcm

    # Unitree SDK2 core headers
    ${UNITREE_INCLUDE}

    # Unitree thirdparty headers
    ${UNITREE_THIRDPARTY}
    ${UNITREE_THIRDPARTY_INC}

    # !!! 关键新的一行: 让 <dds/dds.hpp> 指向 ddscxx/dds/dds.hpp
    ${UNITREE_THIRDPARTY_INC}/ddscxx

    # 你项目自己的 LCM 消息 struct 头文件 (.hpp)
    ${CMAKE_CURRENT_SOURCE_DIR}/lcm_types
)

#####################################################
# 库搜索路径
#####################################################
link_directories(
    ${UNITREE_LIB_X86}              # libunitree_sdk2.a
    ${UNITREE_THIRDPARTY_LIB}       # libddsc.so, libddscxx.so
)

#####################################################
# 可执行文件
#####################################################
add_executable(lcm_position_go2
    ${UNITREE_DIR}/lcm_position_go2.cpp
)

add_executable(lcm_receive
    ${UNITREE_DIR}/lcm_receive.cpp
)

#####################################################
# 链接库
#####################################################
target_link_libraries(lcm_position_go2
    ${LCM_LIBRARY}
    ${UNITREE_STATIC_LIB}
    ddsc
    ddscxx
    rt
    pthread
)

target_link_libraries(lcm_receive
    ${LCM_LIBRARY}
    ${UNITREE_STATIC_LIB}
    ddsc
    ddscxx
    rt
    pthread
)

#####################################################
# rpath 设置，运行时能找到 libddsc.so / libddscxx.so
#####################################################
set_target_properties(lcm_position_go2 PROPERTIES
    BUILD_RPATH "${LCM_BUILD}/lcm;${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
    INSTALL_RPATH "${LCM_BUILD}/lcm;${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
)

set_target_properties(lcm_receive PROPERTIES
    BUILD_RPATH "${LCM_BUILD}/lcm;${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
    INSTALL_RPATH "${LCM_BUILD}/lcm;${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
)

add_executable(release_mcf
    ${UNITREE_DIR}/release_mcf.cpp
)

target_link_libraries(release_mcf
    ${UNITREE_STATIC_LIB}
    ddsc
    ddscxx
    rt
    pthread
)

set_target_properties(release_mcf PROPERTIES
    BUILD_RPATH "${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
    INSTALL_RPATH "${UNITREE_LIB_X86};${UNITREE_THIRDPARTY_LIB}"
)
