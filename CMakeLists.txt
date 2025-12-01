cmake_minimum_required(VERSION 3.5)
project(pp_infer)
set(PROJECT_NAME pp_infer)
EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCH )
message( STATUS "Architecture: ${ARCH}" )

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(CUDA REQUIRED)
find_package(rclcpp_components REQUIRED)

set(CUDA_VERSION 11.3)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-${CUDA_VERSION})

SET(CMAKE_BUILD_TYPE "Release")
add_compile_options(-W)
add_compile_options(-std=c++11)

set(SMS 50 52 53 60 61 62 70 72 75 80 86)
foreach(sm ${SMS})
    set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
list(GET SMS -1 LATEST_SM)
set(GENCODE "${GENCODE} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xcompiler -I/usr/${ARCH}-linux-gnu/include/
    -Xlinker -lsocket
    -Xlinker -rpath=/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -rpath=/usr/${ARCH}-linux-gnu/lib/
    -Xlinker -L/usr/lib/${ARCH}-linux-gnu/
    -Xlinker -L/usr/${ARCH}-linux-gnu/lib/
)

set(TENSORRT_INCLUDE_DIRS /usr/include/${ARCH}-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/${ARCH}-linux-gnu/)

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIRS}
    ../include/
)

link_directories(
    ${TENSORRT_LIBRARY_DIRS}
    /usr/lib/${ARCH}-linux-gnu
    /usr/${ARCH}-linux-gnu/lib/
)

# ========================================================
# 1. Original pp_infer Target
# ========================================================
# オリジナルのソースファイルを明示的に指定
set(ORIGINAL_SOURCES
    src/pointpillar.cpp
    src/postprocess.cpp
)

cuda_add_executable(${PROJECT_NAME} 
    src/pp_inference_node.cpp 
    ${ORIGINAL_SOURCES}
)
ament_target_dependencies(${PROJECT_NAME} rclcpp std_msgs vision_msgs geometry_msgs PCL CUDA)
target_link_libraries(${PROJECT_NAME}
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)

# ========================================================
# 2. HARRP Custom Target (pp_infer_harrp)
# ========================================================
set(HARRP_SOURCES
    src/pointpillar_harrp.cpp
    src/postprocess.cpp
)

# 【変更】cuda_add_executable から cuda_add_library (SHARED) に変更
# ターゲット名も少し変えておくと分かりやすいです: pp_infer_harrp_component
cuda_add_library(${PROJECT_NAME}_harrp_component SHARED
    src/pp_inference_node_harrp.cpp 
    ${HARRP_SOURCES}
)

# Windows対応等のための定型句（Linuxでも入れておいて損はありません）
target_compile_definitions(${PROJECT_NAME}_harrp_component
  PRIVATE "COMPOSITION_BUILDING_DLL"
)

# 依存関係
ament_target_dependencies(${PROJECT_NAME}_harrp_component
  rclcpp
  rclcpp_components # 【追加】
  std_msgs
  vision_msgs
  geometry_msgs
  PCL
  CUDA
)

# リンク設定
target_link_libraries(${PROJECT_NAME}_harrp_component
    libnvinfer.so
    libnvonnxparser.so
    libnvinfer_plugin.so
)

# 【追加】コンポーネントの登録
# 第1引数: ライブラリターゲット名
# 第2引数: 登録したクラス名 (namespace込み)
rclcpp_components_register_nodes(${PROJECT_NAME}_harrp_component
    "pp_infer::PointPillarsHarrpNode"
)

# ========================================================
# Install Rules
# ========================================================
# ライブラリのインストール設定に変更
install(TARGETS
    ${PROJECT_NAME}_harrp_component
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# もしオリジナルの実行ファイルも残しているならそれもinstall記述が必要
# install(TARGETS ${PROJECT_NAME} DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch
  include # ヘッダ類も必要なら
  DESTINATION share/${PROJECT_NAME}/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
