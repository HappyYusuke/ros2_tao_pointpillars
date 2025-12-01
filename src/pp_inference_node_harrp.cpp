/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#define BOOST_BIND_NO_PLACEHOLDERS

#include <chrono>
#include <memory>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cassert>
#include <sstream>
#include <unistd.h>
#include <string>
#include <cmath> // 追加: cos, sin, abs用

#include "cuda_runtime.h"
#include "../include/pp_infer/pointpillar.h"

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp" // 追加: QoS設定用
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "pcl_conversions/pcl_conversions.h"
#include "../include/pp_infer/point_cloud2_iterator_harrp.hpp"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp_components/register_node_macro.hpp> // コンポーネント登録用ヘッダ

using std::placeholders::_1;
using namespace std::chrono_literals;

/* This example creates a subclass of Node and uses std::bind() to register a
 * member function as a callback from the timer. */

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

// 検出ボックスごとの事前計算情報を保持する構造体
struct BoxInfo {
    int index; // 元の検出リストのインデックス
    float cx, cy, cz;
    float l, w, h;
    float half_l, half_w, half_h;
    float cos_r, sin_r;
    
    // AABB (Axis-Aligned Bounding Box) 事前フィルタリング用
    float min_x, max_x, min_y, max_y, min_z, max_z;
    
    // 抽出された点群
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;

    BoxInfo(int idx, const Bndbox& box) : index(idx) {
        cx = box.x; cy = box.y; cz = box.z;
        l = box.l; w = box.w; h = box.h;
        half_l = l / 2.0f; half_w = w / 2.0f; half_h = h / 2.0f;
        cos_r = cos(-box.rt);
        sin_r = sin(-box.rt);
        
        // クラウド初期化
        cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());

        // AABBの計算（回転した矩形を包含する最小の軸平行矩形）
        // x範囲: cx ± (|l*cos| + |w*sin|)/2
        // y範囲: cy ± (|l*sin| + |w*cos|)/2
        float x_extent = (std::abs(l * cos_r) + std::abs(w * sin_r)) / 2.0f;
        float y_extent = (std::abs(l * sin_r) + std::abs(w * cos_r)) / 2.0f;
        
        min_x = cx - x_extent; max_x = cx + x_extent;
        min_y = cy - y_extent; max_y = cy + y_extent;
        min_z = cz - half_h;   max_z = cz + half_h;
    }
};

namespace pp_infer
{

class PointPillarsHarrpNode : public rclcpp::Node
{
public:
  PointPillarsHarrpNode(const rclcpp::NodeOptions & options)
  : Node("point_pillars_harrp_node", options)
  {
    this->declare_parameter("class_names");
    this->declare_parameter<float>("nms_iou_thresh", 0.01);
    this->declare_parameter<int>("pre_nms_top_n", 4096);
    this->declare_parameter<std::string>("model_path", "");
    this->declare_parameter<std::string>("engine_path", "");
    this->declare_parameter<std::string>("data_type", "fp16");
    this->declare_parameter<float>("intensity_scale", 1.0);
    
    rclcpp::Parameter class_names_param = this->get_parameter("class_names");
    class_names = class_names_param.as_string_array();
    nms_iou_thresh = this->get_parameter("nms_iou_thresh").as_double();
    pre_nms_top_n = this->get_parameter("pre_nms_top_n").as_int();
    model_path = this->get_parameter("model_path").as_string();
    engine_path = this->get_parameter("engine_path").as_string();
    data_type = this->get_parameter("data_type").as_string();
    intensity_scale = this->get_parameter("intensity_scale").as_double();
    
    cudaStream_t stream = NULL;
    pointpillar = new PointPillar(model_path, engine_path, stream, data_type);

    publisher_ = this->create_publisher<vision_msgs::msg::Detection3DArray>("bbox", 700);

    // 修正: QoSをReliableに戻して、データの取りこぼしを防ぐ
    // 処理速度が十分高速化したため(14ms)、Reliableでも処理遅延は発生しません
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/point_cloud", 10, std::bind(&PointPillarsHarrpNode::topic_callback, this, _1));

  }

private:
  std::vector<std::string> class_names;
  float nms_iou_thresh;
  int pre_nms_top_n;
  bool do_profile{false};
  std::string model_path;
  std::string engine_path;
  std::string data_type;
  float intensity_scale;
  tf2::Quaternion myQuaternion;
  cudaStream_t stream = NULL;
  PointPillar* pointpillar;  

  void topic_callback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr msg)
  { 
    // 処理時間計測開始（コールバック全体）
    auto start_time = std::chrono::high_resolution_clock::now();

    assert(data_type == "fp32" || data_type == "fp16");
    cudaEvent_t start, stop;
    float inferElapsedTime = 0.0f;
    cudaStream_t stream = NULL;

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaStreamCreate(&stream));

    std::vector<Bndbox> nms_pred;
    nms_pred.reserve(100);

    // PCL形式に変換
    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *pcl_cloud);

      unsigned int num_point_values = pcl_cloud->size();
      unsigned int points_size = pcl_cloud->points.size();

      std::vector<float> pcl_data;
      // 事前リザーブで再割り当てを防ぐ
      pcl_data.reserve(points_size * 4);

      for (const auto& point : pcl_cloud->points) {
        pcl_data.push_back(point.x);
        pcl_data.push_back(point.y);
        pcl_data.push_back(point.z);
        pcl_data.push_back(point.intensity/intensity_scale);
      }

      float* points = static_cast<float *>(pcl_data.data());
      
      unsigned int points_data_size = points_size * sizeof(float) * 4;

      float *points_data = nullptr;
      unsigned int *points_num = nullptr;

      checkCudaErrors(cudaMallocManaged((void **)&points_data, points_data_size));
      checkCudaErrors(cudaMallocManaged((void **)&points_num, sizeof(unsigned int)));
      checkCudaErrors(cudaMemcpy(points_data, points, points_data_size, cudaMemcpyDefault));
      checkCudaErrors(cudaMemcpy(points_num, &points_size, sizeof(unsigned int), cudaMemcpyDefault));
      checkCudaErrors(cudaDeviceSynchronize());

      // 推論実行
      cudaEventRecord(start, stream);
      pointpillar->doinfer(
        points_data, points_num, nms_pred,
        nms_iou_thresh,
        pre_nms_top_n,
        class_names,
        do_profile
      );
      cudaEventRecord(stop, stream);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&inferElapsedTime, start, stop);

      // ==========================================
      // 【超高速化済み】点群抽出処理
      // アルゴリズム:
      // 1. 各検出BBoxについて、AABB(軸平行矩形)を事前計算する
      // 2. 全点群を1回だけ走査する
      // 3. 各点について、AABBチェック(超高速)を行い、通過した場合のみ詳細な回転判定を行う
      // ==========================================
      
      // 1. Box情報の準備
      std::vector<BoxInfo> box_infos;
      box_infos.reserve(nms_pred.size());
      for(int i=0; i<nms_pred.size(); i++) {
          box_infos.emplace_back(i, nms_pred[i]);
      }

      // 2. 点群走査 (全点に対して1回ループ)
      for (const auto& pt : pcl_cloud->points) {
          for (auto& info : box_infos) {
              // AABB 事前フィルタリング (これで90%以上の不要な計算をスキップ)
              if (pt.x < info.min_x || pt.x > info.max_x ||
                  pt.y < info.min_y || pt.y > info.max_y ||
                  pt.z < info.min_z || pt.z > info.max_z) {
                  continue;
              }

              // 詳細判定 (回転考慮)
              float dx = pt.x - info.cx;
              float dy = pt.y - info.cy;
              
              float local_x = dx * info.cos_r - dy * info.sin_r;
              float local_y = dx * info.sin_r + dy * info.cos_r;
              // local_z は AABBチェックで既に z 範囲内であることが確定しているのでチェック不要に近いが、念のため
              // float local_z = pt.z - info.cz; 

              if (std::abs(local_x) <= info.half_l &&
                  std::abs(local_y) <= info.half_w) {
                  info.cloud->push_back(pt);
              }
          }
      }

      // 3. ROSメッセージの構築
      auto pc_detection_arr = std::make_shared<vision_msgs::msg::Detection3DArray>();
      std::vector<vision_msgs::msg::Detection3D> detections;
      
      for(const auto& info : box_infos) {
        vision_msgs::msg::Detection3D detection;
        detection.results.resize(1); 
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        
        // BBox情報のセット (元のnms_predの情報を参照してもよいが、infoから再構築も可能)
        // ここではnms_predを参照
        const auto& pred = nms_pred[info.index];

        detection.bbox.center.position.x = pred.x;
        detection.bbox.center.position.y = pred.y;
        detection.bbox.center.position.z = pred.z;
        detection.bbox.size.x = pred.l;
        detection.bbox.size.y = pred.w;
        detection.bbox.size.z = pred.h;

        myQuaternion.setRPY(0, 0, pred.rt);
        auto orientation = tf2::toMsg(myQuaternion);
        detection.bbox.center.orientation = orientation;

        hyp.id = std::to_string(pred.id);
        hyp.score = pred.score;
        detection.header = msg->header;
        detection.results[0] = hyp;

        // 抽出した点群をセット
        if (!info.cloud->empty()) {
            sensor_msgs::msg::PointCloud2 ros_object_cloud;
            pcl::toROSMsg(*info.cloud, ros_object_cloud);
            ros_object_cloud.header = msg->header;
            detection.source_cloud = ros_object_cloud;
        }

        detections.push_back(detection);
      }

      pc_detection_arr->header = msg->header;
      pc_detection_arr->detections = detections;
      publisher_->publish(*pc_detection_arr);

      checkCudaErrors(cudaFree(points_data));
      checkCudaErrors(cudaFree(points_num));
      nms_pred.clear();

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaStreamDestroy(stream));

    // 全体処理時間の計測終了
    auto end_time = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

    auto message = std_msgs::msg::String();
    // 推論時間(GPU) と 全体処理時間(CPU込み) を両方表示
    message.data = "Infer(GPU): " + std::to_string(inferElapsedTime) + " ms, Total(CPU+GPU): " + std::to_string(total_duration) + " ms, Objects: " + std::to_string(detections.size());
    RCLCPP_INFO(this->get_logger(), "%s", message.data.c_str());
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
  size_t count_;
};

} // namespace pp_infer

RCLCPP_COMPONENTS_REGISTER_NODE(pp_infer::PointPillarsHarrpNode)
