/* Copyright 2016 Carnegie Mellon University, NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "lightscan/storage/storage_config.h"
#include "lightscan/storage/storage_backend.h"
#include "lightscan/util/common.h"
#include "lightscan/util/video.h"
#include "lightscan/util/caffe.h"

#include <opencv2/opencv.hpp>

#include <mpi.h>
#include <pthread.h>
#include <cstdlib>
#include <string>
#include <libgen.h>

extern "C" {
#include "libavformat/avformat.h"
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace lightscan;

const std::string DB_PATH = "/Users/abpoms/kcam";
const std::string IFRAME_PATH_POSTFIX = "_iframes";
const std::string METADATA_PATH_POSTFIX = "_metadata";
const std::string PROCESSED_VIDEO_POSTFIX = "_processed";
const int NUM_GPUS = 1;
const int BATCH_SIZE = 1;

#define THREAD_RETURN_SUCCESS() \
  do {                                           \
    void* val = malloc(sizeof(int));             \
    *((int*)val) = EXIT_SUCCESS;                 \
    pthread_exit(val);                           \
  } while (0);

///////////////////////////////////////////////////////////////////////////////
/// Path utils

std::string dirname_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* dir = dirname(path_copy);
  return std::string(dir);
}

std::string basename_s(const std::string& path) {
  char* path_copy = strdup(path.c_str());
  char* base = basename(path_copy);
  return std::string(base);
}

std::string processed_video_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + PROCESSED_VIDEO_POSTFIX + ".mp4";
}

std::string metadata_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + METADATA_PATH_POSTFIX + ".bin";
}

std::string iframe_path(const std::string& video_path) {
  return dirname_s(video_path) + "/" +
    basename_s(video_path) + IFRAME_PATH_POSTFIX + ".bin";
}

///////////////////////////////////////////////////////////////////////////////
/// MPI utils
inline bool is_master(int rank) {
  return rank == 0;
}

///////////////////////////////////////////////////////////////////////////////
///

void startup(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  av_register_all();
  FLAGS_minloglevel = 2;
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously load video
void convert_av_frame_to_rgb(
  SwsContext*& sws_context,
  AVFrame* frame,
  char* buffer)
{
  size_t buffer_size =
    av_image_get_buffer_size(AV_PIX_FMT_RGB24, frame->width, frame->height, 1);

  // Convert image to RGB
  sws_context = sws_getCachedContext(
    sws_context,

    frame->width, frame->height,
    static_cast<AVPixelFormat>(frame->format),

    frame->width, frame->height, AV_PIX_FMT_RGB24,
    SWS_BICUBIC, 0, 0, 0);

  if (sws_context == nullptr) {
    fprintf(stderr, "Error trying to get sws context\n");
    assert(false);
  }

  AVFrame rgb_format;
  int alloc_fail = av_image_alloc(rgb_format.data,
                                  rgb_format.linesize,
                                  frame->width,
                                  frame->height,
                                  AV_PIX_FMT_RGB24,
                                  1);

  if (alloc_fail < 0) {
    fprintf(stderr, "Error while allocating avpicture for conversion\n");
    assert(false);
  }

  sws_scale(sws_context,
            frame->data /* input data */,
            frame->linesize /* input layout */,
            0 /* x start location */,
            frame->height /* height of input image */,
            rgb_format.data /* output data */,
            rgb_format.linesize /* output layout */);

  av_image_copy_to_buffer(reinterpret_cast<uint8_t*>(buffer),
                          buffer_size,
                          rgb_format.data,
                          rgb_format.linesize,
                          AV_PIX_FMT_RGB24,
                          frame->width,
                          frame->height,
                          1);

  av_freep(&rgb_format.data[0]);
}

struct LoadVideoArgs {
  // Input arguments
  StorageConfig* storage_config;
  std::string video_path;
  std::string iframe_path;
  int frame_start;
  int frame_end;
  VideoMetadata metadata;
  // Output arguments
  size_t frames_buffer_size;
  char* decoded_frames_buffer; // Should have space for start - end frames
  std::atomic<int>* frames_written;
};

void* load_video_thread(void* arg) {
  // Setup connection to load video
  LoadVideoArgs& args = *reinterpret_cast<LoadVideoArgs*>(arg);

  // Setup a distinct storage backend for each IO thread
  StorageBackend* storage =
    StorageBackend::make_from_config(args.storage_config);

  // Open the iframe file to setup keyframe data
  std::vector<int> keyframe_positions;
  std::vector<int64_t> keyframe_timestamps;
  {
    RandomReadFile* iframe_file;
    storage->make_random_read_file(args.iframe_path, iframe_file);

    (void)read_keyframe_info(
      iframe_file, 0, keyframe_positions, keyframe_timestamps);

    delete iframe_file;
  }

  // Open the video file for reading
  RandomReadFile* file;
  storage->make_random_read_file(args.video_path, file);

  VideoDecoder decoder(file, keyframe_positions, keyframe_timestamps);
  decoder.seek(args.frame_start);

  size_t frame_size =
    av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                             args.metadata.width,
                             args.metadata.height,
                             1);

  SwsContext* sws_context;
  int current_frame = args.frame_start;
  while (current_frame < args.frame_end) {
    AVFrame* frame = decoder.decode();
    assert(frame != nullptr);

    size_t frames_buffer_offset =
      frame_size * (current_frame - args.frame_start);
    assert(frames_buffer_offset < args.frames_buffer_size);
    char* current_frame_buffer_pos =
      args.decoded_frames_buffer + frames_buffer_offset;

    convert_av_frame_to_rgb(sws_context, frame, current_frame_buffer_pos);

    *args.frames_written += 1;
    current_frame += 1;
  }

  // Cleanup
  delete file;
  delete storage;

  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Thread to asynchronously save out results
struct SaveVideoArgs {
};

void* save_video_thread(void* arg) {
  // Setup connection to save video
  THREAD_RETURN_SUCCESS();
}

///////////////////////////////////////////////////////////////////////////////
/// Main processing thread that runs the read, evaluate net, write loop
struct ProcessArgs {
  int gpu_device_id;
  StorageConfig* storage_config;
  std::string video_path;
  std::string iframe_path;
  int frame_start;
  int frame_end;
  VideoMetadata metadata;
};

void* process_thread(void* arg) {
  ProcessArgs& args = *reinterpret_cast<ProcessArgs*>(arg);

  size_t frame_size =
    av_image_get_buffer_size(AV_PIX_FMT_RGB24,
                             args.metadata.width,
                             args.metadata.height,
                             1);
  size_t frame_buffer_size = frame_size * (args.frame_end - args.frame_start);
  char* frame_buffer = new char[frame_buffer_size];
  std::atomic<int> frames_written{0};

  // Create IO threads for reading and writing
  LoadVideoArgs load_args;
  load_args.storage_config = args.storage_config;
  load_args.video_path = args.video_path;
  load_args.iframe_path = args.iframe_path;
  load_args.frame_start = args.frame_start;
  load_args.frame_end = args.frame_end;
  load_args.metadata = args.metadata;
  load_args.frames_buffer_size = frame_buffer_size;
  load_args.decoded_frames_buffer = frame_buffer;
  load_args.frames_written = &frames_written;
  pthread_t load_thread;
  pthread_create(&load_thread, NULL, load_video_thread, &load_args);

  // pthread_t* save_thread;
  // pthread_create(save_thread, NULL, save_video_thread, NULL);

  // Setup caffe net
  NetInfo net_info = load_neural_net(NetType::ALEX_NET, args.gpu_device_id);
  caffe::Net<float>* net = net_info.net;

  // Resize net input blob for batch size
  const boost::shared_ptr<caffe::Blob<float>> data_blob{
    net->blob_by_name("data")};
  if (data_blob->shape(0) != BATCH_SIZE) {
    data_blob->Reshape({
        BATCH_SIZE, 3, net_info.input_size, net_info.input_size});
  }

  int dim = net_info.input_size;

  cv::Mat unsized_mean_mat(
    net_info.mean_width, net_info.mean_height, CV_32FC3, net_info.mean_image);
  cv::Mat mean_mat;
  cv::resize(unsized_mean_mat, mean_mat, cv::Size(dim, dim));

  int current_frame = args.frame_start;
  while (current_frame + BATCH_SIZE < args.frame_end) {
    // Read batch of frames
    if ((current_frame - args.frame_start) >= frames_written) continue;

    // Decompress batch of frame
    printf("processing frame %d\n", current_frame);

    // Process batch of frames
    caffe::Blob<float> net_input{BATCH_SIZE, 3, dim, dim};
    float* net_input_buffer = net_input.mutable_cpu_data();

    for (int i = 0; i < BATCH_SIZE; ++i) {
      char* buffer = frame_buffer + frame_size * (i + current_frame);
      cv::Mat input_mat(
        args.metadata.height, args.metadata.width, CV_8UC3, buffer);
      cv::cvtColor(input_mat, input_mat, CV_RGB2BGR);
      cv::Mat conv_input;
      cv::resize(input_mat, conv_input, cv::Size(dim, dim));
      cv::Mat float_conv_input;
      conv_input.convertTo(float_conv_input, CV_32FC3);
      cv::Mat normed_input = float_conv_input - mean_mat;
      //to_conv_input(&std::get<0>(in_vec[i]), &conv_input, &mean);
      memcpy(net_input_buffer + i * (dim * dim * 3),
             normed_input.data,
             dim * dim * 3 * sizeof(float));
    }

    net->Forward({&net_input});

    // Save batch of frames

    current_frame += BATCH_SIZE;
  }

  // Epilogue for processing less than a batch of frames

  // Cleanup
  delete[] frame_buffer;
  delete net;

  THREAD_RETURN_SUCCESS();
}

void shutdown() {
  MPI_Finalize();
}

int main(int argc, char **argv) {
  startup(argc, argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string video_path =
    "kcam-videos-20140910_195012_247.mp4";

  // Setup storage config
  StorageConfig* config =
    StorageConfig::make_disk_config(DB_PATH);
  StorageBackend* storage = StorageBackend::make_from_config(config);

  // Check if we have already preprocessed the video
  FileInfo video_info;
  StoreResult result =
    storage->get_file_info(processed_video_path(video_path), video_info);
  if (result == StoreResult::FileDoesNotExist) {
    // Preprocess video and then exit
    if (is_master(rank)) {
      log_ls.print("Video not processed yet. Processing now...\n");
      //video_path = "../../../tmp/lightscan3n1YnH";
      //video_path = "../../../tmp/lightscanLNaRk3";
      preprocess_video(storage,
                       video_path,
                       processed_video_path(video_path),
                       metadata_path(video_path),
                       iframe_path(video_path));
    }
  } else {
    // Get video metadata to pass to all workers and determine work distribution
    // from frame count
    VideoMetadata metadata;
    {
      std::unique_ptr<RandomReadFile> metadata_file;
      exit_on_error(
        make_unique_random_read_file(storage,
                                     metadata_path(video_path),
                                     metadata_file));
      (void) read_video_metadata(metadata_file.get(), 0, metadata);
    }

    // Parse args to determine video offset

    // Create processing threads for each gpu
    ProcessArgs processing_thread_args[NUM_GPUS];
    pthread_t processing_threads[NUM_GPUS];
    for (int i = 0; i < NUM_GPUS; ++i) {
      ProcessArgs& args = processing_thread_args[i];
      args.gpu_device_id = i;
      args.storage_config = config;
      args.video_path = video_path;
      args.iframe_path = iframe_path(video_path);
      args.frame_start = 0;
      args.frame_end = 2000;
      args.metadata = metadata;
      pthread_create(&processing_threads[i],
                     NULL,
                     process_thread,
                     &processing_thread_args[i]);
    }

    // Wait till done
    for (int i = 0; i < NUM_GPUS; ++i) {
      void* result;

      int err = pthread_join(processing_threads[i], &result);
      if (err != 0) {
        fprintf(stderr, "error in pthread_join\n");
        exit(EXIT_FAILURE);
      }

      printf("Joined with thread %d; returned value was %d\n",
             i, *((int *)result));
      free(result);      /* Free memory allocated by thread */
    }
  }

 // Cleanup
 delete storage;
 delete config;

 shutdown();

 return EXIT_SUCCESS;
}
