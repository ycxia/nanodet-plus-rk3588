// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "postprocess.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <set>
#include <vector>
#include <algorithm>
#define LABEL_NALE_TXT_PATH "./model/coco_80_labels_list.txt"

static char* labels[OBJ_CLASS_NUM];

inline static int clamp(float val, int min, int max) { return val > min ? (val < max ? val : max) : min; }

char* readLine(FILE* fp, char* buffer, int* len)
{
  int    ch;
  int    i        = 0;
  size_t buff_len = 0;

  buffer = (char*)malloc(buff_len + 1);
  if (!buffer)
    return NULL; // Out of memory

  while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
    buff_len++;
    void* tmp = realloc(buffer, buff_len + 1);
    if (tmp == NULL) {
      free(buffer);
      return NULL; // Out of memory
    }
    buffer = (char*)tmp;

    buffer[i] = (char)ch;
    i++;
  }
  buffer[i] = '\0';

  *len = buff_len;

  // Detect end
  if (ch == EOF && (i == 0 || ferror(fp))) {
    free(buffer);
    return NULL;
  }
  return buffer;
}

int readLines(const char* fileName, char* lines[], int max_line)
{
  FILE* file = fopen(fileName, "r");
  char* s;
  int   i = 0;
  int   n = 0;

  if (file == NULL) {
    printf("Open %s fail!\n", fileName);
    return -1;
  }

  while ((s = readLine(file, s, &n)) != NULL) {
    lines[i++] = s;
    if (i >= max_line)
      break;
  }
  fclose(file);
  return i;
}

int loadLabelName(const char* locationFilename, char* label[])
{
  printf("loadLabelName %s\n", locationFilename);
  readLines(locationFilename, label, OBJ_CLASS_NUM);
  return 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0, float xmin1, float ymin1, float xmax1,
                              float ymax1)
{
  float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0);
  float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0);
  float i = w * h;
  float u = (xmax0 - xmin0 + 1.0) * (ymax0 - ymin0 + 1.0) + (xmax1 - xmin1 + 1.0) * (ymax1 - ymin1 + 1.0) - i;
  return u <= 0.f ? 0.f : (i / u);
}

void mynms(std::vector<BoxInfo>& input_boxes, float NMSTHRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
            * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMSTHRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices)
{
  float key;
  int   key_index;
  int   low  = left;
  int   high = right;
  if (left < right) {
    key_index = indices[left];
    key       = input[left];
    while (low < high) {
      while (low < high && input[high] <= key) {
        high--;
      }
      input[low]   = input[high];
      indices[low] = indices[high];
      while (low < high && input[low] >= key) {
        low++;
      }
      input[high]   = input[low];
      indices[high] = indices[low];
    }
    input[low]   = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
  }
  return low;
}

static float sigmoid(float x) { return 1.0 / (1.0 + expf(-x)); }

static float unsigmoid(float y) { return -1.0 * logf((1.0 / y) - 1.0); }

inline static int32_t __clip(float val, float min, float max)
{
  float f = val <= min ? min : (val >= max ? max : val);
  return f;
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float fast_sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale)
{
  float  dst_val = (f32 / scale) + zp;
  int8_t res     = (int8_t)__clip(dst_val, -128, 127);
  return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

template<typename _Tp>
int activation_function_softmax(const _Tp* src, float* dst, int length)
{
    const _Tp alpha = *std::max(src, src + length);
    _Tp denominator{ 0 };

    for (int i = 0; i < length; ++i)
    {
        dst[i] = fast_exp(src[i] - alpha);
        denominator += dst[i];
    }

    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
    }

    return 0;
}

double _get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int post_process(float* output, int model_in_h, int model_in_w, float conf_threshold,
                 float nms_threshold, float scale_w, float scale_h, std::vector<int32_t>& qnt_zps,
                 std::vector<float>& qnt_scales, detect_result_group_t* group)
{
  static int init = -1;
  if (init == -1) {
    int ret = 0;
    ret     = loadLabelName(LABEL_NALE_TXT_PATH, labels);
    if (ret < 0) {
      return -1;
    }
    init = 0;
  }

  memset(group, 0, sizeof(detect_result_group_t));

  std::vector<float> filterBoxes;
  std::vector<float> objProbs;
  std::vector<int>   classId;  
  
  std::vector<std::vector<BoxInfo>> res;
  int32_t zp = qnt_zps[0];
  float scale = qnt_scales[0];

  res.resize(OBJ_CLASS_NUM);
  
  int total_index = 0;
  //printf("conf_thresh: %f\n", conf_threshold);
  // FILE* fp;
  // fp = fopen("box_index.txt","w");

  float unsigmod_conf_thresh = unsigmoid(conf_threshold);
  struct timeval start_time, stop_time;
  
  for (int i = 0; i < STRIDE_SIZE; ++i)
  {    
    int stride = STRIDE[i];
    //printf("stride: %d\n", stride);
    int feature_h = ceil(double(model_in_h) / stride);
    int feature_w = ceil(double(model_in_w) / stride);
    //printf("feature_hw: %d  %d  %d\n", feature_h, feature_w, feature_h * feature_w);
    //float maxscoretime = 0.0;
    for(int j = total_index; j < feature_h * feature_w + total_index; j++)
    {            
      int row = (j - total_index) / feature_w;
      int col = (j - total_index) % feature_w;
      //printf("row col: %d  %d\n", row, col);
      float max_score = -10000;
      int cur_label = 0;
      int class_index_start = j * (OBJ_CLASS_NUM + 4 * (REG_MAX + 1));
      //gettimeofday(&start_time, NULL);
      for(int label_idx=0; label_idx < OBJ_CLASS_NUM; label_idx++)
      {        
        //int class_index = label_idx + class_index_start;
        //int8_t i8_cur_score = output[class_index];        
        float cur_score = output[label_idx + class_index_start];  //this line code caused long time                 
        if(cur_score > max_score)
        {
          max_score = cur_score;
          cur_label = label_idx;
        }        
      }
      //gettimeofday(&stop_time, NULL);
      //maxscoretime += (_get_us(stop_time) - _get_us(start_time)) / 1000;
      //printf("find max score use %f ms\n", maxscoretime);  
      //fprintf(fp, "%d\n", j * (OBJ_CLASS_NUM + 4 * (REG_MAX + 1)) + OBJ_CLASS_NUM );
      //max_score = sigmoid(max_score);
      if(max_score > unsigmod_conf_thresh) //decode box
      {                
        max_score = sigmoid(max_score);
        //printf("cur_label:%d %s %f\n", cur_label, labels[cur_label], max_score);
        int box_index = j * (OBJ_CLASS_NUM + 4 * (REG_MAX + 1)) + OBJ_CLASS_NUM;        
        float ct_x = (col+0.5) * stride;
        float ct_y = (row+0.5) * stride;
        std::vector<float> dis_pred;
        dis_pred.resize(4);
        for(int i = 0; i < 4; i++)
        {
          float dis = 0;
          float *dis_after_sm = (float*)malloc(sizeof(float)*(REG_MAX + 1));          
          activation_function_softmax(output + box_index + i*(REG_MAX+1), dis_after_sm, REG_MAX+1);
          for(int j = 0; j< (REG_MAX+1); j++)
          {
            dis += j * dis_after_sm[j];
          }
          dis *= stride;
          dis_pred[i] = dis;
          if(dis_after_sm)
          {
            free(dis_after_sm);
          }
        }
        float xmin = (std::max)(ct_x - dis_pred[0], .0f);
        float ymin = (std::max)(ct_y - dis_pred[1], .0f);
        float xmax = (std::min)(ct_x + dis_pred[2], (float)model_in_w);
        float ymax = (std::min)(ct_y + dis_pred[3], (float)model_in_h);
        res[cur_label].push_back(BoxInfo{xmin, ymin, xmax, ymax, max_score, cur_label});    
               
      } 
      
    }
    total_index += feature_h * feature_w;
    
  }
  
  std::vector<BoxInfo> dets;
  for (int i = 0; i < (int)res.size(); i++)
  {
      mynms(res[i], NMS_THRESH);
      for (auto box : res[i])
      {
          dets.push_back(box);
      }
  }
  
  //printf("dets.size: %d \n", (int)dets.size());

  group->count   = (int)dets.size();
  /* box valid detect target */
  for (int i = 0; i < dets.size(); ++i) {   

    float x1       = dets[i].x1;
    float y1       = dets[i].y1;
    float x2       = dets[i].x2;
    float y2       = dets[i].y2;
    int   id       = dets[i].label;
    float obj_conf = dets[i].score;

    group->results[i].box.left   = (int)(clamp(x1, 0, model_in_w) / scale_w);
    group->results[i].box.top    = (int)(clamp(y1, 0, model_in_h) / scale_h);
    group->results[i].box.right  = (int)(clamp(x2, 0, model_in_w) / scale_w);
    group->results[i].box.bottom = (int)(clamp(y2, 0, model_in_h) / scale_h);
    group->results[i].prop       = obj_conf;
    char* label                  = labels[id];
    strncpy(group->results[i].name, label, OBJ_NAME_MAX_SIZE);

    // printf("result %2d: (%4d, %4d, %4d, %4d), %s %f\n", i, group->results[i].box.left,
    // group->results[i].box.top,
    //        group->results[i].box.right, group->results[i].box.bottom, label, group->results[i].prop);
    
  }
 
  return 0;
}

void deinitPostProcess()
{
  for (int i = 0; i < OBJ_CLASS_NUM; i++) {
    if (labels[i] != nullptr) {
      free(labels[i]);
      labels[i] = nullptr;
    }
  }
}
