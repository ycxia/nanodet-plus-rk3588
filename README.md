# rk3588-nanodetplus

Deploy nanodet-plus model on rk3588

1. Change some parameters for your model in include/postprocess.h. 

   example: nanodet-EfficientNet-Lite2_512
   
   #define REG_MAX           10
   
   #define STRIDE_SIZE       3
   
   If you develop nanodet-plus-m-1.5x_416, the parameters shoule be:
   
    #define REG_MAX           7
    
    #define STRIDE_SIZE       4
    
2. For speeding up postprocess, I used unsigmoid threshhold in postprocess.cc, because sigmoid function is time consuming. 

   line255 :  float unsigmod_conf_thresh = unsigmoid(conf_threshold);
