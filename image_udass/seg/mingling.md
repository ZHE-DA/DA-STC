transformer_gta
python run_experiments.py --config configs/mic/gtaHR2csHR_mic_hrda.py
cfg.resume_from = '/home/ZZ/MIC-master/seg/work_dirs/local-basic/240810_0952_gtaHR2csHR_mic_hrda_s2_a891a/iter_4000.pth'
self.local_iter = 4000
< 6000
----->   /home/ZZ/MIC-master/seg/work_dirs/local-basic/240811_0032_gtaHR2csHR_mic_hrda_s2_aa701
断点续训：5500  
----->   














transformer_syn
python run_experiments.py --config configs/mic/synthiaHR2csHR_mic_hrda.py
cfg.resume_from = '/home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-basic/240810_0955_synthiaHR2csHR_mic_hrda_s2_ade8e/iter_3000.pth'
self.local_iter = 3000
< 6000
----->   /home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-basic/240811_0350_synthiaHR2csHR_mic_hrda_s2_78eb2
断点续训  cfg.resume_from = '/home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-basic/240811_0350_synthiaHR2csHR_mic_hrda_s2_78eb2/iter_10000.pth'
----->   /home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-basic/240812_0057_synthiaHR2csHR_mic_hrda_s2_bd3a6





cnn_syn
python run_experiments.py --exp 821
cfg.resume_from = '/home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-exp821/240810_1333_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_b6485/iter_6000.pth'
self.local_iter = 6000
< 7000
----->   /home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-exp821/240811_0403_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_69796
断点续训     cfg.resume_from = '/home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-exp821/240811_0403_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_69796/iter_7000.pth'
----->   /home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-exp821/240812_0037_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_2c1ad






















cnn_gta
python run_experiments.py --exp 811
cfg.resume_from = '/home/customer/Desktop/ZZ/MIC-master/seg/work_dirs/local-exp811/240810_1332_gtaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_710e9/iter_6000.pth'
self.local_iter = 6000
< 7000
----->   









需要替换gpu  id , 
dacs文件换成dacs_syn或者dacs_gta
cfg.resume_from 
self.local_iter = 6000
< 7000


