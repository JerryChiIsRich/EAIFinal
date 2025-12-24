按照 command.txt 的指令逐個執行

好笑的是某次推論居然出現
(test) jerry@LAPTOP-F51GU41B:/mnt/d/該死的大學碩士作業/碩一上/AI/finalProject$ python src/infer_student_lcm_lora.py \
>   --edge tmp/edge_0001.png \
>   --teacher_controlnet_dir outputs/teacher_controlnet \
>   --student_lora_dir outputs/student_lora_lcm \
>   --steps 50 \
>   --out student_50stepsv2.png
Loading pipeline components...:   0%|                                                                                                                     | 0/7 [00:00<?, ?it/s]`torch_dtype` is deprecated! Use `dtype` instead!
Loading pipeline components...: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:09<00:00,  1.42s/it]100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:10<00:00,  4.70it/s]`Potential NSFW content was detected in one or more images. A black image will be returned instead. Try again with a different prompt and/or seed.`
[OK] saved: student_50stepsv2.png
