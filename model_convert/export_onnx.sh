python export_onnx.py --restore_ckpt ../models/raftstereo-realtime.pth \
                --shared_backbone \
                --n_downsample 3 \
                --n_gru_layers 2 \
                --slow_fast_gru \
                --valid_iters 7 \
                --corr_radius 4 \
                --corr_implementation alt \
                --output_directory ../models \
                --width 1280 \
                --height 384 