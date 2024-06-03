for pred_len in 25 50 75 
do
	python -u run.py --is_training 1 --root_path ./dataset/humaneva/ --data_path Walking.npy --model_id Walking --model IDOL --data Humaneva --features M --seq_len 125 --label_len 125 --pred_len $pred_len --enc_in 45 --dec_in 45 --c_out 45 --des 'Exp' --d_model 24 --itr 1 --learning_rate 0.015 --dropout 0.1 --batch_size 128 --train_epoch 50
done

