for pred_len in 100 125 150
do
	python -u run.py --is_training 1 --root_path ./dataset/humaneva/ --data_path Gestures.npy --model_id Gestures --model IDOL --data Humaneva --features M --seq_len 125 --label_len 125 --pred_len $pred_len --enc_in 45 --dec_in 45 --c_out 45 --des 'Exp' --d_model 18 --itr 1 --learning_rate 0.02 --dropout 0.0 --batch_size 32 --train_epochs 50
done

