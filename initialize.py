from build import encoder
from layers import tanH, Sigmoid, RelU


def test_ae(	
			init_params,
			build_params,
			train_params,
			flagy = True,
		):
	
	ae = encoder(
			init_params = init_params,
			flagy = flagy 
		       )
			   
	ae.buildit(
			build_params = build_params,
			flagy = flagy
	          )    
	   
	ae.train(
	   		train_params = train_params,
			flagy = flagy
		     )
				
### Boiler Plate ###
if __name__ == '__main__':
    
	init_params = {
					"n_train_batches" : 60,
					"batch_size"	  : 100,
					"img_ht"          : 32,
					"img_wdt"         : 32,
					"tile_ht"         : 10,
					"tile_wdt"        : 10,
					"output_folder"   : 'op_images_256_lr_point4',
					"disp_flag"       : True
					
					}
	#demo 1 (no. of layers=2, learning rate = 0.4, activation func = Sigmoid) DEFAULT CASE				
	build_params = {
					"learning_rate"  :	0.4,
					"n_hidden_enc"   :  [ 256 ],
					"n_hidden_dec"   :  [ 256 ],
					"activation"	 :	[ Sigmoid ], 	
					"cost_fun"		 :	'rmse',
					"tied_weights"	 :  True,
					"LR_decay"		 :  0.0007,
					"begin_mom"		 :  0.4,
					"end_mom"		 :  0.85,
					"mom_thrs"		 :  100
					}
	
	#demo 2 (no. of layers=4, learning rate = 0.4, activation func = Sigmoid) 
	# build_params = {
	# 				"learning_rate"  :	0.4,
	# 				"n_hidden_enc"   :  [ 256, 128 ],
	# 				"n_hidden_dec"   :  [ 128, 256 ],
	# 				"activation"	 :	[ Sigmoid, Sigmoid ], 	
	# 				"cost_fun"		 :	'rmse',
	# 				"tied_weights"	 :  True,
	# 				"LR_decay"		 :  0.0007,
	# 				"begin_mom"		 :  0.4,
	# 				"end_mom"		 :  0.85,
	# 				"mom_thrs"		 :  200
	# 				}

	#demo 3 (no. of layers=4, learning rate = 0.5, activation func = Sigmoid) 
	# build_params = {
	# 				"learning_rate"  :	0.5,
	# 				"n_hidden_enc"   :  [ 512, 256 ],
	# 				"n_hidden_dec"   :  [ 256, 512 ],
	# 				"activation"	 :	[ Sigmoid, Sigmoid ], 	
	# 				"cost_fun"		 :	'rmse',
	# 				"tied_weights"	 :  True,
	# 				"LR_decay"		 :  0.0007,
	# 				"begin_mom"		 :  0.4,
	# 				"end_mom"		 :  0.85,
	# 				"mom_thrs"		 :  200
	# 				}
				
				
	train_params = {
					"training_epochs" : 200
					}
	
	flagy = True
					
test_ae(
			init_params		= init_params,
			build_params	= build_params,
			train_params	= train_params,
			flagy			= flagy,
	)
