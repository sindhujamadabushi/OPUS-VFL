def set_environment(organization_num, dname):
	total_tokens = []
	client_costs_ratio = []
	client_actual_resources = []
	num_warmup_epochs = 0
	epochs = 0

	if dname == 'MNIST':
		if organization_num == 2:
			total_tokens = [100,120]
			client_costs_ratio = [1, 0.7, 0.2]
			client_actual_resources = [10, 40]
			num_warmup_epochs = 20
			epochs = 60

		if organization_num == 3:
			total_tokens = [100,120,130]
			client_costs_ratio = [1, 0.7, 0.2, 0.1]
			client_actual_resources = [10, 40, 50]
			num_warmup_epochs = 20
			epochs = 60

		if organization_num == 4:
			total_tokens = [100,120,130,110]
			client_costs_ratio = [1, 0.7, 0.2, 0.6, 0.1]
			client_actual_resources = [10, 40, 50, 30]
			num_warmup_epochs = 20
			epochs = 60

		if organization_num == 5:
			total_tokens = [100,120,130,110,150]
			client_costs_ratio = [1, 0.7, 0.001, 0.2, 1000, 0.1]
			client_actual_resources = [10, 40, 50, 30, 60]
			num_warmup_epochs = 20
			epochs = 60

		if organization_num == 7:
			total_tokens = [100,120,130,110,150,100,125]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10]
			num_warmup_epochs = 40
			epochs = 100

		if organization_num == 9:
			total_tokens = [100,120,130,110,150,100,125, 110, 150]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2, 0.9, 0.3]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70, 40]
			num_warmup_epochs = 60
			epochs = 60

		if organization_num == 10:
			total_tokens = [100,120,130,110,150,100,125, 110, 150, 170]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.7,  0.4, 0.2, 0.9, 0.3]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70, 40, 50]
			num_warmup_epochs = 60
			epochs = 60

		if organization_num == 12:
			total_tokens = [100,120,130,110,150,100,120, 110, 150, 140, 170, 130]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2, 0.9, 0.3, 0.6, 0.4, 0.7]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70, 40, 30, 70, 20]
			num_warmup_epochs = 60
			epochs = 60

		if organization_num == 15:
			total_tokens = [100,120,130,110,150,100,120, 110, 150, 140, 170, 130, 100, 140, 120]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2, 0.9,0.3, 0.6, 0.4, 0.7, 0.1, 0.5, 0.9]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70,40, 30, 70, 20, 30, 50, 10]
			num_warmup_epochs = 60
			epochs = 150

		if organization_num == 20:
			total_tokens = [100,120,130,110,150,100,120, 110, 150, 140, 170, 130, 100, 140, 120, 120, 110, 150, 140, 170]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2, 0.9,0.3, 0.6, 0.4, 0.7, 0.1, 0.5, 0.9,0.7, 0.5, 0.9, 0.4, 0.8]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70,40, 30, 70, 20, 30, 50, 10,30, 70, 20, 30, 50]
			num_warmup_epochs = 60
			epochs = 60

		if organization_num == 25:
			total_tokens = [100,120,130,110,150,100,120, 110, 150, 140, 170, 130, 100, 140, 120, 120, 110, 150, 140, 170,120, 110, 150, 140, 170]
			client_costs_ratio = [1, 0.7, 0.5, 0.9, 0.4, 0.8, 0.4, 0.2, 0.9,0.3, 0.6, 0.4, 0.7, 0.1, 0.5, 0.9,0.7, 0.5, 0.9, 0.4, 0.8,0.7, 0.5, 0.9, 0.4, 0.8]
			client_actual_resources = [20, 40, 50, 30, 60, 50, 10, 70,40, 30, 70, 20, 30, 50, 10,30,50, 10, 70,40, 30, 70, 20, 30, 50]
			num_warmup_epochs = 60
			epochs = 60
        
	if dname == 'CIFAR10':
		
		if organization_num == 2:
			total_tokens = [100,120]
			client_costs_ratio = [1, 0.7, 0.2]
			client_actual_resources = [10, 40]
			num_warmup_epochs = 20
			epochs = 100

		if organization_num == 3:
			total_tokens = [100,120,130]
			client_costs_ratio = [1, 0.7, 0.2, 0.1]
			client_actual_resources = [10, 40, 50]
			num_warmup_epochs = 20
			epochs = 60
		
		if organization_num == 4:
			print("here")
			total_tokens = [100,120,130,110]
			client_costs_ratio = [1, 0.7, 0.2, 0.6, 0.1]
			client_actual_resources = [10, 40, 50, 30]
			num_warmup_epochs = 20
			epochs = 60

		if organization_num == 5:
			print("here")
			total_tokens = [100,120,130,110, 60]
			client_costs_ratio = [1, 0.7, 0.2, 0.6, 0.1, 0.6]
			client_actual_resources = [10, 40, 50, 30, 20]
			num_warmup_epochs = 20
			epochs = 60


	return total_tokens, client_costs_ratio, client_actual_resources, num_warmup_epochs

def set_learning_rates(dname):
	learning_rates = [0,0]
	if dname == 'MNIST':
		learning_rates[0] = 0.000015
		learning_rates[1] = 0.0000855
		epochs = 60

	if dname == 'CRITEO':
		learning_rates[0] = 0.00001
		learning_rates[1] = 0.00002
		epochs = 10
	return learning_rates, epochs

