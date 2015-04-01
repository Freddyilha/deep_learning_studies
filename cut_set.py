with open('dataset_cifar-10_full.txt') as fid:
	i = 0
	for line in fid:
		print line.rsplit('\n')[0]
		i += 1
		if i == 5000:
			break
		