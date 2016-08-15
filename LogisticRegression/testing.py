import numpy as np


degree = 6
c = 2


powers = []
indexList = [[]]

length = 0

while (length < c):

	while(len(indexList[0]) == length):

		last_list = indexList.pop(0)
		for i in range(degree + 1):
			if sum(last_list) + i <= degree:
				new_list = last_list + [i]
				indexList.append(new_list)


	length += 1


for power in indexList:
	if power not in powers:
		powers.append(power)


print(len(powers))


test_data = np.array([[2, 3]])
new_data = np.zeros((1, 1))

for feat_index in range(len(powers)):
	new_column = np.ones((1, 1))
	
	for index in range(c):
		new_column = new_column * np.power(test_data[0, index], powers[feat_index][index])


	new_data = np.column_stack((new_data, new_column))


print(new_data)
print(np.shape(new_data))