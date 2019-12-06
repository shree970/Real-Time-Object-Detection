def sort(belt,list1):
	
	for item in belt:
		if item in list1[0]:
			out.append(1)
			list1[0].remove(item)
		elif item in list1[1]:
			out.append(2)
			list1[1].remove(item)
		else:
			out.append('None');
	
	return out
'''
belt = ['Dove', 'Moti', 'Med','Moti', 'Med', 'Dove']
out = []
list1 = [['Dove', 'Moti', 'Med'],['Moti', 'Med', 'Dove']]

sort(belt,list1)

print(belt)
print('This items are yet to be delivered') 
print(list1)
#list1[0].remove('Dove')
print(out)
'''
