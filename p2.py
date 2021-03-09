string = input("Enter a long string")
print(string)
key = input("enter a substring to search")
if(key in string):
	print("letter found in string")
else: 
	print("letter not found")
input(" press enter")