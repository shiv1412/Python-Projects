import math
a = int(input("Enter first number"))
b = int(input("Enter second number"))
gcd1 = math.gcd(a,b)
print ("THe gcd of two numbers is ",end="")
print(gcd1)
lcm = str(a*b/gcd1)
print(" lcm of two numbers is ",end="")
print(lcm)