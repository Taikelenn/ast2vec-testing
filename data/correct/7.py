num = int(input("Enter a number: "))

input_valid = num > 100
if input_valid:
    print(num * 2)
else:
    print("Number too small")
