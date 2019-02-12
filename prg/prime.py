def prime(number):
    for i in range(2, number/2 + 1):
        if(number % i == 0):
            print(number + " isn't prime because " + i + " is a factor!")
            return
    print(number + " is a prime number!")
    return

