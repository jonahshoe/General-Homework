# number guessing game

import random

number = random.randint(0,100) #secret number

guess = None
guessnumber = 0

while guess != number:
    if guessnumber == 0:
        guess = int(input("Guess the number between 1 and 100: "))
    elif guessnumber > 0:
        guess = int(input("Guess again: "))

    if guess == number:
        print("You guessed correctly.")
        print("You needed " , guessnumber+1 , " guesses.")
    elif guess < number:
        print("Your guess " , guess , " is too low")
    elif guess > number:
        print("Your guess " , guess , " is too high")
    guessnumber = guessnumber + 1
