# -*- coding: utf-8 -*-

# -- Sheet --

# # Data analysis with Python
# This is a notebook for the University of Helsinki course "Data analysis with Python". The course is available as a MOOC here: https://csmastersuh.github.io/data_analysis_with_python_summer_2021/basics.html
# 
# ## Exercise 1
# In the first exercise, you were expected to print hello world. The actual course expects you to use Test My Code platform and clunky opensource environments. I opted for using the nifty cloud-based Datalore instead.


print("Hello, World!")

# ## Exercise 2
# 
# In exercise 2 the goal was to read an input and the input to a text string.


print("What country are you from?")
country = input()
print("I have heard that " + country + " is a beautiful country.")

# ## Exercise 3
# The third exercise was about doing multiplication in a for loop


base = 4
for iteration in range(0,11):
    result = base * iteration
    print(base, "multiplied by", iteration, "is", result)

# 


