# -*- coding: utf-8 -*-

# -- Sheet --

# # Basics of AI MOOC by Reaktor and Helsinki Uni
# In this notebook, I worked on the exercises for the Basics of AI free MOOC by Reaktor and Helsinki University. If you want to take the course, you can find it at https://buildingai.elementsofai.com
# ## Exercise 1: Advanced
# In this exercise, you had to recursively iterate all the possible port combinations starting from Panama (PAN). Note: This solution isn't a direct answer as it uses a dash instead of a space to separate the routes. However, I like that presentation more.
# 
# ### TODO
# Use the let's plot geocode library and see if the routes could be plotted on a map


portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]
 
def permutations(route, ports):
    if len(ports) <= 0:
        print('-'.join([portnames[i] for i in route]))
        return
    for port in ports:
        subroute = route.copy()
        subroute.append(port)
        subports = ports.copy()
        subports.remove(port)
        permutations(subroute, subports)

# this will start the recursion with 0 as the first stop
permutations([0], list(range(1, len(portnames))))

# ## Exercise 2: Advanced
# This exercise expanded on Exercise 1 by adding an emission table and solving which route combo emits the least amount of CO2. As in the first exercise, I opted to use a dash as to me it illustrates a route better.
# 
# ### TODO
# If geocode plotting works in Exercise 1, add it here too.


portnames = ["PAN", "AMS", "CAS", "NYC", "HEL"]

# https://sea-distances.org/
# nautical miles converted to km

D = [
        [0,8943,8019,3652,10545],
        [8943,0,2619,6317,2078],
        [8019,2619,0,5836,4939],
        [3652,6317,5836,0,7825],
        [10545,2078,4939,7825,0]
    ]

# https://timeforchange.org/co2-emissions-shipping-goods
# assume 20g per km per metric ton (of pineapples)

co2 = 0.020

# DATA BLOCK ENDS

# these variables are initialised to nonsensical values
# your program should determine the correct values for them
smallest = 1000000
bestroute = [0, 0, 0, 0, 0]

def permutations(route, ports):
    global smallest, bestroute
    if len(ports) <= 0:
        #print('-'.join([portnames[i] for i in route]))
        emissions = 0
        for port in range(0,len(route)-1):
            emissions += D[route[port]][route[port+1]]
        emissions *= co2
        if emissions < smallest:
            smallest = emissions
            bestroute = route
        return
    for port in ports:
        subroute = route.copy()
        subroute.append(port)
        subports = ports.copy()
        subports.remove(port)
        permutations(subroute, subports)

def main():
    # this will start the recursion 
    permutations([0], list(range(1, len(portnames))))

    # print the best route and its emissions
    print('-'.join([portnames[i] for i in bestroute]) + " %.1f kg" % smallest)

main()

# ## Exercise 3: Advanced
# This exercise explored the hill climbing algorithm. The solution might crash in the edge cases, but it did pass so whatever (read: too laze to iterate and find out for sure). The plot draws the "mountains" and marks the found peak.


from lets_plot import *

import math
import random             	# just for generating random mountains                                 	 

# generate random mountains                                                                               	 

w = [.05, random.random()/3, random.random()/3]
h = [1.+math.sin(1+x/.6)*w[0]+math.sin(-.3+x/9.)*w[1]+math.sin(-.2+x/30.)*w[2] for x in range(100)]

def climb(x, h):
    # keep climbing until we've found a summit
    summit = False

    # edit here
    while not summit:
        summit = True         # stop unless there's a way up
        rightbound = min(x + 5, len(h))
        leftbound = max(x - 5, 0)
        for step in range(0, rightbound):
            if h[step] > h[x]:
                x = step        # right is higher, go there
                summit = False    # and keep going
        for step in range(0, leftbound, -1):
            if h[step] > h[x]:
                x = step
                summit = False
    return x

def main(h):
    # start at a random place                                                                                  	 
    x0 = random.randint(1, 98)
    x = climb(x0, h)
    plot = ggplot(dict(X=list(range(0,100)), Y=h), aes('X', 'Y')) + geom_line() + geom_point(x=x, y=h[x], size=10, shape=9, color='red')
    return plot
    #return x0, x

main(h)

# ## Exercise 4: Advanced
# This exercise was teaching random in Python. Probably overcomplicated the solution a bit but just goes to show there are multiple ways to skin a cat. And a dog. And a bat.


import random

def main():
    probs = [0.8, 0.9, 1.0]
    animals = ["dogs", "cats", "bats"]
    prob = random.random()
    favorite = ""
    if prob < probs[0]:
        favorite = animals[0]
    elif prob >= probs[0] and prob < probs[1]:
        favorite = animals[1]
    elif prob >= probs[1] and prob < probs[2]:
        favorite = animals[2]
    else:
        favorite = "coding errors"
    print("I love " + favorite) 

main()

# ## Exercise 5: Advanced
# This exercise implements simulated annealing. The return function had some challenges with working in Safari importing Numpy. The instructions weren't also super clear if you actually need to output something. ~~Might need to test on a different browser.~~ Tested on Vivaldi, and got kind of a cryptic assertion.
# 
# Turns out that as the mathematical definition for a probability means clamping between 0 and 1, the test actually ran also with OLD being smaller than NEW, which resulted in a like a 270 percent probability. I just fixed it with a simple clamp (editor's note: Why doesn't Python have something as simple as clamp? coder's note: Did you check numpy?), which might or might not be the correct way.


import random
import numpy as np

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def accept_prob(S_old, S_new, T):
    # this is the acceptance "probability" in the greedy hill-climbing method
    # where new solutions are accepted if and only if they are better
    # than the old one.
    # change it to be the acceptance probability in simulated annealing
    return clamp(np.exp(-(S_old - S_new)/T), 0.0, 1.0)

# the above function will be used as follows. this is shown just for
# your information; you don't have to change anything here
def accept(S_old, S_new, T):
    if random.random() < accept_prob(S_old, S_new, T):
        print(True)
    else:
        print(False)
accept(150, 140, 15)

# ## Exercise 6: Advanced
# The last exercise combined all of the above to try improve the hill climbing algorithm. This challenge had the same problem as the previous one, so no-go to run on Safari. ~~Maybe try with a different browser on the site later.~~ Tried with Vivaldi. Was able to test and submit (after updating the accept_prob with clamp from the previous).
# 
# The plot draws all the tracks and renders succesful ones in green. If the function is working, you should see more green than grey.
# 
# ### TODO
# + Figure out how to plot the peaks like in the last hillclimbing. The dataframes are a bit different this time around.
# + Numpy apparently is easy to get to overflow. Figure out how to get the exp to play nice and not overflowing.


import matplotlib.pyplot as plt 
import numpy as np
import random

N = 100     # size of the problem is N x N                                      
steps = 3000    # total number of iterations                                        
tracks = 50

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def accept_prob(S_old, S_new, T):
    # this is the acceptance "probability" in the greedy hill-climbing method
    # where new solutions are accepted if and only if they are better
    # than the old one.
    # change it to be the acceptance probability in simulated annealing
    if T <= 0:
        return 0
    return clamp(np.exp(-(S_old - S_new)/T), 0.0, 1.0)

# generate a landscape with multiple local optima                                          
def generator(x, y, x0=0.0, y0=0.0):
    return np.sin((x/N-x0)*np.pi)+np.sin((y/N-y0)*np.pi)+\
        .07*np.cos(12*(x/N-x0)*np.pi)+.07*np.cos(12*(y/N-y0)*np.pi)

x0 = np.random.random() - 0.5
y0 = np.random.random() - 0.5
h = np.fromfunction(np.vectorize(generator), (N, N), x0=x0, y0=y0, dtype=int)
peak_x, peak_y = np.unravel_index(np.argmax(h), h.shape)

# starting points                                                               
x = np.random.randint(0, N, tracks)
y = np.random.randint(0, N, tracks)

def main():
    global x
    global y

    for step in range(steps):
        # add a temperature schedule here
        T = max(0, ((steps - step)/steps)**3-.005)
        # update solutions on each search track                                     
        for i in range(tracks):
            # try a new solution near the current one                               
            x_new = np.random.randint(max(0, x[i]-2), min(N, x[i]+2+1))
            y_new = np.random.randint(max(0, y[i]-2), min(N, y[i]+2+1))
            S_old = h[x[i], y[i]]
            S_new = h[x_new, y_new]

            # change this to use simulated annealing
            if random.random() < accept_prob(S_old, S_new, T):
                x[i], y[i] = x_new, y_new   # new solution is better, go there       
            else:
                pass                        # if the new solution is worse, do nothing

    # Number of tracks found the peak
    print(sum([x[j] == peak_x and y[j] == peak_y for j in range(tracks)]))
    for j in range(tracks):
        plotcolor="grey"
        if x[j] == peak_x and y[j] == peak_y:
            plotcolor="green"
        plt.plot(list(range(0,100)), h[j], color=plotcolor)
    plt.show()
main()

# ## Exercise 7: Advanced
# This exercise was about coin flipping, checking times there's at least 5 heads (or tails whichever you want to view it), and checking the count is close to a certain value. As the before examples, the Basics of AI site doesn't like Numpy and Safari combo, so need to verify this in a different browser.
# 
# ### TODO
# Add a visualization for the coin flips.


import numpy as np

def generate(p1):
    # change this so that it generates 10000 random zeros and ones
    # where the probability of one is p1
    seq = np.random.choice([0,1], p=[1-p1, p1], size=10000)
    return seq

def count(seq):
    five_ones = 0
    ones = 0
    for number in seq:
        if number == 1:
            ones += 1
            if ones >= 5:
                five_ones += 1
        else:
            ones = 0
    return five_ones

def main(p1):
    seq = generate(p1)
    return count(seq)

print(main(2/3))

# ## Exercise 8: Advanced
# This exercise was about probabilities, including Nordic fishermen and lottery.


countries = ['Denmark', 'Finland', 'Iceland', 'Norway', 'Sweden']
populations = [5615000, 5439000, 324000, 5080000, 9609000]
male_fishers = [1822, 2575, 3400, 11291, 1731]
female_fishers = [69, 77, 400, 320, 26] 

def guess(winner_gender):
    if winner_gender == 'female':
        fishers = female_fishers
    else:
        fishers = male_fishers

    guess_country = None
    biggest = 0.0
    for country in range(0, len(countries)):
        probability = fishers[country] / sum(fishers) * 100
        if probability < biggest:
            continue
        biggest = probability
        guess_country = countries[country]
    return (guess_country, biggest)  

def main():
    country, fraction = guess("male")
    print("if the winner is male, my guess is he's from %s; probability %.2f%%" % (country, fraction))
    country, fraction = guess("female")
    print("if the winner is female, my guess is she's from %s; probability %.2f%%" % (country, fraction))

main()

# ## Exercise 9: Advanced
# This exercise was social media blocking exercise based on the Bayes rule.


def bot8(pbot, p8_bot, p8_human):
    pbot_8 = p8_bot * pbot / (p8_bot * pbot + p8_human * (1 - pbot))
    print(pbot_8)

# you can change these values to test your program with different values
pbot = 0.1
p8_bot = 0.8
p8_human = 0.05

bot8(pbot, p8_bot, p8_human)

# ## Exercise 10: Advanced
# Exercise 10 was all about the naive Bayes.
# 
# ### TODO
# Add a visualization for the dice throws


import numpy as np

p1 = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]   # normal
p2 = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]   # loaded

def roll(loaded):
    if loaded:
        print("rolling a loaded die")
        p = p2
    else:
        print("rolling a normal die")
        p = p1

    # roll the dice 10 times
    # add 1 to get dice rolls from 1 to 6 instead of 0 to 5
    sequence = np.random.choice(6, size=10, p=p) + 1 
    for roll in sequence:
        print("rolled %d" % roll)
        
    return sequence

def bayes(sequence):
    odds = 1.0           # start with odds 1:1
    for roll in sequence:
        ratio = p2[roll-1]/p1[roll-1]
        odds *= ratio
    if odds > 1:
        return True
    else:
        return False

sequence = roll(True)
if bayes(sequence):
    print("I think loaded")
else:
    print("I think normal")

# ## Exercise 11: Advanced
# This excercise was about linear regression. Predicting prices for Finnish summer cottages. The usual Nordic things.


# input values for three mökkis: size, size of sauna, distance to water, number of indoor bathrooms, 
# proximity of neighbors
X = [[66, 5, 15, 2, 500], 
     [21, 3, 50, 1, 100], 
     [120, 15, 5, 2, 1200]]
c = [3000, 200, -50, 5000, 100]    # coefficient values

def predict(X, c):
    for cabin in X:
        price = 0
        for index in range(0, len(c)):
            price += c[index] * cabin[index]
        print(price)

predict(X, c)

# ## Exercise 12: Advanced
# In this exercise, a "poor man's" Least squares algorithm was tried out.


import numpy as np

# data
X = np.array([[66, 5, 15, 2, 500], 
              [21, 3, 50, 1, 100], 
              [120, 15, 5, 2, 1200]])
y = np.array([250000, 60000, 525000])

# alternative sets of coefficient values
c = np.array([[3000, 200 , -50, 5000, 100], 
              [2000, -250, -100, 150, 250], 
              [3000, -100, -150, 0, 150]])   

def find_best(X, y, c):
    smallest_error = np.Inf
    best_index = -1
    current_index = 0
    for coeff in c:
        error = 0
        for index in range(0, len(y)):
            coeffed = coeff @ X[index]
            diff = y[index] - coeffed
            error += diff * diff
        if error < smallest_error:
            smallest_error = error
            best_index = current_index
        current_index += 1
    print("the best set is set %d" % best_index)


find_best(X, y, c)

# ## Exercise 13: Advanced
# This exercise added more data to showcase how prediction fails (in the expansion of the previous one the prediction didn't, due to math magic).


import numpy as np
from io import StringIO

input_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

np.set_printoptions(precision=1)    # this just changes the output settings for easier reading
 
def fit_model(input_file):
    df = np.genfromtxt(input_file, skip_header=1)
    x = df[:,0:-1]      # input data to the linear regression
    y = df[:,-1]
    c = np.linalg.lstsq(x, y, rcond=-1)[0]
    print(c)
    print(x @ c)

# simulate reading a file
input_file = StringIO(input_string)
fit_model(input_file)

# ## Exercise 14: Advanced
# This one took a step towards machine learning, splitting training data and test data, while the previous used the same data for both.


import numpy as np
from io import StringIO

train_string = '''
25 2 50 1 500 127900
39 3 10 1 1000 222100
13 2 13 1 1000 143750
82 5 20 2 120 268000
130 6 10 2 600 460700
115 6 10 1 550 407000
'''

test_string = '''
36 3 15 1 850 196000
75 5 18 2 540 290000
'''

def main():
    np.set_printoptions(precision=1)    # this just changes the output settings for easier reading

    # read in the training data and separate it to x_train and y_train
    input_file = StringIO(train_string)
    df = np.genfromtxt(input_file, skip_header=1)
    x_train = df[:,0:-1]
    y_train = df[:,-1]

    # fit a linear regression model to the data and get the coefficients
    c = np.linalg.lstsq(x_train, y_train, rcond=-1)[0]

    # read in the test data and separate x_test from it
    input_file = StringIO(test_string)
    df = np.genfromtxt(input_file, skip_header=1)
    x_test = df[:,0:-1]
    #y_test = df[:,-1]

    # print out the linear regression coefficients
    print(c)

    # this will print out the predicted prics for the two new cabins in the test data set
    print(x_test @ c)


main()

# ## Exercise 15: Advanced
# This exercise moved on to nearest neighbor methods. The first one is calculating vector distances.


import numpy as np

x_train = np.random.rand(10, 3)   # generate 10 random vectors of dimension 3
x_test = np.random.rand(3)        # generate one more random vector of the same dimension

def dist(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)
    
def nearest(x_train, x_test):
    nearest = -1
    min_distance = np.Inf
    # add a loop here that goes through all the vectors in x_train and finds the one that
    # is nearest to x_test. return the index (between 0, ..., len(x_train)-1) of the nearest
    # neighbor
    for index in range(0, len(x_train)):
        distance = dist(x_test, x_train[index])
        if(distance < min_distance):
            min_distance = distance
            nearest = index
    print(nearest)

nearest(x_train, x_test)

# ## Exercise 16: Advanced
# The second part of doing nearest neighbor algorithms. This investigates k number of neighbors for classifications.


import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# create random data with two classes
X, Y = make_blobs(n_samples=16, n_features=2, centers=2, center_box=(-2, 2))

# scale the data so that all values are between 0.0 and 1.0
X = MinMaxScaler().fit_transform(X)

# split two data points from the data as test data and
# use the remaining n-2 points as the training data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=2)

# place-holder for the predicted classes
y_predict = np.empty(len(y_test), dtype=np.int64)

# produce line segments that connect the test data points
# to the nearest neighbors for drawing the chart
lines = []

# distance function
def dist(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)


def main(X_train, X_test, y_train, y_test):

    global y_predict
    global lines

    k = 3    # classify our test items based on the classes of 3 nearest neighbors
    
    # process each of the test data points
    for i, test_item in enumerate(X_test):
        # calculate the distances to all training points
        distances = [dist(train_item, test_item) for train_item in X_train]

        # add your code here
        #nearest = np.argmin(distances)       # this just finds the nearest neighbour (so k=1)
        sdistances = np.argsort(distances)
        # create a line connecting the points for the chart
        # you may change this to do the same for all the k nearest neigbhors if you like
        # but it will not be checked in the tests
        ylist = []
        for index in range(0, 3):
            lines.append(np.stack((test_item, X_train[sdistances[index]])))
            ylist.append(y_train[sdistances[index]])
        y_predict[i] = np.round(np.mean(ylist))
    
    print(y_predict)

main(X_train, X_test, y_train, y_test)

# ## Exercise 17: Advanced
# Moving from numbers to words, this exercise was bag of words mixed with Manhattan distance. In essence, you were given the words already as numbers, so not really working with words. But since computers only know ones and zeros, all is well.


import numpy as np

data = [[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 3, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]]

def distance(row1, row2):
    sum = 0
    for index in range(0, len(row1)):
        sum += abs(row1[index] - row2[index])
    return sum
    
def find_nearest_pair(data):
    N = len(data)
    dist = np.empty((N, N), dtype=np.float)
    outerrowindex = 0
    for row in data:
        innerrowindex = 0
        for secondrow in data:
            if innerrowindex == outerrowindex:
                dist[outerrowindex, innerrowindex] = np.inf
            else:
                dist[outerrowindex, innerrowindex] = distance(row, secondrow)
            innerrowindex += 1
        outerrowindex += 1
            
    print(np.unravel_index(np.argmin(dist), dist.shape))

find_nearest_pair(data)

# ## Excercise 18: Advanced
# In this excercise, the simple bag of words is replaced with the tf-idf algorigthm. We also finally work with words themselves. Obviously, the words get converted to numbers for analyzing, but hey, at least we can say we are working with text (editor's note: Last excercise was the "little piggy went to market" if it wasn't clear from the 0s and 1s)


import math
import numpy as np

text = '''Humpty Dumpty sat on a wall
Humpty Dumpty had a great fall
all the king's horses and all the king's men
couldn't put Humpty together again'''

def distance(a, b):
    sum = 0
    for ai, bi in zip(a, b):
        sum = sum + (ai - bi)**2
    return np.sqrt(sum)

def main(text):
    # tasks your code should perform:

    # 1. split the text into words, and get a list of unique words that appear in it
    # a short one-liner to separate the text into sentences (with words lower-cased to make words equal 
    # despite casing) can be done with 
    # docs = [line.lower().split() for line in text.split('\n')]
    docs = [line.lower().split() for line in text.split('\n')]
    docdictionary = {}
    linedictionaries = []
    # 2. go over each unique word and calculate its term frequency, and its document frequency
    for line in docs:
        linedictionary = {}
        for word in line:
            if word in linedictionary.keys():
                linedictionary[word] += 1
            else:
                linedictionary[word] = 1
            if word in docdictionary.keys():
                docdictionary[word] += 1
            else:
                docdictionary[word] = 1
        linedictionaries.append(linedictionary)
    # 3. after you have your term frequencies and document frequencies, go over each line in the text and 
    # calculate its TF-IDF representation, which will be a vector
    tfidf_vector = []
    lineindex = 0
    for line in docs:
        line_vector = []
        for word in line:
            line_vector.append(linedictionaries[lineindex][word] * math.log(1/docdictionary[word]))
        tfidf_vector.append(line_vector)
        lineindex += 1
    # 4. after you have calculated the TF-IDF representations for each line in the text, you need to
    # calculate the distances between each line to find which are the closest.
    #N = len(tfidf_vector)
    outerrow = []
    outerrowindex = 0
    for row in tfidf_vector:
        innerrowindex = 0
        innerrow = []
        for secondrow in tfidf_vector:
            if innerrowindex == outerrowindex:
                innerrow.append(np.inf)
            else:
                innerrow.append(distance(row, secondrow))
            innerrowindex += 1
        outerrow.append(innerrow)
        outerrowindex += 1
    dist = np.stack(outerrow)        
    print(np.unravel_index(np.argmin(dist), dist.shape))

main(text)

# ## Exercise 19: Advanced
# This exercise investigates combating overfitting.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np

# do not edit this
# create fake data
x, y = make_moons(
    n_samples=500,  # the number of observations
    random_state=42,
    noise=0.3
)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Create a classifier and fit it to our data
k_values = [1, 250, 42, 100]
for value in k_values:
    print("Testing with k:", value)
    knn = KNeighborsClassifier(n_neighbors=value)
    knn.fit(x_train, y_train)
    training_accuracy = knn.score(x_train, y_train)
    testing_accuracy = knn.score(x_test, y_test)
    print("training accuracy: %f" % training_accuracy)
    print("testing accuracy: %f" % testing_accuracy)

# ## Excercise 20: Advanced
# Exercises from here on out were not required for completion certificate, but figured I'll do them anyways. First topic covered logistical regression, an expansion to the linear regression covered earlier.


import math
import numpy as np

x = np.array([4, 3, 0])
c1 = np.array([-.5, .1, .08])
c2 = np.array([-.2, .2, .31])
c3 = np.array([.5, -.1, 2.53])

def linear(x, c):
    sum = 0
    for index in range(0, len(c)):
        sum += c[index] * x[index]
    return sum
    
def sigmoid(z):
    # add your implementation of the sigmoid function here
    sigmoid = 1/(1+math.exp(-z))
    return sigmoid

# calculate the output of the sigmoid for x with all three coefficients
s = sigmoid(linear(x, c1))
print(s)
s = sigmoid(linear(x, c2))
print(s)
s = sigmoid(linear(x, c3))
print(s)

# ## Excercise 21: Advanced
# After going through all of the above, we jump to neural networks. Which in essence are just what we did above, but networked in a certain way. Time to do some passes. Like good old Raimo Helminen.


import numpy as np

w0 = np.array([[ 1.19627687e+01,  2.60163283e-01],
               [ 4.48832507e-01,  4.00666119e-01],
                   [-2.75768443e-01,  3.43724167e-01],
                   [ 2.29138536e+01,  3.91783025e-01],
                   [-1.22397711e-02, -1.03029800e+00]])

w1 = np.array([[11.5631751 , 11.87043684],
                   [-0.85735419,  0.27114237]])

w2 = np.array([[11.04122165],
                   [10.44637262]])

b0 = np.array([-4.21310294, -0.52664488])
b1 = np.array([-4.84067881, -4.53335139])
b2 = np.array([-7.52942418])

x = np.array([[111, 13, 12, 1, 161],
                 [125, 13, 66, 1, 468],
                 [46, 6, 127, 2, 961],
                 [80, 9, 80, 2, 816],
                 [33, 10, 18, 2, 297],
                 [85, 9, 111, 3, 601],
                 [24, 10, 105, 2, 1072],
                 [31, 4, 66, 1, 417],
                 [56, 3, 60, 1, 36],
                 [49, 3, 147, 2, 179]])
y = np.array([335800., 379100., 118950., 247200., 107950., 266550.,  75850.,
                93300., 170650., 149000.])


def hidden_activation(z):
    return np.max(z, 0)

def output_activation(z):
    return z

x_test = [[82, 2, 65, 3, 516]]
for item in x_test:
    h1_in = np.dot(item, w0) + b0 # this calculates the linear combination of inputs and weights
    h1_out = hidden_activation(h1_in) # apply activation function
    h2_in = np.dot(h1_out, w1) + b1
    h2_out = hidden_activation(h2_in)
    out_in = np.dot(h2_out, w2) + b2
    out = output_activation(out_in)
    print(out)

# ## Conclusion
# And that's all she wrote.
# 
# ### TODO
# Revisit the above and add plotting where it seems fun.


