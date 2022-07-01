from webbrowser import get
import numpy as np
import math as math

####################################################
# GOAL: AUTO NAVIGATION FOR DOORDASH
# Learned from v1: Q-learning algorithms can't backtrack within the same path, by definition because the Probability of any action + Reward of any 
# state/action combo converges to a constant with respect to time/step#.  

# Therefore, this needs to be a 2-step algorithm -- 
# first training & mapping the path to the restaurant, and 
# secondly to the home.
####################################################

##########################
# Create the Environment
##########################

#defining the shape of the environment [STATES]
# 10 x 10 grid: 100 possible positions.
city_rows = 10
city_columns = 10

# 3D numpy array for Q-values
q_values = np.zeros((city_rows, city_columns, 4))
# first 2 dimensions: STATES in the coordinate grid
# 3rd dimension: ACTIONS: up, left, down, right.
# the data held in the array are REWARDS (Q-values), organized by STATE & 

# numeric action codes:
# 0 = up, 1 = right, 2 = down, 3 = left
actions = ['up', 'right', 'down', 'left']

#presets all positions to be -100 
rewards = np.full((city_rows, city_columns), -100.)

#define road locations 
roads = {} #store locations in a dictionary
roads[0] = [i for i in range(3,9)]     #includes 3-9
roads[1] = [0,1,2,3,5,7,8]   # includes 1-9
roads[2] = [1,3,5,7,8,9]
roads[3] = [1,3,5,7,9]
roads[4] = [1,3,4,5,6,7,9]
roads[5] = roads[3]      # includes 0-10
roads[6] = roads[3]
roads[7] = [0,1,2,3,5,7,9]
roads[8] = [i for i in range(10)]
roads[9] = [0,2,3,5,7,8,9]

# At points where aisles are, reward = -1
# negative incentive for taking long trips to the packaging zone
for row_index in range(0, 10):
  for column_index in roads[row_index]:        #aka if the aisle exists at that column index, give it -1.
    rewards[row_index, column_index] = -1.      #else leave it at -100.

##########################
# DETERMINING THE DESTINATIONS
##########################

print("Please select your home")
home = input()
print("Please select your desired restaurant")
restaurant = input()



restaurant_bonus = 100
# Establish the desired restaurant
if restaurant == "R1":
    rewards[0,0] = restaurant_bonus
    restaurant_coords = [0,0]
elif restaurant == "R2":
    rewards[4,2] = restaurant_bonus
    restaurant_coords = [4,2]
elif restaurant == "R3":
    rewards[6,2] = restaurant_bonus
    restaurant_coords = [6,2]
elif restaurant == "R4":
    rewards[2,4] = restaurant_bonus
    restaurant_coords = [2,4]
elif restaurant == "R5":
    rewards[7,4] = restaurant_bonus
    restaurant_coords = [7,4]
elif restaurant == "R6":
    rewards[1,6] = restaurant_bonus
    restaurant_coords = [1,6]
elif restaurant == "R7":
    rewards[6,5] = restaurant_bonus
    restaurant_coords = [6,5]

print("Your restaurant is located at {}".format(restaurant_coords))

# HOME
if home == "H1":
    home_coords = [9,1]
elif home == "H2":
    home_coords = [2,2]
elif home == "H3":
    home_coords = [3,6]
elif home == "H4":
    home_coords = [9,6]
elif home == "H5":
    home_coords = [1,9]

print("Your home is located at {}".format(home_coords))

#verify it's correct
for row in rewards:
    print(row)


##################################
# Training the Model
##################################
#Helper Functions

#Terminal State?
def is_terminal_state(current_row_index, current_column_index):
    if (rewards[current_row_index, current_column_index] == -100):
        return True
    elif (rewards[current_row_index, current_column_index] == 100):
        return True
    else:
        return False

#Generate Viable Starting Point
def get_starting_location():
    #get a random row and column index
    current_row_index = int(np.random.randint(city_rows))
    current_column_index = int(np.random.randint(city_columns))

    #######DEBUG   
   # print(type(current_row_index))

    #if it's terminal, re-pick until it's not terminal.
    while is_terminal_state(current_row_index, current_column_index):
        current_row_index = int(np.random.randint(city_rows))
        current_column_index = int(np.random.randint(city_columns))
    return current_row_index, current_column_index

#Choose Next Move
def get_next_action(current_row_index, current_column_index, epsilon): #epsilon = ratio of promising inference vs random exploration

    if np.random.random() < epsilon:      #Generates random 0-1.  If < epsilon, choosies most promising Q-value
        return np.argmax(q_values[current_row_index, current_column_index])
    else: #if > epsilon, choose a random action
        return np.random.randint(4)             ###################Perhaps needs to be randint(0,5)

#Find Next Location based on Action
def get_next_location(current_row_index, current_column_index, action_index):
    new_row_index = current_row_index
    new_column_index = current_column_index
    if actions[action_index] == 'up' and current_row_index > 0:
        new_row_index -= 1
    elif actions[action_index] == 'right' and current_column_index < city_columns - 1:
        new_column_index += 1
    elif actions[action_index] == 'down' and current_row_index < city_rows - 1:
        new_row_index += 1
    elif actions[action_index] == 'left' and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

    #Define a function that will get the shortest path between any location within the warehouse that 
    #the robot is allowed to travel and the item packaging location.
def get_shortest_path(start_row_index, start_column_index):
  
    #Avoid generating path from a terminal state
    if is_terminal_state(start_row_index, start_column_index):
        return []
    
    else: 
        current_row_index, current_column_index = start_row_index, start_column_index
        shortest_path = []
        shortest_path.append([current_row_index, current_column_index])
        
        #continue moving along the path until we reach the goal 
        while not is_terminal_state(current_row_index, current_column_index):
            #get the best action to take
            action_index = get_next_action(current_row_index, current_column_index, 1.)
            #move to the next location on the path, and add the new location to the list
            current_row_index, current_column_index = get_next_location(current_row_index, current_column_index, action_index)
            shortest_path.append([current_row_index, current_column_index])
        return shortest_path # "trail of breadcrumbs" historical path


#define training parameters
epsilon = 0.9 #the percentage of time when we should take the best action (instead of a random action)
discount_factor = 0.9 #discount factor for future rewards
learning_rate = 0.9 #the rate at which the AI agent should learn

def training():
    #1000 training episodes
    for episode in range(1000):

        #get the starting location for this episode
        row_index, column_index = get_starting_location()

        #Act until Terminal
        while not is_terminal_state(row_index, column_index):
            
            #Choose Next Move
            action_index = get_next_action(row_index, column_index, epsilon)

            #Update locations
            old_row_index, old_column_index = row_index, column_index #Store old location
            row_index, column_index = get_next_location(row_index, column_index, action_index) #Import new location
            
            #Rewards & Temporal Difference
            reward = rewards[row_index, column_index]       #Calculate reward
            old_q_value = q_values[old_row_index, old_column_index, action_index]       #Record old Q
            temporal_difference = reward + (discount_factor * np.max(q_values[row_index, column_index])) - old_q_value

            #update the Q-value for the previous state and action pair
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
    return

training()
####DEBUG
print("Post-Training ONE")
print(rewards)
print("Q-vals")
print(q_values)

print('Training complete!')

print("Where is the Doordasher starting? X-coordinate:")
start_x = int(input())
print("Y-coordinate:")
start_y = int(input())

path_to_restaurant = get_shortest_path(start_x, start_y)
print(path_to_restaurant)

##############################
# Restuarant --> home algorithm
##############################

# Reset Q values
q_values = np.zeros((city_rows, city_columns, 4))

### Clear Restaurant Goal
rewards[restaurant_coords[0], restaurant_coords[1]] = -1

### Set Home as goal
rewards[home_coords[0], home_coords[1]] = 100


####DEBUG
print("Pre-Training TWO")
print(rewards)
print("Q-vals")
print(q_values)

training()
print("second training complete!")

####DEBUG
print("Post-Training TWO")
print(rewards)
print("Q-vals")
print(q_values)

print(restaurant_coords)

path_to_home = get_shortest_path(restaurant_coords[0], restaurant_coords[1])

print("Path to Restaurant: {}".format(path_to_restaurant))
print()
print("Path to Home: {}".format(path_to_home))

complete_path = path_to_restaurant + path_to_home

print()
print("Complete Path: {}".format(complete_path))

##################
# Outstanding Questions
# 1. How do I pass this output data to Matlab?