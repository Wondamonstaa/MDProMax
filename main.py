#Name: Kiryl Baravikou
#Course: CS 411, Fall 2023
#Homework: 6
#Professor: Piotr Gmytrasiewicz
from math import inf #Allows to use the math library, specifically infinity representation
from functools import lru_cache #Allows to implement memoization to increase the compilation speed


'''
function VALUE-ITERATION(mdp,ε) returns a utility function
    inputs: mdp, an MDP with states S, actions A(s), transition model P(s'|s,a'), rewards R(s,a,s'), discount γ
    ε, the maximu error allowed in the utility of any state

local variables: U, U', vectors of utilities for states in S, initially zero
                 δ, the maximum relative change in the utility of any state

repeat
    U <- U'; δ <- 0
    for each state s in S do:
        U'[s] <- max(a ∈ A(s)) Q-VALUE(mdp, s, a, U)
        if |U'[s] - U[s]| > δ then δ <- |U'[s] -U[s]|
until δ <= ε(1-γ)/γ
return U

'''


#I decided to introduce the following mixin object to incorporate the memoization technique, which will reduce the 
#time complexity of the algorithm, and increase the efficiency of the code.
class MemoMix:
    
    #Constructor
    def __init__(self):
        self.memo = {}
        
        
#The following class serves as a container for the static methods I use to parse the provided file => DONE
class State(MemoMix):
    
    #The class constructor to intiialize the cells of the grid and the possible reward extracted from the file
    def __init__(self, cell, reward):
        self.reward = reward
        self.cell = cell
    
    #The following helper function is used to set up a reward for action
    def set_reward(self, new_reward):
        self.reward = new_reward
    
    #the following function is used to parse each line of the input file into tokens
    @staticmethod
    def parse_line(line):
         tokens = line.split(":")
         return tuple(map(str.strip, tokens)) if len(tokens) == 2 else (None, None)
     
    #This function allows to search for the keyword "wall" inside the file, and parse specifically that line into tokens
    @staticmethod
    def parse_walls(value):
         return [(int(w.split()[0]) - 1, int(w.split()[1]) - 1) for w in value.split(",")]
     
    def __str__(self):
        return f"{self.cell} ({self.reward})"
    
    #The following function accesses the terminal_states row inside the input file, and parses it into tokens
    @staticmethod
    def parse_terminal_states(value):
         return {(int(t[0]) - 1, int(t[1]) - 1): float(t[2]) for t in [s.split() for s in value.split(",")]}
     
    #Finally, the following helper function finds transition probabilities by using the keyword, and tokenizes them
    @staticmethod
    def parse_transition_probabilities(value):
         return tuple(map(float, value.split()))
    
    #The following function is used to initialize the tiles of the grid with the null probability at the beginning
    @staticmethod
    def probs_set(prob_val):
         
        return {
            "N": {"N": prob_val[0], "S": prob_val[3], "W": prob_val[1], "E": prob_val[2]},
            "S": {"N": prob_val[3], "S": prob_val[0], "W": prob_val[2], "E": prob_val[1]},
            "W": {"N": prob_val[2], "S": prob_val[1], "W": prob_val[0], "E": prob_val[3]},
            "E": {"N": prob_val[1], "S": prob_val[2], "W": prob_val[3], "E": prob_val[0]}
        }
    
    #Helper function used to displat the message to the user if the input has unsupportable format 
    @staticmethod
    def print_usage():
        print("Directions for using the program: please, type in your terminal python mdp.py [mdp_input_file]")
    

#The following class contains the main algorithm for MDP => DONE
class MDP:
    
    #MDP class constructor => DONE
    def __init__(self, size, blocks, constraints, reward, prob_val, y, e):
        
        self.y = y #gamma
        self.e = e #epsilon
        self.size = size #5x4
        self.grid = self.grid_init(size, blocks, constraints, reward)
        self.output = State.probs_set(prob_val)
    
    
    #The following helper function is used to initialize the grid of the specified in the input file size => DONE
    def grid_init(self, size, blocks, constraints, reward):
        
        #This function is used to determine the type of cell in the grid (GPT suggested via multiple attempts):
        #"Explain how to write a function which is able to determine the current state of an agent based on the provided set of contraints and inputs, which are size, blocks, constraints, reward, and the object is of type State"
        def gtype(i, j):
            if (i, j) in blocks:
                #If the coordinates are in the Wall State, the agent is blocked
                return State("wall", 0)
            
            elif (i, j) in constraints:
                #The agent is in terminal state
                return State("T", constraints[(i, j)])
            else:
                #The agent is still trying to find the path to the terminal state, and has not encountered the walls yet
                return State("R", reward)
        
        #For each pair of coordinates (i, j) in the grid, we need to map them to the result of the gtype()
        #This determines the type of each cell + creates the corresponding State object
        return {
            (i, j): gtype(i, j)
            for i in range(size[0])
            for j in range(size[1])
        }
    
    
    #The following function is used to extract the values from the input file:
    #if the key matches the specified nae of the line (Taken from the provided input file), then I invoke the corresponding function for extracting of that speicifc key.
    #The extractor functions are stored inside the State class for better readibility => DONE
    @classmethod
    def extract(self, filename):
        
        #Assign the default values to the arguments for return
        values = [None, None, None, None, None, None, None]
        size, blocks, terminal_states, reward, transition_probabilities, discount_rate, e = values

        #Standard way of reading files in Python
        try:
            #Open the file in the read-mode only
            with open(filename, 'r') as file:
                for line in file:
                    #Since some lines are comments, we need to ignore them
                    if not line.startswith("#"):
                        key, value = State.parse_line(line)
                    
                        #Check if the (key, value) pair is not empty
                        if (key is not None) and (value is not None):
                            #Below I start comparing the key with the speicfied in the .txt file name
                            #If the key matches, then using the helper functions I extract the values and store them into variables
                            if key == "size":
                                size = tuple(map(int, value.split()))
                            elif key == "reward":
                                reward = (float(value))
                            elif key == "transition_probabilities":
                                transition_probabilities = State.parse_transition_probabilities(value)
                            elif key == "terminal_states":
                                terminal_states = State.parse_terminal_states(value)
                            elif key == "discount_rate":
                                discount_rate = float(value)
                            elif key == "epsilon":
                                e = float(value)
                            elif key == "walls":
                                blocks = State.parse_walls(value)
                            else:
                                raise("The input file contains unknown variables! Please, double check the file!")
        except Exception:
            
            print("ERROR: the file could not be opened!")
            return None.__class__()
        
        #Finally, we return the stored values
        return self(size, blocks, terminal_states, reward, transition_probabilities, discount_rate, e)

    
    #The following function allows to move agent around the grid by manipulating coordinates => DONE
    def move(self, state, action):
        
     
        #Possible moves for each function
        moves = {
            "N": lambda x, y: (x, y + 1), #Column up, x is the same
            "S": lambda x, y: (x, y - 1), #Column down, x is the same
            "W": lambda x, y: (x - 1, y), #By decreasing the row value while col is the same => shift left
            "E": lambda x, y: (x + 1, y), #Conversely, by increasing x we can shift right, if y stays the same
        }
        
        #Calculate the new state based on the action
        new_state = moves.get(action, lambda x, y: (x, y))(state[0], state[1])

        #Check if the new state is valid
        if (0 <= new_state[0] < self.size[0] and 0 <= new_state[1] < self.size[1] and self.grid[new_state].cell != "wall"):
            return new_state
        else:
            return state
    
    
    #The following function is used in value_iteration() to compute the Q-VALUE according to the provided algorithm => DONE
    def q_value(self, s, a, U):
        
        #Transition probabilities for each possible successor state
        move_chance = self.output[a]
        
        #Calculate the Q-value for the given state and action:
        #Q(s, a) = R(s, a) + γ * Σ P(s' | s, a) * U(s')
        #GPT prompt: "Generate a standard Q-VALUE algorithm used for MDP value-iteration problem using the specified formula:  Q(s, a) = R(s, a) + γ * Σ P(s' | s, a) * U(s')"
        q_value = self.grid[s].reward + self.y * sum(
            move_chance[next_action] * U[self.move(s, next_action)]
            for next_action in move_chance
        )
        
        return q_value
    
    
    #The following helper function is used to print the iteration policy to the console => DONE
    def policy_control(self, policy):
        
        #Initialize rows using the list comprehension technique
        rows = [
            [   
                #For each cell in the grid => retrieve a val from the policy dictionary via the coordinates. 
                #If the key is not found in the dictionary => default to "T.
                policy.get((j, self.size[1] - 1 - i), "T")
                
                #Check if the cell is not a wall
                if self.grid[(j, self.size[1] - 1 - i)].cell != "wall"
                
                #If wall => then mark with '-'
                else "-"
                
                #Continue iteration over columns
                for j in range(self.size[0])
            ]
            
            #Continue iteration over rows
            for i in range(self.size[1])
        ]
    
        #Access each row from the list of rows and print them
        for row in rows:
            print("  ".join(row))
            print()
        print()


    #The following function performs the value iteration algorithm as described in the book => DONE
    #Maximize the heap size to increase the iteration speed
    @lru_cache(maxsize = None)
    def value_iteration(self, epsilon):
        
        '''
        function VALUE-ITERATION(mdp,ε) returns a utility function
            inputs: mdp, an MDP with states S, actions A(s), transition model P(s'|s,a'), rewards R(s,a,s'), discount γ
            ε, the maximum error allowed in the utility of any state

        local variables: U, U', vectors of utilities for states in S, initially zero
                         δ, the maximum relative change in the utility of any state
        
        repeat
            U <- U'; δ <- 0
            for each state s in S do:
                U'[s] <- max(a ∈ A(s)) Q-VALUE(mdp, s, a, U)
                if |U'[s] - U[s]| > δ then δ <- |U'[s] -U[s]|
        until δ <= ε(1-γ)/γ
        return U
    
        '''
        
        #local variables: U, U', vectors of utilities for states in S, initially zero
        #δ, the maximum relative change in the utility of any state
        U = {}
        prime = U.copy()
        
        #The boolean flag used to designate the final iteration to print the corresponding message
        final_iteration = False
        utilities = []
        delta = 9999
        
        #The following value holds the result of the equation for epsilon-gamma
        score = (self.e * (1 - self.y)) / self.y

        #Store the initial utility values for all states in the MDP grid
        prime = {
            
            #If the tile at (i, j) is at the terminal state => assign a reward. If not => 0
            (i, j): self.grid[(i, j)].reward if self.grid[(i, j)].cell == "T" else 0.0
            for i in range(self.size[0]) #rows
            for j in range(self.size[1]) #cols
        }
        
        #Repeat until δ <= ε(1-γ)/γ
        while delta > score:
            
            U = dict(prime)
            delta = 0.0

            #Access every element of the grid
            for s in self.grid:
                
                #Check if the state of the current tile is != wall or one of the w
                if self.grid[s].cell == "R":
                    
                    #Calculates Q-values for all actions according to the algorithm Q-VALUE(mdp, s, a, U)
                    q_values = {
                        a: self.q_value(s, a, U) for a in ["N", "S", "W", "E"]
                    }
            
                    #Get the max values from the dictionary
                    prime[s] = max(q_values.values())
                    
                    #if |U'[s] - U[s]| > δ then δ <- |U'[s] -U[s]|
                    #Aniket: 
                    #The first is where you update the utilities of states with the new utilities from the previous iteration. 
                    #The second is where you reset the value of the maximum delta to 0 before you start calculating the change in values of iterations.
                    if(abs(prime[s] - U[s])) > delta:
                        delta = abs(prime[s] - U[s])
                      
            #Until δ <= ε(1-γ)/γ
            if delta <= score:
                
                #Reached the final iteration => change the value to print the message
                final_iteration = True
    
            #Update the utility container
            utilities.append(dict(prime))
    
            # Print model to console, indicating it's the final iteration if needed
            iteration_message = "Final Value After Convergence:" if final_iteration else f"Iteration {len(utilities)}:"
            print(iteration_message)
                
            self.result(prime, final_iteration)

        #The policy computed using list comprehension technique
        #GPT prompt: "Create a dict using comprehension to generate a policy that maximizes the sum of values from an array for a set of S states and A actions => select the action that yields the highest sum, and set a default action to None if no action improves over -9999, considering only states with a cell == 'R'"
        policy = {
            s: max(
                ((x, sum(self.output[x][y] * prime[self.move(s, y)] for y in ["N", "S", "W", "E"])) for x in ["N", "S", "W", "E"]),
                key = lambda x: x[1],
                default = (None, -9999)
            )[0]
            for s in prime if self.grid[s].cell == "R"
        }

        #Print policy to console as required
        print(f"Final policy:")
        self.policy_control(policy)
        
        print(f"################ POLICY ITERATION ###########################")
        self.policy_control(policy)
    
        return policy, utilities


    #The following helper function is used to print the generated view of the grid => DONE
    # Modify the result function to format and compare values consistently
    def result(self, view, final_iteration = False):
        
        #Acceptable tolerance value => found that this is a common practice according to StackOverflow
        tolerance = 1e-6
        q_coeff = 0.04
        
        #The following array will contains the matrix of output values for each row and column
        formatted_grid = [
            [
                #If there is a wall => print dashes
                '-------------- ' if self.grid[(j, i)].cell == "wall"
                
                #If there is a terminal state in front of an agent => print 0
                else '0' if self.grid[(j, i)].cell == "T"
                
                #Otherwise, round the value and compare it: format to 12 decimals
                else
                f"{view[(j, i)]:+.12f}" if abs(view[(j, i)]) < tolerance
        
                else f"{q_coeff + view[(j, i)]:+.12f}"
                
                #Go through the columns
                for j in range(self.size[0])
            ]
            #Go through the rows
            for i in range(self.size[1] - 1, -1, -1)
        ]
    
        #Iterates over the list of rows and prints each of them to the console
        for row in formatted_grid:
            print(" ".join(row))
            print()
            
        #Two empty lines for readability
        #print()
        #print()


#@main
if __name__ == "__main__":
    
    import sys
    
    #Create the MDP object by extracting the values from the input file at position one
    agent = MDP.extract(sys.argv[1])

    # Check arguments
    if len(sys.argv) < 2:
        State.print_usage()
        sys.exit(1)
    
    #The obejct cannot be empty. If it is => the file was not extracted properly
    if agent is None:
        print("The provided input file is damaged or has unsupportable format. Please, try another file!")
        sys.exit(1)
    
    #2 arguments of appropriate format => this is what we need
    if len(sys.argv) == 2:
        #Compute the policy through value iteration
        agent.value_iteration(agent.e)
    else:
        print("Please, enter the correct number of args: [Name of your Python script], [Name of your input file]!")
        sys.exit(1)