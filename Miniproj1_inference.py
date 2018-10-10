#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
#import collections
import sys

import graphics
import numpy as np
import robot
from collections import defaultdict


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_hidden_states_vec = np.array(all_possible_hidden_states)
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

##Get prior distribution as a numpy array across all possible hidden states
prior_distribution_dict = {x:(1/96 if x[2]=='stay' else 0) for x in all_possible_hidden_states }
prior_distribution2 = np.array(list(prior_distribution_dict.values())).reshape(1,440)
##Index to observation
index_to_location = dict(enumerate(all_possible_observed_states))
location_to_index  = {y:x for(x,y) in index_to_location.items()}

##Index to  hidden state
index_to_state = dict(enumerate(all_possible_hidden_states))
state_to_index  = {y:x for(x,y) in index_to_state.items()}

##Create a transition matrix
tm_dim = len(all_possible_hidden_states)
transition_matrix = np.zeros(shape = (tm_dim,tm_dim))
for i in range(tm_dim):
    for j in range(tm_dim):
        transition_matrix[i,j] = transition_model(all_possible_hidden_states[i])[all_possible_hidden_states[j]]
    
##Create an observation matrix
om_height =  len(all_possible_hidden_states) #hidden states along height
om_width = len(all_possible_observed_states) # observed states along width
observation_matrix = np.zeros(shape = (om_height,om_width))
    
for i in range(om_height):
    location = observation_model(all_possible_hidden_states[i]).keys()
    prob = list(observation_model(all_possible_hidden_states[i]).values())
    col_index = [location_to_index.get(x) for x in location]
    observation_matrix[i,col_index] = prob


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)

careful_log_vec = np.vectorize(careful_log)
      
def get_non_zero_items(X):
    return({x:y for (x,y) in X.items() if y !=0})
    
def vec_to_dict_non_inf(X):
    '''Take a vector of hideen state values and return a dictionary of non zero non inf values'''
    X_dict = defaultdict(float)
    X_dict = {index_to_state[x]:y for (x,y) in list(enumerate(X.ravel())) if y != np.inf}
    return(X_dict)
    
def vec_to_dict_non_zero(X):
    '''Take a vector of hideen state values and return a dictionary of non zero non inf values'''
    X_dict = defaultdict(float)
    X_dict = {index_to_state[x]:y for (x,y) in list(enumerate(X.ravel())) if y != 0}
    return(X_dict)
# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE

    
    

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = np.reshape([prior_distribution2],(1,440))
    # TODO: Compute the forward messages
    for i in range(1,num_time_steps):
        if(observations[i-1]==None):
            obs = np.ones((1,440))
        else:
            obs_index = location_to_index[observations[i-1]]
            obs = np.reshape(observation_matrix[:,obs_index],(1,440))
        
        forward_message = (forward_messages[i-1]*obs)@transition_matrix
        forward_messages[i] = forward_message / np.sum(forward_message)

    backward_messages = [None] * num_time_steps
    backward_messages[-1] = np.ones((1,440))/440
    reverse_transition_matrix = transition_matrix.T
    
    for i in range(num_time_steps-2,-1,-1):
        if(observations[i+1]==None):
            obs = np.ones((1,440))
        else:
            obs_index = location_to_index[observations[i+1]]
            obs = np.reshape(observation_matrix[:,obs_index],(1,440))
        
        backward_message = (backward_messages[i+1]*obs)@reverse_transition_matrix 
        backward_messages[i] = backward_message/np.sum(backward_message)
    
    # TODO: Compute the backward messages

    marginals =  [robot.Distribution() for i in range(num_time_steps)]

    # TODO: Compute the marginals 
    for i in range(num_time_steps):
        if(observations[i]==None):
            obs = np.ones((440))
        else:
            obs_index = location_to_index[observations[i]]
            obs = np.ndarray.flatten(observation_matrix[:,obs_index])
        
        forward_message_i = np.ndarray.flatten(forward_messages[i])
        backward_message_i = np.ndarray.flatten(backward_messages[i])

        
        marginal_prob = np.exp(careful_log_vec(np.exp(careful_log_vec(forward_message_i)))+
                               careful_log_vec(np.exp(careful_log_vec(backward_message_i))) + 
                               careful_log_vec(np.exp(careful_log_vec(obs))))
        marginals[i].update(dict(zip(all_possible_hidden_states,marginal_prob)))
        marginals[i].renormalize()
        
    
    return marginals




def Viterbi(observations):
    
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_hidden_states = len(all_possible_hidden_states)
    num_time_steps = len(observations)
    min_sum_messages = [None] * (num_time_steps)
    trace_back_messages = [None] * (num_time_steps-1)
    neg_log_transition_matrix = -careful_log_vec(transition_matrix.T)
    
    min_sum_messages[0] = -careful_log_vec(np.array([1]*num_hidden_states)).reshape(1,num_hidden_states) 
    for i in range(1,num_time_steps):
        #Message will be a matrix with same dimension as no of hidden states
        #Need to multiply first observation with prior
        if(observations[i-1])==None:
            log_obs = np.ones((1,num_hidden_states))
        else:
            obs_index = location_to_index[observations[i-1]]
            log_obs = -careful_log_vec(np.reshape(observation_matrix[:,obs_index],(1,num_hidden_states)))
        if(i==1):
            log_obs = log_obs -  careful_log_vec(prior_distribution2)
            
        message = log_obs + neg_log_transition_matrix + min_sum_messages[i-1]
        min_sum_messages[i] = np.min(message,axis=1).reshape(1,440)
        trace_back_messages[i-1] = np.argmin(message,axis=1).reshape(1,440)
     
        
    ##Tracing back to time 0
    estimated_hidden_states = [None] * num_time_steps # remove this
    final_obs_index = location_to_index[observations[-1]]
    final_obs = np.reshape(observation_matrix[:,final_obs_index],(1,num_hidden_states))
    final_hidden_state = np.argmin(np.ravel(-careful_log_vec(final_obs) + min_sum_messages[-1]))
    estimated_hidden_states[-1] = all_possible_hidden_states[final_hidden_state]
    

    for i in range(num_time_steps-1,0,-1):
        next_index = state_to_index[estimated_hidden_states[i]]
        current_index = np.ravel(trace_back_messages[i-1])[next_index]
        estimated_hidden_states[i-1]= index_to_state[current_index]
        
    
    

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    
    estimated_hidden_states = None

    return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 99
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
