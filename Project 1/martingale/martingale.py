""""""
"""Assess a betting strategy.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		   	 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		   	 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		   	 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		   	 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 		  		  		    	 		 		   		 		  
or edited.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		   	 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		   	 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
Student Name: Allen Worthley (replace with your name)  		  	   		   	 		  		  		    	 		 		   		 		  
GT User ID: mworthley3 (replace with your User ID)  		  	   		   	 		  		  		    	 		 		   		 		  
GT ID: 903646612 (replace with your GT ID)  		  	   		   	 		  		  		    	 		 		   		 		  
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT username of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: str  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return "mworthley3"  # replace tb34 with your Georgia Tech username.


def gtid():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The GT ID of the student  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: int  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    return 903646612  # replace with your GT ID number


def get_spin_result(win_prob):
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.  		  	   		   	 		  		  		    	 		 		   		 		  
  		  	   		   	 		  		  		    	 		 		   		 		  
    :param win_prob: The probability of winning  		  	   		   	 		  		  		    	 		 		   		 		  
    :type win_prob: float  		  	   		   	 		  		  		    	 		 		   		 		  
    :return: The result of the spin.  		  	   		   	 		  		  		    	 		 		   		 		  
    :rtype: bool  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result


def experiment_1(num_episodes, num_spins, win_prob):

    dim = [num_episodes, num_spins + 1]  # add one for initial spin result of zero

    # initialize result spaces
    episode_winnings = np.zeros(dim)

    ## Run spins and store results per episode ##
    for i in range(num_episodes):
        episode_winnings[i][0] = 0  # redundant given np.zeros setup, but this is a reminder
        spin_counter = 1  # skip initial zero result spin
        while episode_winnings[i][spin_counter - 1] < 80 and spin_counter < dim[1]:
            won = False
            bet_amount = 1
            while not won and spin_counter < dim[1]:
                # bet on black
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings[i][spin_counter] = episode_winnings[i][spin_counter - 1] + bet_amount
                else:
                    episode_winnings[i][spin_counter] = episode_winnings[i][spin_counter - 1] - bet_amount
                    bet_amount = bet_amount * 2
                spin_counter += 1

        while episode_winnings[i][spin_counter - 1] >= 80 and spin_counter < dim[1]:
            episode_winnings[i][spin_counter] = 80
            spin_counter += 1

    ## Calculate Stats for spins ##
    t_winnings = episode_winnings.transpose()  # transpose arrays for spins on columns

    return t_winnings


def experiment_2(num_episodes, num_spins, win_prob):

    dim = [num_episodes, num_spins + 1]  # add one for initial spin result of zero

    # initialize result spaces
    episode_winnings = np.zeros(dim)

    ## Run spins and store results per episode ##
    for i in range(num_episodes):
        episode_winnings[i][0] = 256  # start with 256 in cash, simplifies logic
        spin_counter = 1  # skip initial zero ("256") result spin
        while 0 < episode_winnings[i][spin_counter - 1] < 80 + 256 and spin_counter < dim[1]:
            won = False
            bet_amount = 1
            while not won and spin_counter < dim[1]:
                # bet on black
                won = get_spin_result(win_prob)
                if won:
                    episode_winnings[i][spin_counter] = episode_winnings[i][spin_counter - 1] + bet_amount
                else:
                    episode_winnings[i][spin_counter] = episode_winnings[i][spin_counter - 1] - bet_amount
                    if bet_amount * 2 < episode_winnings[i][spin_counter - 1]:
                        bet_amount = bet_amount * 2
                    else:
                        bet_amount = episode_winnings[i][spin_counter] - 1  # Just bet what is left
                spin_counter += 1

        # carry forward if -256 or 80
        while (episode_winnings[i][spin_counter - 1] >= 80 + 256 or episode_winnings[i][spin_counter - 1] <= 0) and \
                spin_counter < dim[1]:
            if episode_winnings[i - 1][spin_counter] == 80 + 256:
                episode_winnings[i][spin_counter] = 80 + 256
            elif episode_winnings[i - 1][spin_counter] <= 0:
                episode_winnings[i][spin_counter] = 0
            spin_counter += 1

    # Transpose for easier calc's with numpy and matlibplot's plots
    t_winnings = episode_winnings.transpose() - 256  # transpose arrays for spins on columns, subtract bankroll from
    # winnings

    return t_winnings


def test_code():
    """  		  	   		   	 		  		  		    	 		 		   		 		  
    Method to test your code  		  	   		   	 		  		  		    	 		 		   		 		  
    """
    win_prob = 0.473684210526316  # set appropriately to the probability of a win, based off of odds from roulette wiki
    np.random.seed(gtid())  # do this only once  		  	   		   	 		  		  		    	 		 		   		 		  
    print(get_spin_result(win_prob))  # test the roulette spin  		  	   		   	 		  		  		    	 		 		   		 		  
    # add your code here to implement the experiments

    ### -------------------------------------------- ###
    ### --------------- Experiment 1 --------------- ###
    ### -------------------------------------------- ###

    ### --------------- Figure 1 --------------- ###
    # Parameters for this figure
    episodes = 10
    spins = 1000
    winnings = experiment_1(episodes, spins, win_prob)
    # Plot
    plt.figure(1)
    plt.axis([0, 300, -256, 100])
    plt.plot(winnings)
    # note legend goes after plot
    plt.legend(["Episode " + str(a) for a in range(1, episodes + 1)])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("Figure 1")
    plt.show()

    ### --------------- Setup for Figures 2 & 3 --------------- ###
    episodes = 1000
    spins = 1000
    winnings = experiment_1(episodes, spins, win_prob)

    # Create numpy arrays for stats
    avg = np.zeros((spins + 1, 1))
    med = np.zeros((spins + 1, 1))
    std = np.zeros((spins + 1, 1))
    for i in range(1, spins + 1):
        avg[i] = np.average(winnings[i])
        med[i] = np.median(winnings[i])
        std[i] = np.std(winnings[i])

    # Bounds
    avg_upper = avg + std
    avg_lower = avg - std
    med_upper = med + std
    med_lower = med - std

    ### --------------- Figures 2 --------------- ###
    data = np.concatenate((avg_upper, avg, avg_lower), axis=1)

    plt.figure(2)
    plt.axis([0, 300, -256, 100])
    plt.plot(data)
    # note legend goes after plot
    plt.legend(["Upper Bound", "Mean", "Lower Bound"])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("Figure 2")
    plt.show()

    ### --------------- Figures 3 --------------- ###
    data = np.concatenate((med_upper, med, med_lower), axis=1)

    plt.figure(3)
    plt.axis([0, 300, -256, 100])
    plt.plot(data)
    # note legend goes after plot
    plt.legend(["Upper Bound", "Median", "Lower Bound"])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("Figure 3")
    plt.show()

    ### -------------------------------------------- ###
    ### --------------- Experiment 2 --------------- ###
    ### -------------------------------------------- ###

    ### --------------- Setup for Figures 4 & 5 --------------- ###
    episodes = 1000
    spins = 1000
    winnings = experiment_2(episodes, spins, win_prob)

    # Create numpy arrays for stats
    avg = np.zeros((spins + 1, 1))
    med = np.zeros((spins + 1, 1))
    std = np.zeros((spins + 1, 1))
    for i in range(1, spins + 1):
        avg[i] = np.average(winnings[i])
        med[i] = np.median(winnings[i])
        std[i] = np.std(winnings[i])

    # Bounds
    avg_upper = avg + std
    avg_lower = avg - std
    med_upper = med + std
    med_lower = med - std

    ### --------------- Figure 4 --------------- ###
    data = np.concatenate((avg_upper, avg, avg_lower), axis=1)

    plt.figure(3)
    plt.axis([0, 300, -256, 100])
    plt.plot(data)
    # note legend goes after plot
    plt.legend(["Upper Bound", "Mean", "Lower Bound"])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("Figure 4")
    plt.show()

    ### --------------- Figures 5 --------------- ###
    data = np.concatenate((med_upper, med, med_lower), axis=1)

    plt.figure(5)
    plt.axis([0, 300, -256, 100])
    plt.plot(data)
    # note legend goes after plot
    plt.legend(["Upper Bound", "Median", "Lower Bound"])
    plt.xlabel("Spins")
    plt.ylabel("Winnings")
    plt.title("Figure 5")
    plt.show()




