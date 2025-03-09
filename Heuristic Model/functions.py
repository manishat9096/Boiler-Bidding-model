import numpy as np
import matplotlib.pyplot as plt

def ideal_battery_model(total_simulation_time, schedule):
    delta_t_minutes = 15         # Time step in minutes
    delta_t_hours = delta_t_minutes / 60  # Convert time step to hours
    num_blocks = int(total_simulation_time * 60 / delta_t_minutes)  # Number of blocks

    # Time array
    time_hours = np.arange(0, total_simulation_time, delta_t_hours)
    
    # Initialize action array
    action_array = np.zeros(num_blocks)

    # Function to set actions based on merged schedule
    def set_actions_from_schedule(action_array, schedule, delta_t_hours, num_blocks):
        for period in schedule:
            start_hour = period[0]
            duration_hours = period[1]
            action = period[2]  # 1 for charging, -1 for discharging
            start_idx = int(start_hour / delta_t_hours)
            end_idx = int((start_hour + duration_hours) / delta_t_hours)
            # Ensure indices are within bounds
            start_idx = max(start_idx, 0)
            end_idx = min(end_idx, num_blocks)
            # Set action, handling overlaps
            for idx in range(start_idx, end_idx):
                if action_array[idx] == 0:
                    action_array[idx] = action
                else:
                    # Conflict detected
                    if action_array[idx] != action:
                        # Prioritize discharging over charging
                        action_array[idx] = -1
            # Set action
            
            # action_array[start_idx:end_idx] += action
        return action_array

    # Set actions from schedule
    action_array = set_actions_from_schedule(action_array, schedule, delta_t_hours, num_blocks)

    # Ensure action values are within [-1, 1]
    action_array = np.clip(action_array, -1, 1)

    # Initialize SoC array
    soc = np.zeros(num_blocks)
    soc[0] = 0  # Initial SoC at 0%

    # SoC change per action (percentage per block)
    # Adjust this value based on charging/discharging rates
    soc_change_per_action = 10/ (num_blocks/total_simulation_time)  # 10% SoC change per hour

    # Update SoC over time
    for t in range(1, num_blocks):
        soc[t] = soc[t-1] + action_array[t-1] * soc_change_per_action
        # Ensure SoC stays within 0% to 100%
        soc[t] = np.clip(soc[t], 0, 100)

    return soc, time_hours, action_array



def bid_probability(soc, bid_direction):
    if soc <= 0.1:
        prob = 0.0
    elif soc >= 0.9:
        prob = 1.0
    else:
        prob = (soc - 0.1) / 0.8  # Linearly increase from 0 to 1 between 10% and 90% SoC
    if prob >= 0.5:
        prob = np.round(prob,1)
    else:
        prob = np.fix(prob * 10) /10
    if bid_direction == 'UP':
        return prob
    elif bid_direction == 'DOWN':
        return np.round(1 - prob, 1)

def survival_probability(p_s, mu, s):
    # determine the Bid price from the probability
    # if np.any(p_s <= 0.1):
    #     return np.where(p_s == 0, 15000, mu + s * np.log((1 - p_s) / p_s))
    if p_s <= 0.1:
        return 15000
    else:
        return mu + s * np.log((1 - p_s)/ p_s)

def update_soc(current_soc, Emax, bid_quantity, alpha):
    """
    Here assumed that change of soc is instantaneous and irrespective of MW input signal of the Boiler
    input - activation rate (0 to 1), current soc, bid quantity - -ve if UPBid, +ve if DownBid
    """
    new_soc = (current_soc * Emax + (bid_quantity * alpha * 0.25))/Emax
    new_soc = max(new_soc, 0)
    new_soc = min(new_soc, 1)
    return new_soc

def bid_size(soc, Emax, direction):
    # bid all power available based on current soc and direction
    if direction == 'UP':
        soc = max(soc - 0.05*soc, 0)
        bid_size = Emax * soc #assuming 5% error in soc calculation
    if direction == 'DOWN':
        soc = min(soc + 0.05 *soc, 1)
        bid_size = Emax * (1 - soc)
    return bid_size


def bid_price(p_up, p_down, spotprice, t):
    mu_est_up = 68.3744023681381
    mu_est_down = 40.589105032292615
    s_est_up = 26.23855531775929
    s_est_down = 32.67458258396837
    # if p_up < 0.75:
    #     bid_up = survival_probability(p_up, mu_est_up, s_est_up)
    # else:
    #     bid_up = max(spotprice[t], 0, survival_probability(p_up, mu_est_up, s_est_up))
    if spotprice[t] > 0:
        bid_up = max(spotprice[t], survival_probability(p_up, mu_est_up, s_est_up))
    else:
        bid_up = survival_probability(p_up, mu_est_up, s_est_up)
    if p_down < 0.75:
        bid_down = -survival_probability(p_down, mu_est_down, s_est_down)
    else:
        bid_down = min(spotprice[t], 0, -survival_probability(p_down, mu_est_down, s_est_down))
    return bid_up, bid_down

import numpy as np

def activation_estimate():
    """
    Returns an activation fraction (alpha) that can be zero at times and
    otherwise is sampled from a Beta distribution.

    - With probability p_zero, returns 0.0 (no activation).
    - Otherwise, draws a value from a Beta distribution with parameters (a, b).
      You can adjust these parameters to shape the distribution as desired.
    """
    p_zero = 0.4  # 20% chance of zero activation
    a, b = 2, 2   # Beta distribution parameters (adjust as needed)

    # Decide if we return zero or sample from the Beta distribution
    if np.random.rand() < p_zero:
        alpha = 0.0
    else:
        alpha = np.random.beta(a, b)
    return alpha

def soc_on_schedule(Pch, Pdis, soc, storage, production, Emax, Pmax, wch = 1.2, wdis = 1):
    # this function is called when no bids are cleared. The model should converge towards the original schedule
    P_norm = production / Pmax
    if storage >= 0: # charge
        Pch_adjusted = Pch * P_norm * wch
        soc += (Pch_adjusted * 0.25)/Emax
    elif storage <= 0: # discharge
        Pdis_adjusted = Pdis * (1 - P_norm) / wdis
        soc += (Pdis_adjusted * 0.25)/Emax
    else: #idle
        soc = soc
    soc = max(0, min(1, soc))
    return soc

if __name__ == '__main__':

    total_simulation_time = 24  # Total simulation time in hours

    # Merged schedule: array of [start_hour, duration_hours, action]
    # action: 1 for charging, -1 for discharging
    # schedule = np.array([
    #     [0, 3, 1],
    #     [4, 2, -1],
    #     [6, 5, 1],
    #     [12, 5, -1],
    #     [18, 5, 1]
    # ])

    # schedule = np.array([[0, 15, 1], [14, 8, -1]])
    schedule = np.array([[0, 10, 1], [10, 20, -1], [20, 23, 1]])

    soc, time_hours, action_array = buffer(total_simulation_time, schedule)

    # Calculate bid probabilities over time
    probabilities = np.array([bid_probability(s, 'up') for s in soc])

    # Plot the results
    plt.figure(figsize=(14, 8))

    # SoC over time
    plt.subplot(3, 1, 1)
    plt.plot(time_hours, soc, label='SoC (%)', color='blue')
    plt.xlabel('Time (hours)')
    # plt.xticks(time_hours)
    plt.ylabel('State of Charge (%)')
    plt.title('Battery SoC Over Time')
    plt.grid(True)
    plt.legend()

    # Action over time
    plt.subplot(3, 1, 2)
    plt.step(time_hours, action_array, label='Action (1=Charge, -1=Discharge)', where='post', color='purple')
    plt.xlabel('Time (hours)')
    plt.ylabel('Action')
    plt.title('Charging and Discharging Actions Over Time')
    plt.grid(True)
    plt.legend()

    # Bid Probability over time
    plt.subplot(3, 1, 3)
    plt.plot(time_hours, probabilities, label='Bid Probability', color='orange')
    plt.xlabel('Time (hours)')
    plt.ylabel('Probability of Winning Bid')
    plt.title('Bid Probability Over Time')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    """
    mu_est_up = 68.3744023681381
    mu_est_down = 40.589105032292615
    s_est_up = 26.23855531775929
    s_est_down = 32.67458258396837
    """