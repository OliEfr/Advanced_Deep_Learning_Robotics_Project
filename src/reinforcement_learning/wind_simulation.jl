using Rotations;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;

using Plots;
using Statistics;


"""
For now a 1D wind simulation is implemented.
This function will return a Array of 3D Vectors for each simulation step.


N - number of samples
mean_wind - mean speed
std_wind - std to sample the wind speed difference from
attraction_to_mean - adjust how much the wind speed is attracted to the mean value, i.e. the amount and strenght of gusts
max_wind_acceleration - max wind acceleration within one timestep
max_gust_difference_to_mean - max absolut difference between max gust velocity and mean velocity
only_positive_gusts - limits gusts to only positive values, i.e. only gusts making the wind stronger than the mean occure
"""



function simulate_wind(;use_wind = false, N::Int = 1200, mean_wind_min=0.0, mean_wind_max=2.0, std_wind=0.03, std_wind_direction = 0.04, attraction_to_mean = 0.005, max_wind_acceleration=0.1, max_gust_difference_to_mean_factor = 0.4, only_positive_gusts = false)

    mean_wind = rand(Uniform(mean_wind_min, mean_wind_max),1)[1]
    max_gust_difference_to_mean = mean_wind * max_gust_difference_to_mean_factor
    
    # return vector with zero wind speed
    if use_wind == false
        return fill(fill(0.0, 3), N+1)
    end

    # generate random unit vector for wind direction
    direction = rand(Normal(), 3)
    direction = direction / norm(direction)

    # array to store wind speeds in
    wind_array = Vector{Float64}[]

    # first wind gust has mean value
    push!(wind_array, mean_wind * direction)

    # generate array of wind speeds
    for t in 1:(N+1)

        # get new_mean for the distribution to sample from
        new_mean = (- (norm(wind_array[end]) - mean_wind)) * attraction_to_mean
        new_std_wind = std_wind
        
        # stay longer in gust region
        if norm(wind_array[end]) < (mean_wind + 0.2) #&& only_positive_gusts
            new_std_wind = std_wind * 0.7
        end
        
        # generate distribution and get one sample
        distribution = truncated(Normal(new_mean, new_std_wind), -max_wind_acceleration, max_wind_acceleration) #limit max wind acceleration
        wind_diff = rand(distribution, 1) 
        
        # calculate new wind speed
        low = mean_wind - max_gust_difference_to_mean #upper bound
        high = mean_wind + max_gust_difference_to_mean #lower bound
        new_wind_speed = clamp(norm(wind_array[end]) + wind_diff[1], low , high) #get new wind speed

        # get new slightly disturbed direction
        direction_dist = direction + rand(Normal(0.0, std_wind_direction) , 3)
        direction_dist = direction_dist / norm(direction_dist)

        # push new wind speed to array
        push!(wind_array, direction_dist * new_wind_speed)
    end
    return wind_array
end
