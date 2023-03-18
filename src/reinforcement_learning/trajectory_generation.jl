using Rotations;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;



"""
generate_1D_trajectory(...)
Trajectory generation (only velocity)

generation_mode - Trajectory generation mode: zero, constant, step, linear, ramp
N - number of samples
v_max - maximum velocity of trajectory
v_const - constant velocity of trajectory
N_rise - time/sample number for the up-ramp
N_hold - time/sample number for how long the ramp const. value is held
N_fall - time/sample number for the down-ramp
"""

function generate_1D_trajectory(;generation_mode::String = "zero", N::Int = 1200,
                                 v_max::Float64 = 5.0, v_const::Float64 = 1.0,
                                 N_rise::Float64 = 10.0, N_hold::Float64 = 10.0, N_fall::Float64 = 10.0)

    # Array to store velocity setpoints
    trajectory = Float64[]

    # Generate array of velocities
    for k in 0:(N+1)

        if generation_mode == "zero" # Zero velocity
            new_velocity = 0.0
        elseif generation_mode == "constant" # Constant velocity
            new_velocity = v_const
        elseif generation_mode == "step" # Step velocity
            if k <= N_rise + N_hold
                new_velocity = 0.0
            else
                new_velocity = v_const
            end
        elseif generation_mode == "linear" # Linear velocity
            new_velocity = k/N * v_const
        elseif generation_mode == "ramp" # Ramp velocity
            if k <= N_rise
                new_velocity = k/N_rise * v_const
            elseif k > N_rise && k <= N_rise + N_hold
                new_velocity = v_const
            elseif k > N_rise + N_hold && k <= N_rise + N_hold + N_fall
                new_velocity = v_const - (k-(N_hold+N_rise))/(N_fall) * v_const
            elseif k > N_rise + N_hold + N_fall
                new_velocity = 0.0
            end
        end

        # Clamp the velocity to the maximum velocity
        new_velocity = clamp(new_velocity, -v_max, v_max)

        append!(trajectory, new_velocity)

    end
    
    return trajectory

end



"""
generate_3D_trajectory(...)
Trajectory generation (only velocity)

generation_mode - Trajectory generation mode: zero, constant, step, linear, ramp
N - number of samples
v_max - maximum velocity of trajectory
v_const - array of constant velocity of trajectory
N_rise - time/sample number for the up-ramp
N_hold - time/sample number for how long the ramp const. value is held
N_fall - time/sample number for the down-ramp
"""

function generate_3D_trajectory(;generation_mode::String = "zero", N::Int = 1200,
                                 v_max::Float64 = 5.0, v_const::Vector{Float64} = [0.0, 0.0, 1.0],
                                 N_rise::Float64 = 10.0, N_hold::Float64 = 10.0, N_fall::Float64 = 10.0)

    v_const = clamp!(v_const, -v_max, v_max)

    trajectory_1 = generate_1D_trajectory(generation_mode=generation_mode, N=N, v_max=v_max, v_const=v_const[1], N_rise=N_rise, N_hold=N_hold, N_fall=N_fall)
    trajectory_2 = generate_1D_trajectory(generation_mode=generation_mode, N=N, v_max=v_max, v_const=v_const[2], N_rise=N_rise, N_hold=N_hold, N_fall=N_fall)
    trajectory_3 = generate_1D_trajectory(generation_mode=generation_mode, N=N, v_max=v_max, v_const=v_const[3], N_rise=N_rise, N_hold=N_hold, N_fall=N_fall)

    trajectory = Vector{Float64}[]
    for k in 1:length(trajectory_1)
        velocity_vector = [trajectory_1[k]; trajectory_2[k]; trajectory_3[k]]
        push!(trajectory, velocity_vector)
    end

    return trajectory

end



"""
generate_random_3D_trajectory(...)
Random trajectory generation (only velocity)

randomness_mode - Mode for how random the trajectories should be, i.e. how many parameters the 3 dimensions share during generation
N - number of samples
v_max - maximum velocity of trajectory
N_rise_lower_limit - lower limit for time/sample number for the up-ramp
N_rise_upper_limit - upper limit for time/sample number for the up-ramp
N_hold_lower_limit - lower limit for time/sample number for how long the ramp constant value is held
N_hold_upper_limit - upper limit for time/sample number for how long the ramp constant value is held
N_fall_lower_limit - lower limit for time/sample number for the down-ramp
N_fall_upper_limit - upper limit for time/sample number for the down-ramp
"""

function generate_random_3D_trajectory(;randomness_mode::Int = 3, N::Int = 1200, v_max::Float64,
                                     N_rise_lower_limit::Float64, N_rise_upper_limit::Float64,
                                     N_hold_lower_limit::Float64, N_hold_upper_limit::Float64,
                                     N_fall_lower_limit::Float64, N_fall_upper_limit::Float64)

    # Sample random trajectory category
    generation_mode = rand(["zero", "constant", "step", "linear", "ramp"],3)

    # Sample random values
    v_const_distribution = Uniform(-v_max, v_max)
    v_const = rand(v_const_distribution, 3)
    v_const[3] = abs(v_const[3]) # currently only non-negative z velocities are allowed, change later
    N_rise_distribution = Uniform(N_rise_lower_limit, N_rise_upper_limit)
    N_rise = rand(N_rise_distribution, 3)
    N_hold_distribution = Uniform(N_hold_lower_limit, N_hold_upper_limit)
    N_hold = rand(N_hold_distribution, 3)
    N_fall_distribution = Uniform(N_fall_lower_limit, N_fall_upper_limit)
    N_fall = rand(N_fall_distribution, 3)

    # Generate trajectory
    if randomness_mode == 1 # the trajectories share most parameters, except the constant speed
        random_1D_trajectory_1 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[1], N_rise=N_rise[1], N_hold=N_hold[1], N_fall=N_fall[1])
        random_1D_trajectory_2 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[2], N_rise=N_rise[1], N_hold=N_hold[1], N_fall=N_fall[1])
        random_1D_trajectory_3 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[3], N_rise=N_rise[1], N_hold=N_hold[1], N_fall=N_fall[1])
    elseif randomness_mode == 2 # the trajectories share the same generation_mode, the other parameters are different
        random_1D_trajectory_1 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[1], N_rise=N_rise[1], N_hold=N_hold[1], N_fall=N_fall[1])
        random_1D_trajectory_2 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[2], N_rise=N_rise[2], N_hold=N_hold[2], N_fall=N_fall[2])
        random_1D_trajectory_3 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[3], N_rise=N_rise[3], N_hold=N_hold[3], N_fall=N_fall[3])
    elseif randomness_mode == 3 # the trajectories all have different parameters
        random_1D_trajectory_1 = generate_1D_trajectory(generation_mode=generation_mode[1], N=N, v_max=v_max, v_const=v_const[1], N_rise=N_rise[1], N_hold=N_hold[1], N_fall=N_fall[1])
        random_1D_trajectory_2 = generate_1D_trajectory(generation_mode=generation_mode[2], N=N, v_max=v_max, v_const=v_const[2], N_rise=N_rise[2], N_hold=N_hold[2], N_fall=N_fall[2])
        random_1D_trajectory_3 = generate_1D_trajectory(generation_mode=generation_mode[3], N=N, v_max=v_max, v_const=v_const[3], N_rise=N_rise[3], N_hold=N_hold[3], N_fall=N_fall[3])
    end

    random_trajectory = Vector{Float64}[]
    for k in 1:length(random_1D_trajectory_1)
        velocity_vector = [random_1D_trajectory_1[k]; random_1D_trajectory_2[k]; random_1D_trajectory_3[k]]
        push!(random_trajectory, velocity_vector)
    end

    return random_trajectory

end


"""
generate_fixed_wing_trajectory(...)
Create trajectory for fixed wing environment
"""

function generate_fixed_wing_trajectory()
    vel_lin = generate_1D_trajectory(;generation_mode = "linear", v_const = 32.0, v_max = 8.0, N = 800)
    vel_0 = generate_1D_trajectory(;generation_mode = "zero", N = 800)
    trajectory = Vector{Float64}[]
    for k in 1:length(vel_lin)
        vel = [vel_lin[k]; vel_0[k]; vel_0[k]]
        push!(trajectory, vel)
    end
    return trajectory
end
