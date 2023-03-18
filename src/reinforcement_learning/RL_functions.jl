using .Flyonic;
using Rotations;

using ReinforcementLearning;
using StableRNGs;
using Flux;
using Flux.Losses;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;

using Plots;
using Statistics;
using DelimitedFiles;

using Logging;
using TensorBoardLogger;

using Colors: RGBA, RGB

include("trajectory_generation.jl")
include("wind_simulation.jl")
include("custom_functions.jl")

using BSON: @save, @load


# Connect RL-library with simulation via this environment
mutable struct VtolEnv{A,T,ACT,R<:AbstractRNG} <: AbstractEnv # Parametric Constructor for a subtype of AbstractEnv
    # Required
    action_space::A
    state_space::Space{Vector{ClosedInterval{T}}} # technically, this is the observation space (disturbed state space) that goes into policy
    state::Vector{T} # technically, this is the observation (disturbed state) that goes into the policy
    action::ACT # next action executed by the system
    previous_action::ACT # previous action executed by the system
    done::Bool # environment termination boolean e.g. drone crashed
    t::T
    rng::R

    name::String # for multiple environments

    state_history_length::Int
    n_states::Int

    enable_visualization::Bool
    enable_realtime::Bool
    enable_logging::Bool
    enable_resampling::Bool
    enable_random_state_init::Bool
    enable_actuator_dist::Bool
    enable_sensor_dist::Bool
    enable_z_termination::Bool
    
    # Everything you need additionally can also go in here
    # States defined here are only for simulation, not for the policy
    # technically, this is the (undisturbed) state of the system (used in simulation)
    x_W::Vector{T}
    v_B::Vector{T}
    a_B::Vector{T}
    R_W::Matrix{T}
    ω_B::Vector{T}
    α_B::Vector{T}

    # CNN state list
    # Keep separate to normal states
    cnn_states::Vector{Vector{T}}
    
    # Wind and trajectory
    wind_W # should be Vector{Vector{T}}, but saving and loading from .bson destroys type
    v_W_trajectory # should be Vector{Vector{T}}, but saving and loading from .bson destroys type
    step::Int # step counter

    # Logging
    log_thrust_left::Vector{T}
    log_thrust_right::Vector{T}
    log_flaps_right::Vector{T}
    log_flaps_left::Vector{T}
    log_velocity::Vector{Vector{T}} 
    log_target_velocity::Vector{Vector{T}}
    log_ω_B::Vector{Vector{T}}
    log_orientation_diff::Vector{T} # for observing drone to fixed wing mode
    log_orientation_diff_xz::Vector{T} # for observing drone to fixed wing mode
    
    Δt::T
end;


Random.seed!(env::VtolEnv, seed) = Random.seed!(env.rng, seed)
RLBase.action_space(env::VtolEnv) = env.action_space
RLBase.state_space(env::VtolEnv) = env.state_space # technically, this is the observation space (disturbed state space) that goes into the policy
RLBase.is_terminated(env::VtolEnv) = env.done
RLBase.state(env::VtolEnv) = env.state # technically, this is the observation (disturbed state) that goes into policy
RLBase.reward(env::VtolEnv{A,T}) where {A,T} = compute_reward(env)
RLBase.reset!(env::VtolEnv{A,T}) where {A,T} = reset(env)


function compute_reward(env::VtolEnv{A,T}) where {A,T}

    stay_alive = 4.0

    # inverse-apply normalization factor here, because this reward function was designed without standardization of states
    velocity_error_cost = norm([env.cnn_states[7][end], env.cnn_states[8][end], env.cnn_states[9][end]].*VEL_ERROR_MAX_SCALING, 1)
    w_velocity_error = 0.5

    action_cost = norm(env.action, 1)
    w_action = 0.01

    action_diff_cost = norm(env.action - env.previous_action, 1)
    w_action_diff = 0.0

    ω_B_cost = norm(env.ω_B, 1) # dont use env.state here, because env.state is standardized
    w_ω_B = 0.01

    return stay_alive - w_velocity_error * velocity_error_cost - w_action * action_cost - w_action_diff * action_diff_cost - w_ω_B * ω_B_cost
end;


function reset(env::VtolEnv{A,T}) where {A,T}

    # Visualize initial state
    if env.enable_visualization
        set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        set_actuators(env.name, [0.0; 0.0; 0.0; 0.0]);
        transform_arrow(env.name*"/v_B_setpoint", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0])
        transform_arrow(env.name*"/v_B", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0])
        transform_arrow(env.name*"/wind", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0]);
    end

    env.t = 0.0
    env.step = 0
    env.action = [0.0, 0.0, 0.0, 0.0]
    env.previous_action = [0.0, 0.0, 0.0, 0.0]
    env.done = false

    # Initialize drone state randomly
    if env.enable_random_state_init
        env.x_W = [0.0; 0.0; 0.0];
        env.v_B = rand(env.rng, Uniform(v_B_MIN_INIT, v_B_MAX_INIT), 3);
        env.a_B = rand(env.rng, Uniform(a_B_MIN_INIT, a_B_MAX_INIT), 3);
        env.R_W = get_random_rotation_matrix();
        env.ω_B = rand(env.rng, Uniform(ω_B_MIN_INIT, ω_B_MAX_INIT), 3);
        env.α_B = rand(env.rng, Uniform(α_B_MIN_INIT, α_B_MAX_INIT), 3);
    else
        env.x_W = [0.0; 0.0; 0.0];
        env.v_B = [0.0; 0.0; 0.0];
        env.a_B = [0.0; 0.0; 0.0];
        env.R_W = Matrix(UnitQuaternion(RotY(-pi/2.0)*RotX(pi)));
        env.ω_B = [0.0; 0.0; 0.0];
        env.α_B = [0.0; 0.0; 0.0];
    end

    # Resample the trajectoy and wind
    if env.enable_resampling
        env.v_W_trajectory = create_trajectory(USE_RANDOM_TRAJECTORY=true)
        env.wind_W = simulate_wind(use_wind = USE_WIND, N = Int(floor(TOTAL_SIMULATION_TIME/SIMULATION_STEP_TIME)), mean_wind_max = MEAN_WIND_MAX , mean_wind_min = MEAN_WIND_MIN)
    end

    v_B_setpoint = transpose(env.R_W) * env.v_W_trajectory[env.step+1]
    vel_error = (env.v_B - v_B_setpoint)

    # Set initial state, standardized in [-1, 1]
    current_state = [Rotations.params(RotYXZ(env.R_W))./ROT_W_MAX; env.ω_B; vel_error./VEL_ERROR_MAX_SCALING]
    env.cnn_states = [zeros(Float64, env.state_history_length) for i in 1:env.n_states]
    for i in 1:env.n_states
        push!(env.cnn_states[i], current_state[i]) # add new state
        env.cnn_states[i] = env.cnn_states[i][2:end] # remove oldest entry
    end
    env.state = reduce(vcat, env.cnn_states)
    
    # Logging
    env.log_thrust_left = [0]
    env.log_thrust_right = [0]
    env.log_flaps_right = [0]
    env.log_flaps_left = [0]
    env.log_velocity = [[0,0,0]]
    env.log_target_velocity = [[0,0,0]]
    env.log_ω_B = [[0,0,0]]
    env.log_orientation_diff = [0.0]
    env.log_orientation_diff_xz = [0.0]

    nothing
end;


function (env::VtolEnv)(a)
    next_action = [a[1], a[2], a[3], a[4]]
    env.previous_action = env.action
    env.action = next_action
    _step!(env, next_action)
end;


function _step!(env::VtolEnv, next_action)

    # step counter
    env.step += 1

    # Add actuator noise to action
    if env.enable_actuator_dist
        next_action[1] += rand(env.rng, Normal(0.0, STD_ACTUATOR_DIST_THRUST))
        next_action[2] += rand(env.rng, Normal(0.0, STD_ACTUATOR_DIST_THRUST))
        next_action[3] += rand(env.rng, Normal(0.0, STD_ACTUATOR_DIST_FLAPS))
        next_action[4] += rand(env.rng, Normal(0.0, STD_ACTUATOR_DIST_FLAPS))
    end
    
    # Calculate wind impact
    v_in_wind_B = vtol_add_wind(env.v_B, env.R_W, env.wind_W[env.step]);

    # Caluclate aerodynamic forces
    torque_B, force_B = vtol_model(v_in_wind_B, next_action, eth_vtol_param);

    # Integrate rigid body dynamics for Δt
    # W = world KOS, B = body KOS
    env.x_W, env.v_B, env.a_B, env.R_W, env.ω_B, env.α_B, time = rigid_body_simple(torque_B, force_B, env.x_W, env.v_B, env.R_W, env.ω_B, env.t, env.Δt, eth_vtol_param)

    env.t += env.Δt

    # Get all state measurements
    x_W = env.x_W
    v_B = env.v_B
    a_B = env.a_B
    rot_W = Rotations.params(RotYXZ(env.R_W))
    ω_B = env.ω_B
    α_B = env.α_B

    # Add sensor noise to sensor measurements
    if env.enable_sensor_dist
        x_W += rand(env.rng, Normal(0.0, LIN_POS_STD), 3) # position in x,y,z
        v_B += rand(env.rng, Normal(0.0, LIN_VEL_STD), 3) # linear velocity in x,y,z
        a_B += rand(env.rng, Normal(0.0, LIN_ACC_STD), 3) # linear acceleration in x,y,z
        rot_W += rand(env.rng, Normal(0.0, ROT_ANG_STD), 3) # rotation angle around x,y,z
        ω_B += rand(env.rng, Normal(0.0, ROT_VEL_STD), 3) # rotational velocity around x,y,z
        α_B += rand(env.rng, Normal(0.0, ROT_ACC_STD), 3) # rotational acceleration around x,y,z
    end

    # Calculate the linear velocity error
    v_B_setpoint = transpose(env.R_W) * env.v_W_trajectory[env.step]
    vel_error = (v_B - v_B_setpoint)

    # function to determine min and max state values
    # this is only required for system analysis, not for the actual simulation
    unnormed_state = [rot_W; ω_B; vel_error]
    for i in 1:9
        if MinMaxStates[i][1] > unnormed_state[i]
            MinMaxStates[i][1] = unnormed_state[i]
        end
        if MinMaxStates[i][2] < unnormed_state[i]
            MinMaxStates[i][2] = unnormed_state[i]
        end
    end

    # Normalize the network inputs
    # ATTENTION: This defines which states go into the policy network
    current_state = [rot_W./ROT_W_MAX; ω_B./ω_B_MAX; vel_error./VEL_ERROR_MAX_SCALING]

    for i in 1:env.n_states
        push!(env.cnn_states[i], current_state[i]) # add new state
        env.cnn_states[i] = env.cnn_states[i][2:end] # remove oldest entry
    end
    env.state = reduce(vcat, env.cnn_states)

    # Logging
    if env.enable_logging
        push!(env.log_thrust_left, env.action[1])
        push!(env.log_thrust_right, env.action[2])
        push!(env.log_flaps_right, env.action[4])
        push!(env.log_flaps_left, env.action[3])
        push!(env.log_velocity, env.R_W * env.v_B)
        push!(env.log_target_velocity, env.v_W_trajectory[env.step])
        push!(env.log_ω_B, env.ω_B)
        a = env.v_W_trajectory[env.step] # target vector
        b = env.R_W * [1, 0, 0] # current vector
        orientation_diff = acosd(clamp(a⋅b/(norm(a)*norm(b)), -1, 1)) # angle in degree
        push!(env.log_orientation_diff, orientation_diff)
        a_xz = [a[1], a[3]]
        b_xz = [b[1], b[3]]
        orientation_diff_xz = acosd(clamp(a_xz⋅b_xz/(norm(a_xz)*norm(b_xz)), -1, 1)) # angle in degree
        push!(env.log_orientation_diff_xz, orientation_diff_xz)
    end
    
    # Visualize the new state
    if env.enable_visualization
        if env.enable_realtime
            sleep(env.Δt) # just a dirty hack, slower than real time
        end
        set_transform(env.name, env.x_W, QuatRotation(env.R_W));
        set_actuators(env.name, next_action)
        transform_arrow(env.name*"/v_B_setpoint", [0.0; 0.0; 0.0], v_B_setpoint * env.Δt * 10) # factor 10 for better visualization
        transform_arrow(env.name*"/v_B", [0.0; 0.0; 0.0], env.v_B * env.Δt * 10) # factor 10 for better visualization
        wind_B = transpose(env.R_W) * env.wind_W[env.step]
        transform_arrow(env.name*"/wind", [0.0; 0.0; 0.0], wind_B)
    end

    # Termination criteria
    env.done =
        norm(env.v_B, Inf) > v_B_MAX || # stop if drones linear velocity is too high
        norm(env.a_B, Inf) > a_B_MAX || # stop if drones linear acceleration is too high
        norm(env.ω_B, Inf) > ω_B_MAX || # stop if drones rotational velocity is too high
        norm(env.α_B, Inf) > α_B_MAX || # stop if drones rotational acceleration is too high
        env.t > TOTAL_SIMULATION_TIME # stop after maximum simulation time
        # (Optional) Stop if the deviation from the target velocity is too large
    # extra if statement required (!please keep!)
    if env.enable_z_termination
        env.done = env.done || (env.x_W[3] < -1.0)
    end
    nothing
end;


function create_experiment_from_parameters(parameter_dict, experiment_folder)

    experiment_ID = parameter_dict["experiment_ID"]

    ################################################
    #              LEARNING PARAMETERS             #
    ################################################
    global N_ENV = 50; # Number of environments
    global test_data_path = "./test_data/" # path to the test set
    global lr = 4e-4 # learning rate
    global UPDATE_FREQ = 1024


    ################################################
    #           NEURAL NETWORK PARAMETERS          #
    ################################################
    global USE_CNN = parameter_dict["USE_CNN"]
    global STATE_HISTORY_LENGTH = parameter_dict["STATE_HISTORY_LENGTH"] # length of cnn states; equals time history length; SET TO "1" TO GET STANDARD FNN
    global N_NEURONS = parameter_dict["N_NEURONS"] # number of neurons in one hidden layer
    global N_FILTERS_CNN = parameter_dict["N_FILTERS_CNN"] # number of filters used in cnn


    ################################################
    #            TRAJECTORY PARAMETERS             #
    ################################################
    global TRAJECTORY_GENERATION_MODE = "constant"
    global V_CONST = 1.0
    global V_MAX = 5.0
    global N_RISE_LOWER_LIMIT = 10.0 # must be strictly smaller than N_RISE_UPPER_LIMIT
    global N_RISE_UPPER_LIMIT = 200.0 # must be strictly larger than N_RISE_LOWER_LIMIT
    global N_HOLD_LOWER_LIMIT = 10.0 # must be strictly smaller than N_HOLD_UPPER_LIMIT
    global N_HOLD_UPPER_LIMIT = 200.0 # must be strictly larger than N_HOLD_LOWER_LIMIT
    global N_FALL_LOWER_LIMIT = 10.0 # must be strictly smaller than N_FALL_UPPER_LIMIT
    global N_FALL_UPPER_LIMIT = 200.0 # must be strictly larger than N_FALL_LOWER_LIMIT


    ################################################
    #            DISTURBANCE PARAMETERS            #
    ################################################
    # Wind parameters
    # TODO: add wrapping function to make wind more versatile, e.g. multiple MEAN_WINDs
    global USE_WIND = parameter_dict["USE_WIND"] # set true, if drone should get distrubed by wind gusts
    global MEAN_WIND_MIN = 0.0 # mean wind speed | m/s
    global MEAN_WIND_MAX = 5.0 # mean wind speed | m/s

    # Actuator noise parameters
    # ASSUMPTION: Additive zero-mean Gaussian noise
    global USE_ACTUATOR_DIST = parameter_dict["USE_ACTUATOR_DIST"] # set true, if noise should be applied to the actuators
    global STD_ACTUATOR_DIST_FLAPS = 0.04 # range: 0..2
    global STD_ACTUATOR_DIST_THRUST = 0.02 # range: -1..1

    # Sensor noise parameters
    # ASSUMPTION: Additive zero-mean Gaussian noise
    global USE_SENSOR_DIST = parameter_dict["USE_SENSOR_DIST"] # set true, if noise should be applied to the sensor measurements
    global LIN_POS_STD = 0.000005 # UNUSED
    global LIN_VEL_STD = 0.03 # linear velocity noise | m/s
    global LIN_ACC_STD = 0.7 # UNUSED linear acceleration noise | m/s² # Finn provided 1.5 I think, but 0.7 is fine for now
    global ROT_ANG_STD = 0.00002 # rotation angle noise | rad | 0.001 rad = 0.05 deg
    global ROT_VEL_STD = 0.007 # rotational velocity noise | rad/s
    global ROT_ACC_STD = 0.7 # UNUSED rotational acceleration noise | rad/s²


    ################################################
    #            SIMULATION PARAMETERS             #
    ################################################
    global TOTAL_SIMULATION_TIME = 20; # total simulation time | s
    global SIMULATION_STEP_TIME = 0.025; # step size for simulation | s

    # Simulation termination criteria; also used for standardizing states in [-1, 1]
    # Attention: these values are quite sensitive. Do not change untested!
    # Attention: these values highly influence lr and reward
    # INFO: I dont think, that these values are fully optimized yet. e.g. the standardization value for ω_B is too high,
    #       I think it might be best to split termination criteria and standardization values
    global v_B_MAX = 2 * V_MAX # maximum drone linear velocity | m/s
    global a_B_MAX = 1000.0 # maximum drone linear acceleration | m/s²
    global ω_B_MAX = 200.0  # maximum drone rotational velocity | rad/s
    global α_B_MAX = 1000.0 # maximum drone rotational acceleration | rad/s²
    global ROT_W_MAX = pi    # TODO: pi or 2*pi normalization? experimental data suggests to use pi
    global WIND_SPEED_MAX_SCALING = (1 + 0.4) * MEAN_WIND_MAX # depends on implementation in wind_simulation.jl!
    global VEL_ERROR_MAX_SCALING = V_MAX
    global VEL_MAX_SCALING = V_MAX


    ################################################
    #              DRONE PARAMETERS                #
    ################################################
    # Drone initialization parameters
    # TODO: parameters should be equal to termination criteria or e.g. 90% of termination criteria
    global ENABLE_RANDOM_STATE_INITIALIZATION = parameter_dict["ENABLE_RANDOM_STATE_INITIALIZATION"] # set true, if drone should get initialized with random state
    global v_B_MAX_INIT = V_MAX
    global v_B_MIN_INIT = -v_B_MAX_INIT
    global a_B_MAX_INIT = 5.0
    global a_B_MIN_INIT = -a_B_MAX_INIT
    global ω_B_MAX_INIT = 5.0
    global ω_B_MIN_INIT = -ω_B_MAX_INIT
    global α_B_MAX_INIT = 5.0
    global α_B_MIN_INIT = -α_B_MAX_INIT


    ################################################
    #               HOOK PARAMETERS                #
    ################################################
    EVALUATION_FREQ = 10_000
    SAVING_FREQ = 100_000


    # Check if the environment is working
    env = VtolEnv();
    RLBase.test_runnable!(env)


    seed = 123
    rng = StableRNG(seed)

    ################################################
    #            CREATE TRAINING ENVS              #
    ################################################
    # Define multiple environments for parallel training
    env = MultiThreadEnv([
        VtolEnv(; rng = StableRNG(hash(seed+i)), name = "vtol$i") for i in 1:N_ENV
    ])


    ################################################
    #              CREATE TEST SETS                #
    ################################################
    # ATTENTION: make sure to not accidentally overwrite the test data
    N_ENV_TESTSET = 100
    #generate_test_data(test_data_path, N_ENV_TESTSET, TOTAL_SIMULATION_TIME, SIMULATION_STEP_TIME, MEAN_WIND_MIN, MEAN_WIND_MAX)
    test_winds, test_trajectories = load_test_data(test_data_path)


    ################################################
    #              CREATE TEST ENVS                #
    ################################################
    # All disturbances should be set to 'true' for the testing
    test_set_envs = MultiThreadEnv([
        VtolEnv(;rng = StableRNG(hash(seed+i)),
                 name = "vtol$i",
                 wind_W = test_winds[i],
                 v_W_trajectory = test_trajectories[i],
                 enable_resampling = false, # envs should NOT resample a random wind/trajectory on reset
                 enable_logging = true,
                 enable_sensor_dist = true,
                 enable_actuator_dist = true,
                 enable_z_termination = false) for i in 1:N_ENV_TESTSET
    ])


    # Get number of states and actions
    n_states = env[1].n_states
    n_actions = length(action_space(env[1]))

    global cnn_length = Int(env[1].state_history_length)

    if USE_CNN
        ###############################################
        #                     CNN                     #
        ###############################################
        approximator = ActorCritic(
            # n_states - number of states as input
            # n_actions - number of actions as output
            # N_NEURONS - number of hidden neurons in one layer of the CNN
            # 3 layers; last layer splitted in mean and variance; then action is sampled
            actor = GaussianNetwork(
                pre = Chain(
                            cnn_reshape,
                            Conv((cnn_length,), 1 => N_FILTERS_CNN, relu; stride=(cnn_length,), init=init), # (n_states, N_FILTERS_CNN, N_ENV)
                            Flux.flatten, # reshape
                            Dense(n_states * N_FILTERS_CNN, N_NEURONS, relu; initW = glorot_uniform(rng)),
                            Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
                    ),
                μ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
                logσ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
                ),
            critic = Chain(
                        cnn_reshape,
                        Conv((cnn_length,), 1 => N_FILTERS_CNN, relu; stride=(cnn_length,), init=init),
                        Flux.flatten,
                        Dense(n_states * N_FILTERS_CNN, N_NEURONS, relu; initW = glorot_uniform(rng)),
                        Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
                        Dense(N_NEURONS, 1; initW = glorot_uniform(rng)),
            ),
            optimizer = ADAM(lr),
        );
    else
        ###############################################
        #                     FNN                     #
        ###############################################
        approximator = ActorCritic(
            # n_states - number of states as input
            # n_actions - number of actions as output
            # N_NEURONS - number of hidden neurons in one layer of the FNN
            # 3 layers; last layer splitted in mean and variance; then action is sampled
            actor = GaussianNetwork(
                pre = Chain(
                            Dense(n_states, N_NEURONS, relu; initW = glorot_uniform(rng)),
                            Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
                    ),
                μ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
                logσ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
                ),
            critic = Chain(
                        Dense(n_states, N_NEURONS, relu; initW = glorot_uniform(rng)),
                        Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
                        Dense(N_NEURONS, 1; initW = glorot_uniform(rng)),
            ),
            optimizer = ADAM(lr),
        );

        ###############################################
        # Custom implementation of state dependet filters #
        ###############################################

        # approximator = ActorCritic(
        #     # n_states - number of states as input
        #     # n_actions - number of actions as output
        #     # N_NEURONS - number of hidden neurons in one layer of the FNN
        #     # 3 layers; last layer splitted in mean and variance; then action is sampled
        #     actor = GaussianNetwork(
        #         pre = Chain(Parallel(
        #             vcat,
        #             Chain(select_1, Dense(20,1; init = init2)),
        #             Chain(select_1, Dense(20,1; init = init2)),
        #             Chain(select_1, Dense(20,1; init = init2)),
        #             Chain(select_1, Dense(20,1; init = init2)),
        #             Chain(select_2, Dense(20,1; init = init2)),
        #             Chain(select_2, Dense(20,1; init = init2)),
        #             Chain(select_2, Dense(20,1; init = init2)),
        #             Chain(select_2, Dense(20,1; init = init2)),
        #             Chain(select_3, Dense(20,1; init = init2)),
        #             Chain(select_3, Dense(20,1; init = init2)),
        #             Chain(select_3, Dense(20,1; init = init2)),
        #             Chain(select_3, Dense(20,1; init = init2)),
        #             Chain(select_4, Dense(20,1; init = init2)),
        #             Chain(select_4, Dense(20,1; init = init2)),
        #             Chain(select_4, Dense(20,1; init = init2)),
        #             Chain(select_4, Dense(20,1; init = init2)),
        #             Chain(select_5, Dense(20,1; init = init2)),
        #             Chain(select_5, Dense(20,1; init = init2)),
        #             Chain(select_5, Dense(20,1; init = init2)),
        #             Chain(select_5, Dense(20,1; init = init2)),
        #             Chain(select_6, Dense(20,1; init = init2)),
        #             Chain(select_6, Dense(20,1; init = init2)),
        #             Chain(select_6, Dense(20,1; init = init2)),
        #             Chain(select_6, Dense(20,1; init = init2)),
        #             Chain(select_7, Dense(20,1; init = init2)),
        #             Chain(select_7, Dense(20,1; init = init2)),
        #             Chain(select_7, Dense(20,1; init = init2)),
        #             Chain(select_7, Dense(20,1; init = init2)),
        #             Chain(select_8, Dense(20,1; init = init2)),
        #             Chain(select_8, Dense(20,1; init = init2)),
        #             Chain(select_8, Dense(20,1; init = init2)),
        #             Chain(select_8, Dense(20,1; init = init2)),
        #             Chain(select_9, Dense(20,1; init = init2)),
        #             Chain(select_9, Dense(20,1; init = init2)),
        #             Chain(select_9, Dense(20,1; init = init2)),
        #             Chain(select_9, Dense(20,1; init = init2)),
                   
        #         ),
        #                     Dense(36 , N_NEURONS, relu; initW = glorot_uniform(rng)),
        #                     Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
        #             ),
        #         μ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
        #         logσ = Chain(Dense(N_NEURONS, n_actions; initW = glorot_uniform(rng))),
        #         ),
        #     critic = Chain(Parallel(
        #         vcat,
        #         Chain(select_1, Dense(20,1; init = init2)),
        #         Chain(select_1, Dense(20,1; init = init2)),
        #         Chain(select_1, Dense(20,1; init = init2)),
        #         Chain(select_1, Dense(20,1; init = init2)),
        #         Chain(select_2, Dense(20,1; init = init2)),
        #         Chain(select_2, Dense(20,1; init = init2)),
        #         Chain(select_2, Dense(20,1; init = init2)),
        #         Chain(select_2, Dense(20,1; init = init2)),
        #         Chain(select_3, Dense(20,1; init = init2)),
        #         Chain(select_3, Dense(20,1; init = init2)),
        #         Chain(select_3, Dense(20,1; init = init2)),
        #         Chain(select_3, Dense(20,1; init = init2)),
        #         Chain(select_4, Dense(20,1; init = init2)),
        #         Chain(select_4, Dense(20,1; init = init2)),
        #         Chain(select_4, Dense(20,1; init = init2)),
        #         Chain(select_4, Dense(20,1; init = init2)),
        #         Chain(select_5, Dense(20,1; init = init2)),
        #         Chain(select_5, Dense(20,1; init = init2)),
        #         Chain(select_5, Dense(20,1; init = init2)),
        #         Chain(select_5, Dense(20,1; init = init2)),
        #         Chain(select_6, Dense(20,1; init = init2)),
        #         Chain(select_6, Dense(20,1; init = init2)),
        #         Chain(select_6, Dense(20,1; init = init2)),
        #         Chain(select_6, Dense(20,1; init = init2)),
        #         Chain(select_7, Dense(20,1; init = init2)),
        #         Chain(select_7, Dense(20,1; init = init2)),
        #         Chain(select_7, Dense(20,1; init = init2)),
        #         Chain(select_7, Dense(20,1; init = init2)),
        #         Chain(select_8, Dense(20,1; init = init2)),
        #         Chain(select_8, Dense(20,1; init = init2)),
        #         Chain(select_8, Dense(20,1; init = init2)),
        #         Chain(select_8, Dense(20,1; init = init2)),
        #         Chain(select_9, Dense(20,1; init = init2)),
        #         Chain(select_9, Dense(20,1; init = init2)),
        #         Chain(select_9, Dense(20,1; init = init2)),
        #         Chain(select_9, Dense(20,1; init = init2)),
               
        #     ),
        #                 Dense(36, N_NEURONS, relu; initW = glorot_uniform(rng)),
        #                 Dense(N_NEURONS, N_NEURONS, relu; initW = glorot_uniform(rng)),
        #                 Dense(N_NEURONS, 1; initW = glorot_uniform(rng)),
        #     ),
        #     optimizer = ADAM(lr),
        # );


    end

    ## Define the agent
    agent = Agent( # A wrapper of an AbstractPolicy
        # AbstractPolicy: the policy to use
        policy = PPOPolicy(;
                    approximator = approximator |> gpu,
                    update_freq = UPDATE_FREQ,
                    dist = Normal,
                    ),

        # AbstractTrajectory: used to store transitions between an agent and an environment source
        trajectory = PPOTrajectory(;
            capacity = UPDATE_FREQ,
            state = Matrix{Float64} => (length(state(env[1])), N_ENV),
            action = Matrix{Float64} => (length(action_space(env[1])), N_ENV),
            action_log_prob = Vector{Float64} => (N_ENV,),
            reward = Vector{Float64} => (N_ENV,),
            terminal = Vector{Bool} => (N_ENV,),
        ),
    );


    # Hook variables
    global best_reward = -10000.0

    # Define hook which is called during the training
    total_batch_reward_per_episode = TotalBatchRewardPerEpisode(N_ENV, is_display_on_exit = false)
    episode_test_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
    episode_test_envs_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)
    single_test_env = VtolEnv(;rng = rng,
                               name = "single_test_env",
                               enable_visualization = false,
                               enable_realtime = false,
                               enable_logging = true,
                               enable_sensor_dist = true,
                               enable_actuator_dist = true,
                               enable_resampling = false,
                               enable_z_termination = false,
                               wind_W = test_winds[1],
                               v_W_trajectory = create_trajectory(USE_RANDOM_TRAJECTORY=false));

    fixed_wing_trajectory = generate_fixed_wing_trajectory()
    fixed_wing_env = VtolEnv(;rng = rng,
                              name = "fixed_wing_env",
                              enable_visualization = false,
                              enable_realtime = false,
                              enable_logging = true,
                              enable_sensor_dist = false,
                              enable_actuator_dist = false,
                              enable_resampling = false,
                              enable_z_termination = false,
                              wind_W = simulate_wind(use_wind = false),
                              v_W_trajectory = fixed_wing_trajectory);


    hook = ComposedHook(
                total_batch_reward_per_episode,
                DoEveryNStep(;n=SAVING_FREQ) do t, agent, env
                    save_model(t, agent, env, experiment_folder, "vtol_ppo_$t")
                end,
                DoEveryNStep() do t, agent, env
                    p = agent.policy
                    with_logger(logger) do
                        @info "training" loss = mean(p.loss)  actor_loss = mean(p.actor_loss)  critic_loss = mean(p.critic_loss)  entropy_loss = mean(p.entropy_loss)
                    end
                end,
                DoEveryNStep() do t, agent, env
                    with_logger(logger) do
                        rewards = [ total_batch_reward_per_episode.rewards[i][end] for i in 1:length(env) if is_terminated(env[i]) ]
                        if length(rewards) > 0
                            @info "training" rewards = mean(rewards)
                        end
                    end
                end,
                ###############################################
                #           Test on single test env           #
                ###############################################
                DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
                    run(agent.policy, single_test_env, StopAfterEpisode(1), episode_test_reward_hook)
                    abs_vel_error_x = abs.(getindex.(single_test_env.log_velocity,1) - getindex.(single_test_env.log_target_velocity,1))
                    abs_vel_error_y = abs.(getindex.(single_test_env.log_velocity,2) - getindex.(single_test_env.log_target_velocity,2))
                    abs_vel_error_z = abs.(getindex.(single_test_env.log_velocity,3) - getindex.(single_test_env.log_target_velocity,3))
                    # Divide by the step counter, since simulation can terminate early
                    single_test_env_abs_vel_error_per_timestep = sum([abs_vel_error_x; abs_vel_error_y; abs_vel_error_z]) / (single_test_env.step)
                    # Save to logs
                    with_logger(logger) do
                        @info "testing single_test_env" reward = episode_test_reward_hook.rewards[end]
                        @info "testing single_test_env" abs_vel_error_per_timestep = single_test_env_abs_vel_error_per_timestep
                    end
                end,
                ###############################################
                #          Test on multiple test envs         #
                ###############################################
                DoEveryNStep(;n=EVALUATION_FREQ) do t, agent, env
                    rewards = []
                    multiple_test_env_abs_vel_error_per_timestep_mean = []
                    # Loop through all envs, since StopAfterEpisode is not supported for MultiThreadEnv
                    for test_set_env in test_set_envs
                        run(agent.policy, test_set_env, StopAfterEpisode(1), episode_test_envs_reward_hook)
                        push!(rewards, episode_test_envs_reward_hook.rewards[end])
                        abs_vel_error_x = abs.(getindex.(test_set_env.log_velocity,1) - getindex.(test_set_env.log_target_velocity,1))
                        abs_vel_error_y = abs.(getindex.(test_set_env.log_velocity,2) - getindex.(test_set_env.log_target_velocity,2))
                        abs_vel_error_z = abs.(getindex.(test_set_env.log_velocity,3) - getindex.(test_set_env.log_target_velocity,3))
                        push!(multiple_test_env_abs_vel_error_per_timestep_mean, (sum([abs_vel_error_x; abs_vel_error_y; abs_vel_error_z]) / (test_set_env.step)))
                    end
                    # Save model if it is performing better than the current best model
                    if mean(rewards) > best_reward
                        global best_reward = mean(rewards)
                        print("New best performing model with average reward: " * string(best_reward) * "\n")
                        print("New best performing model with multiple_test_env_abs_vel_error_per_timestep_mean: " * string(mean(multiple_test_env_abs_vel_error_per_timestep_mean)) * "\n")
                        save_model(t, agent, env, experiment_folder, "best_model")
                    end
                    # Save to logs
                    with_logger(logger) do
                        if length(rewards) > 0
                            @info "testing multiple_test_env" reward = mean(rewards)  average_abs_vel_error_per_timestep_mean = mean(multiple_test_env_abs_vel_error_per_timestep_mean)
                        end
                    end
                end,
            );

    return env, test_set_envs, single_test_env, fixed_wing_env, agent, hook
end;

function select_1(x)
    return x[1:20,:]
end
function select_2(x)
    return x[21:40,:]
end
function select_3(x)
    return x[41:60,:]
end
function select_4(x)
    return x[61:80,:]
end
function select_5(x)
    return x[81:100,:]
end
function select_6(x)
    return x[101:120,:]
end
function select_7(x)
    return x[121:140,:]
end
function select_8(x)
    return x[141:160,:]
end
function select_9(x)
    return x[161:180,:]
end

# Custom CNN weight init function
function init(a,b,c)
    ones(a,b,c)/(N_FILTERS_CNN*cnn_length)
end;


# Custom reshape function, because anonymous functions cannot be saved and loaded in .bson
function cnn_reshape(x)
    return reshape(x, (size(x)[1], 1, :))
end;


function save_model(t, agent, env, path, model_name)
    model = cpu(agent.policy.approximator)
    f = joinpath(path, "$model_name.bson")
    @save f model
    println("Saved parameters at step $t to $f")
end;


function init2(a, b)
    ones(a,b)/(2*20)
end


function load_model(model_path)
    @load model_path model
    return model
end;


# Define a keyword-based constructor for the type declared in the mutable struct typedef
# It could also be done with the macro Base.@kwdef
function VtolEnv(;rng = Random.GLOBAL_RNG, # Random number generation
                  name = "vtol",
                  enable_visualization = false,
                  enable_realtime = false,
                  enable_logging = false,
                  enable_resampling = true,
                  enable_z_termination = true, # only disable for test environments
                  enable_random_state_init = ENABLE_RANDOM_STATE_INITIALIZATION,
                  enable_actuator_dist = USE_ACTUATOR_DIST,
                  enable_sensor_dist = USE_SENSOR_DIST,
                  v_W_trajectory = create_trajectory(USE_RANDOM_TRAJECTORY=true),
                  wind_W = simulate_wind(use_wind = USE_WIND, N = Int(floor(TOTAL_SIMULATION_TIME/SIMULATION_STEP_TIME)), mean_wind_min = MEAN_WIND_MIN,  mean_wind_max = MEAN_WIND_MAX), # wind_W
                  n_states = 9,
                  state_history_length = STATE_HISTORY_LENGTH,
                  kwargs...) # let the function take an arbitrary number of keyword arguments
    
    T = Float64; # explicit type which is used e.g. in state. Cannot be altered due to the poor matrix definition
    
    # 4D action space for independent rotors and flaps
    action_space = Space(
        ClosedInterval{T}[
            0.0..2.0, # left thrust | N # maybe adjust range later
            0.0..2.0, # right thrust | N # maybe adjust range later
            -1.0..1.0, # left flap | rad
            -1.0..1.0, # right flap | rad
            ],
    )

    # technically, this is the observation space (disturbed state space) that goes into the policy
    state_space = Space(ClosedInterval{T}[typemin(T)..typemax(T) for i in 1:(n_states*state_history_length)])
    # rotation around x,y,z
    # rotation velocity around x,y,z
    # linear body velocity error x,y,z

    # Initialize CNN states and state space
    cnn_states = [zeros(T, state_history_length) for i in 1:n_states]
    init_state = reduce(vcat, cnn_states)
    
    if enable_visualization
        create_VTOL(name, actuators = true, color_vec=[1.0; 1.0; 0.6; 1.0]);
        set_transform(name, [0.0; 0.0; 0.0] ,QuatRotation([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]));
        set_actuators(name, [0.0; 0.0; 0.0; 0.0])
        set_arrow(name*"/v_B_setpoint", color=RGBA{Float32}(0.0, 0.8, 0.0, 0.2));
        transform_arrow(name*"/v_B_setpoint", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0])
        set_arrow(name*"/v_B", color=RGBA{Float32}(0.8, 0.8, 0.0, 0.2));
        transform_arrow(name*"/v_B", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0]);
        set_arrow(name*"/wind", color=RGBA{Float32}(0.0, 0.8, 0.0, 0.2));
        transform_arrow(name*"/wind", [0.0; 0.0; 0.0], [0.0; 0.0; 0.0]);
    end


    # Instantiates and initializes the struct from above
    environment = VtolEnv(
        action_space,
        state_space, # technically, this is the observation space (disturbed state space) that goes into policy
        init_state, # technically, this is the observation (disturbed state) that goes into policy
        zeros(T, length(action_space)),
        zeros(T, length(action_space)),
        false, # episode done?
        0.0, # time
        rng, # random number generator
        name,
        state_history_length,
        n_states,
        enable_visualization,
        enable_realtime, # enable realtime visualization
        enable_logging,
        enable_resampling,
        enable_random_state_init,
        enable_actuator_dist,
        enable_sensor_dist,
        enable_z_termination,
        zeros(T, 3), # x_W
        zeros(T, 3), # v_B
        zeros(T, 3), # a_B
        Array{T}([1 0 0; 0 1 0; 0 0 1]), # R_W
        zeros(T, 3), # ω_B
        zeros(T, 3), # α_B
        cnn_states,
        wind_W,
        v_W_trajectory,
        0, # step counter
        [0.0], # log_thrust_left
        [0.0], # log_thrust_right
        [0.0], # log_flaps_right
        [0.0], # log_flaps_left
        [[0.0,0.0,0.0]], # log_velocity
        [[0.0,0.0,0.0]], # log_target_velocity
        [[0.0,0.0,0.0]], # log_ω_B
        [0.0], # log_orientation_diff
        [0.0], # log_orientation_diff_Xz
        T(SIMULATION_STEP_TIME), # Δt
    )

    # Do this for simulation start
    RLBase.reset!(environment)

    return environment

end;


function create_trajectory(;USE_RANDOM_TRAJECTORY)

    if USE_RANDOM_TRAJECTORY
        v_W_trajectory = generate_random_3D_trajectory(randomness_mode = 3,
                                                    N = Int(floor(TOTAL_SIMULATION_TIME/SIMULATION_STEP_TIME)),
                                                    v_max = V_MAX,
                                                    N_rise_lower_limit = N_RISE_LOWER_LIMIT,
                                                    N_rise_upper_limit = N_RISE_UPPER_LIMIT,
                                                    N_hold_lower_limit = N_HOLD_LOWER_LIMIT,
                                                    N_hold_upper_limit = N_HOLD_UPPER_LIMIT,
                                                    N_fall_lower_limit = N_FALL_LOWER_LIMIT,
                                                    N_fall_upper_limit = N_FALL_UPPER_LIMIT)
    else
        v_W_trajectory = generate_3D_trajectory(generation_mode = TRAJECTORY_GENERATION_MODE,
                                                N = Int(floor(TOTAL_SIMULATION_TIME/SIMULATION_STEP_TIME)),
                                                v_max = V_MAX,
                                                v_const = [0.0,0.0,V_CONST])
    end

    return v_W_trajectory
end;



"""
plot_env_data(...)
"""

function plot_env_data(env)
    data_vel_x = [
        getindex.(env.log_velocity,1),
        getindex.(env.log_target_velocity,1)
    ]

    data_vel_y = [
        getindex.(env.log_velocity,2),
        getindex.(env.log_target_velocity,2)
    ]

    data_vel_z = [
        getindex.(env.log_velocity,3),
        getindex.(env.log_target_velocity,3)
    ]

    data_vel_abs = [
        getindex.(env.log_velocity,1) - getindex.(env.log_target_velocity,1),
        getindex.(env.log_velocity,2) - getindex.(env.log_target_velocity,2),
        getindex.(env.log_velocity,3) - getindex.(env.log_target_velocity,3)
    ]

    data_ω_B = [
        getindex.(env.log_ω_B,1),
        getindex.(env.log_ω_B,2),
        getindex.(env.log_ω_B,3)
    ]

    data_orientation_diff = [
        env.log_orientation_diff
    ]

    data_orientation_diff_xz = [
        env.log_orientation_diff_xz
    ]

    data_actions_thrust = [
        env.log_thrust_left,
        env.log_thrust_right,
    ]

    data_actions_flaps = [
        env.log_flaps_left,
        env.log_flaps_right,
    ]

    data_wind_x = getindex.(env.wind_W,1)[1:env.step]
    data_wind_y = getindex.(env.wind_W,2)[1:env.step]
    data_wind_z = getindex.(env.wind_W,3)[1:env.step]
    data_wind = [data_wind_x, data_wind_y, data_wind_z, norm.(env.wind_W)[1:env.step]]

    t = range(start=0, stop=(env.step)*env.Δt, step=env.Δt)
    p1 = plot(t, data_vel_x, label = ["actual vel" "target vel"], title = "velocity in global x direction", xlabel="time [s]", ylabel="v_x [m/s]", left_margin = 20*Plots.mm)
    p2 = plot(t, data_vel_y, label = ["actual vel" "target vel"], title = "velocity in global y direction", xlabel="time [s]", ylabel="v_y [m/s]", left_margin = 20*Plots.mm)
    p3 = plot(t, data_vel_z, label = ["actual vel" "target vel"], title = "velocity in global z direction", xlabel="time [s]", ylabel="v_z [m/s]", left_margin = 20*Plots.mm)
    p4 = plot(t, data_vel_abs, label = ["x" "y" "z"], title = "absolute velocity errors in global directions", xlabel="time [s]", left_margin = 20*Plots.mm)
    p5 = plot(t[1:end-1], data_wind, label = ["x" "y" "z" "norm"], title = "global wind speeds", xlabel="time [s]", ylabel="speed [m/s]", left_margin = 20*Plots.mm)
    p6 = plot(t, data_actions_thrust, label = ["thrust left" "thrust right"], title = "thrust actions", xlabel="time [s]", ylabel="thrust [N]", left_margin = 20*Plots.mm)
    p7 = plot(t, data_actions_flaps, label = ["flap left" "flap right"], title = "flap actions", xlabel="time [s]", ylabel="flap angle [rad]", left_margin = 20*Plots.mm)
    p8 = plot(t, data_ω_B, label = ["ω x" "ω y" "ω z"], title = "Rotational speed", xlabel="time [s]", ylabel="rotational speed [rad/s]", left_margin = 20*Plots.mm)
    p9 = plot(t, data_orientation_diff, label = false, title = "angle between current and target velocity vector", xlabel="time [s]", ylabel="orientation diff. [deg]", left_margin = 20*Plots.mm)
    p10 = plot(t, data_orientation_diff_xz, label = false, title = "angle between current and target velocity vector in xz plane", xlabel="time [s]", ylabel="orientation diff. [deg]", left_margin = 20*Plots.mm)
    p = plot(p1, p2, p3, p9, p10, p4, p5, p6, p7, p8, layout=(10,1), size=(1000,2000))
    return p
end



"""
plot_cnn_weights(;number_cnn_filters, agent)
"""

function plot_cnn_weights(;number_cnn_filters, agent)
    bars = []
    for i in 1:number_cnn_filters
        title = "CNN weights for filter " * string(i)
        push!(bars, bar(agent.policy.approximator.actor.pre[2].weight[:, : , i], xlabel ="weight number", ylabel ="activation", title = title, legend=false))
    end
    p = plot(bars... , layout=(number_cnn_filters,1), size=(1000,200*number_cnn_filters), left_margin = 20*Plots.mm)
    return p
end;



"""
plot_first_layer_activation(...)
"""

function plot_first_layer_activation(agent, n_states)
    # Abs sum of weights of first FC layer
    abs_weights = abs.(agent.policy.approximator.actor.pre[4].weight)
    sum_weights = sum(abs_weights, dims=1)
    # Create a seperate plot for each state and collect them in list
    plots = []
    for i in 1:(n_states)
        title = "Absolute filter activation for feature " * string(i)
        push!(plots, bar(yticks = 0:1:N_FILTERS_CNN, sum_weights[(i*N_FILTERS_CNN-(N_FILTERS_CNN-1)):(i*N_FILTERS_CNN)], orientation='h', label=false, xlabel="Absolute value of filter activation", ylabel="Filter number", title=title, titlefontsize=10, ylabelfontsize=8, xlabelfontsize=8, left_margin=20*Plots.mm))
    end
    p = plot(plots... , layout=(n_states,1), size=(500,170*n_states))
    return p
end


"""
save_env_metrics(...)
"""

function save_env_metrics(metrics, file_path)
    open(file_path, "w") do file
        println(file, "Reward: " * string(metrics["reward"]))
        println(file, "Number of early terminated environments: " * string(metrics["n_early_terminated"]))
        println(file, "Indices of early terminated environments: " * string(metrics["early_terminated_env_index_list"]))
        println(file, "Sum velocity error x: " * string(metrics["sum_vel_error_x"]))
        println(file, "Sum velocity error y: " * string(metrics["sum_vel_error_y"]))
        println(file, "Sum velocity error z: " * string(metrics["sum_vel_error_z"]))

        println(file, "Sum of absolute velocity error x: " * string(metrics["sum_abs_vel_error_x"]))
        println(file, "Sum of absolute velocity error y: " * string(metrics["sum_abs_vel_error_y"]))
        println(file, "Sum of absolute velocity error z: " * string(metrics["sum_abs_vel_error_z"]))
        println(file, "Maximum absolute velocity error: " * string(metrics["abs_vel_error_max"]))
        println(file, "Average absolute velocity error per timestep: " * string(metrics["average_abs_vel_error_per_timestep"]))

        println(file, "Total thrust energy used: " * string(metrics["thrust_energy"]))
        println(file, "Total flap energy used: " * string(metrics["flap_energy"]))
    end
end

"""
save_additional_statistics(...)
"""

function save_additional_statistics(file_path)
    open(file_path, "w") do file
        # MinMaxStates is global
        println(file, "MinMaxValues of states:")
        for i in 1:length(MinMaxStates)
            println(file, "State $i min-max-values: " * string(MinMaxStates[i]))
        end
    end
end


"""
print_env_metrics(...)
"""

function print_env_metrics(metrics)
    println("Reward: " * string(metrics["reward"]))
    println("Number of terminated environments: " * string(metrics["n_early_terminated"]))
    println("Indices of early terminated environments: " * string(metrics["early_terminated_env_index_list"]))
    println("Sum velocity error x: " * string(metrics["sum_vel_error_x"]))
    println("Sum velocity error y: " * string(metrics["sum_vel_error_y"]))
    println("Sum velocity error z: " * string(metrics["sum_vel_error_z"]))
    println("Sum of absolute velocity error x: " * string(metrics["sum_abs_vel_error_x"]))
    println("Sum of absolute velocity error y: " * string(metrics["sum_abs_vel_error_y"]))
    println("Sum of absolute velocity error z: " * string(metrics["sum_abs_vel_error_z"]))
    println("Maximum absolute velocity error: " * string(metrics["abs_vel_error_max"]))
    println("Average absolute velocity error per timestep: " * string(metrics["average_abs_vel_error_per_timestep"]))
    println("Total thrust energy used: " * string(metrics["thrust_energy"]))
    println("Total flap energy used: " * string(metrics["flap_energy"]))
end


"""
test_agent_on_single_env(...)
"""

function test_agent_on_single_env(agent, env)
    metrics = Dict()

    # initialize everything as list
    metrics["reward"] = []
    metrics["n_early_terminated"] = []
    metrics["early_terminated_env_index_list"] = []
    metrics["sum_vel_error_x"] = []
    metrics["sum_vel_error_y"] = []
    metrics["sum_vel_error_z"] = []
    metrics["sum_abs_vel_error_x"] = []
    metrics["sum_abs_vel_error_y"] = []
    metrics["sum_abs_vel_error_z"] = []  
    metrics["abs_vel_error_max"] = []
    metrics["average_abs_vel_error_per_timestep"]  = []
    metrics["thrust_energy"] = []
    metrics["flap_energy"] = []

    episode_test_envs_reward_hook = TotalRewardPerEpisode(;is_display_on_exit=false)

    number_test_runs = 15

    # run tests number_test_runs times and push to dict
    for i in 1:number_test_runs
        run(agent.policy, env, StopAfterEpisode(1), episode_test_envs_reward_hook)

        # reward
        push!(metrics["reward"], episode_test_envs_reward_hook.rewards[end])
        
        # n terminated env
        if (env.t < TOTAL_SIMULATION_TIME)
            push!(metrics["n_early_terminated"], 1)
        else
            push!(metrics["n_early_terminated"], 0)
        end

        # Absolute velocities
        push!(metrics["sum_vel_error_x"] , sum(getindex.(env.log_velocity,1) - getindex.(env.log_target_velocity,1)))
        push!(metrics["sum_vel_error_y"] , sum(getindex.(env.log_velocity,2) - getindex.(env.log_target_velocity,2)))
        push!(metrics["sum_vel_error_z"] , sum(getindex.(env.log_velocity,3) - getindex.(env.log_target_velocity,3)))
        abs_vel_error_x = abs.(getindex.(env.log_velocity,1) - getindex.(env.log_target_velocity,1))
        abs_vel_error_y = abs.(getindex.(env.log_velocity,2) - getindex.(env.log_target_velocity,2))
        abs_vel_error_z = abs.(getindex.(env.log_velocity,3) - getindex.(env.log_target_velocity,3))
        push!(metrics["sum_abs_vel_error_x"] , sum(abs_vel_error_x))
        push!(metrics["sum_abs_vel_error_y"] , sum(abs_vel_error_y))
        push!(metrics["sum_abs_vel_error_z"] , sum(abs_vel_error_z))
        abs_vel_errors = [abs_vel_error_x; abs_vel_error_y; abs_vel_error_z]
        push!(metrics["abs_vel_error_max"] , maximum(abs_vel_errors))
        push!(metrics["average_abs_vel_error_per_timestep"] , sum(abs_vel_errors) / env.step)

        # Actuator energy
        push!(metrics["thrust_energy"] , sum([abs.(env.log_thrust_left); abs.(env.log_thrust_right)]) / env.step)
        push!(metrics["flap_energy"] , sum([abs.(env.log_flaps_left); abs.(env.log_flaps_right)]) / env.step)

    end

    # now calculate average of runs
    metrics["reward"] = mean(metrics["reward"])
    metrics["sum_vel_error_x"] = mean(metrics["sum_vel_error_x"])
    metrics["sum_vel_error_y"] = mean(metrics["sum_vel_error_y"])
    metrics["sum_vel_error_z"] = mean(metrics["sum_vel_error_z"])
    metrics["sum_abs_vel_error_x"] = mean(metrics["sum_abs_vel_error_x"])
    metrics["sum_abs_vel_error_y"] = mean(metrics["sum_abs_vel_error_y"])
    metrics["sum_abs_vel_error_z"] = mean(metrics["sum_abs_vel_error_z"])  
    metrics["abs_vel_error_max"] = mean(metrics["abs_vel_error_max"])
    metrics["average_abs_vel_error_per_timestep"]  = mean(metrics["average_abs_vel_error_per_timestep"])
    metrics["thrust_energy"] = mean(metrics["thrust_energy"])
    metrics["flap_energy"] = mean(metrics["flap_energy"])

    # for n_early_terminated a treshold is defined
    # if more than 20% terminated -> count as terminated
    # ATTENTION: this approach is somewhat prone to overestimation!
    if mean(metrics["n_early_terminated"]) > 0.3
        metrics["n_early_terminated"] = 1
        metrics["early_terminated_env_index_list"] = 1
    else
        metrics["n_early_terminated"] = 0
        metrics["early_terminated_env_index_list"] = []
    end

    return metrics
end


"""
test_agent_on_multiple_envs(...)
"""

function test_agent_on_multiple_envs(agent, envs)

    new_metrics = Dict()

    rewards = 0
    n_early_terminated = 0
    early_terminated_env_index_list = []
    sum_vel_error_x = 0
    sum_vel_error_y = 0
    sum_vel_error_z = 0
    sum_abs_vel_error_x = 0
    sum_abs_vel_error_y = 0
    sum_abs_vel_error_z = 0
    abs_vel_error_max = 0
    average_abs_vel_error_per_timestep = 0
    thrust_energy = 0
    flap_energy = 0
    n_envs = length(envs)

    # Calculate metrics for every environment
    for i in 1:n_envs
        metrics = test_agent_on_single_env(agent, envs[i])
        if (metrics["n_early_terminated"] == 1)
            n_early_terminated += 1
            push!(early_terminated_env_index_list,i)
        end
        rewards += metrics["reward"]
        sum_vel_error_x += metrics["sum_vel_error_x"]
        sum_vel_error_y += metrics["sum_vel_error_y"]
        sum_vel_error_z += metrics["sum_vel_error_z"]
        sum_abs_vel_error_x += metrics["sum_abs_vel_error_x"]
        sum_abs_vel_error_y += metrics["sum_abs_vel_error_y"]
        sum_abs_vel_error_z += metrics["sum_abs_vel_error_z"]
        abs_vel_error_max += metrics["abs_vel_error_max"]
        average_abs_vel_error_per_timestep += metrics["average_abs_vel_error_per_timestep"]
        thrust_energy += metrics["thrust_energy"]
        flap_energy += metrics["flap_energy"]
    end

    # Average metrics over all environments
    new_metrics["reward"] = rewards/n_envs
    new_metrics["n_early_terminated"] = n_early_terminated
    new_metrics["early_terminated_env_index_list"] = early_terminated_env_index_list
    new_metrics["sum_vel_error_x"] = sum_vel_error_x/n_envs
    new_metrics["sum_vel_error_y"] = sum_vel_error_y/n_envs
    new_metrics["sum_vel_error_z"] = sum_vel_error_z/n_envs
    new_metrics["sum_abs_vel_error_x"] = sum_abs_vel_error_x/n_envs
    new_metrics["sum_abs_vel_error_y"] = sum_abs_vel_error_y/n_envs
    new_metrics["sum_abs_vel_error_z"] = sum_abs_vel_error_z/n_envs
    new_metrics["abs_vel_error_max"] = abs_vel_error_max/n_envs
    new_metrics["average_abs_vel_error_per_timestep"] = average_abs_vel_error_per_timestep/n_envs
    new_metrics["thrust_energy"] = thrust_energy/n_envs
    new_metrics["flap_energy"] = flap_energy/n_envs

    return new_metrics
end


"""
test_agent_on_env(...)
"""

function test_agent_on_env(agent, env)
    if env isa VtolEnv # hack to distinguish between single and multiple environments
        metrics = test_agent_on_single_env(agent, env)
    else
        metrics = test_agent_on_multiple_envs(agent, env)
    end
    return metrics
end



function load_test_data(test_data_path)

    f = joinpath(test_data_path, "test_winds.bson")
    @load f test_winds
    f = joinpath(test_data_path, "test_trajectories.bson")
    @load f test_trajectories
    return test_winds, test_trajectories

end;


function generate_test_data(test_data_path, N_ENV_TESTSET, TOTAL_SIMULATION_TIME, SIMULATION_STEP_TIME, MEAN_WIND_MIN, MEAN_WIND_MAX)

    # Code block to generate and load a test set
    test_winds = []
    test_trajectories = []

    # Generate test wind and trajectories
    for i in 1:N_ENV_TESTSET
        wind = simulate_wind(use_wind = true, N = Int(floor(TOTAL_SIMULATION_TIME/SIMULATION_STEP_TIME)), mean_wind_min = MEAN_WIND_MIN,  mean_wind_max = MEAN_WIND_MAX)
        push!(test_winds, wind)
        trajectory = create_trajectory(USE_RANDOM_TRAJECTORY=true)
        push!(test_trajectories, trajectory)
    end

    # Save test wind and trajectories
    f = joinpath(test_data_path, "test_winds.bson")
    @save f test_winds
    f = joinpath(test_data_path, "test_trajectories.bson")
    @save f test_trajectories

end;
