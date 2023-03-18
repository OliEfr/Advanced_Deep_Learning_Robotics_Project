## Start this script from terminal with
# julia --project=../.. --threads=4 run_experiment.jl ./experiment_001/experiment_001.json train save none
# NOTE: you must be in the folder ./src/reinforcement_learning/
# NOTE: the command requires four arguments, see below. For all of the arguments one value must be provided.


include("../Flyonic.jl");
using .Flyonic;
using ReinforcementLearning;
using TensorBoardLogger;

include("custom_functions.jl")
include("RL_functions.jl")

# Parse args
json_file_path = ARGS[1] # path to your .json configuration file for the experiment.
run_mode = ARGS[2] # set to 'train' to perform the training. Set to 'none' otherwise.
save_mode = ARGS[3] # set to 'save' to save training and test data. Set to 'none' otherwise.
pre_load = ARGS[4] # set to 'path-to-model' to load a pre-trained model. Set to 'none' otherwise.

experiment_folder = dirname(json_file_path)

# Global variables
MinMaxStates = [[0.0,0.0] for x in 1:9] # used to track min-max-values of drone states (only used for statistics)

# Start tensorboard from the terminal
# make sure to be in the folder .../src/examples
# tensorboard --logdir tensorboard

# Create tensorboard logger
logger = TBLogger("$experiment_folder/tensorboard", tb_overwrite)


print("#------------------------------------------------#\n\n")
print("Experiment folder: " * experiment_folder * "\n")
print("JSON parameter file location: " * json_file_path * "\n")
print("\n#------------------------------------------------#\n\n")

# Load parameters
parameter_dict = get_experiment_parameters(json_file_path);

training_envs, test_set_envs, single_test_env, fixed_wing_env, agent, hook = create_experiment_from_parameters(parameter_dict, experiment_folder);

if pre_load != "none"
    print("Pre-loading model...\n")
    model_path = joinpath(experiment_folder, pre_load);
    agent.policy.approximator = load_model(model_path);
    print("Pre-loading done.\n")
end



# Run experiment
if run_mode == "train"
    print("\n#---------------STARTING TRAINING----------------#\n\n")
    run(
        agent,
        training_envs,
        StopAfterStep(2_000_000, is_show_progress = false),
        hook
    )
    print("\n#---------------FINISHED TRAINING----------------#\n")
end


print("Loading best performing agent...\n")
model_path = joinpath(experiment_folder, "best_model.bson");
agent.policy.approximator = load_model(model_path);
print("done\n")


print("\n#--------------- LOADED BEST MODEL ---------------#\n")

print("Testing agent on test set...\n")
metrics = test_agent_on_env(agent, test_set_envs)
print_env_metrics(metrics)
print("done\n")
if save_mode == "save"
    print("Saving metrics...\n")
    file_path = joinpath(experiment_folder, "test_set_envs_metrics.txt")
    save_env_metrics(metrics, file_path)
    print("done\n")
end

print("\n")

print("Testing agent on single test environment...\n")
metrics = test_agent_on_env(agent, single_test_env)
print_env_metrics(metrics)
print("done\n")
if save_mode == "save"
    print("Saving metrics...\n")
    file_path = joinpath(experiment_folder, "single_test_env_metrics.txt")
    save_env_metrics(metrics, file_path)
    print("done\n")
    print("Saving plots...\n")
    p = plot_env_data(single_test_env)
    savefig(p, joinpath(experiment_folder, "single_test_env_plots.pdf"))
    print("done\n")
end

print("\n")

print("Testing agent for fixed wing transition...\n")
metrics = test_agent_on_env(agent, fixed_wing_env)
print_env_metrics(metrics)
print("done\n") 
if save_mode == "save"
    print("Saving metrics...\n")
    file_path = joinpath(experiment_folder, "fixed_wing_env_metrics.txt")
    save_env_metrics(metrics, file_path)
    print("done\n")
    print("Saving plots...\n")
    p = plot_env_data(fixed_wing_env)
    savefig(p, joinpath(experiment_folder, "fixed_wing_env_plots.pdf"))
    print("done\n")
end

print("\n")

if parameter_dict["USE_CNN"] && save_mode == "save"
    print("Saving CNN plots...\n")
    p = plot_cnn_weights(number_cnn_filters=parameter_dict["N_FILTERS_CNN"], agent=agent)
        savefig(p, joinpath(experiment_folder, "cnn_weights.pdf"))
    p = plot_first_layer_activation(agent, training_envs[1].n_states)
        savefig(p, joinpath(experiment_folder, "cnn_first_layer_activation.pdf"))
    print("done\n")
end

print("\n")

print("MinMaxStates:\n")
for i in 1:length(MinMaxStates)
    println("State $i min-max-values: " * string(MinMaxStates[i]))
end
if save_mode == "save"
    file_path = joinpath(experiment_folder, "additional_statistics.txt")
    # MinMaxStates is global -> dont need to pass as argument
    save_additional_statistics(file_path)
end

print("\nc")

print("\n#--------------- ALL DONE ---------------#\n")
