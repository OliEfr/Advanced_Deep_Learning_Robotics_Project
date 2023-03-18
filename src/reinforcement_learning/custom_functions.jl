using Rotations;
using Random;
using IntervalSets;
using LinearAlgebra;
using Distributions;
using JSON;



"""
get_random_rotation_matrix(...)

(Correct) Uniform orientation sampling using Householder matrix,
for more details see https://www.blopig.com/blog/2021/08/uniformly-sampled-3d-rotation-matrices/
"""

function get_random_rotation_matrix()
    x_distribution = Uniform(0, 1)
    x = rand(x_distribution, 3)
    R = RotZ(x[1])
    v = [cos(2*pi*x[2]) * sqrt(x[3]);
        sin(2*pi*x[2]) * sqrt(x[3]);
        sqrt(1-x[3]);]
    H = I - 2 * v * transpose(v)
    R_W = -H * R
    return R_W
end



"""
plot_trajectory(...)

This function plots the passed trajectory array.
It expects a vector of vectors, where the inner vector is of shape (3,) and the outer is of form (N,)
Example:
    [[0.0, -0.0, 0.0],  <- single entry is of form (3,)
     [1.0, -0.5, 0.0],
     [2.0, -1.0, 0.0]]
"""

function plot_trajectory(v_setpoint)
    v_x = getindex.(v_setpoint,1)
    v_y = getindex.(v_setpoint,2)
    v_z = getindex.(v_setpoint,3)
    p = plot([v_x v_y v_z], label=["v_W_x" "v_W_y" "v_W_z"])
    return p
end



"""
get_experiment_parameters(...)

This function returns a dictionary containing the parameters from a given json parameter file.
"""

function get_experiment_parameters(json_file_path)
    dict = Dict()
    open(json_file_path, "r") do f
        dict = JSON.parse(f)
    end
    return dict
end