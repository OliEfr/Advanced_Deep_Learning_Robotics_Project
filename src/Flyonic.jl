module Flyonic



# RigidBodies.jl
include("RigidBodies.jl");
using .RigidBodies;
export vtol_add_wind, rigid_body_simple

# Visualization.jl
include("Visualization.jl");
using .Visualization;
export transform_Vtol, init_visualization, close_visualization, set_Arrow, transform_Vtol2
export create_remote_visualization, create_visualization, create_VTOL, create_sphere, set_transform, close_visualization, create_sphere, set_arrow, transform_arrow, set_actuators

# VtolModel.jl
include("VtolModel.jl");
using .VtolModel;
export vtol_model

# VtolsConfig.jl
include("VtolsConfig.jl");
export eth_vtol_param

end # end of module
