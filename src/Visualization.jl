module Visualization

export create_remote_visualization, create_visualization, create_VTOL, create_sphere, set_transform, close_visualization, create_sphere, set_arrow, transform_arrow, set_actuators


using MeshCat
using CoordinateTransformations
using Rotations
#using ReferenceFrameRotations;
#using GeometryTypes
using GeometryBasics
using Colors: RGBA, RGB
using MeshIO
using FileIO
using StaticArrays
using LinearAlgebra
using Sockets


"""
    create_visualization()

Create and open a new visualizer.
"""
function create_visualization()
    global vis = Visualizer();
    render(vis); # render MeshCat in Browser. Also possible inline.
    return vis
end

"""
    create_remote_visualization()

Create and open a new visualizer but with remote IP. Can be used to Visualize the VTOL on a Webpage at the UpBoard during flight.
"""
function create_remote_visualization()
    global vis = Visualizer(MeshCat.CoreVisualizer(getipaddr(), 8700),  ["meshcat"]);
    return vis
end


"""
    set_transform(x::Array{Real,1}, R::Array{Real,2}, name::AbstractString)
    
Select the object by its name and transform it by position x and rotation matrix R.
"""
function set_transform(name::AbstractString, x::AbstractVector{T}, R::QuatRotation{T}=QuatRotation(1.0,0,0,0)) where T
        settransform!(vis[name], Translation(x) ∘ LinearMap(R)); 
end

#"""
#    set_transform(x::Array{Real,1}, R::MVector{Real,2}, name::AbstractString)
#    
#Select the object by its name and transform it by position x and rotation matrix R.
#"""
#function set_transform(name::AbstractString, x::MVector; R::QuatRotation{<:Real}=QuatRotation(1.0,0,0,0))
#        settransform!(vis[name], Translation(x) ∘ LinearMap(R)); 
#end





"""
    set_actuators(name::AbstractString, actuators::AbstractVector{T})
    
Actuator visualisation 
"""
function set_actuators(name::AbstractString, actuators::AbstractVector{T}) where T

    f_r = actuators[1]
    f_l = actuators[2]
    δ_r = actuators[3]
    δ_l = actuators[4]
    

    base = [0.05; 0.359; 0.0];
    transform_arrow(name*"/thrust_right", base, [f_r; 0.0; 0.0])
    base = [0.05; -0.359; 0.0];
    transform_arrow(name*"/thrust_left", base, [f_l; 0.0; 0.0])
    
    x = [0.0; 0.0; 0.0];
    R = RotZ(0.0)

    x = [-0.27; -0.255; 0.0];
    R = (RotZ(-0.16)*RotY(-δ_l)*RotX(pi))
    settransform!(vis[name]["elevon_large_left"], Translation(x) ∘ LinearMap(R));
    x = [-0.27; 0.255; 0.0];
    R = RotZ(0.16)*RotY(-δ_r)
    settransform!(vis[name]["elevon_large_right"], Translation(x) ∘ LinearMap(R));
end

"""
    create_VTOL(name::AbstractString; actuators::Bool=false, color::RGBA{Float32}=RGBA{Float32}(0.8, 0.8, 0.8, 1.0))
    
Creates a Vtol object with the specified name. Model and color are optional. In addition, visualisations for the actuator values ( flaps, motors) can be activated.
"""
function create_VTOL(name::AbstractString; actuators::Bool=false, color_vec::AbstractVector=[0.8; 0.8; 0.8; 1.0])
    # https://threejs.org/docs/index.html?q=mesh#api/en/materials/MeshPhongMaterial
    color::RGBA{Float32}=RGBA{Float32}(color_vec[1], color_vec[2], color_vec[3], color_vec[4])
    vtol_material = MeshPhongMaterial(color=color);

    path = joinpath(@__DIR__, "..", "visualization", "models", "open_vtol/fuselage.dae");
    setobject!(vis[name]["fuselage"], MeshFileGeometry(path), vtol_material);

    if actuators
        path_elevon = joinpath(@__DIR__, "..", "visualization", "models", "open_vtol/elevon_large.dae")
        setobject!(vis[name]["elevon_large_left"], MeshFileGeometry(path_elevon), vtol_material);
        setobject!(vis[name]["elevon_large_right"], MeshFileGeometry(path_elevon), vtol_material);

        set_arrow(name*"/thrust_left")
        set_arrow(name*"/thrust_right")
    end
end


"""
    set_arrow(name::AbstractString; color::RGBA{Float32}=RGBA{Float32}(0.8, 0.0, 0.0, 0.2))
    
Initialises arrow with name and RGBA colour
"""
function set_arrow(name::AbstractString; color::RGBA{Float32}=RGBA{Float32}(0.8, 0.0, 0.0, 0.2))
    arrow_material = MeshPhongMaterial(color=color);

    shaft = Cylinder(zero(Point{3, Float64}), Point(0.0, 0.0, 1.0), 1.0)
    setobject!(vis[name]["shaft"], shaft, arrow_material)

    head = Cone(Point(0.0, 0.0, 0.0), Point(0.0, 0.0, 1.), 1.0)
    setobject!(vis[name]["head"], head, arrow_material) 
end


"""
    transform_arrow(name::AbstractString, base::AbstractVector, vec::AbstractVector; shaft_radius=0.01, max_head_radius=2*shaft_radius, max_head_length=max_head_radius)
    
Transforms arrow with name and support vector.
"""
function transform_arrow(name::AbstractString, base::AbstractVector, vec::AbstractVector;
    shaft_radius=0.01,
    max_head_radius=2*shaft_radius,
    max_head_length=max_head_radius)

    vec_length = norm(vec)
    T = typeof(vec_length)
    rotation = if vec_length > eps(T)
        rotation_between(SVector(0., 0., 1.), vec)
    else
        one(QuatRotation{Float64})
    end |> LinearMap

    vis_tform = Translation(base) ∘ rotation
    settransform!(vis[name], vis_tform)

    shaft_length = max(vec_length - max_head_length, 0.0)
    shaft_scaling_diag = SVector(shaft_radius, shaft_radius, shaft_length)
    if iszero(shaft_length)
        # This case is necessary to ensure that the shaft
        # completely disappears in animations.
        shaft_scaling_diag = zero(shaft_scaling_diag)
    end

    shaft_scaling = LinearMap(Diagonal(shaft_scaling_diag))
    settransform!(vis[name]["shaft"], shaft_scaling)

    head_length = vec_length - shaft_length
    head_radius = max_head_radius * head_length / max_head_length
    head_scaling = LinearMap(Diagonal(SVector(head_radius, head_radius, head_length)))
    head_tform = Translation(shaft_length * Vec(0, 0, 1)) ∘ head_scaling
    settransform!(vis[name]["head"], head_tform)
end

"""
    create_sphere(name::AbstractString, radius::Real; color::RGBA{Float32}=RGBA{Float32}(0, 1, 0, 0.5))
    
Creates a sphere object with the specified name and radius. Color is optional.
"""
function create_sphere(name::AbstractString, radius::Real; color::RGBA{Float32}=RGBA{Float32}(0, 1, 0, 0.5))

        vtol_material = MeshPhongMaterial(color=color);
        geom = HyperSphere(Point(0.0, 0.0, 0.0),radius);
        setobject!(vis[name], geom, vtol_material);
end

"""
    close_visualization()
    
Deletes the visualization.
"""
function close_visualization()
    delete!(vis); 
end


end # end of module 