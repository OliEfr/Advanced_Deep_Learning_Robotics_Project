module RigidBodies

using LinearAlgebra;
using StaticArrays;
using CoordinateTransformations
using Rotations

using ReferenceFrameRotations



export vtol_add_wind, rigid_body_simple


"""
    rigid_body_simple(torque_B, force_B, x_W_0, v_B_0, R_W_0, ω_B_0, t_0, Δt, vtol_parameters)

Simple rigid body simulation which is differentiable e.g. with Zygote.jl. Takes the state at reference time 0 ``(\\bm{x}_0^W, \\bm{v}_0^B,\\bm{R}_0^W, \\bm{ω}_0^B, t_0)``, the forces that should act in the next step ``\\bm{τ}^B, \\bm{f}^B`` and the step width ``Δt``. Returns the state of the next time step 1. 
A superscript W indicates the world coordinate system, a B indicates the VTOL body coordinate system.

**Translation - velority**

``  m \\dot{\\bm{v}}^W  = ( \\bm{R}^W  f^B + m  \\bm{g}^W )  ``

`` \\hspace{3em}  \\rightarrow d\\bm{v}^W = ( \\bm{R}^W_0  f^B + m  \\bm{g}^W )  Δt \\frac{1}{m} ``

`` \\bm{v}^W_0 = \\bm{R}^W_0  \\bm{v}^B_0 `` 

`` \\bm{v}^W_1 = \\bm{v}^W_0  + d\\bm{v}^W ``


**Translation - position**

`` d\\bm{x}^W = \\bm{v}^W_1 Δt ``

`` \\bm{x}^W_1 = \\bm{x}^W_0  + d\\bm{x}^W  ``


**Rotation - velority**

(principle of angular momentum)

`` J \\dot{\\bm{ω}}^B = τ^B - \\bm{ω}^B \\times J \\bm{ω}^B + \\bm{l}^B \\times (m \\bm{R}^B \\bm{g}^W)   ``   

`` \\hspace{3em}  \\rightarrow   d\\bm{ω}^B = J^{-1} ( τ^B - \\bm{ω}^B_0 \\times J \\bm{ω}^B_0  + \\bm{l}^B \\times (m (\\bm{R}^W_0)^\\top \\bm{g}^W) ) Δt ``

`` d\\bm{ω}^B = J^{-1} ( τ^B - \\bm{ω}^B_0 \\times J \\bm{ω}^B_0 ) Δt `` 

`` \\bm{ω}^B_1 = \\bm{ω}^B_0  + d\\bm{ω}^B  ``

**Rotation - damping**

The rotation is damped by the air resistance. The damping coefficient is calculated as follows.

`` F_{air}(r,ω) = c_w  h(r) \\frac{ρ}{2} v(r)^2 ``

`` v(r) = 2 ω r `` and `` τ = F r `` 

`` τ_{air}(ω) = \\int_0^{R} c_w  h(r) \\frac{ρ}{2} (2 ω r)^2 r dr = c_w ρ \\int_0^{R}   h(r) r^3 dr   ω^2  `` 

`` R `` is the distance between the centre of the drone and the end of the surface which is ortogonal to the rotation. ``h(r)`` is the height of the surface at radius ``r``. `` ρ `` The air density and `` c_w `` the drag coefficient of a flat plate.

The damping coefficient `` D  = c_w ρ \\int_0^{R}   h(r) r^3 dr ``.

**Rotation - orientation**

For the rotation, an integration was chosen which takes us along a paths that is constrained to rotation group. [The description can be found here](https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/).

A constant angular acceleration was assumed for the rotation. With an approximation of the Magnus expansion by its first three therms we get our rotation in the tangent space ``\\mathfrak{so}(3,\\mathbb{R})``, to the Lie-group  ``SO(3,\\mathbb{R})`` at the identity.

``\\bm{Ω}_1 = \\frac{1}{2} (\\bm{ω}^B_0 + \\bm{ω}^B_1) Δt``

``\\bm{Ω}_2 = \\frac{1}{12} (\\bm{ω}^B_1 \\times \\bm{ω}^B_0) Δt^2``

``\\bm{Ω}_3 = \\frac{1}{240} (d\\bm{ω}^B \\times (d\\bm{ω}^B \\times \\bm{ω}^B_0)) Δt^5``

``\\bm{Ω}^W = \\bm{R}^W_0 ( \\bm{Ω}_1 + \\bm{Ω}_2 + \\bm{Ω}_3)``


With the exponential map `` exp(\\bm{Ω}^W_\\times) ``evaluated through Rodrigues’ Formula, the Lie-algebra ``\\bm{Ω}^W_\\times \\in \\mathfrak{so}(3)`` is maped back to the Lie-group `` exp(\\bm{Ω}^W_\\times) \\ in SO(3)``.


`` exp(\\bm{Ω}^W_\\times) = \\bm{I}^{3 \\times 3} + \\frac{\\bm{Ω}^W_\\times}{\\| \\bm{Ω}^W \\|} sin(\\| \\bm{Ω}^W \\|) + \\frac{(\\bm{Ω}^W_\\times)^2}{\\| \\bm{Ω}^W \\|^2} (1 - cos(\\| \\bm{Ω}^W \\|)) ``.

Where ``\\bm{Ω}^W_\\times`` is the screw symmetric cross product matrix

``\\bm{Ω}^W_\\times = \\left( \\begin{array}{rrr} 0 & -Ω^W[3] & Ω^W[2] \\\\ Ω^W[3] & 0 & -Ω^W[1] \\\\ -Ω^W[2] & Ω^W[1] & 0 \\end{array} \\right)``


The mapping of the old rotation onto the new one is then given by.

`` R^W_1 = exp(\\bm{Ω}^W_\\times) R^W_0 ``

**Time**

`` t_1 = t_0 + Δt ``

"""
function rigid_body_simple(torque_B, force_B, x_W_0, v_B_0, R_W_0, ω_B_0, t_0, Δt, vtol_parameters)
    # Variable naming   attribute_frame_time
    # frame ... Body or World
    # time  ... 0    or 1

    # Limit velocities for numerical stability
    v_B_0 = max.(-100.0, min.(v_B_0, 100.0)) # ±100 m/s
    ω_B_0 = max.(-500.0, min.(ω_B_0, 500.0)) # ±500 rad/s ≈ ±80 rotations/s


    gravity = vtol_parameters["gravity"]
    J_B = vtol_parameters["inertia"]
    J_B = vtol_parameters["inertia"]
    J_B_inv = vtol_parameters["inertia_inv"]
    mass = vtol_parameters["mass"]
    rotation_damping = vtol_parameters["rotation_damping"]

    # --------- Translation ------------------------------------
    gravity_W = [0.0, 0.0, -gravity]
    a_W_1 = (R_W_0 * force_B + (mass .* gravity_W)) ./ mass
    v_W_0 = R_W_0 * v_B_0 # transform Body Velocity in World frame
    v_W_1 = v_W_0 + Δt * a_W_1 # integrate velocity
    
    dx_W = v_W_1 * Δt # position change
    x_W_1 = x_W_0 .+ dx_W # integrate position

    
    # --------- Rotation ------------------------------------
    torque_damping = sign.(ω_B_0) .* (ω_B_0 .^ 2) .* rotation_damping;
    α_B_1 = J_B_inv*(torque_B - torque_damping - LinearAlgebra.cross(ω_B_0, J_B * ω_B_0));
    ω_B_1 = ω_B_0 + α_B_1 * Δt;

    
    # first three elements of the Magnus expansion (only an approximation !!!)
    Ω_1 = (1/2)*(ω_B_0 + ω_B_1)*Δt
    Ω_W = R_W_0 * Ω_1 # Transform in world frame

    
    # Screw Symmetric Cross product matrix
    Ω_mat = [ 0.0       -Ω_W[3]   Ω_W[2];
              Ω_W[3]   0.0       -Ω_W[1];
             -Ω_W[2]   Ω_W[1]   0.0     ]
    
    Ω_norm = norm(Ω_W) # Length of vector is rotation angle, direction is rotation axis
    
    if Ω_norm == 0.0
        R_W_1 = R_W_0
    else
        # Rodrigues’ Formula maps the Lie-algebra 𝒘_mat ∈ 𝑠𝑜(3): to Lie-group 𝑅: 𝒆^𝒘_mat = 𝑹
        exponential_map = Matrix(1.0I, 3, 3) + (Ω_mat * (sin(Ω_norm) /Ω_norm)) + ((Ω_mat^2)* ((1.0 - cos(Ω_norm))/(Ω_norm^2))) 
        R_W_1 =  exponential_map * R_W_0
    end

    
    # --------- Time ------------------------------------
    t_1 = t_0 + Δt # Next time
    
    v_B_1 = transpose(R_W_1) * v_W_1; # transform World Velocity in Body frame
    a_B_1 = transpose(R_W_1) * a_W_1
    
    return x_W_1, v_B_1, a_B_1, R_W_1, ω_B_1, α_B_1, t_1
end;



"""
    vtol_add_wind(v_B::Vector{Float64}, R_W::Matrix{Float64}, wind_vector_W::Vector{Float64})

Adds the wind velocity ``\\bm{v}_w^\\mathcal{W}`` to the body velocity ``\\bm{v}_a^\\mathcal{B}``.

`` \\bm{v}_g^\\mathcal{B} = \\bm{v}_a^\\mathcal{B} - \\bm{R}_\\mathcal{W}^\\mathcal{B} \\bm{v}_w^\\mathcal{W} ``


With the velocity of the airframe relative to the ground ``\\bm{v}_g^\\mathcal{B}``,

the velocity of the air mass relative to the ground

``\\bm{v}_w^\\mathcal{W} = `` wind direction ``*`` wind speed

and the velocity of the airframe relative to the surrounding air mass ``\\bm{v}_a^\\mathcal{B}``.


"""
function vtol_add_wind(v_B::Vector{Float64}, R_W::Matrix{Float64}, wind_vector_W::Vector{Float64})
    v_in_wind_B = v_B + transpose(R_W) * wind_vector_W
    return v_in_wind_B
end



end # end of module