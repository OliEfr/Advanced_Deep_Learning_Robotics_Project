function testRotationMatrix(R)

    determinante = (det(R) ≈ 1)
    #inverse = (transpose(R) == inv(R))
    return determinante# && inverse
end




function projectSO3C(R)
    
    R_sym = transpose(R) * R
    eigval = eigvals(R_sym)
    eigvec = eigvecs(R_sym)
    
    u = reverse(eigvec, dims=2)
    
    sign = 1;
    if (det(R) < 0)
        sign = -1;
    end
    
    dMat = [eigval[3]^(-0.5); eigval[2]^(-0.5); sign*eigval[1]^(-0.5)]
    
    R_SO3 = R * u * Diagonal(dMat) * transpose(u)
    
    if !testRotationMatrix(R_SO3)
        println("ERROR: Rotation not on SO3")
    end
    return R_SO3
end


        # TODO: Docu !!!
        # Projection on to SO3 (Projection onto Essential Space: https://vision.in.tum.de/_media/teaching/ss2016/mvg2016/material/multiviewgeometry5.pdf)
        #F = svd(exponential_map)
        #σ = (F.S[1] + F.S[2]) / 2.0
        #S = [σ, σ, 0.0]
        #A = F.U * Diagonal(S) * F.Vt



denormalize_data(x,mean,std) = (x .* (std .+ 1e-5)) .+ mean;
normalize_data(x, mean, std) = (x .- mean) ./ (std .+ 1e-5);


Quat2Rot(q) = transpose([2*(q[1]^2 + q[2]^2)-1 2*(q[2]*q[3] + q[1]*q[4]) 2*(q[2]*q[4] - q[1]*q[3]);
                     2*(q[2]*q[3] - q[1]*q[4]) 2*(q[1]^2 + q[3]^2)-1 2*(q[3]*q[4] + q[1]*q[2]);
                     2*(q[2]*q[4] + q[1]*q[3]) 2*(q[3]*q[4] - q[1]*q[2]) 2*(q[1]^2 + q[4]^2)-1])
