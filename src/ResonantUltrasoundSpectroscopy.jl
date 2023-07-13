module ResonantUltrasoundSpectroscopy

using PrecompileTools
using Memoize
using LinearAlgebra

###################
# utility functions
###################

@memoize function double_factorial(n::Int64)::Int64
    if n < 1
        return 1
    else
        return n * double_factorial(n-2)
    end
end
function volume_integral(dimensions::Vector{Float64},l::Int64,m::Int64,n::Int64,shape::Int64)::Float64
    if (l%2==1)||(m%2==1)||(n%2==1)
        return 0.0
    end
    ds = dimensions[1]^(l+1) * dimensions[2]^(m+1) * dimensions[3]^(n+1)
    if shape == 2
        df_lm = double_factorial(l-1) * double_factorial(m-1)
        return 4*pi*ds/(n+1)*df_lm/double_factorial(l+m+2)
    elseif shape == 3
        df_lm = double_factorial(l-1) * double_factorial(m-1)
        df_all = double_factorial(l+m+n+3)
        return 4*pi*ds*df_lm*double_factorial(n-1)/df_all
    else
        return 8.0 / ((l+1) * (m+1) * (n+1)) * ds
    end
end

#############################
# crystal structure functions 
#############################  

abstract type Structure end
abstract type Hexagonal <: Structure end
struct Isotropic <: Structure end
struct Cubic <: Structure end
struct Tetragonal <: Structure end 
struct Orthorhombic <: Structure end
struct VTI <: Hexagonal end 
struct HTI <: Hexagonal end

crystal_type_by_keys = Dict(
    Set([(1,1),(4,4)]) => Isotropic(),
    Set([(1,1),(1,2),(4,4)]) => Cubic(),
    Set([(3,3),(2,3),(1,2),(4,4),(6,6)]) => VTI(),
    Set([(1,1),(3,3),(1,2),(4,4),(6,6)]) => HTI(),
    Set([(1,1),(3,3),(1,2),(2,3),(4,4),(6,6)]) => Tetragonal(),
    Set([(1,1),(2,2),(3,3),(2,3),(1,3),(1,2),(4,4),(5,5),(6,6)]) => Orthorhombic()
);
crystal_keys_by_type = Dict(value => key for (key, value) in crystal_type_by_keys);
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64})
    kind = crystal_type_by_keys[Set(keys(cxx))]
    forward_cm!(cm,cxx,kind)
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::Orthorhombic)
    for i in 1:6
        cm[i,i] = cxx[(i,i)]
    end
    for i in 1:2, j in i+1:3
        cm[j,i] = cm[i,j] = cxx[(i,j)]
    end
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::Tetragonal)
    cm[1,1] = cm[2,2] = cxx[(1,1)]
    cm[3,3] = cxx[(3,3)]
    cm[3,2] = cm[1,3] = cm[3,1] = cm[2,3] = cxx[(2,3)]
    cm[2,1] = cm[1,2] = cxx[(1,2)]
    cm[5,5] = cm[4,4] = cxx[(4,4)]
    cm[6,6] = cxx[(6,6)]
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::HTI)
    for i in [1,3,4,6]
        cm[i,i] = cxx[(i,i)]
    end
    cm[1,2] = cxx[(1,2)]
    cm[2,3] = cm[3,2] = cm[3,3] - 2*cm[4,4]
    cm[1,3] = cm[2,1] = cm[3,1] = cm[1,2]
    cm[2,2] = cm[3,3]
    cm[5,5] = cm[6,6]
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::VTI)
    for i in [3,4,6]
        cm[i,i] = cxx[(i,i)]
    end
    cm[1,2] = cxx[(1,2)]
    cm[2,3] = cxx[(2,3)]
    cm[1,1] = cm[2,2] = 2.0 * cm[6,6] + cm[1,2]
    cm[1,3] = cm[3,1] = cm[3,2] = cm[2,3]
    cm[2,1] = cm[1,2]
    cm[5,5] = cm[4,4]
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::Cubic)
    for i in [1,4]
        cm[i,i] = cxx[(i,i)]
    end
    cm[1,3] = cm[2,3] = cm[3,1] = cm[3,2] = cm[2,1] = cm[1,2] = cxx[(1,2)]
    cm[2,2] = cm[3,3] = cm[1,1]
    cm[5,5] = cm[6,6] = cm[4,4]
    return Nothing
end
function forward_cm!(cm::Matrix{Float64},cxx::Dict{Tuple{Int64,Int64},Float64},kind::Isotropic)
    for i in [1,4]
        cm[i,i] = cxx[(i,i)]
    end
    cm[2,2] = cm[3,3] = cm[1,1]
    cm[5,5] = cm[6,6] = cm[4,4]
    cm[1,2] = cm[1,3] = cm[2,3] = cm[2,1] = cm[3,1] = cm[3,2] = cm[1,1] - 2.0 * cm[4,4]
    return Nothing
end

function _stiffness_index(i::Int64,j::Int64)
    [0 5 4;
     5 1 3;
     4 3 2][i,j] +1
end
function stiffness(cm::Matrix{Float64})::Array{Float64,4}
    c = zeros(3,3,3,3)
    for i in 1:3, j in 1:3, k in 1:3, l in 1:3
        a = _stiffness_index(i,j)
        b = _stiffness_index(k,l)
        c[i,j,k,l] = cm[a,b]
    end
    return c
end


##############################
# matrix  generating functions
##############################

function index_relationship(d::Int64,s::Int64)::Tuple{Matrix{Int64},Vector{Int64}}
    tabs = zeros(Int64,(s,4))
    irk = zeros(Int64,8)
    ir = 1
    for k in 1:8, i in 1:3, l in 0:d, m in 0:d-l, n in 0:d-l-m
        a = i==1 ? k>4 : k<=4
        b = i!=2 ? mod(k-1,4)>=2 : mod(k-1,4)<2
        c = i==3 ? mod(k-1,2)==0 : mod(k-1,2)==1
        if mod.([l,m,n],2) == [a,b,c] 
            tabs[ir,:] = [i,l,m,n]
            ir += 1
            irk[k] += 1
        end
    end
    return (tabs,irk)
end

function symmetrize(A::Matrix{Float64})::Matrix{Float64}
    # uses the upper diagonal to force matrix to be symmetric.
    return Matrix{Float64}(Symmetric(A,:U))
end

function generate_elastic_and_resonance_matrices(
        tabs::Matrix{Int64},irk::Vector{Int64},
        cxx::Dict{Tuple{Int64, Int64}, Float64},
        rho::Float64,dimensions::Vector{Float64},shape::Int64
    )::Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}}}
    # definitions
    E = [zeros(i,i) for i in irk]
    Γ = [zeros(i,i) for i in irk]
    cm = zeros(6,6)
    forward_cm!(cm,cxx)
    c = stiffness(cm)
    # iteration
    irk_cumsum = cumsum(irk)
    prepend!(irk_cumsum,0)
    for k in 1:8
        for (a,i) in enumerate(irk_cumsum[k]+1:irk_cumsum[k+1])
            for (b,j) in enumerate(irk_cumsum[k]+1:irk_cumsum[k+1])
                if b>=a # only the upper diagonal, as all matrices here are symmetric
                    I,lmn_1... = tabs[i,:]
                    J,lmn_2... = tabs[j,:]
                    lmn = lmn_1 .+ lmn_2
                    # fill E
                    if (I==J)&&(mod.(lmn,2)==[0,0,0])
                        E[k][a,b] = rho * volume_integral(dimensions,lmn...,shape)
                    end
                    # fill Γ
                    # we only call volume_integral if we are sure it will be nonzero
                    for μ=1:3, ν=1:3
                        st = c[I,μ,J,ν]
                        if st != 0 # as is a multiplier of volintegral
                            x = lmn_1[μ]
                            y = lmn_2[ν]
                            if (x>0)&&(y>0) # another multiplier
                                LMN = copy(lmn)
                                LMN[μ]-=1
                                LMN[ν]-=1
                                if mod.(LMN,2) == [0,0,0] # see volume_integral for parity conditions
                                    Γ[k][a,b] += st*x*y * volume_integral(dimensions,LMN...,shape)
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    # symmetrize? 
    E = symmetrize.(E)
    Γ = symmetrize.(Γ)
    return E,Γ
end

#######################
# calculation functions 
#######################

function get_freqs(nfreq::Int64,
    order::Int64,cxx::Dict{Tuple{Int64, Int64}, Float64},
    rho::Float64,dimensions::Vector{Float64},shape::Int64)::Vector{Float64}
    d2 = dimensions ./ 2
    a = order + 1
    b = order + 2
    c = order + 3
    problem_size = 3*a*b*c ÷ 6
    tabs,irk = index_relationship(order,problem_size)
    e,g = generate_elastic_and_resonance_matrices(tabs,irk,cxx,rho,dimensions,shape)
    w = Vector{Float64}[]
    for k in 1:8
        push!(w,eigvals(g[k],e[k]))
    end
    w_flat = vcat(w...)
    filter!(x->x>0,w_flat)
    freqs = sqrt.(w_flat) ./ (2pi)
    filter!(x->x>1e-5,freqs)
    partialsort!(freqs,nfreq)
    return freqs[1:nfreq]
end


################
# Precompilation
################

function _sample_cxx(kind::N) where N <: Structure
    k = crystal_keys_by_type[kind]
    return Dict([(x,1.0*sum(x)) for x in k])
end


@setup_workload begin
    order = 2
    a = order + 1
    b = order + 2
    c = order + 3
    problem_size = 3*a*b*c ÷ 6
    c_out = zeros(6,6)
    _cxx = _sample_cxx(Isotropic())
    @compile_workload begin
        tabs,irk = index_relationship(order,problem_size)
        for kind in [Isotropic(),Cubic(),VTI(),HTI(),Tetragonal(),Orthorhombic()]
            _cxx = _sample_cxx(kind)
            forward_cm!(c_out,_cxx)
        end
        for i=1:3
            e,g = generate_elastic_and_resonance_matrices(tabs,irk,_cxx,1.0,[1.,2.,3.],i);
            get_freqs(10,order,_cxx,1.,[1.,2.,3.],i)
        end
    end
end

export get_freqs, index_relationship, forward_cm!, stiffness, generate_elastic_and_resonance_matrices
export Isotropic, Cubic, VTI, HTI, Tetragonal, Orthorhombic


end # module ResonantUltrasoundSpectroscopy
