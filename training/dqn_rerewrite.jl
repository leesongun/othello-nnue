include("./othello.jl");
include("./model.jl");
using Bits
using Flux


nf = 3
#model = Dense(128, 64, tanh)
model = Chain(
    #Conv((1, 1), 2 => nf, mish, pad=SamePad()),
    block(nf), #block(nf), 
    Conv((1, 1), nf => 1, tanh, pad=SamePad()),
    )

toplane(a::UInt64) = reshape(bits(a), 8, 8, 1, 1)
input(a::Game) = cat(toplane(a.a), toplane(a.b), zeros(Float32, 8, 8, nf - 2, 1), dims=3)
input(a::Vector{Game}) = cat((input).(a)..., dims = 4)
output(a::Game) = model(input(a))
output(a::Vector{Game}) = model(input(a))
value(a::Game) = sum(output(a))

#higher is better
function against_random()
    turn = rand(Bool)

    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        else
            turn = !turn
        end
        if turn
            move = model_agent(g, value)
        else
            move = rand_agent(g)
        end
        g = g + move
    end
    score = count_ones(g.a)
    if turn
        score = 64 - score
    end
    return score
end

function generate_traindata!(positions, values)
    turns = []
    turn = false
    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        else
            turn = !turn
        end

        push!(positions, g)
        push!(turns, turn)

        pair = (x->(x,-value(g + x))).(moves(g))
        maxmove = findmax((x->sum(x[2])).(pair))[2]
        
        if rand() < epsilon
            move = rand(pair)[1]
        else
            move = pair[maxmove][1]
        end
        
        g = g + move
    end
    score = toplane(g.b) .* 2 .- 1
    value_data = ((x == turn ? score : -score) for x in turns)
    append!(values, value_data)
end

opt = RADAMW(0.1, (0.9, 0.999), 1)
epsilon = 0.2

for i in 1:10000000000000

    testmode!(model)

    x_train::Vector{Game} = []
    y_train = []
    #y_train::Vector{Array{Int, 4}} = []

    while length(x_train) <= 256
        generate_traindata!(x_train, y_train)
    end

    #model |> gpu
    global opt
    
    testmode!(model, false)

    parameters = params(model)

    data = (x_train, cat(y_train..., dims=4))
    loader = Flux.Data.DataLoader(data, batchsize=64, shuffle=true)

    loss(x, y) = Flux.Losses.mse(output(x), y)
    #loss(x) = Flux.Losses.mse(model(cat(a->a[1] for a in x, dims=4)), x[2])
    #evalcb() = @show(loss(x_train, y_train))

    for epoch in 1:2
        Flux.train!(loss, parameters, loader, opt)
    end

    if i % 5 == 0
        t = (x -> against_random()).(1:10)
        print(sum(t) / length(t), "\n")
    end

    #model |> cpu
end
