include("./othello.jl");
include("./model.jl");
using Bits
using Flux

model = Dense(128, 64, tanh)

toplane(a::UInt64) = reshape(bits(a), 8, 8, 1, 1)
input(a::Game) = cat(toplane(a.a), toplane(a.b), dims=3)
output(a::Game) = model(input(a))
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

function generate_traindata!(data)
    temp = []
    turn = false
    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        else
            turn = !turn
        end

        push!(temp, (g, turn))

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
    value_data = (x[1], (x[2] == turn ? score : -score)) for x in temp
    append!(data, value_data)
end

opt = RADAMW(0.1, (0.9, 0.999), 1)
epsilon = 0.2

for i in 1:10000000000000

    testmode!(model)

    train_data = []

    while length(x_train) <= 256
        generate_traindata!(train_data)
    end

    #model |> gpu
    global opt
    
    testmode!(model, false)

    parameters = params(model)
    
    loader = DataLoader(train_data, batchsize=64, shuffle=true)

    loss(x, y) = Flux.Losses.mse(model(input(x)), y)
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
