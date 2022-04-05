include("./othello.jl");
include("./model.jl");
using Bits
using Flux
#using CUDA

nf = 32
#model1 = Chain(Dense(128, 256, gelu), Dense(256, 64, tanh))
#model2 = Chain(Dense(128, 256, gelu), Dense(256, 64, tanh))

model1 = Chain(block(nf), block(nf), block(nf), block(nf), Conv((1, 1), nf => 1, tanh, pad=SamePad()),)
model2 = Chain(block(nf), block(nf), block(nf), block(nf), Conv((1, 1), nf => 1, tanh, pad=SamePad()),)

toplane(a::UInt64) = reshape(bits(a), 8, 8, 1, 1)
#input(a::Game) = vcat(bits(a.a), bits(a.b))
input(a::Game) = cat(toplane(a.a), toplane(a.b), zeros(Float32, 8, 8, nf - 2), dims=3)
#tood : add tanh
output(x) = x[:, :, 1:1, :]


#higher is better
function against_random()
    value(a::Game) = sum(model1(input(a)))
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

function generate_traindata!(value_net, act_net, position, value)
    g = init()
    while notend(g)
        if rawmoves(g) == 0
            g = flip(g)
        end

        if rand() < epsilon
            move = rand_agent(g)
        else
            move = agent(g) do x
                return sum(output(act_net(input(x))))
            end
        end

        push!(position, g)
        g = g + move
        push!(value, -output(value_net(input(g))))
    end
    push!(position, g)
    push!(value, (x -> 2 * x - 1).(toplane(g.a)))
end

opt = RADAMW(0.1, (0.9, 0.999), 1)
epsilon = 0.2

for i in 1:10000000000000
    model1 |> gpu
    model2 |> gpu

    testmode!(model1)
    testmode!(model2)

    x_train1 = []
    y_train1 = []

    while length(x_train1) <= 256
        generate_traindata!(model2, model1, x_train1, y_train1)
    end

    x_train2 = []
    y_train2 = []

    while length(x_train2) <= 256
        generate_traindata!(model1, model2, x_train2, y_train2)
    end

    x_train1 = map(input, x_train1)
    x_train2 = map(input, x_train2)

    x_train1 = cat(x_train1..., dims=4)
    y_train1 = cat(y_train1..., dims=4)

    x_train2 = cat(x_train2..., dims=4)
    y_train2 = cat(y_train2..., dims=4)

    global opt

    testmode!(model1, false)
    testmode!(model2, false)

    parameters1 = params(model1)
    data1 = [(x_train1, y_train1)]
    loss1(x, y) = Flux.Losses.mse(output(model1(x)), y)
    evalcb1() = @show(loss1(x_train1, y_train1))

    for epoch in 1:5
        Flux.train!(loss1, parameters1, data1, opt, cb=evalcb1)
    end

    parameters2 = params(model2)
    data2 = [(x_train2, y_train2)]
    loss2(x, y) = Flux.Losses.mse(output(model2(x)), y)
    evalcb2() = @show(loss2(x_train2, y_train2))

    for epoch in 1:5
        Flux.train!(loss2, parameters2, data2, opt, cb=evalcb2)
    end

    if i % 5 == 0
        t = (x -> against_random()).(1:10)
        print(sum(t) / length(t), "\n")
    end
    # open("model/weights.txt", "a+") do io
    #    write(io, string(params(model)))
    # end;

    # model |> cpu
end
