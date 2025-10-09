
using FFTW, LinearAlgebra, BenchmarkTools, Statistics, Printf

cheb_nodes(n) = cos.((0:n) .* (pi/n))
f(x) = exp(-x) * sin(pi*x)

function cheb_coeffs_via_dct1!(c, y, plan)
    mul!(c, plan, y)
    n = length(y) - 1
    c .*= 2.0/n
    c[1] *= 0.5
    c[end] *= 0.5
    return c
end

function run_trials(n::Int; repeats::Int=20)
    x = cheb_nodes(n); y = f.(x); c = similar(y)
    plan = plan_dct(y, 1; flags=FFTW.ESTIMATE)
    cheb_coeffs_via_dct1!(c, y, plan) # warmup
    times = Float64[]
    for _ in 1:repeats
        x = cheb_nodes(n); y = f.(x)
        t = @belapsed cheb_coeffs_via_dct1!($c, $y, $plan)
        push!(times, t)
    end
    return times
end

function summarize(times)
    m = mean(times); sd = std(times); med = median(times)
    ci = 1.96 * sd / sqrt(length(times))
    (; mean_s=m, std_s=sd, median_s=med, min_s=minimum(times), max_s=maximum(times), n_samples=length(times), ci95_s=ci)
end

function main()
    sizes = [64,128,256,512,1024,2048,4096]; repeats = 20
    @printf("%5s, %12s, %12s, %12s, %12s, %12s, %4s, %12s
",
            "n","mean_s","std_s","median_s","min_s","max_s","N","ci95_s")
    for n in sizes
        times = run_trials(n; repeats=repeats)
        s = summarize(times)
        @printf("%5d, %12.6e, %12.6e, %12.6e, %12.6e, %12.6e, %4d, %12.6e
",
                n, s.mean_s, s.std_s, s.median_s, s.min_s, s.max_s, s.n_samples, s.ci95_s)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
