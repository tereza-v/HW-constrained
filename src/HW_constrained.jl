

# constrained maximization exercises

## portfolio choice problem

module HW_constrained

	using JuMP, NLopt, DataFrames, FactCheck, Ipopt

	export data, table_NLopt, table_JuMP

	function data(a)
		n = 3
		p = (1,1,1)
		e = (2,0,0)
		z2 = [.72,.92,1.12,1.32]
		z3 = [.86,.96,1.06,1.16]
		states = Any[]
		push!(states,repeat(z2,inner=[1],outer=[4]))
		push!(states,repeat(z3,inner=[4],outer=[1]))
		states = [ones(16) states[1] states[2]]
		d = Dict("n" => n, "p" => p, "e" => e, "z2" => z2, "z3" => z3, "states" => states)
	end

	function obj(x::Vector,grad::Vector,d::Dict,a)
		u(y) = -exp(-a*y)
		du(y) = a*exp(-a*y)
		grad[1] = du(x[1])
		grad[2] = 1/16 * sum(d["states"][i,1] * du(x[2]*d["states"][i,1] + x[3]*d["states"][i,2] + x[4]*d["states"][i,3]) for i in 1:16)
		grad[3] = 1/16 * sum(d["states"][i,2] * du(x[2]*d["states"][i,1] + x[3]*d["states"][i,2] + x[4]*d["states"][i,3]) for i in 1:16)
		grad[4] = 1/16 * sum(d["states"][i,3] * du(x[2]*d["states"][i,1] + x[3]*d["states"][i,2] + x[4]*d["states"][i,3]) for i in 1:16)
		return u(x[1]) + 1/16*sum(u(x[2]*d["states"][i,1] + x[3]*d["states"][i,2] + x[4]*d["states"][i,3]) for i in 1:16)
	end

	function constr(x::Vector,grad::Vector,d::Dict)
	  grad[1] = 1
		grad[2] = d["p"][1]
		grad[3] = d["p"][2]
		grad[4] = d["p"][3]
		return x[1] + sum(d["p"][i]*(x[i+1]-d["e"][i]) for i in 1:3)
	end

	function max_NLopt(a)
		opt = Opt(:LD_MMA, 4)
		lower_bounds!(opt, [0.; -Inf .* ones(3)])
 		xtol_rel!(opt,1e-4)
 		max_objective!(opt, (x,g) -> obj(x,g,data(a),a))
 		inequality_constraint!(opt, (x,g) -> constr(x,g,data(a)), 1e-8)
 		return optimize(opt, zeros(4))
	end

	function table_NLopt()
		return DataFrame(a=[0.5,1.0,5.0],c=[round(max_NLopt(i)[2][1],6) for i in [.5,1.0,5.0]],omega1=[round(max_NLopt(i)[2][2],6) for i in [.5,1.0,5.0]],
		omega2=[round(max_NLopt(i)[2][3],6) for i in [.5,1.0,5.0]],omega3=[round(max_NLopt(i)[2][4],6) for i in [.5,1.0,5.0]],
		fval=[round(max_NLopt(i)[1],6) for i in [.5,1.0,5.0]])
	end

	function max_JuMP(a)
		d=data(a)
		m = Model(solver=IpoptSolver())
		@variable(m, c)
		@variable(m, ω1)
		@variable(m, ω2)
		@variable(m, ω3)

		@NLobjective(m, Max, -exp(-a*c) + 1/16*sum(-exp(-a*( ω1*d["states"][i,1] + ω2*d["states"][i,2] + ω3*d["states"][i,3] )) for i in 1:16))

		@NLconstraint(m, c + d["p"][1]*(ω1-d["e"][1]) + d["p"][2]*(ω2-d["e"][2]) + d["p"][3]*(ω3-d["e"][3]) == 0)
		solve(m)
		return getobjectivevalue(m), getvalue(c), getvalue(ω1), getvalue(ω2), getvalue(ω3)
	end

	function table_JuMP()
		return DataFrame(a=[0.5,1.0,5.0],c=[round(max_JuMP(i)[2],6) for i in [.5,1.0,5.0]],omega1=[round(max_JuMP(i)[3],6) for i in [.5,1.0,5.0]],
		omega2=[round(max_JuMP(i)[4],6) for i in [.5,1.0,5.0]],omega3=[round(max_JuMP(i)[5],6) for i in [.5,1.0,5.0]],
		fval=[round(max_JuMP(i)[1],6) for i in [.5,1.0,5.0]])
	end

	# function `f` is for the NLopt interface, i.e.
	# it has 2 arguments `x` and `grad`, where `grad` is
	# modified in place
	# if you want to call `f` with more than those 2 args, you need to
	# specify an anonymous function as in
	# other_arg = 3.3
	# test_finite_diff((x,g)->f(x,g,other_arg), x )
	# this function cycles through all dimensions of `f` and applies
	# the finite differencing to each. it prints some nice output.
	function test_finite_diff(f::Function,x::Vector{Float64},tol=1e-6)
	  grad = [0.0,0,0,0]
	  f(x, grad)  # get the true gradient from the objective function
		facts("Testing whether finite differences are close to the true gradient") do

			@fact finite_diff(f,x)[1] - grad[1]  --> roughly(0; atol=tol)
			@fact finite_diff(f,x)[2] - grad[2] --> roughly(0; atol=tol)
			@fact finite_diff(f,x)[3] - grad[3] --> roughly(0; atol=tol)
			@fact finite_diff(f,x)[4] - grad[4] --> roughly(0; atol=tol)

			return
		end
	end

	# do this for each dimension of x
	# low-level function doing the actual finite difference
	function finite_diff(f::Function,x::Vector)
		fin_diff = zeros(4)
		h = 1e-6
		fin_diff[1] = (f([x[1]+h,x[2],x[3],x[4]],[0.0,0,0,0]) - f(x,[0.0,0,0,0])) / h
		fin_diff[2] = (f([x[1],x[2]+h,x[3],x[4]],[0.0,0,0,0]) - f(x,[0.0,0,0,0])) / h
		fin_diff[3] = (f([x[1],x[2],x[3]+h,x[4]],[0.0,0,0,0]) - f(x,[0.0,0,0,0])) / h
		fin_diff[4] = (f([x[1],x[2],x[3],x[4]+h],[0.0,0,0,0]) - f(x,[0.0,0,0,0])) / h
		return fin_diff
	end

	function runAll()
		println("running tests:")
		include("test/runtests.jl")
		println("")
		println("JumP:")
		display(table_JuMP())
		println("End of table JumP.")
		println("")
		println("NLopt:")
		display(table_NLopt())
		println("End of table NLopt.")
		#ok = input("enter y to close this session.")
		#if ok == "y"
		#	quit()
		#end
		println("End of homework.")
	end


end
