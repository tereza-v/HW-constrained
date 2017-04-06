


module AssetTests

	using Base.Test, HW_constrained, FactCheck, DataFrames

	a = 0.5
	d = HW_constrained.data(a)
	x = [1.007995,-1.411361,0.801329,1.602037] # optimization results from NLopt
	gr = zeros(4)

	@testset "testing components" begin

		@testset "finite differences" begin
			@test length(HW_constrained.finite_diff((x,gr) -> HW_constrained.obj(x,gr,d,a),x)) == length(x)
			for i in 1:4
				@test HW_constrained.finite_diff((x,gr) -> HW_constrained.obj(x,gr,d,a),x)[i] < 1 # gradient values shouldn't be too big at the solution
			end
		end

		gradient = zeros(4)
		HW_constrained.obj(x,gradient,d,a) # modify gradient in place

		@testset "tests gradient of objective function" begin
			@test length(gradient) == length(x)
			for i in 1:4
				@test gradient[i] != 0.0 # gradient values should be now modified
			end
		end

		gradient = zeros(4)
		HW_constrained.constr(x,gradient,d) # modify gradient in place of constraint function

		@testset "tests gradient of constraint function" begin
			@test length(gradient) == length(x)
			for i in 1:4
				@test gradient[i] != 0.0 # gradient values should be modified now
			end
		end

		truth = DataFrame(a=[0.5,1.0,5.0],c=[1.00801,1.00401,1.008],omega1=[-1.41237,-0.206197,0.758762],
			omega2=[0.801458,0.400729,0.0801456],omega3=[1.60291,0.801462,0.160291],fval=[-1.20821,-0.732819,-0.013422])
		tol = 1e-2

		@testset "testing result of both maximization methods" begin

			@testset "checking result of NLopt maximization" begin

				@test_approx_eq_eps HW_constrained.table_NLopt()[2][1] truth[2][1] tol
				for j in 2:6
					for i in 1:3
						@test_approx_eq_eps HW_constrained.table_NLopt()[j][i] truth[j][i] tol
					end
				end

			end


			@testset "checking result of JuMP maximization" begin
				for j in 2:6
					for i in 1:3
						@test_approx_eq_eps HW_constrained.table_JuMP()[j][i] truth[j][i] tol
					end
				end
			end

			println("Test_finite_diff running:")
			HW_constrained.test_finite_diff((x,gr) -> HW_constrained.obj(x,gr,d,a),x)

		end


	end

end
