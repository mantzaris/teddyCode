using Plots
using LinearAlgebra
using BenchmarkTools
import Random

function make_mtrx(dimentions, list_1, list_2)
  m = zeros(dimentions, dimentions)
  for i in list_1
    for j in list_2
      m[i, j] = 1
    end
  end
  return m
end


function get_zeros_non_zeros(mtrx, x_dim, y_dim)
  s = 0
    for i in range(1, x_dim)
      for j in range(1, y_dim)
        if (a[i, j] == 0.0)
          s = s + 1
        end
      end
    end
    println("Zeros: $s and Non-zeros: ", x_dim * y_dim - s)
  return s
end

function random_matrix(a, b, x_dim, y_dim)
  Random.seed!(1234)
  mtrx = rand(a:b, x_dim, y_dim )
  print(mtrx)
  return mtrx
end

function get_each_element(mtrx, x_dim, y_dim)
  for i in range(1, x_dim)
    for j in range(1, y_dim)
     print("mtrx[$i, $j]  = ", mtrx[i, j], "\n")
    end
  end
end


aa = make_mtrx(80, (1, 2, 3, 30, 22, 14, 5, 8, 34, 42, 48, 68, 49, 32, 22, 14), (2, 7, 9, 11, 15 ,23, 45, 66, 8, 60, 65, 30, 77, 14, 2, 80))
a = random_matrix(0, 1, 80, 80)

x = random_matrix(-4, 4, 80, 10)
th = random_matrix(-2, 2, 10, 5)

A = a* x * th
get_zeros_non_zeros(A, 80, 5)
get_each_element(A, 80, 5)
get_zeros_non_zeros(A, 80, 5)
