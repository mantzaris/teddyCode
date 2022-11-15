using Random
using InvertedIndices
using SparseArrays
using StatsBase
using LinearAlgebra
using Plots
using BenchmarkTools
using SparseArrays



#convert adj to S marix
function A2S(AdjMat)
    AdjMat += I #add the identity to the diagonal, add self-loops
    diag = Diagonal(vec(sum(AdjMat,dims=2) .^ (-1/2)))
    return (diag) * AdjMat * (diag) #return the normalized S matrix
end



#This function reduces the size of a matrix by throughing the last n_col_row_to_reduce rows-columns
#adj = aj matrix
#x = xmatrix 
#th = th_matrix
#n_col_row_to_reduce = how many rows-columns we want to drop

function reduce_size(adj, x, th, n_col_row_to_reduce)
  x_dim = size(adj, 1)
  y_dim = size(adj, 2)
  adj_reduced = adj[Not(x_dim - n_col_row_to_reduce + 1: end), Not(y_dim - n_col_row_to_reduce + 1: end)]
  s_reduced = A2S(adj_reduced)
  x_reduced = x[Not(x_dim - n_col_row_to_reduce + 1: end), :]
  Final_reduced = s_reduced * x_reduced * th
  return Final_reduced
  end
  
  #This function splits the adjacency matrix  matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
 #adj_mtrx = the adj matrix
 #n_lines_to_split is the number of rows, columns the small matrix will have.
 #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.

 function split_adj_matrix(adj_mtrx, n_lines_to_split)
  x = size(adj_mtrx, 1)
  y = size(adj_mtrx, 2)
  n = div(x, n_lines_to_split)
  dictionary = Dict()
  count = 1
  k = 1
  for i in range(1, n_lines_to_split)
    z = 1
    for j in range(1, n_lines_to_split)
      dictionary["a_$count"] = adj_mtrx[k: k + n - 1, z: j*n ]
      count = count + 1
      z = z + n
    end
    k = k + n
  end
  return(dictionary)
end


#This function splits the x matrix in smaller matrices which size is (n_lines_to_split x n_lines_to_split). 
 #a_mtrx = the a_matrix
 #n_lines_to_split is the number of rows, columns the small matrix will have.
 #The function returns a dictionary. Each key element of the dictionary is one of the smaller matrices.
  
  function split_x_matrix(x_mtrx, n_lines_to_split)
  x = size(x_mtrx, 1)
  n = div(x, n_lines_to_split)
  #print(n)
  dictionary = Dict()
  z = 1
  for i in range(1, n_lines_to_split)
    dictionary["x_$i"] = x_mtrx[z : z + n - 1 , 1:end]
    z = z + n
  end
  return(dictionary)
  end


#This matrix turns s_matrix*x_matrix (in dictionary form), to matrix
#dict = the dictionary we want to conver to matrix
#x_mtrx = x_matrix
#n_lines_to_split =  is the number of rows, columns the small matrix will have.
#It returns the s_matrix*x_matrix in a matrix form

function s_matrix_x_xmtrx_dictionary_to_matrix(dict, x_splited_length, x_mtrx, n_lines_to_split)
  i = 1 
  mtrx_reduced_dictionary = Dict()
  count = 1
  num = div(size(x_mtrx, 1), n_lines_to_split)
  while (i <= length(dict)) 
    x_y_dim = size(x_mtrx, 2)
    s = zeros(num, x_y_dim)
    k = 1 
    while k <= x_splited_length  
      s = s + dict["aj_x_$i"] 
      k = k + 1 
      i = i + 1
    end 
  mtrx_reduced_dictionary["A_$count"] = s
  count = count + 1
  end 
  s_ = vcat(mtrx_reduced_dictionary["A_1"])
  for i in range(2, length(mtrx_reduced_dictionary))
    aux = vcat(s_, mtrx_reduced_dictionary["A_$i"])
    s_ = aux
  end
  return s_ 
end


#This function reads a_matrix, x_matrix, th_matrix, and the number of rows, columns the splited matrices will have.
#a_mmtrx = a_matrix
#x_mtrx = x_matrix
#th_mtrx = th_matrix
#n_lines_to_split = number of rows, columns the small matrices will have
#it splits a_matrix, x_matrix into smaller matrices by saving the in dictionaries and also find s matrix for the splited a_matrix
#then it calculates the s_splited*x_splited, saves it in dictionary and then makes it matrix
#It calculates the s_splited*x_splite[as a matrix])*th_matrix
#It returns the s_splited*x_splite*th_matrix as a matrix

function multiply_splited_2(a_mtrx, x_mtrx, th_mtrx, n_lines_to_split)
  a_splited = split_aj_matrix(a_mtrx, n_lines_to_split)
  x_splited = split_x_matrix(x_mtrx, n_lines_to_split)
  #x_y_dim = size(x_mtrx, 2)
  aj_splited = Dict()
  for i in range(1, length(a_splited))
    aj_splited["aj_$i"] = A2S(a_splited["a_$i"])
  end
  aj_x = Dict()
  k = 1
  while k <= length(a_splited)
    for i in range(1, length(x_splited))
      aj_x["aj_x_$k"] = aj_splited["aj_$k"] * x_splited["x_$i"] 
      k = k + 1
    end
  end
  aj_x_mtrx = s_matrix_x_xmtrx_dictionary_to_matrix(aj_x, length(x_splited), x_mtrx, n_lines_to_split)
  final = aj_x_mtrx * th_mtrx
  #show(stdout, "text/plain", final)
  return(final)
end

#This function gets the divisors of a number
function get_divisors(number)
  divisors = Vector{Float64}()
  for i in range(1, div(number, 2))
    if number % i == 0
      append!(divisors, Int(i))
    end
  end
  append!(divisors, Int(number))
  return divisors
end

#adj = adjacency_matrix
#x_mtrx = x_matrix
#th_mtrx = theta_matrix (Θ)

#parameter_1 = identifies the method to use

#parameter_1 = 1 truncates the initial matrix by parameter_2 %. 
#parameter_1 = 1: drops the last parameter_2 % of rows and last parameter_2 % of columns of the INITIAL adjacency_matrix and then does calculations to save memory
#parameter_2 = the percentage we want to decrease row and column. It belongs to (0, 1) interval
#the lower the percentage, the most accurate the algorithm

#parameter_1 = 2 means that we want to split the adjacency_matrix and x_marix in smaller matrices 
#If the rows of adjacency matrix is k, then all it's divisors are saved in a list L
#parameter_2 = which element of list L to choose in order to split the adjacency matrix and the x matrix.
#For example if adjacency_matrix_columns = 10 and parameter_2 = 2, then the  adjacency_matrix will be splitted in (5 x 5) matrices
#where: 5 = adjacency_matrix_columns / parameter_2 = 10 / 2
#The lower the parameter_2 is, the more accurate the result is



function my_function(adj, x_mtrx, th_mtrx, parameter_1, parameter_2)
  #convert all input matrices to sprse matrices

  #convert adjacency_matrix to sparse matrix
  adj_ = sparse(adj)

  #Drop the zeros in the adjacency_matrix we are going to use
  adj_mtrx_to_use = dropzeros(adj_)

  #convert x_matrix to sparse matrix
  x_mtrx_ = sparse(x_mtrx)

  #Drop the zeros in the x_matrix we are going to use
  x_mtrx_to_use = dropzeros(x_mtrx_)

  #convert theta_matrix (Θ) to sparse matrix
  th_mtrx_ = sparse(th_mtrx)

  #Drop the zeros in the theta_matrix (Θ) we are going to use


  th_mtrx_to_use = dropzeros(th_mtrx_) 
  if parameter_1 == 1
    drop = Int(round(size(adj_mtrx_to_use, 1) * parameter_2))
    final_to_return = reduce_size(adj_mtrx_to_use, x_mtrx_to_use, th_mtrx_to_use, drop)
    percentage = parameter_2 * 100
    println("THIS METHOD DROPS THE $percentage% OF THE RANDOM ROWS AND COLUMNS OF THE ADJACENCY MATRIX  AND THE $percentage ROWS OF X MATRIX")
  elseif parameter_1 == 2
    n_lines_to_split = get_divisors(size(adj_mtrx_to_use, 2))[parameter_2]
    print(n_lines_to_split)
    final_to_return = multiply_splited_2(adj_mtrx_to_use, x_mtrx_to_use, th_mtrx_to_use, Int(n_lines_to_split))
  elseif parameter_1 == 3
    Random.seed!(1)
    for i in range(1, Int(round(size(t, 1) * parameter_2)))
      random_number = rand(1: size(adj_mtrx_to_use, 1))
      adj_mtrx_to_use = adj_mtrx_to_use[1:end .!= random_number, 1:end .!= random_number]
      x_mtrx_to_use = x_mtrx_to_use[1:end .!= random_number, 1:end]
    end
    s_matrix = A2S(adj_mtrx_to_use)
    final_to_return = s_matrix * x_mtrx_to_use * th_mtrx_to_use
    percentage = parameter_2 * 100
    println("THIS METHOD DROPS THE $percentage% OF THE RANDOM ROWS AND COLUMNS OF THE ADJACENCY MATRIX  AND THE $percentage% ROWS OF X MATRIX")
  end 
  return dropzeros(sparse(final_to_return))
end

